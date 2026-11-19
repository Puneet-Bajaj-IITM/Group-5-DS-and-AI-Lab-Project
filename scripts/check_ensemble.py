"""Does averaging the full runs' predictions help? A variance check on their saved validation outputs.

Usage: python scripts/check_ensemble.py A1=<run folder> H2=<run folder> A3=<run folder>

Each run folder is a full run's Kaggle output folder holding `val_outputs.npz` (see results/m4/artifacts.md).
Writes results/m4/bias_variance/ensemble.json. CPU only, no new predictions.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from serpentguard import metrics  # noqa: E402

RESULTS = REPO / "results/m4"
OUT = RESULTS / "bias_variance/ensemble.json"
ENSEMBLES = (("A1", "A3"), ("A1", "H2", "A3"))
BASE = "A3"
TOLERANCE = 1e-9


def macro_f1(y, predicted, val_counts):
    n_classes = len(val_counts)
    return metrics.species_report(y, np.eye(n_classes)[predicted], val_counts, val_counts,
                                  np.zeros(n_classes))["macro_f1"]


def main(folders):
    runs = {name: np.load(Path(folder) / "val_outputs.npz", allow_pickle=True) for name, folder in folders.items()}
    y, paths = runs[BASE]["labels"].astype(int), runs[BASE]["paths"]
    for name, run in runs.items():
        if not ((run["paths"] == paths).all() and (run["labels"].astype(int) == y).all()):
            raise ValueError(f"{name}'s validation rows differ from {BASE}'s")
    n_classes = runs[BASE]["species_logits"].shape[1]
    val_counts = np.bincount(y, minlength=n_classes)
    groups = np.where(np.char.strip(runs[BASE]["observer_id"].astype(str)) != "",
                      runs[BASE]["observer_id"].astype(str), paths.astype(str))
    score = lambda labels, predicted: macro_f1(labels, predicted, val_counts)  # noqa: E731
    top1 = lambda labels, predicted: float(np.mean(labels == predicted))  # noqa: E731

    result = {"photos": int(len(y)), "single": {}, "ensembles": {}}
    for name, run in runs.items():
        predicted = run["species_logits"].argmax(1)
        reported = json.loads((RESULTS / f"full/{name}/final_metrics.json").read_text())["head"]
        result["single"][name] = {"macro_f1": score(y, predicted), "top1": top1(y, predicted)}
        if abs(result["single"][name]["macro_f1"] - reported["macro_f1"]) > TOLERANCE:
            raise ValueError(f"{name}: recomputed macro-F1 differs from final_metrics.json")

    base = runs[BASE]["species_logits"].argmax(1)
    for members in ENSEMBLES:
        probs = sum(metrics.softmax(runs[name]["species_logits"]) for name in members) / len(members)
        predicted = probs.argmax(1)
        key = "+".join(members)
        result["ensembles"][key] = {
            "macro_f1": score(y, predicted), "top1": top1(y, predicted),
            f"minus_{BASE}_macro_f1": dict(zip(("diff", "low", "high"),
                                               metrics.paired_difference_ci(y, predicted, base, groups, score))),
            f"minus_{BASE}_top1": dict(zip(("diff", "low", "high"),
                                           metrics.paired_difference_ci(y, predicted, base, groups, top1)))}

    shuffled = runs["A1"]["species_logits"][np.random.default_rng(metrics.SAMPLE_SEED).permutation(len(y))]
    control = (metrics.softmax(shuffled) + metrics.softmax(runs[BASE]["species_logits"])).argmax(1)
    result["control_A1_rows_shuffled"] = {"macro_f1": score(y, control), "top1": top1(y, control)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_json(OUT, result)
    print(OUT.relative_to(REPO), {key: round(value["macro_f1"], 4) for key, value in result["ensembles"].items()})


if __name__ == "__main__":
    main(dict(argument.split("=", 1) for argument in sys.argv[1:]))
