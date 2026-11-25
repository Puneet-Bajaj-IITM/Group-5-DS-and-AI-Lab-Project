"""Safety for the M5 models behind the app's gate: refit the decision thresholds on calibration, then score validation.

As m5-targets/refit_search.py and eval_refit.py did for A3: decide.fit_thresholds on the run's calibration outputs,
its target raised in steps of 0.0025 until the WHO recall lower bound BEHIND THE APP'S GATE reaches 0.95 (a dropped
WHO-listed snake counts as a miss). The gate's call per photo is the app's own (m5-targets/app_outputs, the M4 "both"
gate), joined by path; it does not depend on the classifier. Calibration only for the fit; validation only scored.

Checks: the default-target fit must reproduce the thresholds the kernel's decision_report fitted (instrument), and
the refitted thresholds on calibration outputs with species and WHO logits shuffled across photos must fall below
0.95 (negative control).

Variants: "plain" (the model as trained), "prior" (species logits + tau * log training count, tau fitted on
calibration by prior.py) and "flip + prior" (every head's logits averaged with the mirrored pass first, the
embedding re-normalised). The kernel-fit instrument check applies to "plain".

Usage: python refit_new.py   (writes refit_new.json)
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "m5-wt/src"))
from serpentguard import decide  # noqa: E402

sys.path.insert(0, str(HERE))
import prior  # noqa: E402

APP = WORK / "m5-targets/app_outputs"
RUNS = {"FT": HERE / "out_finetune/M5_FT", "KD": HERE / "out_kd/M5_KD"}
MARK = 0.95
TARGETS = np.round(np.arange(0.950, 0.9801, 0.0025), 4)
SEED, RESAMPLES = 42, 1000
THRESHOLD_TOLERANCE = 1e-9
SHUFFLED = ("species_logits", "who_logits", "genus_logits", "family_logits", "embedding")
HEADS = ("species_logits", "who_logits", "genus_logits", "family_logits")
SETS = {"calib": "calib_outputs", "val": "val_outputs", "unknown": "unknown_outputs"}
VARIANTS = ("plain", "prior", "flip + prior")


def averaged(plain, flipped):
    assert (plain["paths"] == flipped["paths"]).all(), "the mirrored pass is in another order"
    out = dict(plain)
    for head in HEADS:
        out[head] = (plain[head] + flipped[head]) / 2
    embedding = plain["embedding"] + flipped["embedding"]
    out["embedding"] = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return out


def variant_outputs(run_dir, name):
    """{set: outputs} of one variant, and its tau (None for plain)."""
    sets = {n: decide.load_outputs(run_dir / f"{f}.npz") for n, f in SETS.items()}
    if name == "plain":
        return sets, None
    if name == "flip + prior":
        sets = {n: averaged(o, decide.load_outputs(run_dir / f"{SETS[n]}_flip.npz")) for n, o in sets.items()}
    tau, _ = prior.fit_tau(sets["calib"]["species_logits"], sets["calib"]["labels"])
    for o in sets.values():
        o["species_logits"] = prior.corrected(o["species_logits"], tau)
    return sets, tau


def with_gate(outputs, app_file):
    """The outputs of the photos the app path saw, in the app file's order, with its gate_snake."""
    app = decide.load_outputs(app_file)
    index = {p: i for i, p in enumerate(outputs["paths"])}
    rows = np.array([index[p] for p in app["paths"]])
    assert (outputs["labels"][rows] == app["labels"]).all(), "labels differ from the app file"
    joined = {k: v[rows] for k, v in outputs.items() if len(v) == len(outputs["paths"])}
    joined["gate_snake"] = app["gate_snake"]
    return joined


def gated(outputs, thresholds, prototypes, tables):
    decided = decide.decide(outputs, thresholds, prototypes, tables, outputs["gate_snake"].astype(bool), None)
    listed, outside = decide.listed_and_outside(outputs["labels"], tables)
    hits = decided.label[listed] == "dangerous"
    return decided, listed, outside, hits


def interval(hits, groups):
    rng = np.random.default_rng(SEED)
    owners = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in owners}
    draws = [hits[np.concatenate([index[g] for g in rng.choice(owners, len(owners))])].mean()
             for _ in range(RESAMPLES)]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def calib_row(calib, thresholds, prototypes, tables):
    decided, listed, outside, hits = gated(calib, thresholds, prototypes, tables)
    return {"bound_behind_gate": decide.recall_lower_bound(hits, calib["observer_id"][listed]),
            "recall_behind_gate": float(hits.mean()),
            "false_alarm_calibration": float((decided.label[outside] == "dangerous").mean()),
            "unknown_share_calibration": float((decided.label == "unknown").mean())}


def val_row(val, unknown, thresholds, prototypes, tables):
    decided, listed, outside, hits = gated(val, thresholds, prototypes, tables)
    unknown_decided = decide.decide(unknown, thresholds, prototypes, tables, unknown["gate_snake"].astype(bool), None)
    return {"photos": int(len(val["labels"])), "who_listed": int(listed.sum()), "who_recall": float(hits.mean()),
            "who_recall_ci": interval(hits, val["observer_id"][listed]),
            "false_alarm": float((decided.label[outside] == "dangerous").mean()),
            "verdicts": decide.verdict_counts(decided),
            "unknown_species_verdicts": decide.verdict_counts(unknown_decided)}


def refit(run_dir, tables, variant="plain"):
    prototypes = np.load(run_dir / "prototypes.npz")["plain"]
    saved, tau = variant_outputs(run_dir, variant)
    kernel_fit = decide.Thresholds(**json.loads((run_dir / "decision_metrics.json").read_text())["thresholds"])
    gaps = None
    if variant == "plain":
        again = decide.fit_thresholds(saved["calib"], prototypes, tables)
        gaps = {k: abs(v - getattr(kernel_fit, k)) for k, v in asdict(again).items() if isinstance(v, float)}
        assert max(gaps.values()) < THRESHOLD_TOLERANCE, f"the kernel's fit is not reproduced: {gaps}"
    else:  # the default-target fit of this variant stands in for the kernel's
        kernel_fit = decide.fit_thresholds(saved["calib"], prototypes, tables)
    calib, val, unknown = (with_gate(saved[n], APP / f"{n}_outputs.npz") for n in ("calib", "val", "unknown"))
    rows, chosen = [], None
    for target in TARGETS:
        thresholds = decide.fit_thresholds(calib, prototypes, tables, who_recall_target=float(target))
        rows.append({"target": float(target), **calib_row(calib, thresholds, prototypes, tables),
                     "thresholds": asdict(thresholds)})
        print(f"{run_dir.name} {variant} target {target:.4f}: bound behind gate {rows[-1]['bound_behind_gate']:.4f}", flush=True)
        if rows[-1]["bound_behind_gate"] >= MARK:
            chosen = rows[-1]
            break
    result = {"variant": variant, "tau": tau, "instrument_kernel_fit_reproduced": gaps, "calibration_photos_behind_app_path": len(calib["labels"]),
              "kernel_fit": {"calibration": calib_row(calib, kernel_fit, prototypes, tables),
                             "val_behind_app_gate": val_row(val, unknown, kernel_fit, prototypes, tables),
                             "thresholds": asdict(kernel_fit)},
              "search": rows, "chosen": chosen}
    if chosen is not None:
        refitted = decide.Thresholds(**chosen["thresholds"])
        result["refit"] = {"val_behind_app_gate": val_row(val, unknown, refitted, prototypes, tables)}
        order = np.random.default_rng(0).permutation(len(calib["labels"]))
        control = {k: (v[order] if k in SHUFFLED else v) for k, v in calib.items()}
        result["negative_control_shuffled_logits"] = calib_row(control, refitted, prototypes, tables)
        assert result["negative_control_shuffled_logits"]["bound_behind_gate"] < MARK, "the control passed"
    return result


def main():
    tables = decide.repo_tables()
    out = {f"{name} {variant}": refit(run_dir, tables, variant) for name, run_dir in RUNS.items()
           if (run_dir / "calib_outputs.npz").exists() for variant in VARIANTS}
    (HERE / "refit_new.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    for name, result in out.items():
        print(name, json.dumps({k: result[k] for k in ("kernel_fit", "chosen") if k in result}, default=str)[:1500])


if __name__ == "__main__":
    main()
