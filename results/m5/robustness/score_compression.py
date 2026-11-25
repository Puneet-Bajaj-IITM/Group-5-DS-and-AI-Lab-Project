"""Section 9 int8 table: G3 on the G3 slice and G4 for each compressed variant, with score_test.py's metric code.

G3: score_test.g3_block (check_full_run.paired with top1 and shared_macro_f1(62), 1,000 photographer resamples,
seed 42) on frozen logits = flip average + tau * log(n_c), tau 0.5, for each ONNX variant's logits. The PyTorch row is
the locked test run itself (RESULTS.md). G4: evaluate.rule_result with the frozen cut, on each variant's P(snake).

Usage: m3-final/.venv/bin/python results/m5/robustness/score_compression.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path[:0] = [str(WORK / "results/m5/test"), str(HERE)]
import score_test as st  # noqa: E402
from score_robustness import snake_rows  # noqa: E402

cf, g4e, a = st.cf, st.g4e, st.a
COMPRESSION = HERE / "kernel_out/vikas25s/out/compression"
OUT = HERE / "results"
VARIANTS = ("fp32", "fp16", "int8_dynamic", "int8_static_minmax")


def main():
    OUT.mkdir(exist_ok=True)
    meta = st.metadata()
    paths, sets, b1_pred_by_path, _, _ = st.load_sets("test")
    y = sets["plain"]["labels"].astype(int)
    groups = cf.groups_of(sets["plain"]["observer_id"], paths)
    created = pd.Series(meta.loc[paths, "created_at"].fillna("").astype(str).to_numpy())
    b1_pred = b1_pred_by_path.reindex(paths).fillna(-1).astype(int).to_numpy()
    saved = np.load(COMPRESSION / "classifier_g3_slice.npz", allow_pickle=True)
    rows = pd.Series(np.arange(len(paths)), index=paths).loc[saved["paths"].astype(str)].to_numpy()
    assert (saved["labels"] == y[rows]).all()
    keep = np.zeros(len(paths), bool)
    keep[rows] = True
    preds = {"PyTorch (locked test run)": st.softmax(st.frozen_logits(sets)).argmax(1)}
    agreement = {}
    for variant in VARIANTS:
        if f"{variant}_plain" not in saved.files:
            continue
        logits = st.prior.corrected((saved[f"{variant}_plain"] + saved[f"{variant}_flip"]) / 2, st.TAU)
        pred = np.full(len(paths), -1)
        pred[rows] = logits.argmax(1)
        preds[f"ONNX {variant}"] = pred
        agreement[variant] = float((pred[rows] == preds["PyTorch (locked test run)"][rows]).mean())
    g3 = st.g3_block(y, groups, created, b1_pred, preds, keep)
    assert g3["photos"] == 12_203, g3["photos"]

    calib = pd.read_csv(st.G4 / "runs/distil_oi/scores_kernel/scores_calib.csv")["p_snake"].to_numpy()
    non_listing = pd.read_csv(WORK / "results/m5/test/kernel/inputs/nonsnake_test.csv")
    variants = pd.read_csv(COMPRESSION / "snake_check_variants.csv")
    locked = st.kernel_probs(pd.read_csv(st.KERNEL / "snake_check_scores.csv"))
    locked = locked[locked["set"].isin(["held_out", "nonsnake_test"])]
    g4 = {}
    for name, column in [("PyTorch (locked test run)", None)] + [(f"ONNX {v}", f"p_snake_{v}") for v in VARIANTS]:
        if column is not None and column not in variants:
            continue
        scores = locked if column is None else variants[["set", "path"]].assign(p_snake=variants[column])
        snakes, non = snake_rows(scores, meta, non_listing)
        row = g4e.rule_result(name, calib, snakes, snakes["p_snake"].to_numpy(), non, non["p_snake"].to_numpy(),
                              np.random.default_rng(a.SEED))
        g4[name] = {k: row[k] for k in ("cutoff", "snakes_kept_val", "snakes_kept_val_ci", "snake_photos",
                                        "false_snake_val", "false_snake_val_ci", "false_snake_count", "nonsnake_photos")}
    status = json.loads((COMPRESSION.parent / "status.json").read_text())["checks"]["compression"]
    report = {"g3": {k: {m: g3["candidates"][k][m] for m in ("top1", "macro_f1", "absolute", "g3")}
                     for k in g3["candidates"]}, "g3_photos": g3["photos"], "b1": g3["b1"],
              "top1_agreement_with_pytorch_on_slice": agreement, "g4": g4, "bytes": status["bytes"],
              "not_produced": status.get("not_produced", {}), "kernel_cpu": status.get("cpu")}
    (OUT / "compression.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    for k, v in report["g3"].items():
        print(k, v["top1"], v["macro_f1"], v["g3"])
    for k, v in g4.items():
        print(k, v["snakes_kept_val"], v["false_snake_val"])


if __name__ == "__main__":
    main()
