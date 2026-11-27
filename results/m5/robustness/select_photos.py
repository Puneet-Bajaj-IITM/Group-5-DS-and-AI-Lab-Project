"""Photo lists for the extra kernel, fixed before it runs (seed 42), from files already on the laptop.

- gradcam_photos.csv: 12 correctly and 12 wrongly classified held-out photos with a usable mask (frozen pipeline's
  answers in predictions_heldout.csv); the wrong ones from the top confusions in confusions_heldout.csv.
- g3_slice.csv: the G3 slice (created_at >= 2022-01-01, species among B1's 62), as score_test.g3_block builds it.
- quant_calib.csv: 500 calibration photos (M5_KD's calib_outputs, never a test photo) for static int8.
- parity_val.csv: 64 validation photos (from the kernel's validation sample) for a laptop-vs-kernel int8 parity check.

Usage: m3-final/.venv/bin/python -I results/m5/robustness/select_photos.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
TEST = WORK / "results/m5/test"
M5_KD = WORK / "results/m5/g3/out_kd/M5_KD"
B1_DIR = WORK / "kaggle/m4_runs/b1_predictions"
OUT = HERE / "lists"
SEED = 42
SLICE_FROM = "2022-01-01"
N_CORRECT, N_CALIB, N_PARITY = 12, 500, 64
G3_SLICE_PHOTOS = 12_203
WRONG = [("Ahaetulla oxyrhynca", "Ahaetulla prasina", 2), ("Ahaetulla borealis", "Ahaetulla prasina", 2),
         ("Ptyas mucosa", "Naja naja", 2), ("Naja naja", "Ptyas mucosa", 2), ("Ptyas mucosa", "Ptyas korros", 1),
         ("Craspedocephalus gramineus", "Ahaetulla prasina", 1), ("Dendrelaphis chairecacos", "Dendrelaphis tristis", 1),
         ("Lycodon fasciolatus", "Lycodon aulicus", 1)]


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    preds = pd.read_csv(TEST / "predictions_heldout.csv")
    frame = pd.read_csv(TEST / "kernel_out/out/held_out_frame.csv")
    classes = pd.read_csv(WORK / "m5-wt/data/classes.csv")
    class_of = dict(zip(classes["species"], classes["class_id"]))
    preds = preds.merge(frame[["path", "mask_valid"]], on="path", validate="one_to_one")
    masked = preds[preds["mask_valid"]]

    correct = masked[masked["species"].eq(masked["frozen_pred"])]
    species = rng.choice(np.sort(correct["species"].unique()), N_CORRECT, replace=False)
    picks = [correct[correct["species"].eq(s)].sample(1, random_state=SEED).assign(kind="correct") for s in species]
    for true, wrong, n in WRONG:
        pool = masked[masked["species"].eq(true) & masked["frozen_pred"].eq(wrong)]
        picks.append(pool.sample(n, random_state=SEED).assign(kind="misclassified"))
    chosen = pd.concat(picks, ignore_index=True)
    chosen["target_class"] = chosen["frozen_pred"].map(class_of)
    chosen[["path", "kind", "species", "frozen_pred", "target_class", "frozen_p"]].to_csv(OUT / "gradcam_photos.csv",
                                                                                         index=False)

    shared = set(json.loads((B1_DIR / "final_metrics.json").read_text())["coverage"]["covered_class_ids"])
    in_slice = preds["created_at"].fillna("").astype(str).ge(SLICE_FROM) & preds["class_id"].isin(shared)
    g3 = preds.loc[in_slice, ["path"]]
    assert len(g3) == G3_SLICE_PHOTOS, len(g3)
    g3.to_csv(OUT / "g3_slice.csv", index=False)

    # the targets of every held-out photo, for Grad-CAM's mass check (the frozen answer's class)
    preds.assign(target_class=preds["frozen_pred"].map(class_of))[["path", "target_class"]].to_csv(
        OUT / "gradcam_targets.csv", index=False)

    calib = np.load(M5_KD / "calib_outputs.npz", allow_pickle=True)["paths"].astype(str)
    assert not any(p.startswith(("test/", "test_extra/")) for p in calib)
    pd.DataFrame({"path": np.sort(rng.choice(calib, N_CALIB, replace=False))}).to_csv(OUT / "quant_calib.csv",
                                                                                      index=False)
    val = np.load(TEST / "kernel/inputs/instrument/val_sample_expected.npz", allow_pickle=True)["paths"].astype(str)
    pd.DataFrame({"path": np.sort(rng.choice(val, N_PARITY, replace=False))}).to_csv(OUT / "parity_val.csv",
                                                                                     index=False)
    print(chosen[["kind", "species", "frozen_pred"]].to_string())
    print(f"G3 slice {len(g3):,}; calibration {N_CALIB}; parity {N_PARITY}")


if __name__ == "__main__":
    main()
