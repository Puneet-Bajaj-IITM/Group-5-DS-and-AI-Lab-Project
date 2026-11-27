"""Probe on A3's saved outputs: does a training-prior logit correction (fitted on calibration top-1) move G3?

logits + tau * log(n_c), n_c = training photos of class c after caps (the weighted sampler drew ~ sqrt(n_c)).
tau is chosen on CALIBRATION top-1 over all 184 species; validation is only scored.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "m5-targets"))
import g3_ensemble as g3  # noqa: E402
import check_full_run as cf  # noqa: E402

TAUS = np.round(np.arange(0.0, 3.01, 0.25), 2)


def counts():
    full = pd.read_csv(WORK / "kaggle/m4_runs/full_a3/out_jayesh/full/full_train.csv")
    meta = pd.read_csv(g3.METADATA, usecols=["path", "species"])
    cls = pd.read_csv(WORK / "m5-wt/data/classes.csv")
    ids = full.merge(meta, on="path").merge(cls[["species", "class_id"]], on="species")["class_id"]
    return np.bincount(ids, minlength=g3.N_CLASSES)


def main(run_dir=WORK / g3.RUNS["A3"]):
    log_n = np.log(np.maximum(counts(), 1))
    calib = cf.load(run_dir / "calib_outputs.npz")
    val = cf.load(run_dir / "val_outputs.npz")
    paths, y = val["paths"], val["labels"].astype(int)
    b1 = cf.load(cf.B1_DIR / "predictions.npz")
    shared = np.array(sorted(json.loads((cf.B1_DIR / "final_metrics.json").read_text())["coverage"]["covered_class_ids"]))
    pos = pd.Series(np.arange(len(b1["paths"])), index=b1["paths"]).reindex(paths).to_numpy(int)
    bb_all = b1["probs"].argmax(1)[pos]
    meta = pd.read_csv(g3.METADATA, usecols=["path", "created_at"])
    created = pd.DataFrame({"path": paths}).merge(meta, on="path", how="left")["created_at"].fillna("").astype(str)
    rows = created.ge(cf.SLICE_FROM).to_numpy() & np.isin(y, shared)
    groups = cf.groups_of(val["observer_id"], paths)
    macro = cf.shared_macro_f1(shared, g3.N_CLASSES)
    calib_top1 = {float(t): float(((calib["species_logits"] + t * log_n).argmax(1) == calib["labels"]).mean())
                  for t in TAUS}
    best = max(calib_top1, key=calib_top1.get)
    out = {"calibration_top1": calib_top1, "tau_chosen_on_calibration": best}
    for t in sorted({0.0, best}):
        pred = (val["species_logits"] + t * log_n).argmax(1)
        out[f"tau={t}"] = {"val_top1_all": float((pred == y).mean()),
                           **{m: cf.paired(y[rows], pred[rows], bb_all[rows], groups[rows], s)
                              for m, s in (("top1", cf.top1), ("macro_f1", macro))}}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else WORK / g3.RUNS["A3"])
