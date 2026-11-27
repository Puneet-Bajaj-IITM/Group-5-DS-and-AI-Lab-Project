"""G3 for the M5 candidates, on saved validation outputs only, with g3_ensemble.py's exact computation.

First reproduces A3's reported G3 numbers (-0.0808 / -0.0249) through the same code (the instrument check), then
scores each candidate the same way: ours minus B1 on B1's 62 species, the 11,931 photos uploaded from 2022 on,
paired photographer bootstrap (1,000 resamples, seed 42). A shuffled-predictions control must lose clearly.
"Flip" averages a model's softmax on each photo and on its mirror image. "Prior" adds tau * log(training count) to
the species logits, tau fitted on that model's calibration outputs only (prior.py); with flip, the two passes'
logits are averaged first. No model is run here.

Usage: python g3_candidates.py   (reads out_finetune/ and out_kd/ when present; writes g3_candidates.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "m5-targets"))
import g3_ensemble as g3  # noqa: E402  (also puts m4-checks on the path)
import check_full_run as cf  # noqa: E402
import prior  # noqa: E402

A3_REPORTED = {"top1": -0.0808, "macro_f1": -0.0249}
REPORTED_TOLERANCE = 5e-5
CONTROL_BELOW = -0.3
RUNS = {"FT": HERE / "out_finetune/M5_FT", "KD": HERE / "out_kd/M5_KD"}
A3_M4 = {"FT": HERE / "out_finetune/A3_m4crops", "KD": HERE / "out_kd/A3_m4crops"}
FLIP_FILE = "val_outputs_flip.npz"
CALIB_FLIP_FILE = "calib_outputs_flip.npz"


def with_prior(val_logits, calib):
    """The val logits with the prior correction whose tau is fitted on these calibration outputs."""
    tau, _ = prior.fit_tau(calib["species_logits"], calib["labels"])
    return prior.corrected(val_logits, tau), tau


def aligned(path, paths, labels):
    """A saved npz's arrays in the order of `paths`; its labels must agree."""
    saved = cf.load(path)
    position = pd.Series(np.arange(len(saved["paths"])), index=saved["paths"]).reindex(paths)
    assert position.notna().all(), f"{path}: a slice photo is missing"
    position = position.to_numpy(int)
    assert (saved["labels"][position] == labels).all(), f"{path}: labels differ"
    return {key: value[position] for key, value in saved.items() if len(value) == len(saved["paths"])}


def temperature_of(run_dir):
    return float(json.loads((run_dir / "decision_metrics.json").read_text())["thresholds"]["temperature"])


def main():
    a3 = cf.load(WORK / g3.RUNS["A3"] / "val_outputs.npz")
    paths, y = a3["paths"], a3["labels"].astype(int)
    b1 = cf.load(cf.B1_DIR / "predictions.npz")
    shared = np.array(sorted(json.loads((cf.B1_DIR / "final_metrics.json").read_text())["coverage"]
                             ["covered_class_ids"]))
    position = pd.Series(np.arange(len(b1["paths"])), index=b1["paths"]).reindex(paths).to_numpy(int)
    b1_pred = b1["probs"].argmax(axis=1)[position]
    meta = pd.read_csv(g3.METADATA, usecols=["path", "created_at"])
    created = pd.DataFrame({"path": paths}).merge(meta, on="path", how="left")["created_at"].fillna("").astype(str)
    rows = created.ge(cf.SLICE_FROM).to_numpy() & np.isin(y, shared)
    groups = cf.groups_of(a3["observer_id"], paths)
    macro_62 = cf.shared_macro_f1(shared, g3.N_CLASSES)
    yy, gg, bb = y[rows], groups[rows], b1_pred[rows]

    def score(pred):
        return {m: cf.paired(yy, pred[rows], bb, gg, s) for m, s in (("top1", cf.top1), ("macro_f1", macro_62))}

    out = {"photos": int(rows.sum()), "photographers": int(np.unique(gg).size), "candidates": {}}
    check = score(a3["species_logits"].argmax(1))
    for metric, reported in A3_REPORTED.items():
        assert abs(check[metric]["diff"] - reported) < REPORTED_TOLERANCE, f"instrument check failed on {metric}"
    out["instrument_check"] = {"A3 (shipped), M3 crops": check, "reported": A3_REPORTED}
    print(f"instrument check: A3 reproduces {A3_REPORTED} on {rows.sum():,} photos", flush=True)

    probs, logits, taus = {}, {}, {}
    a3_dir = WORK / g3.RUNS["A3"]
    corrected, taus["A3 (shipped), M3 crops + prior"] = with_prior(a3["species_logits"],
                                                                  cf.load(a3_dir / "calib_outputs.npz"))
    probs["A3 (shipped), M3 crops + prior"] = g3.softmax(corrected, 1.0)
    for name, run_dir in RUNS.items():
        if not (run_dir / "val_outputs.npz").exists():
            print(f"{name}: no outputs at {run_dir}")
            continue
        t = temperature_of(run_dir)
        plain = aligned(run_dir / "val_outputs.npz", paths, y)["species_logits"]
        flip = aligned(run_dir / FLIP_FILE, paths, y)["species_logits"]
        logits[name] = plain
        probs[name] = g3.softmax(plain, t)
        probs[f"{name} + flip"] = (g3.softmax(plain, t) + g3.softmax(flip, t)) / 2
        calib, calib_flip = cf.load(run_dir / "calib_outputs.npz"), cf.load(run_dir / CALIB_FLIP_FILE)
        assert (calib["paths"] == calib_flip["paths"]).all()
        corrected, taus[f"{name} + prior"] = with_prior(plain, calib)
        probs[f"{name} + prior"] = g3.softmax(corrected, 1.0)
        both = {"species_logits": (calib["species_logits"] + calib_flip["species_logits"]) / 2,
                "labels": calib["labels"]}
        corrected, taus[f"{name} + flip + prior"] = with_prior((plain + flip) / 2, both)
        probs[f"{name} + flip + prior"] = g3.softmax(corrected, 1.0)
    for name, folder in A3_M4.items():
        if (folder / "val_outputs.npz").exists():
            before = aligned(folder / "val_outputs.npz", paths, y)["species_logits"]
            probs["A3 on M4 crops (before)"] = g3.softmax(before, 1.0)
            corrected, taus["A3 on M4 crops + prior"] = with_prior(before, cf.load(folder / "calib_outputs.npz"))
            probs["A3 on M4 crops + prior"] = g3.softmax(corrected, 1.0)
            break
    if "FT" in probs and "KD" in probs:  # two full models: the fallback, not a shipped answer
        probs["FT + KD average (two models)"] = probs["FT"] + probs["KD"]
        probs["FT + KD average + flip (two models)"] = probs["FT + flip"] + probs["KD + flip"]
    for name, p in probs.items():
        out["candidates"][name] = score(p.argmax(1))
        c = out["candidates"][name]
        print(f"{name}: top-1 {c['top1']['diff']:+.4f} ({c['top1']['low']:+.4f} to {c['top1']['high']:+.4f}), "
              f"macro-F1 {c['macro_f1']['diff']:+.4f} ({c['macro_f1']['low']:+.4f} to {c['macro_f1']['high']:+.4f})",
              flush=True)
    for name, c in out["candidates"].items():
        if name in taus:
            c["tau_fitted_on_calibration"] = taus[name]
        c["val_top1_all_photos"] = float((probs[name].argmax(1) == y).mean())
        c["val_macro_f1_158_species"] = float(cf.head_macro_f1(y, probs[name].argmax(1), g3.N_CLASSES,
                                                                np.bincount(y, minlength=g3.N_CLASSES)))
        c["g3"] = {m: ("higher" if c[m]["diff"] > 0 else "tied" if c[m]["low"] <= 0 <= c[m]["high"] else "lower")
                   for m in ("top1", "macro_f1")}
        c["g3"]["passed"] = "lower" not in c["g3"].values()

    if logits:
        name = next(iter(logits))
        shuffled = np.random.default_rng(0).permutation(logits[name].argmax(1))
        control = score(shuffled)
        assert max(control[m]["high"] for m in control) < CONTROL_BELOW, "the shuffled control did not fail"
        out["negative_control"] = {"model": name, "predictions": "shuffled across photos", **control}
        closed = {n: score(shared[lg[:, shared].argmax(1)]) for n, lg in logits.items()}
        out["diagnostic_closed_set_not_a_candidate"] = closed
    (HERE / "g3_candidates.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
