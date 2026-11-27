"""POST-TEST change: stricter safety thresholds for the app copy (not the frozen M5 pipeline).

The same refit as results/m5/g3/refit_new.py ("KD flip + prior", the app's gate, decide.fit_thresholds on the 7,999
CALIBRATION photos only), with the target raised in steps of 0.0025 until the WHO recall lower bound behind the gate
reaches 0.97 instead of 0.95. Scored on VALIDATION only, next to the frozen refit. The test split is never read.

Usage, from the repo root: python app/tool/m5/refit_strict.py   (writes refit_strict.json beside it)
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "results/m5/g3"))
import refit_new as rn  # noqa: E402

RUN = REPO / "results/m5/g3/out_kd/M5_KD"
VARIANT = "flip + prior"
STRICT_MARK = 0.97
TARGETS = np.round(np.arange(0.955, 0.9951, 0.0025), 4)
OUT = HERE / "refit_strict.json"


def warned_row(val, decided, listed, outside):
    """The app shows "unknown" (a snake found) as a keep-away warning too: WHO recall and false alarms counting
    both labels. "No snake found" is not a warning on screen, so a photo the gate drops is never warned."""
    warned = np.isin(decided.label, ("dangerous", "unknown")) & ~decided.reasons["no_snake"]
    return {"warned_who_recall": float(warned[listed].mean()),
            "warned_who_recall_lower_bound": rn.decide.recall_lower_bound(warned[listed], val["observer_id"][listed]),
            "warned_false_alarm": float(warned[outside].mean())}


def main():
    tables = rn.decide.repo_tables()
    prototypes = np.load(RUN / "prototypes.npz")["plain"]
    saved, tau = rn.variant_outputs(RUN, VARIANT)
    calib, val, unknown = (rn.with_gate(saved[n], rn.APP / f"{n}_outputs.npz") for n in ("calib", "val", "unknown"))
    frozen = rn.decide.Thresholds(**json.loads((REPO / "results/m5/g3/refit_new.json").read_text())
                                  ["KD flip + prior"]["chosen"]["thresholds"])
    rows, chosen = [], None
    for target in TARGETS:
        thresholds = rn.decide.fit_thresholds(calib, prototypes, tables, who_recall_target=float(target))
        rows.append({"target": float(target), **rn.calib_row(calib, thresholds, prototypes, tables),
                     "thresholds": asdict(thresholds)})
        print(f"target {target:.4f}: calibration bound {rows[-1]['bound_behind_gate']:.4f}", flush=True)
        if rows[-1]["bound_behind_gate"] >= STRICT_MARK:
            chosen = rows[-1]
            break
    assert chosen is not None, "no target reached the strict mark"
    strict = rn.decide.Thresholds(**chosen["thresholds"])
    out = {"note": "POST-TEST change for the app copy; calibration fit, validation check only; test never read",
           "variant": f"KD {VARIANT}", "tau": tau, "mark": STRICT_MARK, "search": rows, "chosen": chosen}
    for name, thresholds in (("frozen", frozen), ("strict", strict)):
        decided, listed, outside, hits = rn.gated(val, thresholds, prototypes, tables)
        out[name] = {"thresholds": asdict(thresholds),
                     "calibration": rn.calib_row(calib, thresholds, prototypes, tables),
                     "validation": {**rn.val_row(val, unknown, thresholds, prototypes, tables),
                                    "who_recall_lower_bound": rn.decide.recall_lower_bound(
                                        hits, val["observer_id"][listed]),
                                    **warned_row(val, decided, listed, outside)}}
    order = np.random.default_rng(0).permutation(len(calib["labels"]))
    control = {k: (v[order] if k in rn.SHUFFLED else v) for k, v in calib.items()}
    out["negative_control_shuffled_logits"] = rn.calib_row(control, strict, prototypes, tables)
    assert out["negative_control_shuffled_logits"]["bound_behind_gate"] < STRICT_MARK, "the control passed"
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
    for name in ("frozen", "strict"):
        v = out[name]["validation"]
        print(name, {k: v[k] for k in ("who_recall", "who_recall_ci", "who_recall_lower_bound", "false_alarm")},
              {k: v["verdicts"][k] for k in ("dangerous", "not_listed", "unknown")})


if __name__ == "__main__":
    main()
