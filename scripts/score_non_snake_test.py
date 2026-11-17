"""Score the non-snake TEST split, once (goal G4), on the laptop's CPU.

Nothing is fitted or tuned on these photos. The gate is notebook 20's pick (results/m4/gate/
gate.json: "both"); the other rows are reported on the same scores, not picked: the retrained (M4) box and mask
gates and Milestone 3's box gate, with the cut-offs gates.csv records, all fitted on calibration snakes in notebook
20. It uses the same functions as cell 44 of notebook 21 (front_end.gate_scores, false_snake_rate, gate_calls) on a
frame built as that cell builds it. Snake test photos are never read.

The script refuses to run when its outputs already exist, so the test split is scored only once."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPO = ROOT / "m4-wt-t5"
DRY_RUN = "--dry-run" in sys.argv  # the same code on the val photos notebook 19 kept, written to a temporary folder
SCRATCH = Path(tempfile.mkdtemp()) if DRY_RUN else None
OUT_JSON = SCRATCH / "non_snake_test.json" if DRY_RUN else REPO / "results/m4/gate/non_snake_test.json"
SCORES_CSV = SCRATCH / "non_snake_test_scores.csv" if DRY_RUN else HERE / "non_snake_test_scores.csv"
if OUT_JSON.exists() or SCORES_CSV.exists():
    raise SystemExit(f"the non-snake test split has already been scored ({OUT_JSON.name} or {SCORES_CSV.name} "
                     f"exists); it is scored only once")
sys.path.insert(0, str(REPO / "src"))
from ultralytics import YOLO  # noqa: E402

from serpentguard import front_end  # noqa: E402
from serpentguard.metrics import write_json  # noqa: E402

MANIFEST = ROOT / "kaggle/m4_runs/non_snake/out_full/manifest.csv"
PHOTOS = ROOT / "kaggle/m4_runs/non_snake/check_photos/saved" if DRY_RUN else HERE / "non_snake_test_photos"
SPLIT = "val" if DRY_RUN else "test"
FETCH_REPORT = HERE / "fetch_non_snake_test.json"
GATE_JSON, GATES_CSV, VAL_BY_CATEGORY = (REPO / "results/m4/gate" / name
                                         for name in ("gate.json", "gates.csv", "fpr_by_category.csv"))
RUN = HERE / "out_full/m4_front_end"
M3_FRONT = ROOT / "kaggle/m4_runs/frontend_full/weights_ds/dl/frontends"
WEIGHTS = {"M4 detector": RUN / "detector/M4_detector.pt", "M4 segmenter": RUN / "segmenter/segmenter.pt",
           "M3 detector": M3_FRONT / "D1/yolo26n-ft/detector.pt", "M3 segmenter": M3_FRONT / "S1/unet/segmenter.pt"}
M3_DETECTOR_SHA256 = "099b36bd8271c328b60be8152161f74558a0083ebe4e6fec1f558e7c0951c5b0"
GATE_RULE, GATE_THRESHOLDS = "both", {"box_conf": 0.00499307457357645, "mask_area": 4.8e-05}
FALSE_SNAKE_GOAL = 0.05
PRECISION_NOTE = ("the cut-offs were fitted in notebook 20 on Kaggle T4 scores, where the U-Net ran in float16; "
                  "this scoring runs on a CPU in float32 (mask IoU 0.9995 between the two, measured on the "
                  "team laptop); the detector runs in float32 in both")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


commit = git("rev-parse", "HEAD")
assert not git("status", "--porcelain", "src"), "front_end.py has uncommitted changes"
gate = json.loads(GATE_JSON.read_text())
assert gate["rule"] == GATE_RULE and gate["thresholds"] == GATE_THRESHOLDS, gate
assert gate["limit"] is None and gate["epochs"] is None, "the gate comes from a smoke run"
gates = pd.read_csv(GATES_CSV).set_index(["front_end", "rule"])
assert sha256(WEIGHTS["M3 detector"]) == M3_DETECTOR_SHA256
fetch = {} if DRY_RUN else json.loads(FETCH_REPORT.read_text())
assert DRY_RUN or (fetch["saved"] == 600 and not fetch["size_mismatches"] and not fetch["phash_over_limit"])

manifest = pd.read_csv(MANIFEST)
test = manifest[manifest["split"] == SPLIT].reset_index(drop=True)
if DRY_RUN:
    test = test[[(PHOTOS / name).exists() for name in test["file"]]].reset_index(drop=True)
else:
    assert len(test) == 600 and set(test["split"]) == {"test"}
frame = pd.DataFrame({"path": test["file"], "is_snake": False,
                      "image_file": [str(PHOTOS / name) for name in test["file"]]})
started = time.perf_counter()
m4 = front_end.gate_scores(frame, YOLO(str(WEIGHTS["M4 detector"])),
                           front_end.Segmenter("unet", WEIGHTS["M4 segmenter"].parent))
m3 = front_end.gate_scores(frame, YOLO(str(WEIGHTS["M3 detector"])),
                           front_end.Segmenter("unet", WEIGHTS["M3 segmenter"].parent))
minutes = (time.perf_counter() - started) / 60
box_text = lambda b: "" if b is None else json.dumps([round(v, 2) for v in b])
scores = pd.DataFrame({"path": test["file"], "category": test["category"], "is_snake": False,
                       "m4_box_conf": m4["box_conf"], "m4_mask_area": m4["mask_area"], "m4_box": m4["box"].map(box_text),
                       "m3_box_conf": m3["box_conf"], "m3_mask_area": m3["mask_area"], "m3_box": m3["box"].map(box_text)})

rules = {"M4 both": ("M4", m4, GATE_RULE, GATE_THRESHOLDS, "gate.json (the chosen gate)", True),
         "M4 box": ("M4", m4, "box", json.loads(gates.loc[("M4", "box"), "thresholds"]), "gates.csv", False),
         "M4 mask": ("M4", m4, "mask", json.loads(gates.loc[("M4", "mask"), "thresholds"]), "gates.csv", False),
         "M3 box": ("M3", m3, "box", json.loads(gates.loc[("M3", "box"), "thresholds"]), "gates.csv (before)", False)}
results = {}
for name, (front, table, rule, thresholds, source, picked) in rules.items():
    called = pd.Series(front_end.gate_calls(table, rule, thresholds), index=test["category"].to_numpy())
    results[name] = {"front_end": front, "rule": rule, "thresholds": thresholds, "thresholds_from": source,
                     "picked": picked, "false_snake_rate": front_end.false_snake_rate(table, rule, thresholds),
                     "called_a_snake": int(called.sum()),
                     "by_category": called.groupby(level=0).mean().round(4).to_dict()}
headline = results["M4 both"]
val = pd.read_csv(VAL_BY_CATEGORY).set_index("category")["M4 called a snake"]
scores.to_csv(SCORES_CSV, index=False)  # written last, with the report, so a failure above leaves nothing behind
report = {
    "split": SPLIT, "photos": len(test), "scored_once": not DRY_RUN,
    "photos_per_category": test["category"].value_counts().to_dict(),
    "photo_files": "re-made on the laptop from manifest.csv's source urls with notebook 19's negatives.save_photo; "
                   "every one matches the manifest's size and pHash (fetch_non_snake_test.json)",
    "photo_check": {k: fetch.get(k) for k in ("saved", "size_mismatches", "phash_over_limit", "phash_limit")},
    "g4": {"goal": FALSE_SNAKE_GOAL, "gate": "M4 both", "false_snake_rate": headline["false_snake_rate"],
           "passed": headline["false_snake_rate"] <= FALSE_SNAKE_GOAL,
           "val_false_snake_rate": gate["fpr_nonsnake_val"], "val_snake_recall": gate["recall_snake_val"]},
    "rules": results,
    "val_by_category_for_the_gate": val.round(4).to_dict(),
    "nothing_fitted_here": "every cut-off comes from notebook 20's calibration fit (gate.json, gates.csv)",
    "device": "cpu", "segmenter_precision": "float32", "precision_note": PRECISION_NOTE,
    "weights_sha256": {name: sha256(path) for name, path in WEIGHTS.items()},
    "code_commit": commit, "script_sha256": sha256(__file__),
    "scores_csv": str(SCORES_CSV.relative_to(ROOT) if SCORES_CSV.is_relative_to(ROOT) else SCORES_CSV), "scores_csv_sha256": sha256(SCORES_CSV),
    "minutes": round(minutes, 1),
}
write_json(OUT_JSON, report)
print(json.dumps({"g4": report["g4"], **{n: (r["false_snake_rate"], r["called_a_snake"]) for n, r in results.items()}},
                 indent=1))
print(pd.DataFrame({**{n: r["by_category"] for n, r in results.items()}, "val (M4 both)": val}).round(3).to_string())
