"""Bugs 1 and 4 on the M4 lock (git HEAD video.py) with notebook 24's saved per-frame outputs (A3, PyTorch path)."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd
WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK / "m5-g5/src"))
from serpentguard import decide
spec = importlib.util.spec_from_file_location("video_m4", Path(__file__).parent / "old/video_m4.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
CACHE = WORK / "m4-video-clips/cache"
decision = json.loads((WORK / "m5-g5/results/m4/full/A3/decision_metrics.json").read_text())
thr = decide.Thresholds(**decision["thresholds"])
tables = decide.repo_tables()
protos = np.asarray(json.loads((WORK / "m4-final/for-jayesh/models/tables.json").read_text())["prototypes"])
names = pd.read_csv(WORK / "m5-g5/data/classes.csv").set_index("class_id")["species"]
def load(name):
    f = next(CACHE.glob(f"A3_*__{name}.npz"))
    with np.load(f) as s:
        parts = [k for k in old.PART_NAMES if k in s.files]
        items = [{k: s[k][i] for k in parts} if s["passed"][i] else None for i in range(len(s["shown_at"]))]
        return items, s["shown_at"]
def changes(seq):
    a = [x for x in seq if x is not None]
    return sum(p != q for p, q in zip(a, a[1:]))
rows = []
for clip in pd.read_csv(WORK / "m5-g5/results/m4/video/clips.csv")["clip"].tolist() + pd.read_csv(WORK / "m5-g5/results/m4/video/negatives.csv")["clip"].tolist():
    items, times = load(clip)
    steps = old.pooled_lock(items, times, thr.theta_conf, 0.05, window=5)
    locked = [s["species"] for s in steps]
    displayed = []  # what the app showed: the pooled verdict's top species once locked, or while dangerous
    for s in steps:
        if s["parts"] is None:
            displayed.append(displayed[-1] if displayed else None); continue
        label = old.frame_verdict(s["parts"], thr, protos, tables)
        show = s["state"] == "locked" or label == "dangerous"
        displayed.append(int(np.argmax(s["parts"]["probs"])) if show else None)
    verdicts = old.lock_verdicts(steps, thr, protos, tables)
    rows.append({"clip": clip, "changes_locked_species (notebook 24 scored)": changes(locked),
                 "changes_displayed_species (app showed)": changes(displayed),
                 "frames_displayed_differs_from_locked": sum(d is not None and l is not None and d != l for d, l in zip(displayed, locked)),
                 "ever_locked": any(l is not None for l in locked), "verdicts_shown": sorted({v for v in verdicts if v})})
t = pd.DataFrame(rows); print(t.to_string()); t.to_csv(WORK / "results/m5/g5/bug_check_m4_lock.csv", index=False)
