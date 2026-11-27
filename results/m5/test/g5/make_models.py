"""The frozen G5 model folder and settings: M5_KD's ONNX classifier with the M4 detector and U-Net (clip_outputs
--models), a pipeline.json whose decision thresholds are the frozen refit (the lock confidence is its theta_conf),
and a tables.json whose prototypes are M5_KD's. Also log_counts.npy for reference_frozen.py."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path.insert(0, str(WORK / "results/m5/g3"))
import prior  # noqa: E402

MODELS = HERE / "models_frozen"
M4 = WORK / "m4-final/for-jayesh/models"
APP = WORK / "m5-g5/app/assets/model"
REFIT = json.loads((WORK / "results/m5/g3/refit_new.json").read_text())["KD flip + prior"]["chosen"]["thresholds"]
LOG_COUNTS_SHA = "d6cb68339c75b59af12f18a6a3a1601c18cfd42bb2a1f12a3459a32a286f4425"

MODELS.mkdir(exist_ok=True)
shutil.copy2(WORK / "results/m5/g3/app_models_KD/classifier.onnx", MODELS / "classifier.onnx")
for name in ("detector.onnx", "segmenter.onnx", "pipeline.json", "tables.json"):
    shutil.copy2(M4 / name, MODELS / name)
counts = prior.log_counts()
assert hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest() == LOG_COUNTS_SHA
np.save(HERE / "log_counts.npy", counts)
settings = json.loads((APP / "pipeline.json").read_text())
settings["decision"]["thresholds"] = REFIT
settings["decision"]["source"] = "results/m5/g3/refit_new.json, KD flip + prior, chosen (FREEZE.md)"
(HERE / "pipeline_frozen.json").write_text(json.dumps(settings, indent=1) + "\n")
tables = json.loads((APP / "tables.json").read_text())
prototypes = np.load(WORK / "results/m5/g3/out_kd/M5_KD/prototypes.npz")["plain"]
assert np.asarray(tables["prototypes"]).shape == prototypes.shape
tables["prototypes"] = prototypes.tolist()
(HERE / "tables_frozen.json").write_text(json.dumps(tables) + "\n")
print("theta_conf", REFIT["theta_conf"], "prototypes", prototypes.shape)
