"""The M5 model bundle for the app: assets/model/pipeline.json, assets/model/tables.json and assets/models/*.onnx.

Frozen M5 pipeline (results/m5/test/FREEZE.md):
- classifier: M5_KD's ONNX (results/m5/g3/app_models_KD), run on the crop and its mirror image, every head averaged,
  then the training prior tau * log(n_c) added to the species logits (tau 0.5)
- tables: the app's tables with M5_KD's "plain" prototypes
- snake check: distil_oi fp16, a photo is a snake when probs[0] >= the frozen cut; it replaces the gate's decision,
  the detector and U-Net stay for the crop
- decision thresholds: POST-TEST, the stricter refit of tool/m5/refit_strict.py (bound 0.97); the frozen refit
  (refit_new.json, "KD flip + prior", chosen) is kept beside them as "frozen_thresholds"

Every input file is checked against its FREEZE.md hash. Usage, from the repo root: python app/tool/m5/build_assets.py
"""
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
APP = HERE.parents[1]
REPO = APP.parent
WORKSPACE = REPO.parent  # the team's working folder, which holds the M4 app bundle
sys.path.insert(0, str(REPO / "results/m5/g3"))
import prior  # noqa: E402

G3 = REPO / "results/m5/g3"
KD_RUN = G3 / "out_kd/M5_KD"
SNAKE_RUN = REPO / "results/m5/g4/runs/distil_oi"
M4_MODELS = WORKSPACE / "m4-final/for-jayesh/models"
STRICT = HERE / "refit_strict.json"

TAU = 0.5
SNAKE_CUTOFF = 0.20484977503483762
SNAKE_INPUT = 256
SNAKE_FILE = "snake_check_fp16.onnx"
FROZEN = {
    KD_RUN / "best.pt": "1ec2ee88b95af0d5a0e6e69feed71c9be4bbdbe3584790ffa389fa52bdb1019f",
    KD_RUN / "prototypes.npz": "a7fa7cba23eb791e41e12187d022e1c1d7503e3e241dbec84da75e17820cade5",
    M4_MODELS / "detector.onnx": "1d2da244575a39c23ebab6ee2f4f04b1c672794031a56fab9d49dc5176f4f9ba",
    M4_MODELS / "segmenter.onnx": "da5d59baca34eda5993d837673449b99ccd178c98bbd6755a6b20bbea6f99269",
    SNAKE_RUN / "model.pt": "88fd518c550a4f1502c86e7da1d8a3a8bd9c332b5377a8cc5d7f65f8144b7cfd",
    SNAKE_RUN / SNAKE_FILE: "fb5b5bda5e0dbdb1895448ae1c70e56e93042ef322f9e5595cfcd6e0b8ec9fc3",
}
A3_CLASSIFIER_SHA = "355a975449e48b0b2dd7d633641132cf90c68d5d79a8974849b01b6c4651252e"  # the M4 bundle's SHA256SUMS
LOG_COUNTS_SHA ="d6cb68339c75b59af12f18a6a3a1601c18cfd42bb2a1f12a3459a32a286f4425"
MODEL_SOURCES = {
    "classifier.onnx": G3 / "app_models_KD/classifier.onnx",
    "detector.onnx": M4_MODELS / "detector.onnx",
    "segmenter.onnx": M4_MODELS / "segmenter.onnx",
    SNAKE_FILE: SNAKE_RUN / SNAKE_FILE,
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def committed(path):
    """The A3 version of an asset as committed (HEAD), so a rerun starts from the same base."""
    return json.loads(subprocess.run(["git", "-C", str(APP), "show", f"HEAD:app/{path}"], check=True,
                                     capture_output=True, text=True).stdout)


def json_number(value):
    """JSON has no infinity: write it as the string "Infinity", which Dart's double.parse and Python's float read."""
    return "Infinity" if math.isinf(value) else value


def thresholds(raw):
    return {k: json_number(float(v)) for k, v in raw.items()}


def main():
    for path, want in FROZEN.items():
        assert sha256(path) == want, f"{path} is not the frozen file"
    counts = prior.log_counts()
    assert hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest() == LOG_COUNTS_SHA

    frozen = json.loads((G3 / "refit_new.json").read_text())["KD flip + prior"]
    assert frozen["tau"] == TAU
    strict = json.loads(STRICT.read_text())
    assert strict["frozen"]["thresholds"] == frozen["chosen"]["thresholds"]

    settings = committed("assets/model/pipeline.json")
    settings["gate"]["rule"] = "snake_check"
    settings["gate"]["note"] = "the M4 'both' thresholds are kept for the record; the snake check decides"
    settings["snake_check"] = {
        "file": SNAKE_FILE, "input_size": SNAKE_INPUT, "output": "probs", "snake_index": 0, "cutoff": SNAKE_CUTOFF,
        "input": "the classifier's own path: exposure fix, letterbox on pad_rgb at input_size, ImageNet normalise",
        "source": "results/m5/g4 distil_oi (FREEZE.md section 3)"}
    settings["classifier"] = {
        "flip_tta": True, "prior_tau": TAU, "log_counts": counts.tolist(), "log_counts_sha256": LOG_COUNTS_SHA,
        "rule": "every head = (crop + mirrored crop) / 2, embedding re-normalised; "
                "species_logits += prior_tau * log_counts (FREEZE.md section 3)"}
    settings["decision"]["thresholds"] = thresholds(strict["chosen"]["thresholds"])
    settings["decision"]["source"] = ("POST-TEST change (app copy only): tool/m5/refit_strict.py, the refit_new.py "
                                      "fit on calibration with the WHO recall bound behind the gate raised to 0.97")
    settings["decision"]["frozen_thresholds"] = thresholds(frozen["chosen"]["thresholds"])
    settings["decision"]["frozen_source"] = "results/m5/g3/refit_new.json, KD flip + prior, chosen (FREEZE.md)"
    settings["source_weights_sha256"] = {
        "M5_KD best.pt": FROZEN[KD_RUN / "best.pt"], "M5_KD prototypes.npz": FROZEN[KD_RUN / "prototypes.npz"],
        "detector.pt": settings["source_weights_sha256"]["detector.pt"],
        "segmenter.pt": settings["source_weights_sha256"]["segmenter.pt"],
        "distil_oi model.pt": FROZEN[SNAKE_RUN / "model.pt"]}
    # The app refuses an ONNX file whose SHA-256 differs from this (pipeline.dart extractModel). The classifier is
    # checked against A3's file here: both are MobileViT-S exports of the same byte size.
    settings["model_sha256"] = {name: sha256(source) for name, source in MODEL_SOURCES.items()}
    assert settings["model_sha256"]["classifier.onnx"] != A3_CLASSIFIER_SHA, "classifier.onnx is A3, not M5_KD"
    settings["model_sha256_refused"] = {"classifier.onnx": {
        A3_CLASSIFIER_SHA: "A3, the M4 classifier: same size as M5_KD, wrong for these settings"}}
    settings.pop("model_sha256_missing", None)  # every file is pinned now
    (APP / "assets/model/pipeline.json").write_text(json.dumps(settings, indent=1) + "\n")

    tables = committed("assets/model/tables.json")
    prototypes = np.load(KD_RUN / "prototypes.npz")["plain"]
    assert np.asarray(tables["prototypes"]).shape == prototypes.shape
    tables["prototypes"] = prototypes.tolist()
    (APP / "assets/model/tables.json").write_text(json.dumps(tables) + "\n")
    reference = json.loads((REPO / "results/m5/test/g5/tables_frozen.json").read_text())
    assert reference == tables, "tables differ from the frozen G5 tables"

    target = APP / "assets/models"
    for name, source in MODEL_SOURCES.items():
        shutil.copy2(source, target / name)
    sums = {name: sha256(target / name) for name in MODEL_SOURCES}
    (target / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sums.items()))
    print(json.dumps({name: (target / name).stat().st_size for name in MODEL_SOURCES}, indent=1))


if __name__ == "__main__":
    main()
