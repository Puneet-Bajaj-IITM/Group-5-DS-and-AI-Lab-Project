"""The new model's input as the app builds it: decode upright, exposure fix, letterbox, ImageNet normalise.

Every step is the M4 reference pipeline's own function (m4-final/for-jayesh/reference_pipeline.py), with the
pipeline.json settings; only the letterbox size differs (the model's own input size).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "m4-final/for-jayesh"
sys.path.insert(0, str(REFERENCE))
import reference_pipeline as ref  # noqa: E402

SETTINGS = json.loads((REFERENCE / "models/pipeline.json").read_text())


def model_input(path, size):
    rgb = ref.load_photo(path, SETTINGS["exposure_fix"])
    canvas = ref.letterbox(rgb, size, SETTINGS["letterbox"]["pad_rgb"])
    return ref.normalise(canvas, SETTINGS["normalise"])
