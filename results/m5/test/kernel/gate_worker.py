"""The app's M4 "both" gate on one photo (reference_pipeline.Pipeline.gate), for a spawn process pool."""
import sys
from pathlib import Path

_pipeline = None
_ref = None


def init(models_dir):
    global _pipeline, _ref
    sys.path.insert(0, str(models_dir))
    import reference_pipeline as ref
    _ref = ref
    _pipeline = ref.Pipeline(models_dir, threads=1)


def gate(item):
    path, file = item
    rgb = _ref.load_photo(Path(file), _pipeline.settings["exposure_fix"])
    call, _ = _pipeline.gate(rgb)
    return {"path": path, "gate_snake": bool(call["snake"]),
            "box_conf": float("nan") if call["box_conf"] is None else float(call["box_conf"]),
            "mask_area": float(call["mask_area"])}
