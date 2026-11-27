"""Instrument check of reference_frozen.py: 40 validation photos through the app path with the frozen classifier
answer, against the kernel's saved M5_KD (plain + mirrored) / 2 + prior logits for the same photos. The app's crop is
not the kernel's crop (g3 README: about 1 point of top-1 apart), so agreement is high but not exact. A control with
the prior left out must agree less with the prior-corrected saved answer."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
spec = importlib.util.spec_from_file_location("reference_frozen", HERE / "reference_frozen.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
M5 = WORK / "results/m5/g3/out_kd/M5_KD"
PHOTOS = WORK / "m5-g4-probe/photos/snakes"

plain, flip = np.load(M5 / "val_outputs.npz", allow_pickle=True), np.load(M5 / "val_outputs_flip.npz", allow_pickle=True)
pick = np.random.default_rng(7).choice(len(plain["paths"]), 40, replace=False)
pipeline = ref.Pipeline(HERE / "models_frozen", threads=2)
same, same_no_prior, diffs = 0, 0, []
for i in pick:
    rgb = ref.load_photo(PHOTOS / plain["paths"][i], pipeline.settings["exposure_fix"])
    _, mask = pipeline.gate(rgb)
    h, w = rgb.shape[:2]
    window = ref.crop_window(ref.clean_mask(mask, pipeline.settings["mask_cleanup"]), h, w, pipeline.settings["crop"]["padding"])
    crop = rgb if window is None else rgb[window[0]:window[1], window[2]:window[3]]
    mine = pipeline.classify(crop)["species_logits"]
    saved = (plain["species_logits"][i] + flip["species_logits"][i]) / 2 + ref.TAU * ref.LOG_COUNTS
    same += int(mine.argmax() == saved.argmax())
    same_no_prior += int((mine - ref.TAU * ref.LOG_COUNTS).argmax() == saved.argmax())
    diffs.append(float(np.abs(mine - saved).max()))
out = {"photos": len(pick), "top1_agreement": same / len(pick), "control_without_prior_agreement": same_no_prior / len(pick),
       "median_max_abs_logit_diff": float(np.median(diffs))}
(HERE / "check_reference.json").write_text(json.dumps(out, indent=1) + "\n")
print(out)
