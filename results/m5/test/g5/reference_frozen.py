"""The app's reference pipeline with the frozen classifier answer: each crop and its mirror image through the
classifier, every head's logits averaged (embedding re-normalised, as refit_new.averaged), then the training prior
tau * log(n_c) added to the species logits (tau 0.5, FREEZE.md). Everything else is reference_pipeline's own.
clip_outputs.py loads this file as its --reference."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "m4-final/for-jayesh"))
from reference_pipeline import *  # noqa: E402,F401,F403
import reference_pipeline as _ref  # noqa: E402

TAU = 0.5
LOG_COUNTS = np.load(HERE / "log_counts.npy")
HEADS = ("species_logits", "who_logits", "genus_logits", "family_logits")


class Pipeline(_ref.Pipeline):
    def classify(self, crop):
        plain = super().classify(crop)
        mirrored = super().classify(np.ascontiguousarray(crop[:, ::-1]))
        out = dict(plain)
        for head in HEADS:
            out[head] = (plain[head] + mirrored[head]) / 2
        embedding = plain["embedding"] + mirrored["embedding"]
        out["embedding"] = embedding / np.linalg.norm(embedding)
        out["species_logits"] = out["species_logits"] + TAU * LOG_COUNTS
        return out
