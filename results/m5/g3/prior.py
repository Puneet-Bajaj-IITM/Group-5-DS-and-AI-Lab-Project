"""A training-prior correction of the species logits: logits + tau * log(n_c).

n_c is the number of training-list photos of class c BEFORE the per-class caps (the training photographers' rows of
the train split, data.full_train_rows): the natural frequency the capped, weighted sampler flattened. tau is fitted on
CALIBRATION photos only: the grid value that maximises the mean of calibration top-1 (all photos) and calibration
macro-F1 (species with at least 5 calibration photos), the smallest on a tie. Both scores are kept in the criterion
because G3 judges both; top-1 alone rises with tau over a flat stretch while macro-F1 falls. Validation is never
read here.
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / "m5-wt/src"))
from serpentguard import data  # noqa: E402

N_CLASSES = 184
TAU_GRID = np.round(np.arange(0.0, 3.0001, 0.05), 2)
MIN_PHOTOS = 5


@lru_cache(maxsize=1)
def log_counts():
    meta = pd.read_csv(WORK / "kaggle/clean_build/out/metadata.csv", dtype=data.ID_TYPES, low_memory=False)
    subset = pd.read_csv(WORK / "kaggle/m3_subset/subset.csv", dtype=data.ID_TYPES)
    classes = pd.read_csv(WORK / "m5-wt/data/classes.csv")
    rows = data.full_train_rows(meta, subset).merge(classes[["species", "class_id"]], on="species",
                                                     validate="many_to_one")
    return np.log(np.maximum(np.bincount(rows["class_id"], minlength=N_CLASSES), 1))


def corrected(logits, tau):
    return logits + tau * log_counts()


def criterion(pred, labels):
    measurable = np.flatnonzero(np.bincount(labels, minlength=N_CLASSES) >= MIN_PHOTOS)
    tp = np.bincount(labels[labels == pred], minlength=N_CLASSES)
    total = np.bincount(labels, minlength=N_CLASSES) + np.bincount(pred, minlength=N_CLASSES)
    f1 = np.divide(2 * tp, total, out=np.zeros(N_CLASSES), where=total > 0)
    return (float((pred == labels).mean()) + float(f1[measurable].mean())) / 2


def fit_tau(calib_logits, calib_labels):
    labels = np.asarray(calib_labels).astype(int)
    scores = [criterion(corrected(calib_logits, t).argmax(1), labels) for t in TAU_GRID]
    best = int(np.argmax(scores))  # argmax takes the first, so the smallest tau on a tie
    return float(TAU_GRID[best]), dict(zip(map(float, TAU_GRID), scores))
