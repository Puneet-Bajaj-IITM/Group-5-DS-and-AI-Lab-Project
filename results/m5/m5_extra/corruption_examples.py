"""Figure of every corruption on one local validation photo (not a test photo): figures/corruption_examples.png.

Usage: m3-final/.venv/bin/python results/m5/m5_extra/corruption_examples.py
"""
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import corrupt  # noqa: E402

HERE = Path(__file__).resolve().parent
PHOTO = HERE.parents[1] / "m5-g4-probe/photos/snakes/val/fowlea_piscator/inaturalist_108134482.jpg"
COLS = 5


def main():
    rgb = corrupt.decode(PHOTO)
    names = list(corrupt.CONDITIONS)
    rows = int(np.ceil(len(names) / COLS))
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 2.4, rows * 1.9))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, name in zip(axes.ravel(), names):
        out = corrupt.corrupted(rgb, name, str(PHOTO))
        if isinstance(out, bytes):
            out = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"))
        ax.imshow(out)
        ax.set_title(name, fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/corruption_examples.png", dpi=130)


if __name__ == "__main__":
    main()
