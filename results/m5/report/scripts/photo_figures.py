"""Copies the two photo figures into the report at web size and writes their photo credits.

The figures themselves are drawn by robustness/gradcam_report.py and robustness/corruption_examples.py.
Every photo in them is from the clean dataset, which keeps only Creative Commons licences; the credit
for each photo comes from that dataset's metadata.csv.

Usage: python photo_figures.py <figures folder> <metadata.csv>
"""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

M5 = Path(__file__).resolve().parents[2]
FIGURES = M5 / "report/figures"
GRADCAM_LIST = M5 / "robustness/lists/gradcam_photos.csv"
CORRUPTION_PHOTO = "val/fowlea_piscator/inaturalist_108134482.jpg"
FIGURE_FILES = {"gradcam_grid_stride8.png": "gradcam_stride8.jpg",
                "corruption_examples.png": "corruption_examples.jpg"}
CREDITS_FILE = "photo_credits.csv"
MAX_WIDTH, JPEG_QUALITY = 1600, 88
CREDIT_COLUMNS = ["figure", "path", "species", "license", "attribution", "url"]


def shrink(source, target):
    image = Image.open(source).convert("RGB")
    if image.width > MAX_WIDTH:
        image = image.resize((MAX_WIDTH, round(image.height * MAX_WIDTH / image.width)),
                             Image.LANCZOS)
    image.save(target, quality=JPEG_QUALITY, optimize=True)


def credits(metadata_csv):
    meta = pd.read_csv(metadata_csv, usecols=["path", "species", "license", "attribution", "url"],
                       low_memory=False)
    gradcam = pd.read_csv(GRADCAM_LIST)[["path"]].assign(figure=FIGURE_FILES["gradcam_grid_stride8.png"])
    corruption = pd.DataFrame({"path": [CORRUPTION_PHOTO],
                               "figure": [FIGURE_FILES["corruption_examples.png"]]})
    rows = pd.concat([gradcam, corruption]).merge(meta, on="path", how="left", validate="one_to_one")
    missing = rows["license"].isna()
    if missing.any():
        raise SystemExit(f"no licence for {rows.loc[missing, 'path'].tolist()}")
    if not rows["license"].str.startswith("cc").all():
        raise SystemExit("a photo without a Creative Commons licence would be published")
    return rows[CREDIT_COLUMNS]


def main():
    source_dir, metadata_csv = Path(sys.argv[1]), Path(sys.argv[2])
    table = credits(metadata_csv)
    for source, target in FIGURE_FILES.items():
        shrink(source_dir / source, FIGURES / target)
    table.to_csv(FIGURES / CREDITS_FILE, index=False)
    print(table["license"].value_counts().to_string())


if __name__ == "__main__":
    main()
