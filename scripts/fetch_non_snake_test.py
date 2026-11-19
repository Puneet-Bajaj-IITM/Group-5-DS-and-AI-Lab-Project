"""Re-make the 600 non-snake TEST photos on the laptop, exactly as notebook 19 made them, and prove they match.

The kernel output that holds notebook 19's saved photos was never downloaded to the laptop, and this step may not
use Kaggle. So each test photo is fetched again from the source url in manifest.csv (Open Images bucket or
iNaturalist medium size) and saved with notebook 19's own function, negatives.save_photo (500 px long side, JPEG 90).
Only manifest rows with split "test" are touched; this script scores nothing.

Checks against things this script did not make: each photo's width and height and its 64-bit pHash must match
manifest.csv (written by the Kaggle run), and the photos notebook 19 kept from the kernel output byte for byte
(check_photos/saved) are compared pixel by pixel with their re-made copies."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "m4-wt/src"))  # branch m4, where notebook 19's negatives.py is committed
from serpentguard import negatives  # noqa: E402

MANIFEST = ROOT / "kaggle/m4_runs/non_snake/out_full/manifest.csv"
KERNEL_COPIES = ROOT / "kaggle/m4_runs/non_snake/check_photos/saved"
PHOTOS = HERE / "non_snake_test_photos"
REPORT = HERE / "fetch_non_snake_test.json"
MAX_PHASH_BITS = negatives.NEAR_DUP_BITS

manifest = pd.read_csv(MANIFEST)
test = manifest[manifest["split"] == "test"].reset_index(drop=True)
assert len(test) == 600 and test["file"].is_unique
todo = test[[not (PHOTOS / f).exists() for f in test["file"]]]
done = negatives.download_photos(todo, PHOTOS, workers=4) if len(todo) else todo.assign(error="")
errors = done.loc[done["error"].ne(""), ["file", "error"]].to_dict("records")

rows = []
for row in test.itertuples():
    path = PHOTOS / row.file
    if not path.exists():
        rows.append({"file": row.file, "saved": False})
        continue
    with Image.open(path) as image:
        bits = int(negatives.distances(negatives.hash_bits([row.phash]), negatives.hash_bits([negatives.phash(image)])[0])[0])
        rows.append({"file": row.file, "saved": True, "size_matches": image.size == (row.width, row.height),
                     "phash_bits": bits})
check = pd.DataFrame(rows)
kernel = []
for copy in sorted(KERNEL_COPIES.rglob("*.jpg")):
    name = copy.relative_to(KERNEL_COPIES).as_posix()
    if name in set(test["file"]) and (PHOTOS / name).exists():
        a = np.asarray(Image.open(copy).convert("RGB"), dtype=np.int16)
        b = np.asarray(Image.open(PHOTOS / name).convert("RGB"), dtype=np.int16)
        kernel.append({"file": name, "same_bytes": copy.read_bytes() == (PHOTOS / name).read_bytes(),
                       "same_shape": a.shape == b.shape,
                       "max_pixel_gap": int(np.abs(a - b).max()) if a.shape == b.shape else None})
report = {"photos": len(test), "saved": int(check["saved"].sum()), "download_errors": errors,
          "size_mismatches": int((~check["size_matches"].fillna(False).astype(bool) & check["saved"]).sum()),
          "phash_bits": check["phash_bits"].describe().to_dict(),
          "phash_over_limit": int((check["phash_bits"] > MAX_PHASH_BITS).sum()), "phash_limit": MAX_PHASH_BITS,
          "kernel_copies_compared": kernel,
          "by_category": test["category"].value_counts().to_dict()}
REPORT.write_text(json.dumps(report, indent=1, default=str))
print(json.dumps(report, indent=1, default=str))
assert report["saved"] == 600 and not report["size_mismatches"] and not report["phash_over_limit"], "re-make failed"
print("all 600 test photos re-made and matched to the manifest")
