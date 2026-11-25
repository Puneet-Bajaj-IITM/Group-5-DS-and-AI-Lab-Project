"""Stage the private Kaggle dataset the training kernel reads: lists of snake photos and the non-snake training photos.

- snakes_train.csv: the dataset's training split minus every calibration photographer (paths under data/).
- snakes_score.csv: calibration (7,999) and validation (17,151) snakes, scored by the kernel for the parity check.
- nonsnake_train.csv + images: notebook 19's training split, the 1,317 extra lookalikes, the new negatives.
- nonsnake_val.csv + images: the 600 validation non-snakes, scored by the kernel only (never trained on).
No test non-snake photo is copied. Checks: no snake training photographer is a calibration or
validation photographer, no non-snake training owner is in notebook 19's validation or test split, and no two
non-snake photos (training, validation, test hashes) are within 8 pHash bits across that line.

Usage: python -I stage_dataset.py --new <collected folder> [--new <folder> ...] --out <folder>
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "m4-wt/src"))
from serpentguard import negatives  # noqa: E402

PROBE = ROOT / "m5-g4-probe"
METADATA = ROOT / "kaggle/clean_build/out/metadata.csv"
CALIBRATION = PROBE / "results/snakes_calib.csv"
VALIDATION = PROBE / "results/snakes_val_all.csv"
NB19 = ROOT / "kaggle/m4_runs/non_snake/out_full/manifest.csv"
HARD = PROBE / "hard_negatives"
LOOKALIKES = {"caecilian", "earthworm", "eel", "legless_lizard", "worm_lizard"}
COLUMNS = ["path", "category", "group", "owner", "source", "licence", "source_id", "phash", "origin"]


def snake_lists(out):
    meta = pd.read_csv(METADATA, usecols=["path", "species", "family", "observer_id", "split"])
    calib, val = pd.read_csv(CALIBRATION), pd.read_csv(VALIDATION)
    held = set(calib["observer_id"]) | set(val["observer_id"])
    train = meta[meta["split"].eq("train") & ~meta["observer_id"].isin(held)]
    train[["path", "species", "family", "observer_id"]].to_csv(out / "snakes_train.csv", index=False)
    score = pd.concat([calib.assign(set="calib"), val.assign(set="val")])[["set", "path", "species", "observer_id"]]
    score.to_csv(out / "snakes_score.csv", index=False)
    return {"snakes_train": len(train), "snakes_train_observers": int(train["observer_id"].nunique()),
            "train_observers_in_calib_or_val": int(train["observer_id"].isin(held).sum()),
            "snakes_score": score["set"].value_counts().to_dict()}


def nonsnake_rows(new_folders):
    nb19 = pd.read_csv(NB19)
    train = nb19[nb19["split"].eq("train")].assign(origin="nb19", root=str(PROBE / "photos/train_nonsnake"))
    hard = pd.read_csv(HARD / "manifest.csv").assign(origin="hard_negatives", root=str(HARD))
    frames = [train, hard]
    for folder in new_folders:
        frames.append(pd.read_csv(Path(folder) / "manifest.csv").assign(origin=Path(folder).name, root=str(folder)))
    rows = pd.concat(frames, ignore_index=True)
    rows["group"] = rows["category"].map(lambda c: "lookalike" if c in LOOKALIKES else "other")
    rows["path"] = [f"nonsnake/{o}/{f}" for o, f in zip(rows["origin"], rows["file"])]
    return nb19, rows


def stage_validation(nb19, out):
    """The 600 validation non-snakes, so the kernel scores them on the same path as the snakes. Never trained on."""
    val = nb19[nb19["split"].eq("val")]
    val = val.assign(path=[f"nonsnake_val/{f}" for f in val["file"]], group=val["owner"].astype(str))
    for f, dest in zip(val["file"], val["path"]):
        target = out / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(PROBE / "photos/val_nonsnake" / f, target)
    val[["path", "category", "group", "source", "source_id"]].to_csv(out / "nonsnake_val.csv", index=False)
    return len(val)


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = snake_lists(out)
    nb19, rows = nonsnake_rows(args.new)
    held = nb19[nb19["split"].isin(["val", "test"])]
    names = rows["path"].map(lambda p: Path(p).name)
    if names.duplicated().any():
        raise ValueError(f"{int(names.duplicated().sum())} non-snake file names repeat")
    bank = negatives.hash_bits(list(held["phash"]))
    near = [int(negatives.distances(bank, negatives.hash_bits([h])[0]).min()) for h in rows["phash"]]
    report.update({"nonsnake_train": len(rows), "by_origin": rows["origin"].value_counts().to_dict(),
                   "by_group": rows["group"].value_counts().to_dict(),
                   "by_category": rows["category"].value_counts().to_dict(),
                   "owners_in_nb19_val_or_test": int(rows["owner"].astype(str).isin(set(held["owner"].astype(str)))
                                                     .sum()),
                   "within_8_bits_of_a_val_or_test_photo": int(sum(d <= negatives.NEAR_DUP_BITS for d in near))})
    if report["owners_in_nb19_val_or_test"] or report["within_8_bits_of_a_val_or_test_photo"]:
        raise ValueError(f"leak: {report}")
    for row in rows.itertuples():
        dest = out / row.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(Path(row.root) / row.file, dest)
    rows[COLUMNS].to_csv(out / "nonsnake_train.csv", index=False)
    report["nonsnake_val_scored_only"] = stage_validation(nb19, out)
    (out / "g4_data_manifest.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new", action="append", default=[])
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
