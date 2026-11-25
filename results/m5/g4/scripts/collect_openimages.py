"""More Open Images non-snake TRAINING photos (people, hands, ropes, animals, plants, everyday things).

Notebook 19's source and filters, unchanged: Open Images V7 validation + test splits (labels checked by people),
a photo is out if any snake label is on it (checked, machine-made or a box), CC BY 2.0 only. Also out: every photo
notebook 19 already has (any split), every photo whose Flickr author has a photo in notebook 19's validation or test
split, and anything within 8 pHash bits of a notebook 19 / earlier-negative photo or of another new photo.

Usage: python -I collect_openimages.py --out <folder>
"""
import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "m4-wt/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from serpentguard import negatives  # noqa: E402
from collect_negatives import download  # noqa: E402

NB19 = ROOT / "kaggle/m4_runs/non_snake/out_full/manifest.csv"
EARLIER = (ROOT / "m5-g4-probe/hard_negatives/manifest.csv", ROOT / "results/m5/g4/data/negatives_v1/manifest.csv")
OI_TABLES = "https://storage.googleapis.com/openimages"
OI_IMAGES = "https://open-images-dataset.s3.amazonaws.com"
OI_LICENCE = "https://creativecommons.org/licenses/by/2.0/"
OI_SPLITS = {"validation": "val", "test": "test"}
TARGETS = {  # category: {label: photos wanted}, filled in this order (rarer first)
    "rope_hose_belt": {"Rope": 150, "Belt": 150},
    "tshirt": {"T-shirt": 250},
    "other_animal": {name: 40 for name in ("Lizard", "Frog", "Turtle", "Squirrel", "Monkey", "Bird", "Insect",
                                            "Fish", "Horse", "Cattle", "Goat", "Sheep", "Deer", "Butterfly")},
    "cat": {"Cat": 250},
    "dog": {"Dog": 250},
    "hand_arm": {"Human hand": 300, "Human arm": 250, "Human leg": 150, "Human foot": 100},
    "plant_ground": {name: 80 for name in ("Tree", "Flower", "Houseplant", "Plant", "Grass", "Mushroom")},
    "everyday": {name: 60 for name in ("Necklace", "Scarf", "Tie", "Shoe", "Bag", "Toy", "Food", "Car", "Bicycle",
                                       "Furniture", "Building", "Sculpture")},
    "person": {"Person": 1200, "Man": 200, "Woman": 200, "Boy": 100, "Girl": 100},
}
WORKERS = 32


def table(path, cache):
    local = cache / Path(path).name
    if not local.exists():
        local.write_bytes(negatives.get(f"{OI_TABLES}/{path}").content)
    return local


def candidates(cache, nb19):
    classes = pd.read_csv(table("v7/oidv7-class-descriptions.csv", cache))
    hierarchy = json.loads(table("2018_04/bbox_labels_600_hierarchy.json", cache).read_text())
    snake_mids = negatives.snake_mids(classes, hierarchy)
    mid_of, category_of, wanted = {}, {}, {}
    for category, labels in TARGETS.items():
        for name, n in labels.items():
            mids = classes.loc[classes["DisplayName"] == name, "LabelName"]
            if len(mids) != 1:
                print("skipped label (not exactly one class):", name, len(mids))
                continue
            mid_of[name], category_of[mids.iloc[0]], wanted[mids.iloc[0]] = mids.iloc[0], category, n
    human = pd.concat(pd.read_csv(table(f"v7/oidv7-{s}-annotations-human-imagelabels.csv", cache))
                      for s in OI_SPLITS.values())
    machine = pd.concat(pd.read_csv(table(f"v5/{s}-annotations-machine-imagelabels.csv", cache)) for s in OI_SPLITS)
    boxes = pd.concat(pd.read_csv(table(f"v5/{s}-annotations-bbox.csv", cache), usecols=["ImageID", "LabelName"])
                      for s in OI_SPLITS)
    snake_ids = (set(human.loc[human["LabelName"].isin(snake_mids) & (human["Confidence"] == 1), "ImageID"])
                 | set(machine.loc[machine["LabelName"].isin(snake_mids), "ImageID"])
                 | set(boxes.loc[boxes["LabelName"].isin(snake_mids), "ImageID"]))
    used = set(nb19.loc[nb19["source"].eq("openimages"), "source_id"].astype(str))
    labels = human[~human["ImageID"].isin(used)][["ImageID", "LabelName", "Confidence"]]
    snake_rows = pd.DataFrame({"ImageID": sorted(snake_ids), "LabelName": negatives.OI_SNAKE, "Confidence": 1.0})
    picked = negatives.open_images_candidates(pd.concat([labels, snake_rows]), wanted, snake_mids,
                                              seed=negatives.SEED + 1)
    if set(picked["ImageID"]) & (snake_ids | used):
        raise ValueError("a picked photo carries a snake label or is already in notebook 19")
    info = pd.concat(pd.read_csv(table(f"2018_04/{s}/{s}-images-with-rotation.csv", cache),
                                 usecols=["ImageID", "Subset", "OriginalLandingURL", "License", "Author"])
                     for s in OI_SPLITS).set_index("ImageID")
    oi = picked.join(info, on="ImageID")
    oi = oi[oi["License"] == OI_LICENCE].assign(Author=lambda r: r["Author"].map(html.unescape))
    names = classes.set_index("LabelName")["DisplayName"]
    return pd.DataFrame({
        "source": "openimages", "source_id": oi["ImageID"], "owner": oi["Author"], "licence": "cc-by-2.0",
        "url": oi["OriginalLandingURL"], "category": oi["LabelName"].map(category_of),
        "label": oi["LabelName"].map(names), "image_url": OI_IMAGES + "/" + oi["Subset"] + "/" + oi["ImageID"] + ".jpg",
        "attribution": "(c) " + oi["Author"] + ", CC BY 2.0", "group": "other", "split": "train"}), len(snake_ids)


def main(args):
    out = Path(args.out)
    cache = out / "tables"
    cache.mkdir(parents=True, exist_ok=True)
    nb19 = pd.read_csv(NB19)
    held_out_owners = set(nb19.loc[nb19["split"].isin(["val", "test"]), "owner"].astype(str))
    frame, snake_photos = candidates(cache, nb19)
    before = len(frame)
    frame = frame[~frame["owner"].astype(str).isin(held_out_owners)].reset_index(drop=True)
    frame["file"] = [f"images/{c}/openimages_{s}.jpg" for c, s in zip(frame["category"], frame["source_id"])]
    frame.to_csv(out / "candidates.csv", index=False)
    saved = download(frame, out)
    failed = int(saved["error"].ne("").sum())
    saved = saved[saved["error"].eq("")].copy()
    hashes = []
    for f in saved["file"]:
        with Image.open(out / f) as image:
            hashes.append(negatives.phash(image))
    saved["phash"] = hashes
    earlier = [pd.read_csv(p)["phash"] for p in EARLIER]
    bank = negatives.hash_bits(list(nb19["phash"]) + [h for part in earlier for h in part])
    near_old = np.array([int(negatives.distances(bank, negatives.hash_bits([h])[0]).min()) <= negatives.NEAR_DUP_BITS
                         for h in saved["phash"]])
    saved = saved[~near_old]
    repeats = negatives.near_duplicate_rows(list(saved["phash"]))
    saved = saved.drop(saved.index[sorted(repeats)]) if repeats else saved
    columns = [c for c in nb19.columns if c in saved.columns] + ["group"]
    saved[columns].to_csv(out / "manifest.csv", index=False)
    report = {"photos_with_a_snake_label_excluded": snake_photos, "picked": before,
              "dropped_owner_in_nb19_val_or_test": before - len(frame), "download_failed": failed,
              "near_copies_of_old": int(near_old.sum()), "near_copies_within_new": len(repeats), "kept": int(len(saved)),
              "by_category": saved["category"].value_counts().to_dict(), "owners": int(saved["owner"].nunique())}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
