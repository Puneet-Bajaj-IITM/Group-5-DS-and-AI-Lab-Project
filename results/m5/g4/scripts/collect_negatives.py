"""More non-snake TRAINING photos from iNaturalist for the G4 snake / not-snake model.

Queried by taxon id only (each id's record must carry the expected name). Research grade, one photo per
observation, licences CC0 / BY / BY-SA / BY-NC / BY-NC-SA (training only; the snake photos themselves include NC).
A photo is dropped if notebook 19 or the earlier hard-negative set already has it, if its photographer has any photo
in notebook 19's validation or test split, or if it is within 8 pHash bits of any of those photos or of another new
photo. Writes <out>/manifest.csv in notebook 19's columns plus `group` (lookalike or other), every row split "train".

Usage: python -I collect_negatives.py --out <folder>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "m4-wt/src"))
from serpentguard import negatives  # noqa: E402

NB19_MANIFEST = ROOT / "kaggle/m4_runs/non_snake/out_full/manifest.csv"
HARD_MANIFEST = ROOT / "m5-g4-probe/hard_negatives/manifest.csv"
LICENCES = ("cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa")
INDIA_PLACE_ID = 6681
SPREAD_STARTS = 24
FULL_PAGING_UP_TO = 3000
MAX_OBSERVATION_ID = 330_000_000
WORKERS = 32

# name, taxon id, category, group, wanted, extra query params
TAXA = (
    ("Gymnophiona", 27880, "caecilian", "lookalike", 1100, {}),
    ("Anguinae", 797527, "legless_lizard", "lookalike", 900, {}),
    ("Anguilliformes", 47314, "eel", "lookalike", 900, {}),
    ("Synbranchiformes", 53512, "eel", "lookalike", 500, {}),
    ("Crassiclitellata", 152943, "earthworm", "lookalike", 900, {}),
    ("Squamata", 26172, "other_reptile_india", "other", 900, {"place_id": INDIA_PLACE_ID,
                                                                 "without_taxon_id": negatives.SERPENTES_ID}),
    ("Animalia", 1, "other_animal_india", "other", 900, {"place_id": INDIA_PLACE_ID,
                                                          "without_taxon_id": negatives.SERPENTES_ID}),
)


def observations(taxon_id, extra, wanted):
    """Pages of research-grade observations from evenly spread starting ids, until about 2x wanted are found."""
    found, seen = [], set()
    base = {"taxon_id": taxon_id, "quality_grade": "research", "photo_license": ",".join(LICENCES),
            "photos": "true", "order_by": "id", "order": "asc", "per_page": negatives.INAT_PAGE, **extra}
    total = negatives.get_json(f"{negatives.INAT_API}/observations", {**base, "per_page": 0})["total_results"]
    every_page = total <= FULL_PAGING_UP_TO
    spread = np.linspace(0, MAX_OBSERVATION_ID, SPREAD_STARTS, endpoint=False).astype(int)
    starts = iter(np.random.default_rng(negatives.SEED).permutation(spread))
    start = 0 if every_page else next(starts)
    while True:
        page = negatives.get_json(f"{negatives.INAT_API}/observations", {**base, "id_above": int(start)})["results"]
        for obs in page:
            if obs["id"] not in seen:
                seen.add(obs["id"])
                found.append(obs)
        if not every_page and len(found) >= 2 * wanted:
            return found
        if every_page:
            if not page:
                return found
            start = page[-1]["id"]
        else:
            start = next(starts, None)
            if start is None:
                return found


def candidates(old_sources, held_out_owners):
    picked = []
    for name, taxon_id, category, group, wanted, extra in TAXA:
        record = negatives.inat_taxon(taxon_id)
        if record["name"] != name:
            raise ValueError(f"taxon {taxon_id} is {record['name']}, not {name}")
        rows = [row for obs in observations(taxon_id, extra, wanted)
                if (row := negatives.photo_row(obs, taxon_id, LICENCES)) is not None]
        frame = pd.DataFrame(rows)
        frame = frame[~frame["source_id"].isin(old_sources) & ~frame["owner"].astype(str).isin(held_out_owners)]
        chosen = negatives.take_turns(frame, wanted)
        print(name, "observations with a usable photo", len(rows), "chosen", len(chosen), flush=True)
        picked.append(chosen.assign(taxon=name, taxon_id=taxon_id, category=category, group=group, label=name))
    out = pd.concat(picked, ignore_index=True).drop_duplicates("source_id")
    out["file"] = [f"images/{c}/inaturalist_{s}.jpg" for c, s in zip(out["category"], out["source_id"])]
    return out.assign(split="train")


def whole(path):
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:  # missing or cut short by the interruption
        path.unlink(missing_ok=True)
        return False


def download(frame, out):
    """negatives.download_photos, but a photo saved whole by an earlier, interrupted run is not fetched again."""
    have = frame[[whole(out / f) for f in frame["file"]]]
    fresh = negatives.download_photos(frame[~frame.index.isin(have.index)], out, workers=WORKERS)
    sizes = [Image.open(out / f).size for f in have["file"]]
    have = have.assign(width=[w for w, _ in sizes], height=[h for _, h in sizes], error="")
    return pd.concat([have, fresh]).loc[frame.index]


def main(args):
    out = Path(args.out)
    nb19, hard = pd.read_csv(NB19_MANIFEST), pd.read_csv(HARD_MANIFEST)
    held_out_owners = set(nb19.loc[nb19["split"].isin(["val", "test"]), "owner"].astype(str))
    old_sources = set(nb19["source_id"].astype(str)) | set(hard["source_id"].astype(str))
    frame = candidates(old_sources, held_out_owners)
    frame.to_csv(out / "candidates.csv", index=False)
    saved = download(frame, out)
    failed = int(saved["error"].ne("").sum())
    saved = saved[saved["error"].eq("")].copy()
    hashes = []
    for f in saved["file"]:
        with Image.open(out / f) as image:
            hashes.append(negatives.phash(image))
    saved["phash"] = hashes
    bank = negatives.hash_bits(list(nb19["phash"]) + list(hard["phash"]))
    near_old = np.array([int(negatives.distances(bank, negatives.hash_bits([h])[0]).min()) <= negatives.NEAR_DUP_BITS
                         for h in saved["phash"]])
    saved = saved[~near_old]
    repeats = negatives.near_duplicate_rows(list(saved["phash"]))
    saved = saved.drop(saved.index[sorted(repeats)]) if repeats else saved
    for f in set(frame["file"]) - set(saved["file"]):
        (out / f).unlink(missing_ok=True)
    columns = [c for c in nb19.columns if c in saved.columns] + ["group", "taxon_id"]
    saved[columns].to_csv(out / "manifest.csv", index=False)
    report = {"candidates": int(len(frame)), "download_failed": failed, "near_copies_of_old": int(near_old.sum()),
              "near_copies_within_new": len(repeats), "kept": int(len(saved)),
              "by_category": saved["category"].value_counts().to_dict(), "owners": int(saved["owner"].nunique()),
              "owners_in_nb19_val_or_test": int(saved["owner"].astype(str).isin(held_out_owners).sum()),
              "licences": saved["licence"].value_counts().to_dict()}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
