"""Second round of iNaturalist TRAINING negatives for the G4 retry: more of the lookalikes that still pass.

Chosen after seeing validation errors of distil_oi (caecilians, eels, legless lizards, other animals), so any
number from a model trained on them is optimistic. Same rules as collect_negatives.py: taxon ids only (each
record must carry the expected name; ids found by walking parent ids), research grade, one photo per observation;
owners with any photo in notebook 19's validation or test split are dropped; photos already in notebook 19, the
hard negatives or the first round are dropped; pHash more than 8 bits from all of those and from each other.
Open licences (CC0 / BY / BY-SA) are taken first; BY-NC / BY-NC-SA only fill what is left.

Usage: python -I collect_negatives_v2.py --out <folder>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_negatives import LICENCES, ROOT, download, negatives, observations  # noqa: E402

NB19_MANIFEST = ROOT / "kaggle/m4_runs/non_snake/out_full/manifest.csv"
EARLIER = (ROOT / "m5-g4-probe/hard_negatives/manifest.csv", ROOT / "results/m5/g4/data/negatives_v1/manifest.csv",
           ROOT / "results/m5/g4/data/openimages_v1/manifest.csv")
OPEN_LICENCES = {"cc0", "cc-by", "cc-by-sa"}

# name, taxon id, category, group, wanted
TAXA = (
    ("Gymnophiona", 27880, "caecilian", "lookalike", 1100),
    ("Muraenidae", 47313, "eel", "lookalike", 800),
    ("Anguilliformes", 47314, "eel", "lookalike", 600),
    ("Synbranchiformes", 53512, "eel", "lookalike", 600),
    ("Anguis", 32036, "legless_lizard", "lookalike", 500),
    ("Ophisaurus", 31978, "legless_lizard", "lookalike", 400),
    ("Pseudopus", 31974, "legless_lizard", "lookalike", 400),
    ("Pygopodidae", 36925, "legless_lizard", "lookalike", 500),
    ("Amphisbaenia", 85550, "worm_lizard", "lookalike", 600),
    ("Syngnathoidei", 1303580, "pipefish_seahorse", "other", 400),
)


def open_first(frame, wanted):
    """Owners take turns, open licences first; non-commercial ones only fill the remainder."""
    is_open = frame["licence"].isin(OPEN_LICENCES)
    first = negatives.take_turns(frame[is_open], wanted)
    rest = negatives.take_turns(frame[~is_open], wanted - len(first))
    return pd.concat([first, rest])


def candidates(old_sources, held_out_owners):
    picked = []
    for name, taxon_id, category, group, wanted in TAXA:
        record = negatives.inat_taxon(taxon_id)
        if record["name"] != name:
            raise ValueError(f"taxon {taxon_id} is {record['name']}, not {name}")
        rows = [row for obs in observations(taxon_id, {}, 2 * wanted)  # twice round one, past the pages it used
                if (row := negatives.photo_row(obs, taxon_id, LICENCES)) is not None]
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        frame = frame[~frame["source_id"].isin(old_sources) & ~frame["owner"].astype(str).isin(held_out_owners)]
        chosen = open_first(frame, wanted)
        old_sources |= set(chosen["source_id"])
        print(name, "usable", len(rows), "new after filters", len(frame), "chosen", len(chosen), flush=True)
        picked.append(chosen.assign(taxon=name, taxon_id=taxon_id, category=category, group=group, label=name))
    out = pd.concat(picked, ignore_index=True).drop_duplicates("source_id")
    out["file"] = [f"images/{c}/inaturalist_{s}.jpg" for c, s in zip(out["category"], out["source_id"])]
    return out.assign(split="train")


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nb19 = pd.read_csv(NB19_MANIFEST)
    earlier = [pd.read_csv(p) for p in EARLIER]
    held_out_owners = set(nb19.loc[nb19["split"].isin(["val", "test"]), "owner"].astype(str))
    old_sources = set(nb19["source_id"].astype(str)).union(*(set(e["source_id"].astype(str)) for e in earlier))
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
    bank = negatives.hash_bits(list(nb19["phash"]) + [h for e in earlier for h in e["phash"]])
    near_old = np.array([int(negatives.distances(bank, negatives.hash_bits([h])[0]).min()) <= negatives.NEAR_DUP_BITS
                         for h in saved["phash"]])
    saved = saved[~near_old]
    repeats = negatives.near_duplicate_rows(list(saved["phash"]))
    saved = saved.drop(saved.index[sorted(repeats)]) if repeats else saved
    columns = [c for c in nb19.columns if c in saved.columns] + ["group", "taxon_id"]
    saved[columns].to_csv(out / "manifest.csv", index=False)
    report = {"candidates": int(len(frame)), "download_failed": failed, "near_copies_of_old": int(near_old.sum()),
              "near_copies_within_new": len(repeats), "kept": int(len(saved)),
              "by_category": saved["category"].value_counts().to_dict(),
              "by_taxon": saved["label"].value_counts().to_dict(), "owners": int(saved["owner"].nunique()),
              "owners_in_nb19_val_or_test": int(saved["owner"].astype(str).isin(held_out_owners).sum()),
              "licences": saved["licence"].value_counts().to_dict()}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
