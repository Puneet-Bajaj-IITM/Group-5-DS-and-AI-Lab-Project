"""Builds assets/species.json from the repo's species cards and class table.

Run from the app folder after the class list or danger table changes:
    python3 tool/build_species_asset.py
"""
import csv
import json
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
REPO = APP.parent

classes = {int(r["class_id"]): r for r in csv.DictReader(open(REPO / "data/classes.csv"))}
cards = list(csv.DictReader(open(REPO / "data/m3/kb/species_cards.csv")))

species = []
for card in sorted(cards, key=lambda r: int(r["class_id"])):
    cid = int(card["class_id"])
    species.append({
        "id": cid,
        "species": card["species"],
        "common": card["common_name"] or None,
        "genus": classes[cid]["genus"],
        "family": card["family"],
        # who_category: "1" (WHO Category 1), "2" (WHO Category 2) or "not_listed"
        "who": card["who_category"],
        "dangerous": card["dangerous"] == "True",
        "basis": card["basis"],
    })

assert [s["id"] for s in species] == list(range(len(species))), "class ids must be 0..C-1"
out = APP / "assets/species.json"
out.write_text(json.dumps(species, ensure_ascii=False, separators=(",", ":")))
print(f"wrote {len(species)} species, {sum(s['dangerous'] for s in species)} dangerous -> {out}")
