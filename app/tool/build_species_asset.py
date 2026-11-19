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

# Regional names (tool/regional_names.csv): only names in wide use, kept as a
# draft until a native speaker of each language has checked them. WHO-listed
# species with no widely used regional name are left out rather than guessed.
LANGS = ["hi", "mr", "bn", "ta", "te", "kn", "ml"]
regional = {}
for r in csv.DictReader(open(APP / "tool/regional_names.csv", encoding="utf-8")):
    assert r["lang"] in LANGS, r
    regional.setdefault(r["species"], []).append({"lang": r["lang"], "name": r["name"], "latin": r["latin"]})
known = {c["species"] for c in cards}
assert set(regional) <= known, set(regional) - known

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
        "regional": sorted(regional.get(card["species"], []), key=lambda n: LANGS.index(n["lang"])),
    })

assert [s["id"] for s in species] == list(range(len(species))), "class ids must be 0..C-1"
out = APP / "assets/species.json"
out.write_text(json.dumps(species, ensure_ascii=False, separators=(",", ":")))
print(f"wrote {len(species)} species, {sum(s['dangerous'] for s in species)} dangerous -> {out}")
listed = [s for s in species if s["who"] != "not_listed"]
print(f"regional names: {sum(bool(s['regional']) for s in listed)} of {len(listed)} WHO-listed species,"
      f" {sum(bool(s['regional']) for s in species)} species in all")
print("WHO-listed without one:", ", ".join(s["species"] for s in listed if not s["regional"]))
