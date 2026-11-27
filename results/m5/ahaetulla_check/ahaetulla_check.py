"""Are the held-out Ahaetulla oxyrhynca / A. borealis photos that the frozen model calls A. prasina labelled right?

Picks 15 of each (seed 42) from the per-photo test predictions and asks iNaturalist for each observation's
current community taxon and quality grade, then compares with the label we trained and scored against.
The labels were taken when the dataset was built (October 2026); a changed community ID means our label is stale.
Usage: python results/m5/ahaetulla_check/ahaetulla_check.py
"""
import csv
import json
import random
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# Both inputs are too large for git (results/m5/NOT_IN_GIT.txt); the metadata is in the Kaggle dataset.
PREDICTIONS = REPO / "results/m5/test/predictions_heldout.csv"
METADATA = REPO / "data/metadata.csv"
SPECIES = ("Ahaetulla oxyrhynca", "Ahaetulla borealis")
PREDICTED = "Ahaetulla prasina"
PER_SPECIES, SEED = 15, 42
API = "https://api.inaturalist.org/v1/observations?per_page=50&id="
PAUSE_S = 1.0


def main():
    picked = []
    rows = list(csv.DictReader(open(PREDICTIONS)))
    for species in SPECIES:
        pool = sorted(r["path"] for r in rows if r["species"] == species and r["frozen_pred"] == PREDICTED)
        picked += [(species, p) for p in random.Random(SEED).sample(pool, PER_SPECIES)]
    wanted = {p for _, p in picked}
    meta = {r["path"]: r for r in csv.DictReader(open(METADATA)) if r["path"] in wanted}
    ids = sorted({meta[p]["obs_id"] for p in wanted})
    with urllib.request.urlopen(urllib.request.Request(API + ",".join(ids), headers={"User-Agent": "course-project"})) as r:
        found = {str(o["id"]): o for o in json.load(r)["results"]}
    time.sleep(PAUSE_S)
    out = []
    for species, path in picked:
        m = meta[path]
        o = found.get(m["obs_id"])
        current = (o or {}).get("community_taxon") or (o or {}).get("taxon") or {}
        out.append({"label": species, "path": path, "obs_id": m["obs_id"], "observed_on": m["observed_on"],
                    "in_india": m["in_india"], "url": f"https://www.inaturalist.org/observations/{m['obs_id']}",
                    "current_taxon": current.get("name"), "quality_grade": (o or {}).get("quality_grade"),
                    "identifications": len((o or {}).get("identifications", [])),
                    "label_still_current": current.get("name") == species if o else None})
    with open(HERE / "ahaetulla_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    summary = {s: {"checked": sum(r["label"] == s for r in out),
                   "label_still_current": sum(r["label"] == s and r["label_still_current"] is True for r in out),
                   "now_another_taxon": sorted({r["current_taxon"] for r in out
                                                if r["label"] == s and r["label_still_current"] is False}),
                   "not_found": sum(r["label"] == s and r["label_still_current"] is None for r in out),
                   "research_grade": sum(r["label"] == s and r["quality_grade"] == "research" for r in out)}
               for s in SPECIES}
    (HERE / "ahaetulla_check.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
