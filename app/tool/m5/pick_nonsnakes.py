"""Picks the VALIDATION non-snake golden photos (notebook 19's val split; the test split is never read):
one per category for eight categories, plus two the frozen snake check called a snake on validation.
Writes nonsnake_golden.csv beside it: the served URL, licence and credit of each."""
import csv
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WORKSPACE = REPO.parent  # the team's working folder, which holds the M4 Kaggle downloads
MANIFEST = WORKSPACE / "kaggle/m4_runs/non_snake/out_full/manifest.csv"
SCORES = REPO / "results/m5/g4/runs/distil_oi/scores_kernel/scores_nonsnake_val.csv"
CUTOFF = 0.20484977503483762
CATEGORIES = ("person", "rope_hose_belt", "cat", "dog", "hand_arm", "eel", "legless_lizard", "caecilian")
FALSE_SNAKES = 2
SEED = 42

rows = [r for r in csv.DictReader(MANIFEST.open()) if r["split"] == "val"]
scores = {r["path"]: float(r["p_snake"]) for r in csv.DictReader(SCORES.open())}
rng = random.Random(SEED)
picked = []
for category in CATEGORIES:
    pool = [r for r in rows if r["category"] == category and scores[r["file"]] < CUTOFF]
    picked.append(rng.choice(pool))
false_snakes = [r for r in rows if scores[r["file"]] >= CUTOFF and r not in picked]
picked += rng.sample(false_snakes, FALSE_SNAKES)
with (HERE / "nonsnake_golden.csv").open("w", newline="") as handle:
    out = csv.writer(handle)
    out.writerow(["photo_id", "url", "licence", "credit", "category", "val_p_snake"])
    for r in picked:
        out.writerow([f"val_nonsnake_{r['source_id']}", r["image_url"], r["licence"], r["attribution"],
                      r["category"], round(scores[r["file"]], 4)])
print("\n".join(f"{r['category']} {r['source']} {scores[r['file']]:.3f} {r['image_url']}" for r in picked))
