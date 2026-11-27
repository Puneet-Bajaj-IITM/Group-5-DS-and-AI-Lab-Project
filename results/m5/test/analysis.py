"""Error-analysis tables for RESULTS.md, from the per-photo files score_test.py wrote (no new goal metric).

Writes per_species_heldout.csv (photos, recall and F1 of the frozen model, A3 and B1 per species), confusions_heldout.csv
(the most frequent true -> predicted pairs), and analysis_heldout.json (front-end slices: mask or no mask, gate and
snake-check drops; water and sea snakes; WHO-listed misses by species).

Usage: m3-final/.venv/bin/python results/m5/test/analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "m4-checks"))
import check_full_run as cf  # noqa: E402

N_CLASSES, TOP_CONFUSIONS, MIN_PHOTOS = 184, 25, 5
WATER = {"sea snakes and sea kraits": lambda f: f["genus"].isin(["Hydrophis", "Laticauda"]),
         "other water snakes (Homalopsidae, Acrochordidae)": lambda f: f["family"].isin(["Homalopsidae", "Acrochordidae"])}


def f1_of(true, pred):
    f1, _ = cf.per_class_f1(true, pred, N_CLASSES)
    return f1


def main():
    photos = pd.read_csv(HERE / "predictions_heldout.csv", dtype={"observer_id": str})
    frame = pd.read_csv(HERE / "kernel_out/out/held_out_frame.csv")
    photos = photos.merge(frame[["path", "mask_valid"]], on="path", validate="one_to_one")
    meta = pd.read_csv(WORK / "kaggle/clean_build/out/metadata.csv", usecols=["path", "genus"])
    photos = photos.merge(meta, on="path", how="left", validate="one_to_one")
    classes = pd.read_csv(WORK / "m5-wt/data/classes.csv")
    id_of = dict(zip(classes["species"], classes["class_id"]))
    y = photos["class_id"].to_numpy()
    pred = {name: photos[col].map(id_of).fillna(-1).astype(int).to_numpy()
            for name, col in (("frozen", "frozen_pred"), ("a3", "a3_pred"), ("b1", "b1_pred"))}
    b1_rows = pred["b1"] >= 0
    f1 = {name: f1_of(y, p) for name, p in pred.items() if name != "b1"}
    f1_b1 = f1_of(y[b1_rows], pred["b1"][b1_rows])
    rows = []
    for c, group in photos.groupby("class_id"):
        right = group["frozen_pred"].eq(group["species"])
        wrong = group.loc[~right, "frozen_pred"].value_counts()
        rows.append({"species": group["species"].iloc[0], "family": group["family"].iloc[0],
                     "who_category": group["who_category"].iloc[0], "photos": len(group),
                     "test_photos": int(group["split"].eq("test").sum()),
                     "frozen_recall": float(right.mean()), "frozen_f1": float(f1["frozen"][c]),
                     "a3_recall": float(group["a3_pred"].eq(group["species"]).mean()), "a3_f1": float(f1["a3"][c]),
                     "b1_f1_on_its_photos": float(f1_b1[c]) if b1_rows[y == c].any() else np.nan,
                     "most_common_wrong": wrong.index[0] if len(wrong) else "", "most_common_wrong_n": int(wrong.iloc[0]) if len(wrong) else 0})
    per_species = pd.DataFrame(rows).sort_values(["photos"], ascending=False)
    per_species.to_csv(HERE / "per_species_heldout.csv", index=False)
    wrong = photos[photos["frozen_pred"].ne(photos["species"])]
    confusions = (wrong.groupby(["species", "frozen_pred"]).size().rename("photos").reset_index()
                  .sort_values("photos", ascending=False).head(TOP_CONFUSIONS))
    confusions.to_csv(HERE / "confusions_heldout.csv", index=False)
    measurable = per_species[per_species["photos"] >= MIN_PHOTOS]

    def top1(mask):
        m = mask.to_numpy() if hasattr(mask, "to_numpy") else mask
        return {"photos": int(m.sum()), "frozen_top1": float((pred["frozen"][m] == y[m]).mean()) if m.any() else None,
                "a3_top1": float((pred["a3"][m] == y[m]).mean()) if m.any() else None}

    listed = photos["who_category"].astype(str).ne("not_listed")
    gated = photos["gate_snake"].notna()
    missed = listed & gated & photos["verdict_frozen_behind_gate"].ne("dangerous")
    out = {
        "front_end": {"with_mask": top1(photos["mask_valid"].astype(bool)),
                      "without_mask_whole_or_box": top1(~photos["mask_valid"].astype(bool)),
                      "gate_drops": int((gated & photos["gate_snake"].eq(0)).sum()),
                      "gate_calls": int(gated.sum()),
                      "snake_check_drops": int((~photos["snake_check_kept"].astype(bool)).sum()),
                      "both_drop": int((gated & photos["gate_snake"].eq(0) & ~photos["snake_check_kept"].astype(bool)).sum())},
        "by_split": {s: top1(photos["split"].eq(s)) for s in ("test", "test_extra")},
        "water_snakes": {name: {**top1(select(photos)), "snake_check_kept": float(photos.loc[select(photos), "snake_check_kept"].mean())}
                         for name, select in WATER.items()},
        "who_listed_missed_behind_gate": {"listed": int((listed & gated).sum()), "missed": int(missed.sum()),
                                          "by_species": photos[missed]["species"].value_counts().head(15).to_dict(),
                                          "verdicts_given": photos[missed]["verdict_frozen_behind_gate"].value_counts().to_dict()},
        "worst_species_measurable": measurable.sort_values("frozen_f1").head(15)[
            ["species", "photos", "frozen_f1", "a3_f1", "most_common_wrong", "most_common_wrong_n"]].to_dict("records"),
        "species_f1_zero_measurable": int((measurable["frozen_f1"] == 0).sum()),
        "species_measurable": int(len(measurable)),
    }
    (HERE / "analysis_heldout.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
