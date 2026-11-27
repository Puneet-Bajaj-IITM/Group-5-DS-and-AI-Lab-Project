"""G4 on validation for one scored run: cut-off from calibration snakes, then false snakes and snakes kept.

Rule: a photo is a snake when the model's P(snake) >= cut, the cut being the highest value at which at least 99% of
calibration snakes pass (m5-g4-probe/analyse.cutoff). Intervals: 1,000 resamples of whole owners / photographers,
seed 42 (analyse.owner_interval). Controls that must fail: a random score, and a logistic head on the model's own
features fitted with shuffled labels (half the calibration photographers + training non-snakes; cut-off on the
other half, as in analyse.py).

Usage: python -I evaluate.py --scores <folder from score_onnx.py> --out <json> [--name <label>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "m5-g4-probe"))
import analyse as a  # noqa: E402

TARGET = a.G4_KEEP_MARK
SNAKE_GROUPS = {
    "sea snakes and sea kraits": lambda f: f["genus"].isin(["Hydrophis", "Laticauda"]),
    "blind snakes": lambda f: f["family"].isin(["Typhlopidae", "Gerrhopilidae", "Leptotyphlopidae"]),
    "shieldtails": lambda f: f["family"].eq("Uropeltidae"),
    "other water snakes": lambda f: f["family"].isin(["Homalopsidae", "Acrochordidae"]),
    "vine snakes and bronzebacks": lambda f: f["genus"].isin(["Ahaetulla", "Dendrelaphis"]),
}


def load(folder):
    folder = Path(folder)
    sets = {}
    for name in ("calib", "val", "nonsnake_val", "nonsnake_train"):
        path = folder / f"scores_{name}.csv"
        if path.exists():
            frame = pd.read_csv(path, dtype={"group": str})
            features = folder / f"features_{name}.npy"
            if features.exists():
                frame["features"] = list(np.load(features).astype(np.float32))
            sets[name] = frame
    return sets


def rule_result(name, calib_scores, val_snakes, val_snake_scores, nonsnakes, nonsnake_scores, rng):
    cut = a.cutoff(calib_scores, np.ones(len(calib_scores), dtype=bool), TARGET)
    called, kept = nonsnake_scores >= cut, val_snake_scores >= cut
    row = {"rule": name, "target": TARGET, "cutoff": cut, "calibration_kept": float((calib_scores >= cut).mean()),
           "false_snake_val": float(called.mean()),
           "false_snake_val_ci": a.owner_interval(called, nonsnakes["group"].to_numpy(), rng),
           "false_snake_count": int(called.sum()), "nonsnake_photos": int(len(called)),
           "snakes_kept_val": float(kept.mean()), "snakes_kept_val_ci": a.owner_interval(kept, val_snakes["group"].to_numpy(), rng),
           "snake_photos": int(len(kept))}
    row["meets_g4_on_val"] = row["false_snake_val"] <= a.G4_FALSE_SNAKE_MARK and row["snakes_kept_val"] >= TARGET
    row["false_snake_by_category"] = {c: {"photos": int(m.sum()), "called_a_snake": round(float(called[m].mean()), 3)}
                                      for c, m in ((c, nonsnakes["category"].eq(c).to_numpy())
                                                   for c in sorted(nonsnakes["category"].unique()))}
    source = np.where(nonsnakes["path"].str.contains("inaturalist"), "iNaturalist lookalikes", "Open Images")
    row["false_snake_by_source"] = {s: {"photos": int((source == s).sum()),
                                        "called_a_snake": round(float(called[source == s].mean()), 3)}
                                    for s in ("Open Images", "iNaturalist lookalikes")}
    row["snakes_kept_by_group"] = {}
    for group, select in SNAKE_GROUPS.items():
        m = select(val_snakes).to_numpy()
        row["snakes_kept_by_group"][group] = {
            "photos": int(m.sum()), "kept": round(float(kept[m].mean()), 4) if m.any() else None,
            "ci": a.owner_interval(kept[m], val_snakes["group"].to_numpy()[m], rng) if m.any() else None}
    return row


def species_misses(val_snakes, kept, top=12):
    frame = pd.DataFrame({"species": val_snakes["category"], "kept": kept})
    per = frame.groupby("species")["kept"].agg(["count", "mean"])
    per["missed"] = (per["count"] * (1 - per["mean"])).round().astype(int)
    return per.sort_values("missed", ascending=False).head(top).reset_index().to_dict("records")


def shuffled_control(sets, rng):
    calib = sets["calib"]
    owners = calib["group"].unique()
    half = set(np.random.default_rng(a.SEED).permutation(owners)[: len(owners) // 2])
    fit, cut_half = calib[calib["group"].isin(half)], calib[~calib["group"].isin(half)]
    train_non = sets["nonsnake_train"]
    x = np.vstack(list(fit["features"]) + list(train_non["features"]))
    y = np.random.default_rng(a.SEED).permutation(np.r_[np.ones(len(fit)), np.zeros(len(train_non))])
    head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(x, y)
    score = lambda f: head.decision_function(np.vstack(list(f["features"])))
    return score(cut_half), score


def main(args):
    rng = np.random.default_rng(a.SEED)
    sets = load(args.scores)
    calib, val, non = sets["calib"], sets["val"], sets["nonsnake_val"]
    result = {"name": args.name, "photos": {k: len(v) for k, v in sets.items()}}
    both = pd.concat([non, val])
    result["auroc_val"] = float(roc_auc_score(both["label"], both["p_snake"]))
    main_row = rule_result("P(snake) >= cut", calib["p_snake"].to_numpy(), val, val["p_snake"].to_numpy(), non,
                           non["p_snake"].to_numpy(), rng)
    kept = val["p_snake"].to_numpy() >= main_row["cutoff"]
    main_row["most_missed_species"] = species_misses(val, kept)
    rows = [main_row]
    random_rng = np.random.default_rng(a.SEED)
    rows.append(rule_result("control: random score", random_rng.random(len(calib)), val, random_rng.random(len(val)),
                            non, random_rng.random(len(non)), rng))
    if "nonsnake_train" in sets and "features" in sets["calib"]:
        cut_scores, score = shuffled_control(sets, rng)
        rows.append(rule_result("control: head on the model's features, shuffled labels", cut_scores, val, score(val),
                                non, score(non), rng))
    result["rules"] = rows
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    for r in rows:
        print(f"{r['rule']}: false snakes {r['false_snake_val']:.4f} {np.round(r['false_snake_val_ci'], 4)} "
              f"({r['false_snake_count']}/{r['nonsnake_photos']}), kept {r['snakes_kept_val']:.4f} "
              f"{np.round(r['snakes_kept_val_ci'], 4)}, calib kept {r['calibration_kept']:.4f}, "
              f"meets G4 {r['meets_g4_on_val']}")
    print("AUROC", round(result["auroc_val"], 4))
    print(json.dumps({k: main_row[k] for k in ("false_snake_by_source", "false_snake_by_category",
                                               "snakes_kept_by_group")}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="")
    main(parser.parse_args())
