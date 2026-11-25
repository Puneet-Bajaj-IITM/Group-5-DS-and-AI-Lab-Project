"""Extra M5 report numbers from saved test outputs only (CPU, no model run).

Naive baselines on the G3 slice, slices, held-out composition, leakage overlap, error counts,
misclassified examples and the confusion heatmap. Writes data/report_numbers.json and figures/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
WORK = REPORT.parents[1]
TEST = WORK / "results/m5/test"
for path in (WORK / "m5-wt/src", WORK / "results/m5/g3", WORK / "m5-targets", WORK / "m4-checks"):
    sys.path.insert(0, str(path))
import check_full_run as cf  # noqa: E402
import prior  # noqa: E402

N_CLASSES, SEED, TOP_N_HEATMAP, N_EXAMPLES = 184, 42, 30, 10
VENOMOUS_FAMILIES = ("Viperidae", "Elapidae")
OTHER_SPLITS = ("train", "val", "calib", "unknown_pool")
EXPECTED = {"frozen_top1": 0.7739900024584119, "frozen_macro": 0.7185399349281757,
            "b1_top1": 0.7800540850610506, "b1_macro": 0.7125378620517416, "photos": 12203}


def per_class(y, p):
    f1, _ = cf.per_class_f1(y, p, N_CLASSES)
    return f1


def main():
    classes = pd.read_csv(WORK / "m5-wt/data/classes.csv")
    cid = dict(zip(classes.species, classes.class_id))
    pred = pd.read_csv(TEST / "predictions_heldout.csv", dtype={"observer_id": str})
    meta = pd.read_csv(WORK / "kaggle/clean_build/out/metadata.csv", low_memory=False,
                       usecols=["path", "split", "observer_id", "obs_id", "sha1", "phash", "in_india"],
                       dtype={"observer_id": str, "obs_id": str})
    y = pred.species.map(cid).to_numpy()
    cols = {k: pred[f"{k}_pred"].map(cid).fillna(-1).astype(int).to_numpy() for k in ("frozen", "a3", "b1")}
    out = {}

    shared = np.array(sorted(json.loads((cf.B1_DIR / "final_metrics.json").read_text())
                             ["coverage"]["covered_class_ids"]))
    rows = (pred.created_at.fillna("").astype(str).ge(cf.SLICE_FROM) & np.isin(y, shared)).to_numpy()
    yy, groups = y[rows], pred.observer_id.to_numpy()[rows]
    macro = cf.shared_macro_f1(shared, N_CLASSES)
    check = {"photos": int(rows.sum()), "frozen_top1": cf.top1(yy, cols["frozen"][rows]),
             "frozen_macro": macro(yy, cols["frozen"][rows]), "b1_top1": cf.top1(yy, cols["b1"][rows]),
             "b1_macro": macro(yy, cols["b1"][rows])}
    check["matches_test_scores"] = all(abs(check[k] - v) < 1e-9 for k, v in EXPECTED.items())
    out["instrument_check_g3_slice"] = check

    counts = np.exp(prior.log_counts())
    most_common = int(np.argmax(counts))
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(N_CLASSES, size=len(y), p=counts / counts.sum())
    shuffled_frozen = rng.permutation(cols["frozen"][rows])
    naive = {"always the most common training species": np.full(rows.sum(), most_common),
             "random draw from training frequencies (prior only, seed 42)": sampled[rows],
             "control: frozen predictions shuffled": shuffled_frozen}
    out["most_common_training_species"] = classes.species[most_common]
    out["naive_baselines_g3_slice"] = {}
    for name, p in naive.items():
        out["naive_baselines_g3_slice"][name] = {
            "top1": cf.top1(yy, p), "macro_f1_62": macro(yy, p),
            "vs_b1_top1": cf.paired(yy, p, cols["b1"][rows], groups, cf.top1),
            "vs_b1_macro_f1": cf.paired(yy, p, cols["b1"][rows], groups, macro)}
    measurable = np.flatnonzero(np.bincount(y, minlength=N_CLASSES) >= cf.MIN_VAL)
    out["naive_baselines_all_heldout"] = {
        name: {"top1": float((p == y).mean()), "macro_f1_measurable": float(per_class(y, p)[measurable].mean())}
        for name, p in (("always the most common training species", np.full(len(y), most_common)),
                        ("random draw from training frequencies (prior only, seed 42)", sampled),
                        ("frozen", cols["frozen"]), ("A3", cols["a3"]))}

    m = meta.set_index("path")
    held = m.loc[pred.path]
    india = held.in_india.astype(str).str.lower().eq("true").to_numpy()
    out["india_vs_outside"] = {
        label: {"photos": int(mask.sum()), "frozen_top1": float((cols["frozen"][mask] == y[mask]).mean()),
                "a3_top1": float((cols["a3"][mask] == y[mask]).mean())}
        for label, mask in (("in India", india), ("outside India", ~india))}
    out["india_by_split"] = pd.crosstab(pred.split, india).rename(columns=str).to_dict()

    held_set = meta[meta.split.isin(["test", "test_extra"])]
    other = meta[meta.split.isin(OTHER_SPLITS)]
    out["leakage"] = {
        "heldout_rows_in_metadata": int(len(held_set)), "other_rows": int(len(other)),
        "other_splits_present": sorted(meta.split.unique().tolist()),
        "observers_heldout_and_other": int(len(set(held_set.observer_id) & set(other.observer_id))),
        "observers_heldout_and_train": int(len(set(held_set.observer_id)
                                               & set(meta[meta.split == "train"].observer_id))),
        "obs_id_heldout_and_other": int(len(set(held_set.obs_id) & set(other.obs_id))),
        "sha1_heldout_and_other": int(len(set(held_set.sha1) & set(other.sha1))),
        "phash_exact_heldout_and_other": int(len(set(held_set.phash.dropna()) & set(other.phash.dropna())))}

    per_species = pred.species.value_counts()
    out["composition"] = {
        "photos": int(len(pred)), "species": int(per_species.size),
        "species_with_5_or_more": int((per_species >= cf.MIN_VAL).sum()),
        "photos_per_species_median": float(per_species.median()), "max": int(per_species.max()),
        "min": int(per_species.min()), "top5": per_species.head(5).to_dict(),
        "share_top10_species": float(per_species.head(10).sum() / len(pred)),
        "who_category": pred.who_category.astype(str).value_counts().to_dict(),
        "families_top5": pred.family.value_counts().head(5).to_dict(),
        "photographers": int(pred.observer_id.nunique()),
        "split": pred.split.value_counts().to_dict()}

    listed = ~pred.who_category.astype(str).eq("not_listed")
    dangerous = pred.verdict_frozen_behind_gate.eq("dangerous")
    outside = ~listed & ~pred.family.isin(VENOMOUS_FAMILIES)
    out["errors"] = {
        "species_wrong": int((cols["frozen"] != y).sum()),
        "who_listed": int(listed.sum()), "who_listed_not_dangerous": int((listed & ~dangerous).sum()),
        "who_listed_gate_dropped": int((listed & pred.gate_snake.eq(0)).sum()),
        "harmless_outside_venomous_families": int(outside.sum()),
        "harmless_called_dangerous": int((outside & dangerous).sum()),
        "false_alarm_share": float((outside & dangerous).sum() / outside.sum()),
        "snake_check_dropped": int((~pred.snake_check_kept.astype(bool)).sum())}

    wrong = pred[cols["frozen"] != y].sort_values("frozen_p", ascending=False)
    picks = wrong.drop_duplicates("species").head(N_EXAMPLES)
    out["misclassified_examples"] = picks[["path", "split", "species", "who_category", "frozen_pred",
                                           "frozen_p", "frozen_top3", "verdict_frozen_behind_gate"]] \
        .to_dict(orient="records")
    listed_missed = pred[listed & ~dangerous & pred.gate_snake.eq(1)].sort_values("frozen_p", ascending=False)
    out["who_listed_missed_examples"] = listed_missed.drop_duplicates("species").head(5)[
        ["path", "species", "frozen_pred", "frozen_p", "verdict_frozen_behind_gate"]].to_dict(orient="records")

    heatmap(pred, per_species.head(TOP_N_HEATMAP).index.tolist())
    (REPORT / "data/report_numbers.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


def heatmap(pred, species):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = pred[pred.species.isin(species)]
    pred_col = sub.frozen_pred.where(sub.frozen_pred.isin(species), "other species")
    table = pd.crosstab(sub.species, pred_col).reindex(index=species, columns=species + ["other species"],
                                                        fill_value=0)
    share = table.div(table.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(13, 11), facecolor="white")
    im = ax.imshow(share.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(share.columns)), share.columns, rotation=90, fontsize=7, style="italic")
    ax.set_yticks(range(len(share.index)), [f"{s} ({n})" for s, n in zip(share.index, table.sum(axis=1))],
                  fontsize=7, style="italic")
    ax.set_xlabel("Predicted species (frozen M5)")
    ax.set_ylabel("True species (held-out photos)")
    ax.set_title("Held-out test: share of each true species' photos by predicted species\n"
                 f"{len(species)} species with the most held-out photos; rows sum to 1")
    fig.colorbar(im, ax=ax, fraction=0.03, label="share of row")
    fig.tight_layout()
    fig.savefig(REPORT / "figures/confusion_top30.png", dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
