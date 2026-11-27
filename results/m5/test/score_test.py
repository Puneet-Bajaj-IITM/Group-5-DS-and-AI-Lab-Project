"""Every M5 goal for the frozen pipeline (FREEZE.md), on validation (the instrument check) or the held-out test split.

No new metric: G3 is g3_candidates' computation (check_full_run.paired, top1, shared_macro_f1, photographer bootstrap
seed 42); G2 is check_full_run.per_class_f1 on measurable species; safety is refit_new.gated / interval and
decide.recall_lower_bound; G4 is results/m5/g4/scripts/evaluate.rule_result; G1 is robustness_metrics' own report.

--split val reads the saved validation outputs and must reproduce the READMEs' numbers (it asserts them).
--split test reads the kernel's held-out outputs (kernel_out/out) and writes test_scores.json and the per-photo files.

Usage: m3-final/.venv/bin/python results/m5/test/score_test.py --split val|test
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
G3, G4 = WORK / "results/m5/g3", WORK / "results/m5/g4"
for path in (WORK / "m5-wt/src", G3, WORK / "m5-targets", G4 / "scripts"):
    sys.path.insert(0, str(path))
import g3_candidates as g3c  # noqa: E402  (imports g3_ensemble, check_full_run and prior)
import refit_new  # noqa: E402
import evaluate as g4e  # noqa: E402  (imports m5-g4-probe/analyse)
from serpentguard import decide  # noqa: E402

cf, g3, prior, a = g3c.cf, g3c.g3, g3c.prior, g4e.a
M5_KD = G3 / "out_kd/M5_KD"
A3_FULL = WORK / "kaggle/m4_runs/full_a3/out_jayesh/M4_FULL"
APP_PIPELINE = WORK / "m4-final/for-jayesh/models/pipeline.json"
KERNEL = HERE / "kernel_out/out"
N_CLASSES, TAU, SNAKE_CUT = 184, 0.5, 0.20484977503483762
F1_MARK, MIN_PHOTOS = 0.5, cf.MIN_VAL
MARKS = {"G1": 0.084, "G2": 80, "safety": 0.95, "G4_false": 0.05, "G4_kept": 0.99}
FROZEN = "M5_KD + flip + prior (frozen)"
BEFORE = "A3 on M4 crops (shipped classifier)"
CONTROL_SEED = 0
G4_CLASSES = ("snake", "lookalike", "other")
EXPECTED_VAL = {  # README numbers the validation dry run must reproduce
    "g3": json.loads((G3 / "g3_candidates.json").read_text())["candidates"]["KD + flip + prior"],
    "g3_a3_m3": {"top1": -0.0808, "macro_f1": -0.0249},
    "g2": {FROZEN: 81, "M5_KD plain": 88},
    "safety": json.loads((G3 / "refit_new.json").read_text())["KD flip + prior"]["refit"]["val_behind_app_gate"],
    "g4": json.loads((G4 / "runs/distil_oi/g4_eval.json").read_text())["rules"][0],
    "g1": 0.05483146067415734,
}


def softmax(logits):
    return g3.softmax(logits, 1.0)


def metadata():
    meta = pd.read_csv(g3.METADATA, usecols=["path", "split", "created_at", "species", "genus", "family",
                                             "observer_id", "who_category"], dtype={"observer_id": str})
    return meta.set_index("path")


def aligned(saved, paths):
    """Every array of a saved npz (or dict) in the order of `paths`."""
    index = pd.Series(np.arange(len(saved["paths"])), index=np.asarray(saved["paths"]).astype(str)).reindex(paths)
    assert index.notna().all(), f"{int(index.isna().sum())} photos missing"
    rows = index.to_numpy(int)
    return {k: np.asarray(v)[rows] for k, v in saved.items() if len(np.asarray(v)) == len(saved["paths"])}


def load_sets(split):
    """{name: outputs} aligned on the frozen model's plain outputs, plus B1 probs, gate calls and snake scores."""
    if split == "val":
        files = {"plain": M5_KD / "val_outputs.npz", "flip": M5_KD / "val_outputs_flip.npz",
                 "a3": G3 / "out_kd/A3_m4crops/val_outputs.npz", "a3_m3": WORK / g3.RUNS["A3"] / "val_outputs.npz"}
    else:
        files = {"plain": KERNEL / "m5_kd_heldout.npz", "flip": KERNEL / "m5_kd_flip_heldout.npz",
                 "a3": KERNEL / "a3_m4crops_heldout.npz"}
    plain = decide.load_outputs(files["plain"])
    paths = plain["paths"].astype(str)
    sets = {name: aligned(decide.load_outputs(f), paths) for name, f in files.items()}
    for name, outputs in sets.items():
        assert (outputs["labels"] == sets["plain"]["labels"]).all(), f"{name}: labels differ"
    if split == "val":
        b1 = cf.load(cf.B1_DIR / "predictions.npz")
        app = decide.load_outputs(WORK / "m5-targets/app_outputs/val_outputs.npz")
        gate = pd.Series(app["gate_snake"].astype(bool), index=app["paths"].astype(str))
        snakes = pd.read_csv(G4 / "runs/distil_oi/scores_kernel/scores_val.csv")
        p_snake = pd.Series(snakes["p_snake"].to_numpy(), index=snakes["path"])
    else:
        b1 = dict(np.load(KERNEL / "b1_heldout.npz"))
        rows = pd.read_csv(KERNEL / "gate_heldout.csv")
        gate = pd.Series(rows["gate_snake"].astype(bool).to_numpy(), index=rows["path"])
        scores = held_out_snake_scores()
        p_snake = pd.Series(scores["p_snake"].to_numpy(), index=scores["path"])
    b1_pred = pd.Series(np.asarray(b1["probs"]).argmax(1), index=np.asarray(b1["paths"]).astype(str))
    return paths, sets, b1_pred, gate, p_snake


def kernel_probs(frame):
    logits = frame[[f"logit_{c}" for c in G4_CLASSES]].to_numpy()  # kernel_scores.py's softmax
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    return frame.assign(**{f"p_{c}": probs[:, i] for i, c in enumerate(G4_CLASSES)})


def held_out_snake_scores():
    scores = kernel_probs(pd.read_csv(KERNEL / "snake_check_scores.csv"))
    return scores[scores["set"].eq("held_out")].reset_index(drop=True)


def frozen_logits(sets):
    calib = cf.load(M5_KD / "calib_outputs.npz")
    calib_flip = cf.load(M5_KD / "calib_outputs_flip.npz")
    tau, _ = prior.fit_tau((calib["species_logits"] + calib_flip["species_logits"]) / 2, calib["labels"])
    assert tau == TAU, f"tau refits to {tau}, not the frozen {TAU}"
    return prior.corrected((sets["plain"]["species_logits"] + sets["flip"]["species_logits"]) / 2, TAU)


def standing(c):
    return "higher" if c["diff"] > 0 else "tied" if c["low"] <= 0 <= c["high"] else "lower"


def g3_block(y, groups, created, b1_pred, preds, keep):
    shared = np.array(sorted(json.loads((cf.B1_DIR / "final_metrics.json").read_text())["coverage"]
                             ["covered_class_ids"]))
    rows = keep & created.ge(cf.SLICE_FROM).to_numpy() & np.isin(y, shared)
    yy, gg, bb = y[rows], groups[rows], b1_pred[rows]
    macro_62 = cf.shared_macro_f1(shared, N_CLASSES)
    out = {"photos": int(rows.sum()), "photographers": int(np.unique(gg).size),
           "b1": {"top1": cf.top1(yy, bb), "macro_f1": macro_62(yy, bb)}, "candidates": {}}
    for name, pred in preds.items():
        c = {m: cf.paired(yy, pred[rows], bb, gg, s) for m, s in (("top1", cf.top1), ("macro_f1", macro_62))}
        c["absolute"] = {"top1": cf.top1(yy, pred[rows]), "macro_f1": macro_62(yy, pred[rows])}
        c["g3"] = {m: standing(c[m]) for m in ("top1", "macro_f1")}
        c["g3"]["passed"] = "lower" not in c["g3"].values()
        out["candidates"][name] = c
    return out


def g2_block(y, pred, keep):
    yk, pk = y[keep], pred[keep]
    counts = np.bincount(yk, minlength=N_CLASSES)
    measurable = np.flatnonzero(counts >= MIN_PHOTOS)
    f1, _ = cf.per_class_f1(yk, pk, N_CLASSES)  # over all photos, as the kernel's per_class_counts (species_report)
    return {"measurable_species": int(len(measurable)), "f1_at_least_0.5": int((f1[measurable] >= F1_MARK).sum()),
            "macro_f1_measurable": cf.head_macro_f1(yk, pk, N_CLASSES, counts), "top1_all": float((yk == pk).mean()),
            "per_class_f1": {int(c): float(f1[c]) for c in measurable}, "photos_per_class": counts.tolist()}


def safety_outputs(sets, kind):
    if kind == "frozen":
        out = refit_new.averaged(sets["plain"], sets["flip"])
        out["species_logits"] = prior.corrected(out["species_logits"], TAU)
        thresholds = decide.Thresholds(**json.loads((G3 / "refit_new.json").read_text())["KD flip + prior"]
                                       ["chosen"]["thresholds"])
        prototypes = np.load(M5_KD / "prototypes.npz")["plain"]
    else:  # A3 with the shipped app's thresholds (the M4 app)
        out = dict(sets["a3"])
        thresholds = decide.Thresholds(**json.loads(APP_PIPELINE.read_text())["decision"]["thresholds"])
        prototypes = np.load(A3_FULL / "prototypes.npz")["plain"]
    return out, thresholds, prototypes


def safety_block(outputs, gate_calls, thresholds, prototypes, tables, keep):
    o = {k: v[keep] for k, v in outputs.items()}
    o["gate_snake"] = gate_calls[keep]
    decided, listed, outside, hits = refit_new.gated(o, thresholds, prototypes, tables)
    groups = o["observer_id"][listed]
    return {"photos": int(keep.sum()), "who_listed": int(listed.sum()), "who_recall": float(hits.mean()),
            "who_recall_ci": refit_new.interval(hits, groups), "lower_bound": decide.recall_lower_bound(hits, groups),
            "false_alarm": float((decided.label[outside] == "dangerous").mean()),
            "verdicts": decide.verdict_counts(decided), "labels": decided.label}


def shuffled_logits(outputs):
    order = np.random.default_rng(CONTROL_SEED).permutation(len(outputs["labels"]))
    return {k: (v[order] if k in refit_new.SHUFFLED else v) for k, v in outputs.items()}


def g4_block(split, keep_paths=None):
    calib = pd.read_csv(G4 / "runs/distil_oi/scores_kernel/scores_calib.csv", dtype={"group": str})
    cut = a.cutoff(calib["p_snake"].to_numpy(), np.ones(len(calib), dtype=bool), g4e.TARGET)
    assert cut == SNAKE_CUT, f"the cut re-derives to {cut}"
    if split == "val":
        snakes = pd.read_csv(G4 / "runs/distil_oi/scores_kernel/scores_val.csv", dtype={"group": str})
        non = pd.read_csv(G4 / "runs/distil_oi/scores_kernel/scores_nonsnake_val.csv", dtype={"group": str})
    else:
        meta = metadata()
        scores = held_out_snake_scores()
        info = meta.loc[scores["path"]]
        snakes = scores.assign(group=info["observer_id"].astype(str).to_numpy(), category=info["species"].to_numpy(),
                               genus=info["genus"].to_numpy(), family=info["family"].to_numpy(), label=1)
        all_scores = kernel_probs(pd.read_csv(KERNEL / "snake_check_scores.csv"))
        non_scores = all_scores[all_scores["set"].eq("nonsnake_test")]
        listing = pd.read_csv(HERE / "kernel/inputs/nonsnake_test.csv")
        non = non_scores.merge(listing, on="path", validate="one_to_one")
        non = non.assign(group=non["owner"].astype(str), label=0)
    if keep_paths is not None:
        snakes = snakes[snakes["path"].isin(keep_paths)].reset_index(drop=True)
    rng = np.random.default_rng(a.SEED)
    main_row = g4e.rule_result("P(snake) >= cut", calib["p_snake"].to_numpy(), snakes, snakes["p_snake"].to_numpy(),
                               non, non["p_snake"].to_numpy(), rng)
    main_row["most_missed_species"] = g4e.species_misses(snakes, snakes["p_snake"].to_numpy() >= main_row["cutoff"])
    random_rng = np.random.default_rng(a.SEED)
    control = g4e.rule_result("control: random score", random_rng.random(len(calib)), snakes,
                              random_rng.random(len(snakes)), non, random_rng.random(len(non)), rng)
    both = np.r_[snakes["p_snake"].to_numpy(), non["p_snake"].to_numpy()]
    shuffled = np.random.default_rng(CONTROL_SEED).permutation(both)
    control_shuffled = g4e.rule_result("control: scores shuffled across snakes and non-snakes",
                                       calib["p_snake"].to_numpy(), snakes, shuffled[:len(snakes)], non,
                                       shuffled[len(snakes):], rng)
    called = non["p_snake"].to_numpy() >= main_row["cutoff"]
    false_snakes = non.loc[called, ["path", "category", "p_snake"]].sort_values("p_snake", ascending=False)
    return {"main": main_row, "controls": [control, control_shuffled],
            "false_snakes": false_snakes.to_dict("records")}


def g1_block(split):
    report = json.loads((G3 / "out_kd/full/robustness_metrics.json" if split == "val"
                         else KERNEL / "robustness_heldout.json").read_text())
    c = report["conditions"]
    out = {"as_is_masked_top1": c["as_is_masked"]["top1"], "swapped_top1": c["swapped"]["top1"],
           "gap": c["as_is_masked"]["top1"] - c["swapped"]["top1"], "masked_photos": c["as_is_masked"]["photos"],
           "swapped_photos": c["swapped"]["photos"], "instrument_check": report["instrument_check"]}
    swapped_file = KERNEL / "heldout_robustness_swapped.npz" if split == "test" else M5_KD / "r1/robustness_swapped.npz"
    if swapped_file.exists():
        saved = np.load(swapped_file, allow_pickle=True)
        pred = saved["species_logits"].argmax(1)
        shuffled = np.random.default_rng(CONTROL_SEED).permutation(pred)
        out["control_shuffled_swapped_gap"] = out["as_is_masked_top1"] - float((shuffled == saved["labels"]).mean())
        out["recomputed_swapped_top1"] = float((pred == saved["labels"]).mean())
    return out


def strip(block):
    return {k: v for k, v in block.items() if k != "labels"}


def score(split):
    meta = metadata()
    paths, sets, b1_pred_by_path, gate_by_path, p_snake_by_path = load_sets(split)
    y = sets["plain"]["labels"].astype(int)
    groups = cf.groups_of(sets["plain"]["observer_id"], paths)
    created = pd.Series(meta.loc[paths, "created_at"].fillna("").astype(str).to_numpy())
    b1_pred = b1_pred_by_path.reindex(paths).fillna(-1).astype(int).to_numpy()
    frozen = frozen_logits(sets)
    preds = {FROZEN: softmax(frozen).argmax(1), BEFORE: softmax(sets["a3"]["species_logits"]).argmax(1),
             "M5_KD plain": sets["plain"]["species_logits"].argmax(1)}
    if "a3_m3" in sets:
        preds["A3 on M3 crops (instrument)"] = sets["a3_m3"]["species_logits"].argmax(1)
    preds["control: frozen predictions shuffled"] = np.random.default_rng(CONTROL_SEED).permutation(preds[FROZEN])
    in_test = meta.loc[paths, "split"].eq("test").to_numpy() if split == "test" else np.ones(len(paths), bool)
    subsets = {"all": np.ones(len(paths), bool)}
    if split == "test":
        subsets["test only"] = in_test
    tables = decide.repo_tables()
    has_gate = np.asarray(pd.Series(paths).isin(gate_by_path.index))
    gate = gate_by_path.reindex(paths).fillna(False).astype(bool).to_numpy()
    p_snake = p_snake_by_path.reindex(paths).to_numpy(float)
    result = {"split": split, "photos": len(paths), "photos_with_gate_call": int(has_gate.sum()), "subsets": {}}
    keep_labels = {}
    for name, keep in subsets.items():
        block = {"G3": g3_block(y, groups, created, b1_pred, preds, keep),
                 "G2": {m: g2_block(y, preds[m], keep) for m in (FROZEN, BEFORE, "M5_KD plain",
                                                                 "control: frozen predictions shuffled")}}
        safety = {}
        for kind in ("frozen", "before"):
            outputs, thresholds, prototypes = safety_outputs(sets, kind)
            gated_keep = keep & has_gate
            behind_gate = safety_block(outputs, gate, thresholds, prototypes, tables, gated_keep)
            safety[f"{kind}, behind the M4 gate"] = behind_gate
            if kind == "frozen":
                keep_labels[name] = (gated_keep, behind_gate["labels"])
                checked = keep & ~np.isnan(p_snake)
                safety["frozen, behind the snake check"] = safety_block(
                    outputs, np.nan_to_num(p_snake, nan=-1) >= SNAKE_CUT, thresholds, prototypes, tables, checked)
                safety["control: frozen, logits shuffled"] = safety_block(
                    shuffled_logits(outputs), gate, thresholds, prototypes, tables, gated_keep)
                safety["thresholds"] = asdict(thresholds)
        block["safety"] = {k: strip(v) if isinstance(v, dict) and "labels" in v else v for k, v in safety.items()}
        block["G4"] = g4_block(split, None if name == "all" else set(np.asarray(paths)[keep]))
        result["subsets"][name] = block
    result["G1"] = g1_block(split)
    return result, (paths, y, preds, frozen, b1_pred, gate, has_gate, p_snake, meta, keep_labels, sets)


def check_val(result):
    """The dry run must give the README numbers."""
    s = result["subsets"]["all"]
    got = s["G3"]["candidates"][FROZEN]
    for m in ("top1", "macro_f1"):
        for k in ("diff", "low", "high"):
            assert abs(got[m][k] - EXPECTED_VAL["g3"][m][k]) < 1e-12, ("G3", m, k, got[m][k], EXPECTED_VAL["g3"][m][k])
    a3 = s["G3"]["candidates"]["A3 on M3 crops (instrument)"]
    for m, v in EXPECTED_VAL["g3_a3_m3"].items():
        assert abs(a3[m]["diff"] - v) < g3c.REPORTED_TOLERANCE, ("G3 A3", m)
    for name, v in EXPECTED_VAL["g2"].items():
        assert s["G2"][name]["f1_at_least_0.5"] == v, ("G2", name, s["G2"][name]["f1_at_least_0.5"], v)
    saf, want = s["safety"]["frozen, behind the M4 gate"], EXPECTED_VAL["safety"]
    assert saf["photos"] == want["photos"] and saf["who_listed"] == want["who_listed"], ("safety n", saf["photos"])
    assert abs(saf["who_recall"] - want["who_recall"]) < 1e-12 and np.allclose(saf["who_recall_ci"], want["who_recall_ci"],
                                                                               rtol=0, atol=1e-12), "safety"
    assert abs(saf["false_alarm"] - want["false_alarm"]) < 1e-12, "false alarms"
    g4, want4 = s["G4"]["main"], EXPECTED_VAL["g4"]
    for k in ("cutoff", "false_snake_val", "false_snake_count", "snakes_kept_val", "snake_photos"):
        assert g4[k] == want4[k], ("G4", k, g4[k], want4[k])
    assert np.allclose(g4["false_snake_val_ci"], want4["false_snake_val_ci"]) and np.allclose(
        g4["snakes_kept_val_ci"], want4["snakes_kept_val_ci"]), "G4 intervals"
    assert abs(result["G1"]["gap"] - EXPECTED_VAL["g1"]) < 1e-12, "G1"
    return "every README number reproduced"


def goals(block, g1):
    g3 = block["G3"]["candidates"][FROZEN]
    saf = block["safety"]["frozen, behind the M4 gate"]
    g4 = block["G4"]["main"]
    g2 = block["G2"][FROZEN]
    return {"G1": {"value": g1["gap"], "mark": MARKS["G1"], "passed": g1["gap"] <= MARKS["G1"]},
            "G2": {"value": g2["f1_at_least_0.5"], "of": g2["measurable_species"], "mark": MARKS["G2"],
                   "passed": g2["f1_at_least_0.5"] >= MARKS["G2"]},
            "G3": {"top1": g3["top1"], "macro_f1": g3["macro_f1"], "standing": g3["g3"], "passed": g3["g3"]["passed"],
                   "photos": block["G3"]["photos"], "photographers": block["G3"]["photographers"]},
            "safety": {"lower_bound": saf["lower_bound"], "recall": saf["who_recall"], "ci": saf["who_recall_ci"],
                       "who_listed": saf["who_listed"], "mark": MARKS["safety"],
                       "passed": saf["lower_bound"] >= MARKS["safety"]},
            "G4": {"false_snake": g4["false_snake_val"], "false_snake_ci": g4["false_snake_val_ci"],
                   "false_snake_count": g4["false_snake_count"], "nonsnakes": g4["nonsnake_photos"],
                   "kept": g4["snakes_kept_val"], "kept_ci": g4["snakes_kept_val_ci"], "snakes": g4["snake_photos"],
                   "passed": g4["false_snake_val"] <= MARKS["G4_false"] and g4["snakes_kept_val"] >= MARKS["G4_kept"]}}


def controls_fail(block, g1):
    """Every shuffled control must fail its goal; returns the list of controls and whether each failed."""
    g3 = block["G3"]["candidates"]["control: frozen predictions shuffled"]
    g2 = block["G2"]["control: frozen predictions shuffled"]
    saf = block["safety"]["control: frozen, logits shuffled"]
    g4 = [c for c in block["G4"]["controls"]]
    out = {"G3 shuffled predictions": not g3["g3"]["passed"],
           "G2 shuffled predictions": g2["f1_at_least_0.5"] < MARKS["G2"],
           "safety shuffled logits": saf["lower_bound"] < MARKS["safety"],
           **{f"G4 {c['rule']}": not (c["false_snake_val"] <= MARKS["G4_false"] and c["snakes_kept_val"] >= MARKS["G4_kept"])
              for c in g4}}
    if "control_shuffled_swapped_gap" in g1:
        out["G1 shuffled swapped predictions"] = g1["control_shuffled_swapped_gap"] > MARKS["G1"]
    return out


def write_photos(extra, result):
    paths, y, preds, frozen, b1_pred, gate, has_gate, p_snake, meta, keep_labels, sets = extra
    classes = pd.read_csv(WORK / "m5-wt/data/classes.csv").set_index("class_id")["species"]
    probs = softmax(frozen)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    verdict = np.full(len(paths), "", dtype=object)
    gated_keep, labels = keep_labels["all"]
    verdict[gated_keep] = labels
    info = meta.loc[paths]
    frame = pd.DataFrame({
        "path": paths, "split": info["split"].to_numpy(), "created_at": info["created_at"].to_numpy(),
        "observer_id": info["observer_id"].to_numpy(), "species": classes.loc[y].to_numpy(), "class_id": y,
        "family": info["family"].to_numpy(), "who_category": info["who_category"].to_numpy(),
        "frozen_pred": classes.loc[preds[FROZEN]].to_numpy(), "frozen_p": probs.max(1),
        "frozen_top3": ["|".join(classes.loc[t]) for t in top3],
        "a3_pred": classes.loc[preds[BEFORE]].to_numpy(),
        "b1_pred": [classes.get(int(c), "") if c >= 0 else "" for c in b1_pred],
        "gate_snake": np.where(has_gate, gate, np.nan), "p_snake": p_snake,
        "snake_check_kept": p_snake >= SNAKE_CUT, "verdict_frozen_behind_gate": verdict})
    frame.to_csv(HERE / "predictions_heldout.csv", index=False)
    np.savez_compressed(HERE / "predictions_heldout.npz", paths=paths, labels=y, frozen_logits=frozen.astype(np.float32),
                        a3_logits=sets["a3"]["species_logits"].astype(np.float32), b1_pred=b1_pred)
    non = pd.DataFrame(result["subsets"]["all"]["G4"]["false_snakes"])
    non.to_csv(HERE / "false_snakes_nonsnake_test.csv", index=False)
    scores = kernel_probs(pd.read_csv(KERNEL / "snake_check_scores.csv"))
    listing = pd.read_csv(HERE / "kernel/inputs/nonsnake_test.csv")
    nonsnake = scores[scores["set"].eq("nonsnake_test")].merge(listing, on="path")
    nonsnake.assign(called_a_snake=nonsnake["p_snake"] >= SNAKE_CUT).to_csv(HERE / "predictions_nonsnake_test.csv",
                                                                             index=False)


def main(args):
    result, extra = score(args.split)
    for name, block in result["subsets"].items():
        block["goals"] = goals(block, result["G1"])
        block["controls_fail"] = controls_fail(block, result["G1"])
    if args.split == "val":
        result["instrument_check"] = check_val(result)
        out = HERE / "val_dry_run.json"
    else:
        out = HERE / "test_scores.json"
        write_photos(extra, result)
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    for name, block in result["subsets"].items():
        print(name, json.dumps(block["goals"], default=float))
        print(name, "controls fail:", block["controls_fail"])
    print("G1", {k: v for k, v in result["G1"].items() if k != "instrument_check"})
    if args.split == "val":
        print(result["instrument_check"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("val", "test"), required=True)
    main(parser.parse_args())
