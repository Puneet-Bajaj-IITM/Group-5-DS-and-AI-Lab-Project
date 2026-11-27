"""Section 6 robustness table: the frozen pipeline per corruption, against the locked clean test run.

Metric code is score_test.py's own (imported, unchanged): frozen logits = flip average + tau * log(n_c); macro-F1 =
check_full_run.head_macro_f1 over the 165 measurable species; WHO recall behind the M4 gate = score_test.safety_block
(refit_new.gated with the frozen thresholds and prototypes); snake check = evaluate.rule_result with the frozen cut
re-derived from the calibration scores. The top-1 change has a paired photographer bootstrap (check_full_run.paired,
1,000 resamples, seed 42) of corrupted minus clean on the same photos.

Instrument check: the locked run's own files through this code must give RESULTS.md's numbers exactly
(test_scores.json); and each kernel's clean pass (same write-and-read harness as the corruptions) is scored as a row.

Usage: m3-final/.venv/bin/python results/m5/robustness/score_robustness.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path[:0] = [str(WORK / "results/m5/test"), str(HERE)]
import score_test as st  # noqa: E402  (its imports put g3, g4, m5-targets and m5-wt/src on the path)
import corrupt  # noqa: E402

cf, decide, g4e, a = st.cf, st.decide, st.g4e, st.a
LOCKED = st.KERNEL
RUNS = {"jayeshbansal497": HERE / "kernel_out/jayeshbansal497/out", "vikas25s": HERE / "kernel_out/vikas25s_v2/out"}
OUT = HERE / "results"
TOLERANCE = 1e-12


def condition_files(folder, locked=False):
    suffix = "_heldout" if locked else ""
    return {"plain": folder / f"m5_kd{suffix}.npz", "flip": folder / f"m5_kd_flip{suffix}.npz",
            "gate": folder / "gate_heldout.csv", "snake": folder / "snake_check_scores.csv"}


def load_condition(files, paths):
    sets = {name: st.aligned(decide.load_outputs(files[name]), paths) for name in ("plain", "flip")}
    gate = pd.read_csv(files["gate"]).set_index("path")["gate_snake"].astype(bool)
    scores = st.kernel_probs(pd.read_csv(files["snake"]))
    return sets, gate.reindex(paths), scores


def snake_rows(scores, meta, non_listing):
    snakes = scores[scores["set"].eq("held_out")].reset_index(drop=True)
    info = meta.loc[snakes["path"]]
    snakes = snakes.assign(group=info["observer_id"].astype(str).to_numpy(), category=info["species"].to_numpy(),
                           genus=info["genus"].to_numpy(), family=info["family"].to_numpy(), label=1)
    non = scores[scores["set"].eq("nonsnake_test")].merge(non_listing, on="path", validate="one_to_one")
    return snakes, non.assign(group=non["owner"].astype(str), label=0)


def metrics(sets, gate, scores, y, groups, meta, non_listing, calib, tables):
    frozen = st.prior.corrected((sets["plain"]["species_logits"] + sets["flip"]["species_logits"]) / 2, st.TAU)
    pred = st.softmax(frozen).argmax(1)
    counts = np.bincount(y, minlength=st.N_CLASSES)
    outputs, thresholds, prototypes = st.safety_outputs(sets, "frozen")
    has_gate = gate.notna().to_numpy()
    safety = st.safety_block(outputs, gate.fillna(False).to_numpy(bool), thresholds, prototypes, tables, has_gate)
    snakes, non = snake_rows(scores, meta, non_listing)
    row = g4e.rule_result("P(snake) >= cut", calib, snakes, snakes["p_snake"].to_numpy(), non, non["p_snake"].to_numpy(),
                          np.random.default_rng(a.SEED))
    return pred, {"photos": len(y), "top1": cf.top1(y, pred), "macro_f1": cf.head_macro_f1(y, pred, st.N_CLASSES, counts),
                  "who_recall": safety["who_recall"], "who_recall_ci": safety["who_recall_ci"],
                  "who_lower_bound": safety["lower_bound"], "who_listed": safety["who_listed"],
                  "false_alarm": safety["false_alarm"], "verdicts": safety["verdicts"],
                  "gate_passed": int(gate.fillna(False).sum()), "snakes_kept": row["snakes_kept_val"],
                  "snakes_kept_ci": row["snakes_kept_val_ci"], "snake_photos": row["snake_photos"],
                  "nonsnakes_called_snake": row["false_snake_val"], "nonsnakes_called_snake_ci": row["false_snake_val_ci"],
                  "nonsnake_count": row["false_snake_count"], "cutoff": row["cutoff"]}


def check_locked(clean):
    want = json.loads((WORK / "results/m5/test/test_scores.json").read_text())["subsets"]["all"]
    g2 = want["G2"][st.FROZEN]
    saf = want["safety"]["frozen, behind the M4 gate"]
    g4 = want["G4"]["main"]
    pairs = {"top1": g2["top1_all"], "macro_f1": g2["macro_f1_measurable"], "who_recall": saf["who_recall"],
             "who_lower_bound": saf["lower_bound"], "snakes_kept": g4["snakes_kept_val"],
             "nonsnakes_called_snake": g4["false_snake_val"]}
    for key, value in pairs.items():
        assert abs(clean[key] - value) < TOLERANCE, (key, clean[key], value)
    assert np.allclose(clean["who_recall_ci"], saf["who_recall_ci"], rtol=0, atol=TOLERANCE)
    return {k: {"here": clean[k], "test_scores.json": v} for k, v in pairs.items()}


def main():
    OUT.mkdir(exist_ok=True)
    meta = st.metadata()
    tables = decide.repo_tables()
    non_listing = pd.read_csv(WORK / "results/m5/test/kernel/inputs/nonsnake_test.csv")
    calib = pd.read_csv(st.G4 / "runs/distil_oi/scores_kernel/scores_calib.csv")["p_snake"].to_numpy()
    locked_plain = decide.load_outputs(LOCKED / "m5_kd_heldout.npz")
    paths = locked_plain["paths"].astype(str)
    sets, gate, scores = load_condition(condition_files(LOCKED, locked=True), paths)
    y = sets["plain"]["labels"].astype(int)
    groups = cf.groups_of(sets["plain"]["observer_id"], paths)
    clean_pred, clean = metrics(sets, gate, scores, y, groups, meta, non_listing, calib, tables)
    report = {"instrument_check": check_locked(clean), "clean_locked_run": clean, "conditions": {}}
    print("locked clean run reproduces test_scores.json:", report["instrument_check"])

    for account, folder in RUNS.items():
        if not folder.exists():
            print("no kernel output yet:", folder)
            continue
        for condition_dir in sorted(p for p in folder.iterdir() if p.is_dir() and p.name in corrupt.CONDITIONS):
            files = condition_files(condition_dir)
            if not all(f.exists() for f in files.values()):
                print("incomplete:", condition_dir)
                continue
            c_sets, c_gate, c_scores = load_condition(files, paths)
            assert (c_sets["plain"]["labels"] == y).all(), condition_dir
            pred, row = metrics(c_sets, c_gate, c_scores, y, groups, meta, non_listing, calib, tables)
            row["top1_change"] = cf.paired(y, pred, clean_pred, groups, cf.top1)
            row["prediction_agreement_with_clean"] = float((pred == clean_pred).mean())
            row["gate_call_agreement_with_clean"] = float((c_gate.to_numpy() == gate.to_numpy()).mean())
            row["account"] = account
            name = condition_dir.name if condition_dir.name != "clean" else f"clean (harness rerun, {account})"
            report["conditions"][name] = row
            print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                         if k in ("top1", "macro_f1", "who_recall", "snakes_kept", "nonsnakes_called_snake")},
                  row["top1_change"])
    (OUT / "robustness.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    write_table(report)


def write_table(report):
    clean = report["clean_locked_run"]
    order = ["clean"] + [c for c in corrupt.CONDITIONS if c != "clean"]
    lines = []
    for name in [k for k in report["conditions"] if k.startswith("clean")] + order[1:]:
        row = report["conditions"].get(name)
        if row is None:
            continue
        change = row["top1_change"]
        lines.append({"condition": name, "top1": row["top1"], "top1_change": change["diff"], "top1_change_low": change["low"],
                      "top1_change_high": change["high"], "macro_f1": row["macro_f1"],
                      "macro_f1_change": row["macro_f1"] - clean["macro_f1"], "who_recall": row["who_recall"],
                      "who_recall_change": row["who_recall"] - clean["who_recall"], "who_lower_bound": row["who_lower_bound"],
                      "false_alarm": row["false_alarm"], "false_alarm_change": row["false_alarm"] - clean["false_alarm"],
                      "gate_passed": row["gate_passed"], "snakes_kept": row["snakes_kept"],
                      "snakes_kept_change": row["snakes_kept"] - clean["snakes_kept"],
                      "nonsnakes_called_snake": row["nonsnakes_called_snake"],
                      "nonsnakes_called_snake_change": row["nonsnakes_called_snake"] - clean["nonsnakes_called_snake"],
                      "nonsnake_count": row["nonsnake_count"], "prediction_agreement_with_clean":
                          row["prediction_agreement_with_clean"]})
    pd.DataFrame(lines).to_csv(OUT / "robustness_table.csv", index=False)


if __name__ == "__main__":
    main()
