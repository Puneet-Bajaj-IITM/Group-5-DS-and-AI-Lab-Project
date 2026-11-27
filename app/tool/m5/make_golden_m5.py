"""Golden files for the Dart port of the M5 app path, from the Python reference (reference_m5.py).

test/golden/golden_decisions.json  200 rows of M5_KD's saved VALIDATION outputs (plain and mirrored pass), and the
                                   verdict the repo's decide.verdict gives on the flip-averaged, prior-corrected
                                   outputs with the app's thresholds ("expected") and the frozen ones
                                   ("expected_frozen"); every (verdict, reasons, genus fallback) combination, plus
                                   rows where the two threshold sets disagree
test/golden/golden_photos.json     the five demo photos and ten VALIDATION non-snakes (pick_nonsnakes.py): where to
                                   download them, each step's values and the reference answer

Usage, from the repo root: python app/tool/m5/make_golden_m5.py --photos <folder with <photo_id>.jpg>
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import reference_m5 as m5

APP = m5.APP
sys.path.insert(0, str(APP.parent / "src"))
from serpentguard import decide  # noqa: E402

KD_RUN = m5.REPO / "results/m5/g3/out_kd/M5_KD"
OUTPUT_KEYS = ("embedding", "species_logits", "who_logits", "genus_logits", "family_logits")
HEADS = OUTPUT_KEYS[1:]
DECISION_ROWS = 200
PER_COMBINATION = 8
DISAGREEING_ROWS = 8
SEED = 42
TOP_LOGITS = 5
PHOTO_TOLERANCE = 1e-3
DEMO_PHOTOS = m5.HERE / "demo_photos.json"


def combined(plain, flipped, log_counts, tau):
    """refit_new.averaged + prior.corrected for one photo, in float64."""
    out = {head: (plain[head].astype(np.float64) + flipped[head].astype(np.float64)) / 2 for head in HEADS}
    embedding = plain["embedding"].astype(np.float64) + flipped["embedding"].astype(np.float64)
    out["embedding"] = embedding / np.linalg.norm(embedding)
    out["species_logits"] = out["species_logits"] + tau * log_counts
    return out


def expected_record(record):
    return {"verdict": record.label, "reason": list(record.reason), "message": record.message,
            "genus_fallback": record.genus_fallback, "margin": record.margin,
            "top3": [{"species": s, "probability": p} for s, p in record.top3]}


def golden_decisions(pipeline):
    load = lambda name: dict(np.load(KD_RUN / name, allow_pickle=True))  # noqa: E731
    plain, flipped = load("val_outputs.npz"), load("val_outputs_flip.npz")
    assert (plain["paths"] == flipped["paths"]).all()
    classifier = pipeline.settings["classifier"]
    log_counts, tau = np.asarray(classifier["log_counts"]), classifier["prior_tau"]
    prototypes = np.load(KD_RUN / "prototypes.npz")["plain"]
    tables = decide.repo_tables()
    app = decide.Thresholds(**pipeline.thresholds)
    held = decide.Thresholds(**pipeline.frozen_thresholds)

    def record(i, thresholds):
        outputs = combined({k: plain[k][i] for k in OUTPUT_KEYS}, {k: flipped[k][i] for k in OUTPUT_KEYS},
                           log_counts, tau)
        return expected_record(decide.verdict({k: v[None] for k, v in outputs.items()}, thresholds, prototypes,
                                              tables, scorer="head"))

    n = len(plain["paths"])
    records = [record(i, app) for i in range(n)]
    combination = pd.Series([f"{r['verdict']}|{','.join(r['reason'])}|{r['genus_fallback'] is not None}"
                             for r in records])
    rng = np.random.default_rng(SEED)
    picked = []
    for _, rows in combination.groupby(combination).groups.items():
        picked += rng.choice(np.asarray(rows), min(PER_COMBINATION, len(rows)), replace=False).tolist()
    labels_frozen = {}
    for i in rng.permutation(n):
        if len(labels_frozen) == DISAGREEING_ROWS:
            break
        if i in picked:
            continue
        other = record(int(i), held)
        if other["verdict"] != records[i]["verdict"]:
            labels_frozen[int(i)] = other
    picked += list(labels_frozen)
    rest = np.setdiff1d(np.arange(n), picked)
    picked += rng.choice(rest, max(0, DECISION_ROWS - len(picked)), replace=False).tolist()
    rows = []
    for i in sorted(int(i) for i in picked):
        rows.append({"row": i, "outputs": {k: plain[k][i].astype(float).tolist() for k in OUTPUT_KEYS},
                     "outputs_flip": {k: flipped[k][i].astype(float).tolist() for k in OUTPUT_KEYS},
                     "expected": records[i], "expected_frozen": labels_frozen.get(i) or record(i, held)})
    disagree = sum(r["expected"]["verdict"] != r["expected_frozen"]["verdict"] for r in rows)
    return {"thresholds": pipeline.settings["decision"]["thresholds"],
            "frozen_thresholds": pipeline.settings["decision"]["frozen_thresholds"], "prior_tau": tau,
            "combinations": int(combination.nunique()), "rows_where_frozen_differs": int(disagree), "rows": rows}


def photo_case(pipeline, path, meta):
    s = pipeline.settings
    with m5.ref.Image.open(path) as image:
        raw = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb = m5.ref.fix_exposure(raw, s["exposure_fix"])
    check_in = pipeline.snake_check_input(rgb)
    answer = pipeline.answer_rgb(rgb)
    probs = pipeline.snake_probs(rgb)
    steps = {"height": raw.shape[0], "width": raw.shape[1], "mean_brightness": float(raw.mean()),
             "exposure_fixed": bool((rgb != raw).any()),
             "snake_check": {"probs": probs.astype(float).tolist(), "input_mean": float(check_in.mean()),
                             "input_std": float(check_in.std())}}
    outputs = answer.pop("outputs", None)
    if outputs is not None:
        window = answer["crop_window"]
        crop = rgb if window is None else rgb[window[0]:window[1], window[2]:window[3]]
        image = m5.ref.normalise(m5.ref.letterbox(crop, s["input_size"], s["letterbox"]["pad_rgb"]), s["normalise"])
        order = np.argsort(-outputs["species_logits"])[:TOP_LOGITS]
        steps.update({"crop_window_top_bottom_left_right": window, "classifier_input_mean": float(image.mean()),
                      "classifier_input_std": float(image.std()),
                      "top_species_logits": [{"class_id": int(k), "logit": float(outputs["species_logits"][k])}
                                             for k in order],
                      "best_cosine": m5.ref.best_cosine(outputs["embedding"].astype(np.float64),
                                                        pipeline.tables["prototypes"])})
    answer.pop("crop_window", None)
    answer.pop("who_category", None)
    answer.pop("margin", None)
    return {**meta, "sha1": hashlib.sha1(path.read_bytes()).hexdigest(), "steps": steps, "answer": answer}


def golden_photos(pipeline, folder):
    metas = [{k: p[k] for k in ("photo_id", "url", "credit", "licence")} | {"set": "demo"}
             for p in json.loads(DEMO_PHOTOS.read_text())["photos"]]
    with (m5.HERE / "nonsnake_golden.csv").open() as handle:
        metas += [{"photo_id": r["photo_id"], "url": r["url"], "credit": r["credit"], "licence": r["licence"],
                   "set": f"validation non-snake ({r['category']})"} for r in csv.DictReader(handle)]
    return {"tolerance": PHOTO_TOLERANCE, "photos": [photo_case(pipeline, folder / f"{m['photo_id']}.jpg", m)
                                                     for m in metas]}


def main(args):
    pipeline = m5.AppPipeline()
    decisions = golden_decisions(pipeline)
    (APP / "test/golden/golden_decisions.json").write_text(json.dumps(decisions) + "\n")
    photos = golden_photos(pipeline, args.photos.resolve())
    (APP / "test/golden/golden_photos.json").write_text(json.dumps(photos, indent=1) + "\n")
    print(f"golden_decisions.json: {len(decisions['rows'])} rows, {decisions['combinations']} combinations, "
          f"{decisions['rows_where_frozen_differs']} where the frozen thresholds differ")
    for p in photos["photos"]:
        a = p["answer"]
        print(p["photo_id"], p["set"], f"P(snake) {a['snake_check']['snake_probability']:.4f}", a["verdict"],
              a["top3"][0]["species"] if a["top3"] else "-", "frozen:", a["frozen_verdict"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photos", type=Path, required=True)
    main(parser.parse_args())
