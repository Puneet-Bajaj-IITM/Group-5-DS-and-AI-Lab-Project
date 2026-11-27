"""The M5 video lock on saved per-frame outputs: tuning on dev clips, then G5 on the test clips.

Reads the outputs clip_outputs.py saved (every frame, raw logits), picks the frames a phone with a
given latency would process (video.schedule), turns logits into probabilities with the thresholds
of a pipeline.json (the app's by default), and runs video.pooled_lock and video.lock_verdicts.

Usage:
  python lock_eval.py tune  --outputs DIR --pipeline pipeline.json --tables tables.json --results DIR
  python lock_eval.py score --outputs DIR --pipeline ... --tables ... --results DIR --sets new_test ...
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK / "m5-g5/src"))
from serpentguard import decide, video  # noqa: E402

LATENCIES = (0.1, 0.4, 0.8)           # seconds a frame: the M4 target, the emulator, a slow live camera
REPORT_LATENCIES = (0.1, 0.25, 0.4, 0.8)
GRID = {"window_seconds": (0.5, 1.0, 1.5, 2.0, 3.0), "margin": (0.05, 0.1, 0.2, 0.3),
        "switch_seconds": (1.0, 2.0, 3.0), "gate_share": (0.0, 0.5), "min_frames": (1, 2, 3)}
# second pass: a lock confidence above theta_conf, to see whether it keeps no-snake clips from locking
GRID_CONF = {"window_seconds": (0.5, 1.0), "margin": (0.05, 0.1, 0.2), "switch_seconds": (2.0, 3.0),
             "gate_share": (0.0,), "lock_conf": (None, 0.4, 0.5, 0.6)}
MAX_CHANGES, MAX_FIRST_ANSWER = 1, 3.0
DEV_SETS = ("dev_synthetic", "dev_real")
TUNING_NEGATIVES = ("old_negative",)
PAN_GAP_SECONDS = 3.0                 # no-snake seconds between the two clips of the gap variant
PART_KEYS = {"probs": "species_logits", "who": "who_logits", "genus": "genus_logits",
             "family": "family_logits", "embedding": "embedding"}


def softmax(logits, temperature=1.0):
    z = logits.astype(np.float64) / temperature
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


class Clips:
    """Per clip: probabilities of every frame (NaN rows where the gate found no snake)."""

    def __init__(self, outputs, pipeline, tables_json):
        settings = json.loads(Path(pipeline).read_text())
        self.thresholds = decide.Thresholds(**{k: v for k, v in settings["decision"]["thresholds"].items()
                                               if k in decide.Thresholds.__dataclass_fields__})
        self.prototypes = np.asarray(json.loads(Path(tables_json).read_text())["prototypes"], dtype=float)
        self.tables = decide.repo_tables()
        classes = pd.read_csv(WORK / "m5-g5/data/classes.csv")
        self.class_id = dict(zip(classes.species, classes.class_id))
        self.names = dict(zip(classes.class_id, classes.species))
        manifests = sorted(Path(outputs).glob("manifest_*.csv"))
        assert len(manifests) == 1, f"one model's outputs per folder, found {manifests}"
        self.manifest = pd.read_csv(manifests[0])
        self.outputs = Path(outputs)
        self._cache = {}

    def frames(self, clip):
        if clip not in self._cache:
            row = self.manifest.set_index("clip").loc[clip]
            with np.load(self.outputs / row["outputs"]) as saved:
                t = self.thresholds
                temps = {"probs": t.temperature, "who": t.temperature_who}
                parts = {name: (softmax(saved[key], temps.get(name, 1.0)) if name != "embedding"
                                else saved[key].astype(np.float64)) for name, key in PART_KEYS.items()}
                self._cache[clip] = {"fps": float(saved["fps"]), "passed": saved["passed"].astype(bool),
                                     "parts": parts}
        return self._cache[clip]

    def clips(self, sets):
        return self.manifest[self.manifest["set"].isin(sets)].to_dict("records")


def sequence(frames, latency, full=True):
    """(items, times) a phone busy `latency` s a frame shows; items None where the gate says no snake."""
    kept, shown_at = video.schedule(len(frames["passed"]), frames["fps"], latency)
    items = []
    for i in kept:
        if not frames["passed"][i]:
            items.append(None)
        elif full:
            items.append({name: values[i] for name, values in frames["parts"].items()})
        else:
            items.append(frames["parts"]["probs"][i])
    return items, shown_at


def species_metrics(species, times, true_class):
    """G5's numbers from the species on screen per step (None while checking)."""
    answered = [(s, t) for s, t in zip(species, times) if s is not None]
    changes = sum(a[0] != b[0] for a, b in zip(answered, answered[1:]))
    final = species[-1] if len(species) else None
    return {"changes": int(changes), "answered": bool(answered),
            "first_answer_s": round(float(answered[0][1]), 2) if answered else None,
            "final_species": final, "final_right": bool(final is not None and final == true_class)}


def tune(clips, results, grid=GRID, name="tuning"):
    dev = clips.clips(DEV_SETS)
    negatives = clips.clips(TUNING_NEGATIVES)
    seqs = {(c["clip"], lat): sequence(clips.frames(c["clip"]), lat, full=False)
            for c in dev + negatives for lat in LATENCIES}
    rows = []
    for values in itertools.product(*grid.values()):
        row = dict(zip(grid, values))
        setting = {k: v for k, v in row.items() if k != "lock_conf"}
        conf = row.get("lock_conf") or clips.thresholds.theta_conf
        row["lock_conf"] = conf
        for lat in LATENCIES:
            scored = []
            for c in dev:
                items, times = seqs[(c["clip"], lat)]
                steps = video.pooled_lock(items, times, conf, **setting)
                scored.append(species_metrics([s["species"] for s in steps], times, clips.class_id[c["species"]]))
            first = [s["first_answer_s"] if s["answered"] else np.inf for s in scored]
            locked = sum(any(s["state"] == "locked" for s in video.pooled_lock(*seqs[(c["clip"], lat)], conf,
                                                                               **setting)) for c in negatives)
            row |= {f"max_changes@{lat}": max(s["changes"] for s in scored),
                    f"over_1@{lat}": sum(s["changes"] > MAX_CHANGES for s in scored),
                    f"median_first_s@{lat}": float(np.median(first)),
                    f"never@{lat}": sum(not s["answered"] for s in scored),
                    f"final_right@{lat}": sum(s["final_right"] for s in scored),
                    f"negatives_locked@{lat}": locked}
        rows.append(row)
    table = pd.DataFrame(rows)
    table["negatives_locked"] = sum(table[f"negatives_locked@{lat}"] for lat in LATENCIES)
    table["over_1"] = sum(table[f"over_1@{lat}"] for lat in LATENCIES)
    table["final_right"] = sum(table[f"final_right@{lat}"] for lat in LATENCIES)
    table["slowest_median_first_s"] = table[[f"median_first_s@{lat}" for lat in LATENCIES]].max(axis=1)
    table["fast_enough"] = table.slowest_median_first_s <= MAX_FIRST_ANSWER
    # the rule: no dev clip over 1 change at any latency (G5 on dev), then a median first answer within 3 s
    # at every latency, then fewest tuning negatives locked, then most right final answers, then the
    # fastest, then the shortest window, smallest margin, shortest switch, lowest gate share
    order = table.sort_values(["over_1", "fast_enough", "negatives_locked", "final_right", "slowest_median_first_s",
                               "window_seconds", "margin", "switch_seconds", "gate_share", "min_frames", "lock_conf"],
                              ascending=[True, False, True, False, True, True, True, True, True, True, True])
    table["rank"] = 0
    table.loc[order.index, "rank"] = np.arange(1, len(order) + 1)
    table.sort_values("rank").to_csv(Path(results) / f"{name}.csv", index=False)
    best = order.iloc[0]
    chosen = {k: (int(best[k]) if k == "min_frames" else float(best[k])) for k in grid if k != "lock_conf"}
    (Path(results) / f"{name}_chosen.json").write_text(json.dumps(
        {"settings": {"min_frames": video.LOCK_MIN_FRAMES, "hold_seconds": video.HOLD_SECONDS} | chosen,
         "conf": float(best["lock_conf"]), "theta_conf": clips.thresholds.theta_conf, "latencies": LATENCIES, "dev_clips": len(dev), "tuning_negatives": [c["clip"] for c in negatives],
         "rule": "no dev clip over 1 change at any latency, median first answer <= 3 s at every latency, fewest "
                 "tuning negatives locked, most final right, fastest, shortest window, smallest margin, switch, "
                 "gate share"},
        indent=1))
    print(table.sort_values("rank").head(15).to_string())
    return chosen


def run_clip(clips, clip, latency, setting, items_times=None, true_class=None):
    items, times = items_times or sequence(clips.frames(clip["clip"]), latency)
    conf = clips.thresholds.theta_conf
    steps = video.pooled_lock(items, times, conf, **setting)
    shown = video.lock_verdicts(steps, clips.thresholds, clips.prototypes, clips.tables)
    species = [None if s is None else s[1] for s in shown]
    verdicts = [None if s is None else s[0] for s in shown]
    true = clips.class_id.get(clip["species"]) if true_class is None and isinstance(clip["species"], str) else true_class
    m = species_metrics(species, times, true)
    verdict_changes = sum(a != b for a, b in zip([v for v in verdicts if v], [v for v in verdicts if v][1:]))
    first_any = next((float(t) for v, t in zip(verdicts, times) if v), None)
    return {"clip": clip["clip"], "set": clip["set"], "species": clip["species"],
            "species_source": clip.get("species_source"), "latency_s": latency,
            "frames": len(items), "snake_frames": sum(i is not None for i in items),
            "changes_after_first_answer": m["changes"], "answered": m["answered"],
            "first_answer_s": m["first_answer_s"],
            "first_anything_s": None if first_any is None else round(first_any, 2),
            "final_shown": None if m["final_species"] is None else clips.names[m["final_species"]],
            "final_right": m["final_right"] if true is not None else None,
            "final_verdict": verdicts[-1] if verdicts else None, "verdict_changes": int(verdict_changes),
            "final_pooled_verdict_no_latch": None if steps[-1]["parts"] is None else video.frame_verdict(
                steps[-1]["parts"], clips.thresholds, clips.prototypes, clips.tables),
            "ever_dangerous": "dangerous" in verdicts,
            "species_shown": sorted({clips.names[s] for s in species if s is not None}),
            "locked_frames": sum(s["state"] == "locked" for s in steps),
            "expected_verdict": None if true is None else decide.expected_verdict(true, clips.tables)}


def panning(clips, setting, results, pairs):
    rows = []
    for first, second, gap in pairs:
        for lat in REPORT_LATENCIES:
            a, b = (clips.clips(DEV_SETS + ("old_negative",)),) * 2
            ca = next(c for c in a if c["clip"] == first)
            cb = next(c for c in b if c["clip"] == second)
            ia, ta = sequence(clips.frames(first), lat)
            ib, tb = sequence(clips.frames(second), lat)
            offset = (ta[-1] if len(ta) else 0) + lat
            gap_items, gap_times = [], []
            if gap:
                gap_times = list(np.arange(offset, offset + gap, lat))
                gap_items = [None] * len(gap_times)
                offset += gap
            items = ia + gap_items + ib
            times = np.concatenate([ta, gap_times, tb + offset])
            conf = clips.thresholds.theta_conf
            steps = video.pooled_lock(items, times, conf, **setting)
            shown = video.lock_verdicts(steps, clips.thresholds, clips.prototypes, clips.tables)
            species = [None if s is None else s[1] for s in shown]
            n_a = len(ia) + len(gap_items)
            before = [s for s in species[:len(ia)] if s is not None]
            after = species[n_a:]
            second_true = clips.class_id[cb["species"]]
            switched_at = next((float(times[n_a + i] - times[n_a]) for i, s in enumerate(after) if s == second_true), None)
            rows.append({"first": first, "first_species": ca["species"], "second": second,
                         "second_species": cb["species"], "gap_s": gap, "latency_s": lat,
                         "shown_end_of_first": None if not before else clips.names[before[-1]],
                         "shown_at_end": None if species[-1] is None else clips.names[species[-1]],
                         "switched_to_second": species[-1] == second_true,
                         "seconds_into_second_until_shown": None if switched_at is None else round(switched_at, 2),
                         "species_sequence": [clips.names[s] for i, s in enumerate(species)
                                              if s is not None and (i == 0 or species[i - 1] != s)]})
    table = pd.DataFrame(rows)
    table.to_csv(Path(results) / "panning.csv", index=False)
    print(table.drop(columns=["species_sequence"]).to_string())
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tune", "score", "pan"))
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--sets", nargs="*", default=())
    parser.add_argument("--name", default="score")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    clips = Clips(args.outputs, args.pipeline, args.tables)
    if args.mode == "tune":
        tune(clips, args.results)
        return
    setting = json.loads((args.results / "chosen_settings.json").read_text())["settings"]  # copied from a *_chosen.json by hand
    if args.mode == "pan":
        # pairs of dev clips the lock gets right on their own, different species; the last with a gap
        panning(clips, setting, args.results, [
            ("laticauda_colubrina_philippines.webm", "synthetic_81497741", 0.0),
            ("synthetic_527874978", "synthetic_46690465", 0.0),
            ("synthetic_223029311", "laticauda_colubrina_fiji.webm", 0.0),
            ("synthetic_567656035", "synthetic_658554014", 0.0),
            ("laticauda_colubrina_philippines.webm", "synthetic_81497741", PAN_GAP_SECONDS),
            ("synthetic_527874978", "synthetic_46690465", PAN_GAP_SECONDS)])
        return
    rows = [run_clip(clips, c, lat, setting) for c in clips.clips(args.sets) for lat in REPORT_LATENCIES]
    table = pd.DataFrame(rows)
    table.to_csv(args.results / f"{args.name}.csv", index=False)
    print(table.drop(columns=["species_shown", "expected_verdict"]).to_string())


if __name__ == "__main__":
    main()
