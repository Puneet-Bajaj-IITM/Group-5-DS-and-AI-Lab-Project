"""Python-generated sequences for the Dart VideoLock parity test.

Each sequence is real per-frame outputs (dev clips only, as a phone with the given latency would
process them), its lock settings, and what video.pooled_lock + video.lock_verdicts show at every
step. Values are rounded before the expected steps are computed, so both sides read the same numbers.

Usage: python make_parity.py --outputs DIR --pipeline pipeline.json --tables tables.json --out file.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import lock_eval
from serpentguard import video

DIGITS = 7
PART_NAMES = ("probs", "who", "genus", "family", "embedding")
SEQUENCES = [
    # name, clips (None = seconds of no snake), latency, settings overrides
    ("dev_clip_default", ["laticauda_colubrina_philippines.webm"], 0.4, {}),
    ("pan_switch", ["python_bivittatus_zoo.ogv", "bungarus_fasciatus_milking.webm"], 0.4,
     {"switch_seconds": 1.0, "margin": 0.05}),
    ("pan_gap_reset", ["laticauda_colubrina_fiji.webm", 3.0, "python_bivittatus_zoo.ogv"], 0.4, {}),
    ("dev_not_listed", ["synthetic_81497741"], 0.25, {}),
    ("dev_dangerous_latch", ["synthetic_658554014"], 0.1, {}),
    ("pan_safe_to_dangerous", ["synthetic_81497741", "synthetic_527874978"], 0.4, {}),
    ("negative_gate_share", ["negative_river_wyoming.webm"], 0.25,
     {"gate_share": 0.5, "window_seconds": 1.5, "min_frames": 3}),
]


def build(clips, parts_list, latency):
    items, times, offset = [], [], 0.0
    for part in parts_list:
        if isinstance(part, float):
            gap = list(np.arange(offset, offset + part, latency))
            items += [None] * len(gap)
            times += gap
            offset += part
            continue
        its, ts = lock_eval.sequence(clips.frames(part), latency)
        items += its
        times += list(np.asarray(ts) + offset)
        offset = times[-1] + latency
    rounded = [None if it is None else {k: np.round(np.asarray(it[k], dtype=float), DIGITS) for k in PART_NAMES}
               for it in items]
    return rounded, [round(float(t), 6) for t in times]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    clips = lock_eval.Clips(args.outputs, args.pipeline, args.tables)
    conf = clips.thresholds.theta_conf
    out = {"what": "VideoLock parity: per step, what src/serpentguard/video.py shows",
           "theta_conf": conf, "sequences": []}
    for name, parts_list, latency, overrides in SEQUENCES:
        items, times = build(clips, parts_list, latency)
        settings = video.lock_settings(**overrides)
        steps = video.pooled_lock(items, times, conf, **settings)
        shown = video.lock_verdicts(steps, clips.thresholds, clips.prototypes, clips.tables)
        expected = [{"checking": st["state"] == "checking", "locked": st["species"], "event": st["event"],
                     "label": None if sh is None else sh[0], "species": None if sh is None else sh[1]}
                    for st, sh in zip(steps, shown)]
        out["sequences"].append({
            "name": name, "clips": [str(p) for p in parts_list], "latency_s": latency, "settings": settings,
            "times": times,
            "frames": [None if it is None else {k: it[k].tolist() for k in PART_NAMES} for it in items],
            "expected": expected})
        events = [e["event"] for e in expected if e["event"]]
        labels = sorted({e["label"] for e in expected if e["label"]})
        resets = sum(1 for a, b in zip(expected, expected[1:]) if not a["checking"] and b["checking"])
        print(f"{name}: {len(items)} steps, {sum(i is not None for i in items)} snake, events {events}, "
              f"labels {labels}, resets {resets}, dangerous while checking "
              f"{sum(e['checking'] and e['label'] == 'dangerous' for e in expected)}")
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", args.out, f"{args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
