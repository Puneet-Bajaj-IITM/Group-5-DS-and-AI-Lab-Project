"""Score photos with the exported ONNX model through the app's input path; one row per photo plus its features.

Sets: calibration snakes (7,999), validation snakes (all cached), validation non-snakes (600), training non-snakes
(for the shuffled-label control's head only). The non-snake test split is never read.

Usage: python -I score_onnx.py --model <snake_check.onnx> --out <folder> [--sets calib val nonsnake_val ...]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_inputs import ROOT, model_input  # noqa: E402

PROBE = ROOT / "m5-g4-probe"
STAGED = ROOT / "results/m5/g4/data/kaggle_dataset"
NB19 = ROOT / "kaggle/m4_runs/non_snake/out_full/manifest.csv"
METADATA = ROOT / "kaggle/clean_build/out/metadata.csv"
BATCH = 32
WORKERS = 6
THREADS = 4
PREFETCH = 8
CLASSES = ("snake", "lookalike", "other")


def photo_sets():
    meta = pd.read_csv(METADATA, usecols=["path", "genus", "family"])
    snakes = {}
    for name, csv in (("calib", "snakes_calib.csv"), ("val", "snakes_val_all.csv")):
        rows = pd.read_csv(PROBE / "results" / csv).merge(meta, on="path", how="left")
        rows["file"] = [str(PROBE / "photos/snakes" / p) for p in rows["path"]]
        rows = rows[[Path(f).exists() for f in rows["file"]]]
        snakes[name] = rows.assign(group=rows["observer_id"].astype(str), category=rows["species"], label=1)
    nb19 = pd.read_csv(NB19)
    val = nb19[nb19["split"].eq("val")]
    val = val.assign(path=val["file"], file=[str(PROBE / "photos/val_nonsnake" / f) for f in val["file"]],
                     group=val["owner"].astype(str), label=0)
    train = pd.read_csv(STAGED / "nonsnake_train.csv")
    train = train.assign(file=[str(STAGED / p) for p in train["path"]], group=train["owner"].astype(str), label=0)
    return {"calib": snakes["calib"], "val": snakes["val"], "nonsnake_val": val, "nonsnake_train": train}


def prepare(args):
    files, size = args
    return np.concatenate([model_input(f, size) for f in files])


def score(model_path, frame):
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.log_severity_level = THREADS, 3
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    size = session.get_inputs()[0].shape[2]
    chunks = [list(frame["file"][i:i + BATCH]) for i in range(0, len(frame), BATCH)]
    probs, features = [], []
    with ThreadPoolExecutor(WORKERS) as pool:
        for start in range(0, len(chunks), PREFETCH):  # bounded, so prepared batches cannot pile up in memory
            for batch in pool.map(prepare, [(c, size) for c in chunks[start:start + PREFETCH]]):
                p, f = session.run(["probs", "features"], {"images": batch})
                probs.append(p)
                features.append(f.astype(np.float16))
    return np.concatenate(probs), np.concatenate(features)


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = photo_sets()
    timing = {}
    for name in args.sets:
        frame = sets[name].reset_index(drop=True)
        frame = frame.head(args.limit) if args.limit else frame
        started = time.time()
        probs, features = score(args.model, frame)
        timing[name] = {"photos": len(frame), "seconds": round(time.time() - started, 1)}
        keep = [c for c in ("path", "group", "category", "label", "genus", "family", "source") if c in frame]
        table = frame[keep].copy()
        for i, c in enumerate(CLASSES):
            table[f"p_{c}"] = probs[:, i]
        table.to_csv(out / f"scores_{name}.csv", index=False)
        np.save(out / f"features_{name}.npy", features)
        print(name, timing[name], flush=True)
    (out / "score_timing.json").write_text(json.dumps(timing, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, help="first N photos per set, for a quick check")
    parser.add_argument("--sets", nargs="+", default=["calib", "val", "nonsnake_val", "nonsnake_train"])
    main(parser.parse_args())
