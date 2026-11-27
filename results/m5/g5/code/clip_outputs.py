"""Every frame of every clip through the app's path (ONNX reference pipeline), saved as raw outputs.

The lock is simulated later on these outputs at any phone latency (video.schedule picks the frames),
with any thresholds (temperatures are applied at analysis time). Model-agnostic: --models is a folder
with classifier.onnx, segmenter.onnx, detector.onnx, pipeline.json and tables.json.

Usage: python clip_outputs.py --models DIR --reference reference_pipeline.py --out DIR [--jobs N] [--only SET]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[3]
REPO = WORK / "m5-g5"
OLD_CLIPS = WORK / "m4-video-clips"
NEW_DIR = WORK / "results/m5/g5"
METADATA = WORK / "kaggle/clean_build/out/metadata.csv"
MAX_SIDE, MAX_FPS, MAX_SECONDS = 640, 15, 30
N_SYNTHETIC, SEED, N_LISTED_PER_SPECIES = 20, 42, 2
SYNTH_LICENCES = ("cc0", "cc-by")
OUTPUT_KEYS = ("embedding", "species_logits", "who_logits", "genus_logits", "family_logits")

sys.path.insert(0, str(REPO / "src"))


def load_reference(path):
    spec = importlib.util.spec_from_file_location("reference_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INTERPOLATION = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}  # linear ~ the app's Android bilinear scaling
RESIZE = INTERPOLATION[os.environ.get("RESIZE", "area")]


def read_video(path):
    """Frames as notebook 24 reads them: at most 15 fps, 640 px long side (INTER_AREA), 30 s."""
    capture = cv2.VideoCapture(str(path))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or MAX_FPS
    step = max(1, round(source_fps / MAX_FPS))
    frames, n = [], 0
    while len(frames) < MAX_SECONDS * source_fps / step:
        ok, frame = capture.read()
        if not ok:
            break
        if n % step == 0:
            scale = min(1.0, MAX_SIDE / max(frame.shape[:2]))
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=RESIZE)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        n += 1
    capture.release()
    if not frames:
        raise ValueError(f"no frames in {path}")
    return frames, source_fps / step


def synthetic_frames(photo_id):
    from serpentguard import video
    clip = video.synthesize_clip(video.load_raw_rgb(OLD_CLIPS / "val_photos" / f"{photo_id}.jpg"),
                                 seed=SEED + int(photo_id) % 10_000)
    return clip.frames, float(clip.params["fps"])


def synthetic_pick():
    """Notebook 24's synthetic dev clips: the same draw from the validation metadata."""
    classes = pd.read_csv(REPO / "data/classes.csv")
    meta = pd.read_csv(METADATA, usecols=["species", "split", "license", "photo_id", "who_category"])
    pool = meta[(meta.split == "val") & meta.license.isin(SYNTH_LICENCES) & meta.species.isin(classes.species)]
    random_pick = pool.groupby("species", group_keys=False).sample(1, random_state=SEED)
    random_pick = random_pick.sample(N_SYNTHETIC, random_state=SEED).assign(group="random")
    listed = pool[pool.who_category.astype(str) != "not_listed"]
    listed_pick = (listed.sample(frac=1, random_state=SEED).groupby("species").head(N_LISTED_PER_SPECIES)
                   .assign(group="who_listed"))
    picked = pd.concat([random_pick, listed_pick[~listed_pick.photo_id.isin(random_pick.photo_id)]])
    return picked.reset_index(drop=True)


def clip_table():
    """Every clip: name, set, species (None for no snake), how to read its frames."""
    rows = []
    for row in synthetic_pick().itertuples():
        rows.append({"clip": f"synthetic_{row.photo_id}", "set": "dev_synthetic", "species": row.species,
                     "species_source": "stated", "source": ("synthetic", int(row.photo_id))})
    old = pd.read_csv(REPO / "results/m4/video/clips.csv")
    for row in old.itertuples():
        rows.append({"clip": row.clip, "set": "dev_real" if row.split == "dev" else "old_test",
                     "species": row.species, "species_source": row.species_source,
                     "source": ("file", str(OLD_CLIPS / row.clip))})
    for row in pd.read_csv(REPO / "results/m4/video/negatives.csv").itertuples():
        rows.append({"clip": row.clip, "set": "old_negative", "species": None, "species_source": None,
                     "source": ("file", str(OLD_CLIPS / row.clip))})
    for row in pd.read_csv(NEW_DIR / "new_clips.csv").itertuples():
        rows.append({"clip": row.clip, "set": "new_test", "species": row.species,
                     "species_source": row.species_source, "source": ("file", str(NEW_DIR / "new_clips" / row.clip))})
    for row in pd.read_csv(NEW_DIR / "new_negatives.csv").itertuples():
        rows.append({"clip": row.clip, "set": "new_negative", "species": None, "species_source": None,
                     "source": ("file", str(NEW_DIR / "new_clips" / row.clip))})
    return rows


def model_tag(models):
    digest = hashlib.sha256()
    for name in ("classifier.onnx", "segmenter.onnx", "detector.onnx", "pipeline.json", "tables.json"):
        digest.update(hashlib.sha256((Path(models) / name).read_bytes()).digest())
    return digest.hexdigest()[:12]


def source_tag(source):
    kind, value = source
    if kind == "synthetic":
        return f"synthetic{value}"
    return hashlib.sha256(Path(value).read_bytes()).hexdigest()[:12]


def run_clip(row, models, reference, out_dir, tag):
    target = Path(out_dir) / f"{row['clip']}__{tag}__{source_tag(row['source'])}.npz"
    if target.exists():
        return row["clip"], str(target), "cached"
    ref = load_reference(reference)
    pipeline = ref.Pipeline(models, threads=1)
    kind, value = row["source"]
    frames, fps = synthetic_frames(value) if kind == "synthetic" else read_video(value)
    passed, box, area = [], [], []
    outputs = {key: [] for key in OUTPUT_KEYS}
    sizes = {"embedding": 256, "species_logits": 184, "who_logits": 3, "genus_logits": 69, "family_logits": 13}
    for frame in frames:
        rgb = ref.fix_exposure(frame, pipeline.settings["exposure_fix"])
        gate, mask = pipeline.gate(rgb)
        passed.append(gate["snake"])
        box.append(np.nan if gate["box_conf"] is None else gate["box_conf"])
        area.append(gate["mask_area"])
        if gate["snake"]:
            height, width = rgb.shape[:2]
            window = ref.crop_window(ref.clean_mask(mask, pipeline.settings["mask_cleanup"]), height, width,
                                     pipeline.settings["crop"]["padding"])
            crop = rgb if window is None else rgb[window[0]:window[1], window[2]:window[3]]
            result = pipeline.classify(crop)
            for key in OUTPUT_KEYS:
                outputs[key].append(np.asarray(result[key], dtype=np.float32))
        else:
            for key in OUTPUT_KEYS:
                outputs[key].append(np.full(sizes[key], np.nan, np.float32))
    tmp = target.with_suffix(".tmp.npz")
    np.savez(tmp, fps=fps, times=np.arange(len(frames)) / fps, passed=np.array(passed),
             box_conf=np.array(box), mask_area=np.array(area), **{k: np.stack(v) for k, v in outputs.items()})
    os.replace(tmp, target)
    return row["clip"], str(target), f"{len(frames)} frames, {int(sum(passed))} pass the gate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--manifest", action="store_true", help="only write the manifest of finished clips")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = model_tag(args.models)
    rows = [r for r in clip_table() if not args.only or r["set"] in args.only]
    if args.manifest:
        manifest = []
        for row in rows:
            path = args.out / f"{row['clip']}__{tag}__{source_tag(row['source'])}.npz"
            assert path.exists(), f"missing {path.name}"
            manifest.append({k: row[k] for k in ("clip", "set", "species", "species_source")} | {"outputs": path.name})
        pd.DataFrame(manifest).to_csv(args.out / f"manifest_{tag}.csv", index=False)
        print("model tag", tag, "clips", len(manifest))
        return
    for row in rows[args.shard::args.shards]:
        clip, _, note = run_clip(row, str(args.models), str(args.reference), str(args.out), tag)
        print(clip, note, flush=True)


if __name__ == "__main__":
    main()
