"""Bias or variance? The shipped model A3 on its own training photos against validation photos.

Usage: python scripts/check_train_vs_val.py --run <A3 output> --masks <front-end masks> --metadata <metadata.csv>
       --photos <cache folder> [--per-sample 1500]

--run is the A3 full run's Kaggle output folder (`M4_FULL/` with best.pt and val_outputs.npz, and
`full/full_train.csv`); --masks is the Milestone 3 front-end dataset (frontend_preds.csv and masks_pred/);
--metadata is the clean dataset's metadata.csv. Both samples are drawn the same way (metrics.round_robin,
seed 42) and scored through the training code's evaluation path (data.build_frame, data.eval_loader,
train.predict): the eval transform, no augmentation, no scenery swap, A3's crop to the predicted mask's box.
Training photos come from the photos A3 trained on that have a Milestone 3 mask (the experiment subset).
Photos are fetched from their iNaturalist URL into the cache folder and kept only when their SHA-1 matches the
dataset's. The validation sample's predictions must agree with the run's saved validation outputs.
Writes results/m4/bias_variance/train_vs_val.json. CPU only.
"""
import argparse
import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from serpentguard import data, metrics, train  # noqa: E402

OUT = REPO / "results/m4/bias_variance/train_vs_val.json"
CLASSES = REPO / "data/classes.csv"
THREADS, RETRIES, TIMEOUT_S, CPU_THREADS = 16, 4, 60, 8
MIN_AGREEMENT = 0.98   # the validation sample must reproduce the run's saved top species on this share


def fetch(job):
    """Downloads one photo; True when the file's SHA-1 matches the dataset's."""
    target, url, sha1 = job
    for _ in range(RETRIES):
        if target.exists() and hashlib.sha1(target.read_bytes()).hexdigest() == sha1:
            return True
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSfL", "--max-time", str(TIMEOUT_S), "-o", str(target), url],
                       capture_output=True)
    if target.exists():
        target.unlink()
    return False


def sample(frame, metadata, photos, size):
    """The round-robin sample, its photos fetched; rows whose photo could not be fetched are dropped."""
    picked = metrics.round_robin(frame, size)
    source = metadata.set_index("path").loc[picked["path"], ["url", "sha1"]]
    jobs = [(photos / "data" / path, url, sha1) for path, url, sha1 in
            zip(picked["path"], source["url"], source["sha1"])]
    with ThreadPoolExecutor(THREADS) as pool:
        kept = np.array(list(pool.map(fetch, jobs)))
    return data.with_files(picked[kept], photos), {"asked": len(picked), "kept": int(kept.sum())}


def scores(frame, logits, train_counts, n_classes):
    """Top-1, macro-F1 and macro-F1 by training photos per species, with photographer-bootstrap intervals."""
    y, predicted = frame["class_id"].to_numpy(int), logits.argmax(1)
    val_counts = np.bincount(y, minlength=n_classes)
    report = metrics.species_report(y, metrics.softmax(logits), train_counts, val_counts, np.zeros(n_classes))
    groups = metrics.photographer_groups(frame)
    top1_interval = metrics.grouped_bootstrap(lambda rows: float(np.mean(y[rows] == predicted[rows])), groups)
    return {"photos": int(len(y)), "classes": int((val_counts > 0).sum()), "top1": report["top1"],
            "top1_interval": list(top1_interval), "macro_f1": report["macro_f1"],
            "macro_f1_by_training_photos": report["macro_f1_band"]}


def main(args):
    run, masks, photos = Path(args.run), Path(args.masks), Path(args.photos)
    metadata = pd.read_csv(args.metadata, dtype=data.ID_TYPES, keep_default_na=False)
    classes = pd.read_csv(CLASSES)
    preds = pd.read_csv(masks / "frontend_preds.csv", keep_default_na=False)
    trained_on = pd.read_csv(run / "full/full_train.csv")["path"]
    frames, fetched = {}, {}
    for split in ("train", "val"):
        frame = data.build_frame(metadata, classes, split, frontend_preds=preds, mask_source="predicted",
                                 preds_root=masks)
        frame = frame[frame["path"].isin(preds["path"]) & frame["class_id"].ge(0)]
        if split == "train":
            frame = frame[frame["path"].isin(trained_on)]
        frames[split], fetched[split] = sample(frame, metadata, photos, args.per_sample)

    cfg, model, epoch = train.load_checkpoint(run / "M4_FULL")
    torch.set_num_threads(CPU_THREADS)
    logits = {split: train.predict(model, data.eval_loader(cfg, frame, num_workers=0), torch.device("cpu"),
                                   amp=False)["species_logits"] for split, frame in frames.items()}

    saved = np.load(run / "M4_FULL/val_outputs.npz", allow_pickle=True)
    row_of = {path: i for i, path in enumerate(saved["paths"])}
    rows = [row_of[path] for path in frames["val"]["path"]]
    agreement = float(np.mean(logits["val"].argmax(1) == saved["species_logits"][rows].argmax(1)))
    if agreement < MIN_AGREEMENT:
        raise RuntimeError(f"the validation sample agrees with the run's saved outputs on {agreement:.3f}")

    n_classes = len(classes)
    species_of = metadata.set_index("path").loc[trained_on, "species"]
    train_counts = np.bincount(classes.set_index("species").loc[species_of, "class_id"].to_numpy(int),
                               minlength=n_classes)
    result = {"model": f"A3, best epoch {epoch}", "per_sample": args.per_sample, "fetched": fetched,
              "same_top_species_as_saved_val_outputs": agreement,
              **{split: scores(frames[split], logits[split], train_counts, n_classes) for split in frames}}
    shuffled = np.random.default_rng(metrics.SAMPLE_SEED).permutation(frames["train"]["class_id"].to_numpy(int))
    result["control_train_labels_shuffled_top1"] = float(np.mean(shuffled == logits["train"].argmax(1)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_json(OUT, result)
    print(OUT.relative_to(REPO), "train top-1", round(result["train"]["top1"], 3), "val top-1",
          round(result["val"]["top1"], 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("--run", "--masks", "--metadata", "--photos"):
        parser.add_argument(name, required=True)
    parser.add_argument("--per-sample", type=int, default=1500)
    main(parser.parse_args())
