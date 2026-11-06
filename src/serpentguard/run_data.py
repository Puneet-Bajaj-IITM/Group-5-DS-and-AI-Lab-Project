"""What one training run reads: its tables, frames and loaders, the seeding that keeps them reproducible, the class
mix its sampler draws and the page-cache warm-up."""
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import BaseCompose
from torch.utils.data import DataLoader, get_worker_info

from serpentguard.data import (BOX_CROP, FRONT_END_MODES, ID_TYPES, PK_PHOTOS, PREDICTED, SAM, UNKNOWN_POOL, PKSampler,
                               SnakeDataset, build_frame, cap_train, class_sampling_probs, make_sampler, with_files)
from serpentguard.metrics import assert_no_test_paths
from serpentguard.subset import calibration_set, unknown_split

MASK_INPUT_MODES = ("mask_box_crop", "soft_dim", "hard_mask")  # the model's input reads the mask
PREDICTED_INPUT_MODES = (BOX_CROP, *MASK_INPUT_MODES)           # F1, F1m, F2, F3: input from front-end predictions
METADATA_FILE, CLASSES_FILE = "metadata.csv", "classes.csv"
MASK_STATS_FILE, FRONTEND_PREDS_FILE = "mask_stats.csv", "frontend_preds.csv"
EVAL_BATCH_SCALE = 2           # evaluation keeps no gradients, so it takes twice the training batch
OPENCV_THREADS_PER_WORKER = 0  # 0 runs OpenCV sequentially inside each loader worker
WARM_THREADS = 64              # Kaggle's dataset mount: ~60 ms per cold file; 64 parallel reads reach ~800 files/s


@dataclass
class Tables:
    metadata: pd.DataFrame
    classes: pd.DataFrame
    subset: pd.DataFrame
    mask_stats: pd.DataFrame | None
    frontend_preds: pd.DataFrame | None


@dataclass
class RunFrames:
    """Frames with image_file and mask_file, ready for SnakeDataset; heldout is None without exclude_species."""
    train: pd.DataFrame
    val: pd.DataFrame
    calibration: pd.DataFrame
    unknown: pd.DataFrame
    heldout: pd.DataFrame | None
    mode: str


def input_mode(cfg):
    return FRONT_END_MODES[cfg.front_end]


def uses_masks(cfg):
    """True when training reads masks: a mask head, or a front end whose input is cut from the mask."""
    return cfg.mask_head or input_mode(cfg) in MASK_INPUT_MODES


def check_run_config(cfg):
    """Raises ValueError for a config that would quietly train something other than what its run id says."""
    if "kd" in cfg.losses:
        raise ValueError("the kd loss needs a teacher model, which is not built yet")
    if "triplet" in cfg.losses and cfg.sampler != "pk":
        raise ValueError("a triplet loss needs sampler 'pk', so every batch holds k photos of each class")
    if "arcface" in cfg.losses and cfg.head != "arcface":
        raise ValueError(f"the arcface loss needs head 'arcface', got head {cfg.head!r}")
    if cfg.front_end == "F4" and not (cfg.mask_head and cfg.pooling == "gap"):
        raise ValueError("F4 trains a mask head and pools without it: set mask_head true and pooling 'gap'")
    if (cfg.front_end == "F5") != (cfg.mask_head and cfg.pooling == "mask"):
        raise ValueError("F5, and only F5, pools with its mask: set mask_head true and pooling 'mask'")
    if input_mode(cfg) in PREDICTED_INPUT_MODES and cfg.mask_source != PREDICTED:
        raise ValueError(f"{cfg.front_end} trains and validates on front-end predictions: set mask_source 'predicted'")


def read_tables(cfg, data_root, masks_root, subset_csv, preds_root=None):
    """The CSVs a run needs; mask stats and front-end predictions only when the config reads them."""
    metadata = pd.read_csv(Path(data_root, METADATA_FILE), dtype=ID_TYPES, low_memory=False)
    classes = pd.read_csv(Path(data_root, CLASSES_FILE))
    subset = pd.read_csv(subset_csv, dtype=ID_TYPES)
    mask_stats = frontend_preds = None
    if cfg.mask_source == SAM and uses_masks(cfg):
        mask_stats = pd.read_csv(Path(masks_root, MASK_STATS_FILE), usecols=["path", "mask_path", "flagged"])
    if input_mode(cfg) in PREDICTED_INPUT_MODES or cfg.mask_source == PREDICTED:
        frontend_preds = pd.read_csv(Path(preds_root, FRONTEND_PREDS_FILE))
    return Tables(metadata, classes, subset, mask_stats, frontend_preds)


def first(frame, limit):
    return (frame if limit is None else frame.head(limit)).reset_index(drop=True)


def covered(frame, mode):
    return frame["box"].notna() if mode == BOX_CROP else frame["mask_valid"]


def run_frames(cfg, tables, data_root, masks_root=None, preds_root=None, limit=None):
    """The run's train, val, calibration, unknown and K1 heldout frames; `limit` keeps the first N photos of each.
    exclude_species is dropped after capping, so a K1 run trains on its parent's photos of every other class."""
    mode, excluded, metadata = input_mode(cfg), set(cfg.exclude_species), tables.metadata
    if excluded - set(tables.classes["species"]):
        raise ValueError(f"exclude_species not in classes.csv: {sorted(excluded - set(tables.classes['species']))}")
    assert_no_test_paths(tables.subset)
    subset_rows = metadata[(metadata["split"] == "train") & metadata["path"].isin(tables.subset["path"])]
    train = cap_train(build_frame(subset_rows, tables.classes, ["train"], tables.mask_stats, tables.frontend_preds,
                                  cfg.mask_source, masks_root, preds_root))
    train = first(train[~train["species"].isin(excluded)], limit)
    if not uses_masks(cfg):
        train = train.assign(mask_path="", mask_valid=False)
    if uses_masks(cfg) and not train["mask_valid"].any():
        raise ValueError(f"run {cfg.run_id} uses masks ({cfg.mask_source}), but no training photo has a valid mask")
    if mode == BOX_CROP and not train["box"].notna().any():
        raise ValueError(f"run {cfg.run_id} crops to front-end boxes, but no training photo has a box")

    predicted_input = mode in PREDICTED_INPUT_MODES

    def scored(rows, splits):
        frame = build_frame(rows, tables.classes, splits, frontend_preds=tables.frontend_preds if predicted_input else None,
                            mask_source=PREDICTED if predicted_input else SAM, preds_root=preds_root)
        return frame if mode in MASK_INPUT_MODES else frame.assign(mask_path="", mask_valid=False)

    known = ~metadata["species"].isin(excluded)
    val = first(scored(metadata[known], ["val"]), limit)
    if predicted_input and not covered(val, mode).any():
        raise ValueError(f"run {cfg.run_id} validates on front-end predictions, but frontend_preds.csv gives no "
                         f"validation photo a {'box' if mode == BOX_CROP else 'mask'}")
    calibration = scored(calibration_set(metadata[(metadata["split"] == "train") & known], tables.subset), ["train"])
    unknown = scored(unknown_split(metadata[metadata["split"] == UNKNOWN_POOL])[0], [UNKNOWN_POOL])
    heldout = None
    if excluded:
        rows = metadata[~known & metadata["split"].isin(["train", "val"])]
        heldout = first(scored(rows, ["train", "val"]).merge(rows[["path", "split", "obs_id"]], on="path", how="left",
                                                               validate="one_to_one"), limit)
    frames = {"train": train, "val": val, "calibration": first(calibration, limit), "unknown": first(unknown, limit),
              "heldout": heldout}
    return RunFrames(**{name: None if frame is None else with_files(frame, data_root) for name, frame in frames.items()},
                     mode=mode)


def prediction_coverage(frames):
    """For a front end that reads predictions, the share of each frame's photos with the box or mask it needs."""
    if frames.mode not in PREDICTED_INPUT_MODES:
        return None
    return {name: float(covered(frame, frames.mode).mean()) for name, frame in vars(frames).items()
            if isinstance(frame, pd.DataFrame)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_augmentation(dataset, seed):
    """Gives every augmentation pipeline and each of its transforms its own seed drawn from `seed`; Compose's own
    set_random_seed gives all its transforms one seed, so their draws would move together."""
    composes, transforms = {}, {}

    def walk(node):
        composes[id(node)] = node
        for child in node.transforms:
            if isinstance(child, BaseCompose):
                walk(child)
            else:
                transforms[id(child)] = child

    for pipeline in dataset.augment.values():
        walk(pipeline)
    parts = [*composes.values(), *transforms.values()]
    for part, part_seed in zip(parts, np.random.SeedSequence(seed).generate_state(len(parts))):
        part.set_random_seed(int(part_seed))


def init_worker(worker_id):
    """DataLoader worker_init_fn: reseeds Python, NumPy and the augmentation from the worker's seed, and runs OpenCV
    in one thread so two runs sharing a machine don't each start a thread per core."""
    info = get_worker_info()
    random.seed(info.seed)
    np.random.seed(info.seed % 2 ** 32)
    seed_augmentation(info.dataset, info.seed)
    cv2.setNumThreads(OPENCV_THREADS_PER_WORKER)


def train_loader(cfg, frame, mode, num_workers, device):
    """The weighted or P x K training loader, both drawing class_sampling_probs; weighted batches drop the last
    partial batch, because BatchNorm1d can't take a batch of one."""
    dataset = SnakeDataset(frame, train=True, mode=mode, size=cfg.input_size, aug_strength=cfg.aug_strength)
    seed_augmentation(dataset, cfg.seed)
    common = {"num_workers": num_workers, "pin_memory": device.type == "cuda", "worker_init_fn": init_worker,
              "generator": torch.Generator().manual_seed(cfg.seed), "persistent_workers": num_workers > 0}
    if cfg.sampler == "pk":
        sampler = PKSampler(frame["class_id"].to_numpy(), p=cfg.batch_size // PK_PHOTOS, k=PK_PHOTOS, seed=cfg.seed,
                            class_probs=class_sampling_probs(frame))
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(dataset, batch_size=cfg.batch_size, sampler=make_sampler(frame, seed=cfg.seed), drop_last=True,
                      **common)


def scored_frame(frame, mode):
    """The frame as an evaluation pass reads it: masks only when the input is cut from them."""
    return frame if mode in MASK_INPUT_MODES else frame.assign(mask_file="")


def eval_loader(cfg, frame, mode, num_workers, device):
    """In order, without augmentation."""
    dataset = SnakeDataset(scored_frame(frame, mode), train=False, mode=mode, size=cfg.input_size)
    return DataLoader(dataset, batch_size=cfg.batch_size * EVAL_BATCH_SCALE, shuffle=False, num_workers=num_workers,
                      pin_memory=device.type == "cuda")


def class_mix_report(frame, drawn_counts, classes):
    """Per class in the training frame: the share of draws it got against its class_sampling_probs share."""
    present = np.unique(frame["class_id"])
    expected = class_sampling_probs(frame)
    drawn = drawn_counts[present] / drawn_counts.sum()
    names = classes.set_index("class_id")["species"]
    return {
        "draws": int(drawn_counts.sum()),
        "total_variation": float(np.abs(drawn - expected).sum() / 2),
        "classes": [{"class_id": int(class_id), "species": names[class_id], "expected": float(share),
                     "drawn": float(got), "ratio": float(got / share)}
                    for class_id, share, got in zip(present, expected, drawn)],
    }


def frame_files(frames):
    """Every file a run opens for these frames, once each, as the training and evaluation loaders read them."""
    scored = [scored_frame(frame, frames.mode) for frame in
              (frames.train, frames.val, frames.calibration, frames.unknown, frames.heldout) if frame is not None]
    paths = dict.fromkeys(path for frame in [frames.train, *scored] for column in ("image_file", "mask_file")
                          for path in frame[column] if path)
    return [Path(path) for path in paths]


def run_files(cfg, data_root, masks_root, subset_csv, preds_root=None, limit=None):
    """Every file train() opens for cfg."""
    tables = read_tables(cfg, data_root, masks_root, subset_csv, preds_root)
    return frame_files(run_frames(cfg, tables, data_root, masks_root, preds_root, limit))


def read_size(path):
    try:
        return len(path.read_bytes())
    except OSError:  # one bad path never stops the warm-up
        return None


def warm_page_cache(paths, threads=WARM_THREADS):
    """Reads each file once in parallel so the OS page cache holds it (a cold read on Kaggle's mount is ~60 ms);
    unreadable files count as missing."""
    unique = list(dict.fromkeys(Path(path) for path in paths))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        sizes = [size for size in pool.map(read_size, unique) if size is not None]
    return {"files": len(sizes), "missing": len(unique) - len(sizes), "seconds": time.perf_counter() - started,
            "mb": sum(sizes) / 10 ** 6}
