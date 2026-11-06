"""Photos, masks and loaders: notebook 06's preprocessing, the input modes, augmentation, samplers.

A run's settings are a plain dict; DEFAULTS lists every setting and its default.
"""
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # else albumentations calls PyPI on import

import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler, get_worker_info

from serpentguard.metrics import TEST_SPLITS, WHO_RISK_ID, assert_no_test_paths, taxonomy_index

IMG_SIZE, SEED = 320, 42
NO_ID = -1                    # class, genus or family id of a species outside classes.csv
ID_TYPES = {"obs_id": str, "photo_id": str, "observer_id": str}
TARGETS = ("class_id", "genus_id", "family_id", "who_risk_id")
WHO_BOOST = {0: 1.0, 1: 1.35, 2: 1.6}  # sampling weight by WHO code
GREY_RGB = (124, 116, 104)    # padding, and the background in the masked modes
CROP_PADDING = 0.06           # margin around the mask or box
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PK_PHOTOS = 4                 # photos per class in a P x K batch
FRONT_END_MODES = {"F0": "full", "F1": "box_crop", "F1m": "mask_box_crop", "F2": "soft_dim",
                   "F3": "hard_mask", "F4": "full", "F5": "full"}
MASK_INPUT_MODES = ("mask_box_crop", "soft_dim", "hard_mask")  # input cut from the mask
PREDICTED_INPUT_MODES = ("box_crop", *MASK_INPUT_MODES)       # input from front-end predictions

# A run's settings in config.yaml's key order; None marks a setting every run must give.
DEFAULTS = dict(run_id=None, encoder=None, head=None, losses=None, front_end=None, input_size=320,
                epochs=20, lr=3e-4, weight_decay=0.05, batch_size=64, sampler=None, seed=42,
                subset_path=None, embedding_dim=256, arcface_s=30.0, arcface_m=0.3,
                triplet_margin=0.2, triplet_weight=1.0, mask_weight=1.0, hierarchy=True,
                mask_head=False, mask_keep=0.2, pooling="gap", danger_weight=5.0, aug_strength=1.0,
                mask_source="sam", box_source="none", eval_every=2, warmup_epochs=1,
                exclude_species=[])


def run_config(cfg):
    """cfg on top of DEFAULTS, in config.yaml's key order; an unknown key is an error."""
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}; the known ones are {list(DEFAULTS)}")
    cfg = {**DEFAULTS, **cfg}
    cfg["exclude_species"] = list(cfg["exclude_species"])
    return cfg


def input_mode(cfg):
    return FRONT_END_MODES[cfg["front_end"]]


def uses_masks(cfg):
    """True when training reads masks: a mask head, or an input cut from the mask."""
    return cfg["mask_head"] or input_mode(cfg) in MASK_INPUT_MODES


def load_tables(data_root):
    """(metadata, classes) from the dataset folder."""
    metadata = pd.read_csv(Path(data_root, "metadata.csv"), dtype=ID_TYPES, low_memory=False)
    return metadata, pd.read_csv(Path(data_root, "classes.csv"))


def is_missing_box(box):
    """True for None, NaN, "" and empty sequences."""
    if isinstance(box, str):
        return not box.strip()
    if np.ndim(box) == 0:
        return bool(pd.isna(box))
    return len(box) == 0


def parse_box(text):
    """A front-end box from its JSON text as (x1, y1, x2, y2) pixels, or None."""
    if is_missing_box(text):
        return None
    box = json.loads(text) if isinstance(text, str) else text
    return tuple(float(value) for value in box) if len(box) else None


def build_frame(metadata, classes, splits, mask_stats=None, frontend_preds=None, mask_source="sam",
                masks_root=None, preds_root=None):
    """Notebook 06's join of photos, class ids, masks and front-end boxes; never a test split."""
    splits = [splits] if isinstance(splits, str) else list(splits)
    if set(splits) & set(TEST_SPLITS):
        raise ValueError(f"build_frame never reads the test splits {TEST_SPLITS}; asked for "
                         f"{splits}")
    frame = metadata[metadata["split"].isin(splits)].reset_index(drop=True)
    class_of = {species: int(class_id)
                for species, class_id in zip(classes["species"], classes["class_id"])}
    frame["class_id"] = frame["species"].map(class_of).fillna(NO_ID).astype(int)  # -1: unknown pool
    known = frame["class_id"].ne(NO_ID).to_numpy()
    genus_names, family_names, genus_of_class, family_of_class = taxonomy_index(classes)
    for level, names, of_class in (("genus", genus_names, genus_of_class),
                                   ("family", family_names, family_of_class)):
        by_name = (frame[level].map({name: i for i, name in enumerate(names)}).fillna(NO_ID)
                   .to_numpy(dtype=int))
        frame[f"{level}_id"] = np.where(known, of_class[frame["class_id"].clip(lower=0)], by_name)
    frame["who_risk_id"] = frame["who_category"].astype(str).map(WHO_RISK_ID).astype(int)

    def joined(table, *columns):
        return frame[["path"]].merge(table[["path", *columns]], on="path", how="left",
                                     validate="one_to_one")

    if mask_source == "predicted":
        paths = joined(frontend_preds, "mask_file")["mask_file"]
        usable = paths.fillna("").astype(str).str.strip().ne("").to_numpy()
        root = preds_root
    elif mask_stats is not None:
        stats = joined(mask_stats, "mask_path", "flagged")
        paths, usable = (stats["mask_path"],
                         (stats["mask_path"].notna() & stats["flagged"].eq(False)).to_numpy())
        root = None if masks_root is None else Path(masks_root, "masks")
    else:
        paths, usable, root = [""] * len(frame), np.zeros(len(frame), dtype=bool), None
    # a photo without a usable mask gets mask_path ""
    frame["mask_path"] = ["" if not ok else str(path) if root is None else str(Path(root, path))
                          for path, ok in zip(paths, usable)]
    frame["mask_valid"] = usable
    boxes = (joined(frontend_preds, "box")["box"] if frontend_preds is not None
             else [None] * len(frame))
    frame["box"] = pd.Series([parse_box(text) for text in boxes], index=frame.index, dtype=object)
    frame = frame[["path", "species", "class_id", "genus_id", "family_id", "who_risk_id", "tier",
                   "observer_id", "in_india", "width", "height", "mask_path", "mask_valid", "box"]]
    assert_no_test_paths(frame)
    return frame


def with_files(frame, data_root):
    """The frame with the image_file and mask_file columns SnakeDataset reads."""
    return frame.assign(image_file=[str(Path(data_root, "data", path)) for path in frame["path"]],
                        mask_file=frame["mask_path"].where(frame["mask_valid"], ""))


def load_rgb(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return fix_exposure(rgb)


def fix_exposure(rgb):
    """CLAHE on the lightness of a photo whose mean brightness is under 60 or over 200."""
    if 60 <= rgb.mean() <= 200:
        return rgb
    lightness, a, b = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(lightness), a, b]), cv2.COLOR_LAB2RGB)


def load_mask(path, shape):
    """The cleaned mask, or None when the photo has no usable mask."""
    if not path:
        return None
    with Image.open(path) as image:
        mask = (np.asarray(image.convert("L")) > 127).astype(np.uint8)
    if mask.shape != shape:
        raise ValueError(f"mask {mask.shape} does not match photo {shape}: {path}")
    return clean_mask(mask)


def clean_mask(mask):
    """The 0/1 mask closed, small parts dropped; None when under 1.5% of the photo is left."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    parts, labels, part_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = part_stats[1:, cv2.CC_STAT_AREA]
    if parts <= 1 or areas.max() < 0.015 * mask.size:
        return None
    # keep parts of at least 12% of the largest part and 0.8% of the photo
    keep = 1 + np.flatnonzero(areas >= max(0.12 * areas.max(), 0.008 * mask.size))
    return np.isin(labels, keep).astype(np.uint8)


def prepare(rgb, mask, mode="full", size=IMG_SIZE):
    """The photo and mask in the input mode, letterboxed to size x size (uint8 arrays)."""
    target = np.zeros(rgb.shape[:2], np.uint8) if mask is None else mask
    if mode != "full" and mask is not None:
        if mode != "rgbm_4ch":
            keep = 0.0 if mode == "hard_mask" else 0.20  # soft_dim keeps 20% of the background
            weight = keep + (1 - keep) * cv2.GaussianBlur(mask.astype(np.float32), (5, 5),
                                                          0)[..., None]
            rgb = (rgb * weight + np.array(GREY_RGB) * (1 - weight)).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        pad_x, pad_y = int(np.ptp(xs) * CROP_PADDING), int(np.ptp(ys) * CROP_PADDING)
        window = (slice(max(0, ys.min() - pad_y), ys.max() + pad_y + 1),
                  slice(max(0, xs.min() - pad_x), xs.max() + pad_x + 1))
        rgb, target = rgb[window], target[window]

    height, width = rgb.shape[:2]
    scale = size / max(height, width)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb = cv2.resize(rgb, resized, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    target = cv2.resize(target, resized, interpolation=cv2.INTER_NEAREST)
    top, left = (size - resized[1]) // 2, (size - resized[0]) // 2
    canvas = np.full((size, size, 3), GREY_RGB, dtype=np.uint8)
    canvas_mask = np.zeros((size, size), np.uint8)
    canvas[top:top + resized[1], left:left + resized[0]] = rgb
    canvas_mask[top:top + resized[1], left:left + resized[0]] = target
    return canvas, canvas_mask


def mask_box(mask):
    """The mask's bounding box (x1, y1, x2, y2), inclusive pixels, or None for no or empty mask."""
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_to_box(rgb, mask, box, pad=CROP_PADDING):
    """rgb (and mask) cropped to box plus padding; a missing or degenerate box changes nothing."""
    if is_missing_box(box):
        return rgb, mask
    x1, y1, x2, y2 = (float(value) for value in box)
    if (not all(math.isfinite(value) for value in (x1, y1, x2, y2))
            or x2 - x1 < 1.0 or y2 - y1 < 1.0):
        return rgb, mask
    height, width = rgb.shape[:2]
    pad_x, pad_y = int((x2 - x1) * pad), int((y2 - y1) * pad)
    # pixels floor(x1)..floor(x2) are kept, as prepare keeps them
    left, top = max(0, math.floor(x1) - pad_x), max(0, math.floor(y1) - pad_y)
    right, bottom = (min(width, math.floor(x2) + pad_x + 1),
                     min(height, math.floor(y2) + pad_y + 1))
    if right <= left or bottom <= top:
        return rgb, mask
    window = (slice(top, bottom), slice(left, right))
    return rgb[window], None if mask is None else mask[window]


def apply_mode(rgb, mask, mode="full", box=None, size=IMG_SIZE):
    """One input mode, letterboxed: box_crop crops to box, mask_box_crop to the mask's box."""
    if mode == "box_crop":
        rgb, mask = crop_to_box(rgb, mask, box)
    elif mode == "mask_box_crop":
        rgb, mask = crop_to_box(rgb, mask, mask_box(mask))
    return prepare(rgb, mask, "full" if mode in ("box_crop", "mask_box_crop") else mode, size)


def build_augment(strength=1.0):
    """Notebook 06's tier and eval pipelines, ranges and chances scaled by strength (1 = as is)."""
    def spread(limit):
        return -limit * strength, limit * strength

    def around_one(low, high):
        return 1 + (low - 1) * strength, 1 + (high - 1) * strength

    def chance(p):
        return min(1.0, p * strength)

    to_tensor = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    camera_style = [
        A.ImageCompression(quality_range=(40, 95), p=0.3),
        A.Downscale(scale_range=(0.5, 0.9), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=chance(0.15)),
        A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.3),
    ]
    return {
        "tier1": A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Affine(scale=around_one(0.88, 1.12), rotate=spread(25),
                     translate_percent=spread(0.05), border_mode=cv2.BORDER_CONSTANT, p=0.6),
            A.ColorJitter(brightness=0.2 * strength, contrast=0.2 * strength,
                          saturation=0.2 * strength, hue=0.04 * strength, p=0.5),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 36),
                            hole_width_range=(16, 36), p=chance(0.35)),
            *camera_style,
            *to_tensor,
        ]),
        "tier2": A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.35),
            A.Affine(scale=around_one(0.80, 1.20), rotate=spread(45),
                     translate_percent=spread(0.08), shear=spread(10),
                     border_mode=cv2.BORDER_CONSTANT, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.3 * strength,
                                       contrast_limit=0.3 * strength, p=0.7),
            A.HueSaturationValue(hue_shift_limit=8 * strength, sat_shift_limit=25 * strength,
                                 val_shift_limit=25 * strength, p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=chance(0.25)),
            A.CoarseDropout(num_holes_range=(2, 6), hole_height_range=(20, 44),
                            hole_width_range=(20, 44), p=chance(0.5)),
            *camera_style,
            *to_tensor,
        ]),
        "eval": A.Compose(to_tensor),
    }


AUGMENT = build_augment(1.0)


class SnakeDataset(Dataset):
    """Each frame row as its image, mask, whether the mask is real, and the four targets."""

    def __init__(self, frame, train=False, mode="full", size=IMG_SIZE, aug_strength=1.0):
        self.frame, self.train, self.mode, self.size = (frame.reset_index(drop=True), train,
                                                        mode, size)
        self.augment = AUGMENT if aug_strength == 1.0 else build_augment(aug_strength)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        rgb = load_rgb(row["image_file"])
        mask = load_mask(row["mask_file"], rgb.shape[:2])
        image, target = apply_mode(rgb, mask, self.mode, row.get("box"), self.size)
        sample = self.augment[row["tier"] if self.train else "eval"](image=image, mask=target)
        target = sample["mask"].unsqueeze(0).float()
        image = (torch.cat([sample["image"], target]) if self.mode == "rgbm_4ch"
                 else sample["image"])
        return {"image": image, "mask": target, "mask_valid": mask is not None,
                **{name: int(row[name]) for name in TARGETS}}


def cap_train(train):
    """Up to 50 photos per photographer per class and 1000 per class, photographers in turn."""
    shuffled = train.sample(frac=1, random_state=SEED)
    turn = shuffled.groupby(["species", "observer_id"]).cumcount()
    kept = shuffled.assign(turn=turn)[turn < 50].sort_values("turn", kind="stable")
    return kept.groupby("species").head(1000).drop(columns="turn")


def calibration_set(train, subset):
    """Training photos outside the subset whose photographer has no photo in it."""
    return train[~train["path"].isin(subset["path"])
                 & ~train["observer_id"].isin(subset["observer_id"])]


def unknown_split(unknown, seed=SEED):
    """(half_a, half_b) of the unknown pool; A holds the listed and a seeded half of the rest."""
    listed = set(unknown.loc[unknown["who_category"].astype(str) != "not_listed", "species"])
    others = sorted(set(unknown["species"]) - listed)
    shuffled = np.random.default_rng(seed).permutation(others)
    in_a = unknown["species"].isin(listed | set(shuffled[:len(others) // 2]))
    return unknown[in_a], unknown[~in_a]


def photo_weights(frame, who_boost=WHO_BOOST):
    """Each photo's draw weight: 1/sqrt(class size), boosted for WHO-listed species."""
    class_size = frame.groupby("class_id")["class_id"].transform("size")
    return (1 / np.sqrt(class_size) * frame["who_risk_id"].map(who_boost)).to_numpy()


def make_sampler(frame, seed=SEED):
    """Draws each photo by photo_weights, as many draws as photos."""
    return WeightedRandomSampler(torch.tensor(photo_weights(frame), dtype=torch.double),
                                 len(frame), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def class_sampling_probs(frame, who_boost=WHO_BOOST):
    """Each class's share of make_sampler's draws, in class_id order."""
    by_class = (pd.Series(photo_weights(frame, who_boost), index=frame["class_id"].to_numpy())
                .groupby(level=0).sum())
    return (by_class / by_class.sum()).to_numpy()


class PKSampler(Sampler):
    """Batches of p classes (drawn by class_probs, without replacement) times k photos each."""

    def __init__(self, labels, p=16, k=PK_PHOTOS, seed=SEED, class_probs=None):
        labels = np.asarray(labels)
        self.members = [np.flatnonzero(labels == name) for name in np.unique(labels)]
        weights = (np.array([1 / np.sqrt(len(members)) for members in self.members])
                   if class_probs is None else np.asarray(class_probs, dtype=float))
        self.class_probability = weights / weights.sum()
        self.p, self.k = p, k
        self.batches = max(1, len(labels) // (p * k))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            picked = self.rng.choice(len(self.members), self.p, replace=False,
                                     p=self.class_probability)
            # photos repeat only in a class with fewer than k
            yield [int(index) for name in picked
                   for index in self.rng.choice(self.members[name], self.k,
                                                replace=len(self.members[name]) < self.k)]


def has_input(frame, mode):
    """Per photo: whether it has the box (box_crop) or mask (other modes) its input needs."""
    return frame["box"].notna() if mode == "box_crop" else frame["mask_valid"]


def make_frames(cfg, data_root, masks_root, subset_csv, preds_root=None, limit=None):
    """The run's train, val, calibration, unknown and heldout frames; limit keeps N of each."""
    cfg = run_config(cfg)
    mode, excluded = input_mode(cfg), set(cfg["exclude_species"])
    predicted_input = mode in PREDICTED_INPUT_MODES
    metadata, classes = load_tables(data_root)
    subset = pd.read_csv(subset_csv, dtype=ID_TYPES)
    mask_stats = frontend_preds = None
    if cfg["mask_source"] == "sam" and uses_masks(cfg):
        mask_stats = pd.read_csv(Path(masks_root, "mask_stats.csv"),
                                 usecols=["path", "mask_path", "flagged"])
    if predicted_input or cfg["mask_source"] == "predicted":
        frontend_preds = pd.read_csv(Path(preds_root, "frontend_preds.csv"))
    if excluded - set(classes["species"]):
        raise ValueError("exclude_species not in classes.csv: "
                         f"{sorted(excluded - set(classes['species']))}")
    assert_no_test_paths(subset)

    def first(frame):
        return (frame if limit is None else frame.head(limit)).reset_index(drop=True)

    subset_rows = metadata[(metadata["split"] == "train") & metadata["path"].isin(subset["path"])]
    train = cap_train(build_frame(subset_rows, classes, ["train"], mask_stats, frontend_preds,
                                  cfg["mask_source"], masks_root, preds_root))
    # held-out species go after capping, so such a run keeps its parent's photos of the rest
    train = first(train[~train["species"].isin(excluded)])
    if not uses_masks(cfg):
        train = train.assign(mask_path="", mask_valid=False)
    if uses_masks(cfg) and not train["mask_valid"].any():
        raise ValueError(f"run {cfg['run_id']} uses masks ({cfg['mask_source']}), but no "
                         "training photo has a valid mask")
    if mode == "box_crop" and not train["box"].notna().any():
        raise ValueError(f"run {cfg['run_id']} crops to front-end boxes, but no training photo "
                         "has a box")

    def scored(rows, splits):
        frame = build_frame(rows, classes, splits,
                            frontend_preds=frontend_preds if predicted_input else None,
                            mask_source="predicted" if predicted_input else "sam",
                            preds_root=preds_root)
        return frame if mode in MASK_INPUT_MODES else frame.assign(mask_path="", mask_valid=False)

    known = ~metadata["species"].isin(excluded)
    val = first(scored(metadata[known], ["val"]))
    if predicted_input and not has_input(val, mode).any():
        raise ValueError(f"run {cfg['run_id']} validates on front-end predictions, but "
                         "frontend_preds.csv gives no validation photo a "
                         f"{'box' if mode == 'box_crop' else 'mask'}")
    calibration = scored(calibration_set(metadata[(metadata["split"] == "train") & known], subset),
                         ["train"])
    unknown = scored(unknown_split(metadata[metadata["split"] == "unknown_pool"])[0],
                     ["unknown_pool"])
    heldout = None
    if excluded:
        rows = metadata[~known & metadata["split"].isin(["train", "val"])]
        heldout = first(scored(rows, ["train", "val"]).merge(
            rows[["path", "split", "obs_id"]], on="path", how="left", validate="one_to_one"))
    frames = {"train": train, "val": val, "calibration": first(calibration),
              "unknown": first(unknown), "heldout": heldout}
    return {name: None if frame is None else with_files(frame, data_root)
            for name, frame in frames.items()}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_augmentation(dataset, seed):
    """Gives every augmentation pipeline and transform its own seed drawn from `seed`."""
    composes, transforms = {}, {}  # Compose's own seeding would give its transforms one seed

    def walk(node):
        composes[id(node)] = node
        for child in node.transforms:
            if isinstance(child, A.BaseCompose):
                walk(child)
            else:
                transforms[id(child)] = child

    for pipeline in dataset.augment.values():
        walk(pipeline)
    parts = [*composes.values(), *transforms.values()]
    for part, part_seed in zip(parts, np.random.SeedSequence(seed).generate_state(len(parts))):
        part.set_random_seed(int(part_seed))


def init_worker(worker_id):
    """DataLoader worker setup: reseed from the worker's seed; OpenCV in one thread."""
    info = get_worker_info()
    random.seed(info.seed)
    np.random.seed(info.seed % 2 ** 32)
    seed_augmentation(info.dataset, info.seed)
    cv2.setNumThreads(0)


def train_loader(cfg, frame, num_workers=4):
    """The weighted or P x K training loader; both draw the class_sampling_probs mix."""
    dataset = SnakeDataset(frame, train=True, mode=input_mode(cfg), size=cfg["input_size"],
                           aug_strength=cfg["aug_strength"])
    seed_augmentation(dataset, cfg["seed"])
    common = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available(),
              "worker_init_fn": init_worker,
              "generator": torch.Generator().manual_seed(cfg["seed"]),
              "persistent_workers": num_workers > 0}
    if cfg["sampler"] == "pk":
        sampler = PKSampler(frame["class_id"].to_numpy(), p=cfg["batch_size"] // PK_PHOTOS,
                            k=PK_PHOTOS, seed=cfg["seed"], class_probs=class_sampling_probs(frame))
        return DataLoader(dataset, batch_sampler=sampler, **common)
    # drop_last: BatchNorm1d can't take a batch of one
    return DataLoader(dataset, batch_size=cfg["batch_size"],
                      sampler=make_sampler(frame, seed=cfg["seed"]), drop_last=True, **common)


def scored_frame(frame, mode):
    """The frame as an evaluation pass reads it: masks only when the input is cut from them."""
    return frame if mode in MASK_INPUT_MODES else frame.assign(mask_file="")


def eval_loader(cfg, frame, num_workers=4):
    """In order, no augmentation, twice the training batch (no gradients are kept)."""
    mode = input_mode(cfg)
    dataset = SnakeDataset(scored_frame(frame, mode), train=False, mode=mode,
                           size=cfg["input_size"])
    return DataLoader(dataset, batch_size=cfg["batch_size"] * 2, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def make_loaders(cfg, data_root, masks_root, subset_csv, preds_root=None, num_workers=4,
                 limit=None):
    """(frames, training loader, validation loader) for one run."""
    cfg = run_config(cfg)
    frames = make_frames(cfg, data_root, masks_root, subset_csv, preds_root, limit)
    return (frames, train_loader(cfg, frames["train"], num_workers),
            eval_loader(cfg, frames["val"], num_workers))


def frame_files(frames, mode):
    """Every file a run opens for these frames, once each."""
    present = [frame for frame in frames.values() if frame is not None]
    paths = dict.fromkeys(path
                          for frame in [frames["train"], *(scored_frame(f, mode) for f in present)]
                          for column in ("image_file", "mask_file") for path in frame[column]
                          if path)
    return [Path(path) for path in paths]


def read_size(path):
    try:
        return len(path.read_bytes())
    except OSError:  # one bad path never stops the warm-up
        return None


def warm_cache(files, threads=64):
    """Reads each file once in parallel so the OS page cache holds it before training."""
    # a cold read on Kaggle's dataset mount takes ~60 ms; 64 parallel reads reach ~800 files/s
    unique = list(dict.fromkeys(Path(path) for path in files))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        sizes = [size for size in pool.map(read_size, unique) if size is not None]
    return {"files": len(sizes), "missing": len(unique) - len(sizes),
            "seconds": time.perf_counter() - started, "mb": sum(sizes) / 10 ** 6}
