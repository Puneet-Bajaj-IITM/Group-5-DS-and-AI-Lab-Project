"""Notebook 06's preprocessing and loader, copied so every M3 run reads photos the same way (tests/test_data.py checks
they still match), plus the M3 frames, front-end input modes, augmentation strength and the P x K sampler."""
import json
import math
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from serpentguard.metrics import TEST_SPLITS, assert_no_test_paths

IMG_SIZE, SEED = 320, 42
TRAIN_CAP = 1000             # most training photos kept per class
PHOTOGRAPHER_CAP = 50        # most training photos per photographer per class
DARK_BELOW, BRIGHT_ABOVE = 60, 200  # mean brightness outside this range gets exposure correction
MASK_MODE = "full"           # full, soft_dim, hard_mask or rgbm_4ch
WHO_RISK_ID = {"not_listed": 0, "2": 1, "1": 2}
WHO_BOOST = {0: 1.0, 1: 1.35, 2: 1.6}
GREY_RGB = (124, 116, 104)   # padding, and the background in the masked modes
BACKGROUND_KEEP = 0.20       # how much background soft_dim keeps
CROP_PADDING = 0.06          # margin around the mask or box
MIN_BOX_SIDE = 1.0           # pixels; a thinner or shorter box is treated as missing
MIN_MASK_SHARE = 0.015       # a mask smaller than this share of the photo is not used
MIN_PART_SHARE = 0.008       # smallest separate part of a mask to keep, as a share of the photo
MIN_PART_OF_LARGEST = 0.12   # and as a share of the largest part
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ID_TYPES = {"obs_id": str, "photo_id": str, "observer_id": str}
TARGETS = ("class_id", "genus_id", "family_id", "who_risk_id")
PK_CLASSES, PK_PHOTOS = 16, 4
IMAGE_FOLDER = "data"        # photos at <data root>/data/<path>, SAM masks at <masks root>/masks/<mask_path>
UNKNOWN_POOL = "unknown_pool"
NO_ID = -1                   # class, genus or family id of an unknown-pool species outside classes.csv
FRAME_COLUMNS = ["path", "species", "class_id", "genus_id", "family_id", "who_risk_id", "tier", "observer_id", "in_india",
                 "width", "height", "mask_path", "mask_valid", "box"]
BOX, SAM, PREDICTED = "box", "sam", "predicted"
BOX_CROP = "box_crop"
BOX_MODES = (BOX_CROP, "mask_box_crop")
FRONT_END_MODES = {"F0": "full", "F1": "box_crop", "F1m": "mask_box_crop", "F2": "soft_dim", "F3": "hard_mask",
                   "F4": "full", "F5": "full"}
DEFAULT_AUG_STRENGTH = 1.0

CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def class_index(classes):
    """Species name to class_id, from classes.csv."""
    return {species: int(class_id) for species, class_id in zip(classes["species"], classes["class_id"])}


def taxonomy_index(classes):
    """(genus_names, family_names, genus_of_class, family_of_class): the sorted names, and each class's genus and
    family ids indexed by class_id."""
    ordered = classes.sort_values("class_id")
    assert ordered["class_id"].tolist() == list(range(len(ordered))), "class_id must run 0..C-1"
    names = [sorted(classes[level].unique()) for level in ("genus", "family")]
    of_class = [ordered[level].map({name: i for i, name in enumerate(level_names)}).to_numpy(dtype=int)
                for level, level_names in zip(("genus", "family"), names)]
    return names[0], names[1], of_class[0], of_class[1]


def is_missing_box(box):
    """True for None, NaN, "" and empty sequences."""
    if isinstance(box, str):
        return not box.strip()
    if np.ndim(box) == 0:
        return bool(pd.isna(box))
    return len(box) == 0


def parse_box(text):
    """A front-end box from its JSON text as (x1, y1, x2, y2) floats in pixels, or None when there is none."""
    if is_missing_box(text):
        return None
    box = json.loads(text) if isinstance(text, str) else text
    return tuple(float(value) for value in box) if len(box) else None


def joined(frame, table, *columns):
    return frame[["path"]].merge(table[["path", *columns]], on="path", how="left", validate="one_to_one")


def build_frame(metadata, classes, splits, mask_stats=None, frontend_preds=None, mask_source=SAM, masks_root=None,
                preds_root=None):
    """Notebook 06's join of photos, class ids, masks and front-end boxes for the given splits, never a test split.
    Unknown-pool species outside classes.csv get id -1; a photo without a usable mask has mask_path ""."""
    splits = [splits] if isinstance(splits, str) else list(splits)
    if set(splits) & set(TEST_SPLITS):
        raise ValueError(f"build_frame never reads the test splits {TEST_SPLITS}; asked for {splits}")
    frame = metadata[metadata["split"].isin(splits)].reset_index(drop=True)
    frame["class_id"] = frame["species"].map(class_index(classes)).fillna(NO_ID).astype(int)
    known = frame["class_id"].ne(NO_ID).to_numpy()
    genus_names, family_names, genus_of_class, family_of_class = taxonomy_index(classes)
    for level, names, of_class in (("genus", genus_names, genus_of_class), ("family", family_names, family_of_class)):
        by_name = frame[level].map({name: i for i, name in enumerate(names)}).fillna(NO_ID).to_numpy(dtype=int)
        frame[f"{level}_id"] = np.where(known, of_class[frame["class_id"].clip(lower=0)], by_name)
    frame["who_risk_id"] = frame["who_category"].astype(str).map(WHO_RISK_ID).astype(int)

    if mask_source == PREDICTED:
        paths = joined(frame, frontend_preds, "mask_file")["mask_file"]
        usable = paths.fillna("").astype(str).str.strip().ne("").to_numpy()
        root = preds_root
    elif mask_stats is not None:
        stats = joined(frame, mask_stats, "mask_path", "flagged")
        paths, usable = stats["mask_path"], (stats["mask_path"].notna() & stats["flagged"].eq(False)).to_numpy()
        root = None if masks_root is None else Path(masks_root, "masks")
    else:
        paths, usable, root = [""] * len(frame), np.zeros(len(frame), dtype=bool), None
    frame["mask_path"] = ["" if not ok else str(path) if root is None else str(Path(root, path))
                          for path, ok in zip(paths, usable)]
    frame["mask_valid"] = usable
    boxes = joined(frame, frontend_preds, BOX)[BOX] if frontend_preds is not None else [None] * len(frame)
    frame[BOX] = pd.Series([parse_box(text) for text in boxes], index=frame.index, dtype=object)
    frame = frame[FRAME_COLUMNS]
    assert_no_test_paths(frame)
    return frame


def with_files(frame, data_root):
    """The frame with the image_file and mask_file columns SnakeDataset reads."""
    return frame.assign(image_file=[str(Path(data_root, IMAGE_FOLDER, path)) for path in frame["path"]],
                        mask_file=frame["mask_path"].where(frame["mask_valid"], ""))


def load_rgb(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return fix_exposure(rgb)


def fix_exposure(rgb):
    if DARK_BELOW <= rgb.mean() <= BRIGHT_ABOVE:
        return rgb
    lightness, a, b = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB))
    return cv2.cvtColor(cv2.merge([CLAHE.apply(lightness), a, b]), cv2.COLOR_LAB2RGB)


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
    """The 0/1 mask closed and without its small separate parts, or None when what is left is under MIN_MASK_SHARE."""
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, CLOSE_KERNEL)
    parts, labels, part_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = part_stats[1:, cv2.CC_STAT_AREA]
    if parts <= 1 or areas.max() < MIN_MASK_SHARE * mask.size:
        return None
    keep = 1 + np.flatnonzero(areas >= max(MIN_PART_OF_LARGEST * areas.max(), MIN_PART_SHARE * mask.size))
    return np.isin(labels, keep).astype(np.uint8)


def prepare(rgb, mask, mode=MASK_MODE, size=IMG_SIZE):
    """The photo and mask in the input mode, letterboxed to size, as uint8 HxWx3 and HxW arrays."""
    target = np.zeros(rgb.shape[:2], np.uint8) if mask is None else mask
    if mode != "full" and mask is not None:
        if mode != "rgbm_4ch":
            keep = 0.0 if mode == "hard_mask" else BACKGROUND_KEEP
            weight = keep + (1 - keep) * cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)[..., None]
            rgb = (rgb * weight + np.array(GREY_RGB) * (1 - weight)).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        pad_x, pad_y = int(np.ptp(xs) * CROP_PADDING), int(np.ptp(ys) * CROP_PADDING)
        window = (slice(max(0, ys.min() - pad_y), ys.max() + pad_y + 1), slice(max(0, xs.min() - pad_x), xs.max() + pad_x + 1))
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
    """The mask's bounding box as inclusive pixel corners (x1, y1, x2, y2), or None for no mask or an empty one."""
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_to_box(rgb, mask, box, pad=CROP_PADDING):
    """rgb (and mask) cropped to box (x1, y1, x2, y2), padded by pad x its size and clipped to the photo; pixels
    floor(x1)..floor(x2) are kept, as `prepare` keeps them. A missing or degenerate box changes nothing."""
    if is_missing_box(box):
        return rgb, mask
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
        return rgb, mask
    height, width = rgb.shape[:2]
    pad_x, pad_y = int((x2 - x1) * pad), int((y2 - y1) * pad)
    left, top = max(0, math.floor(x1) - pad_x), max(0, math.floor(y1) - pad_y)
    right, bottom = min(width, math.floor(x2) + pad_x + 1), min(height, math.floor(y2) + pad_y + 1)
    if right <= left or bottom <= top:
        return rgb, mask
    window = (slice(top, bottom), slice(left, right))
    return rgb[window], None if mask is None else mask[window]


def apply_mode(rgb, mask, mode=MASK_MODE, box=None, size=IMG_SIZE):
    """One front-end input mode, letterboxed: box_crop (F1) crops to box, mask_box_crop (F1m) to the mask's box, the
    rest are `prepare`'s modes. A missing box or mask gives the whole photo."""
    if mode == BOX_CROP:
        rgb, mask = crop_to_box(rgb, mask, box)
    elif mode == "mask_box_crop":
        rgb, mask = crop_to_box(rgb, mask, mask_box(mask))
    return prepare(rgb, mask, "full" if mode in BOX_MODES else mode, size)


def build_augment(strength=DEFAULT_AUG_STRENGTH):
    """Notebook 06's tier and eval pipelines with the affine and colour ranges and the cutout and blur chances scaled
    by strength (1 is the notebook's). Each call makes new transforms, so strengths never share random state."""
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
            A.Affine(scale=around_one(0.88, 1.12), rotate=spread(25), translate_percent=spread(0.05),
                     border_mode=cv2.BORDER_CONSTANT, p=0.6),
            A.ColorJitter(brightness=0.2 * strength, contrast=0.2 * strength, saturation=0.2 * strength,
                          hue=0.04 * strength, p=0.5),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 36), hole_width_range=(16, 36), p=chance(0.35)),
            *camera_style,
            *to_tensor,
        ]),
        "tier2": A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.35),
            A.Affine(scale=around_one(0.80, 1.20), rotate=spread(45), translate_percent=spread(0.08), shear=spread(10),
                     border_mode=cv2.BORDER_CONSTANT, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.3 * strength, contrast_limit=0.3 * strength, p=0.7),
            A.HueSaturationValue(hue_shift_limit=8 * strength, sat_shift_limit=25 * strength,
                                 val_shift_limit=25 * strength, p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=chance(0.25)),
            A.CoarseDropout(num_holes_range=(2, 6), hole_height_range=(20, 44), hole_width_range=(20, 44), p=chance(0.5)),
            *camera_style,
            *to_tensor,
        ]),
        "eval": A.Compose(to_tensor),
    }


AUGMENT = build_augment(DEFAULT_AUG_STRENGTH)


class SnakeDataset(Dataset):
    """The image, the mask, whether that mask is real, and the four class targets of each frame row."""

    def __init__(self, frame, train=False, mode=MASK_MODE, size=IMG_SIZE, aug_strength=DEFAULT_AUG_STRENGTH):
        self.frame, self.train, self.mode, self.size = frame.reset_index(drop=True), train, mode, size
        self.augment = AUGMENT if aug_strength == DEFAULT_AUG_STRENGTH else build_augment(aug_strength)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        rgb = load_rgb(row["image_file"])
        mask = load_mask(row["mask_file"], rgb.shape[:2])
        image, target = apply_mode(rgb, mask, self.mode, row.get(BOX), self.size)
        sample = self.augment[row["tier"] if self.train else "eval"](image=image, mask=target)
        target = sample["mask"].unsqueeze(0).float()
        image = torch.cat([sample["image"], target]) if self.mode == "rgbm_4ch" else sample["image"]
        return {"image": image, "mask": target, "mask_valid": mask is not None, **{name: int(row[name]) for name in TARGETS}}


def cap_train(train):
    """Up to PHOTOGRAPHER_CAP photos per photographer per class and TRAIN_CAP per class, photographers in turn."""
    shuffled = train.sample(frac=1, random_state=SEED)
    turn = shuffled.groupby(["species", "observer_id"]).cumcount()
    kept = shuffled.assign(turn=turn)[turn < PHOTOGRAPHER_CAP].sort_values("turn", kind="stable")
    return kept.groupby("species").head(TRAIN_CAP).drop(columns="turn")


def photo_weights(frame, who_boost=WHO_BOOST):
    """Each photo's draw weight: 1/sqrt(class size), boosted for WHO-listed species."""
    class_size = frame.groupby("class_id")["class_id"].transform("size")
    return (1 / np.sqrt(class_size) * frame["who_risk_id"].map(who_boost)).to_numpy()


def make_sampler(frame, seed=SEED):
    """Draws each photo by photo_weights, as many draws as photos."""
    return WeightedRandomSampler(torch.tensor(photo_weights(frame), dtype=torch.double), len(frame), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def class_sampling_probs(frame, who_boost=WHO_BOOST):
    """Each class's share of make_sampler's draws, in class_id order; PKSampler takes it as class_probs, so both
    samplers draw the same class mix."""
    by_class = pd.Series(photo_weights(frame, who_boost), index=frame["class_id"].to_numpy()).groupby(level=0).sum()
    return (by_class / by_class.sum()).to_numpy()


class PKSampler(Sampler):
    """Batch sampler: p classes drawn without replacement by class_probs (else 1/sqrt(class size)), then k photos of
    each, repeating photos only in a class with fewer than k."""

    def __init__(self, labels, p=PK_CLASSES, k=PK_PHOTOS, seed=SEED, class_probs=None):
        labels = np.asarray(labels)
        self.members = [np.flatnonzero(labels == name) for name in np.unique(labels)]
        weights = (np.array([1 / np.sqrt(len(members)) for members in self.members]) if class_probs is None
                   else np.asarray(class_probs, dtype=float))
        self.class_probability = weights / weights.sum()
        self.p, self.k = p, k
        self.batches = max(1, len(labels) // (p * k))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            picked = self.rng.choice(len(self.members), self.p, replace=False, p=self.class_probability)
            yield [int(index) for name in picked
                   for index in self.rng.choice(self.members[name], self.k, replace=len(self.members[name]) < self.k)]
