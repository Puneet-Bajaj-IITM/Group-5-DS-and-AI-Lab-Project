"""S1 and D1 scores against the Open Images masks and boxes and a SAM-mask holdout, the rule that picks a winner, and
the YOLO snake-box finder that B5, D1 and the app share. Rotation-tagged Open Images photos are scored as stored, like
their masks."""
import math

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from serpentguard.data import SEED, clean_mask, load_mask, load_rgb
from serpentguard.metrics import det_scores, seg_scores
from serpentguard.segmenters import (YOLO_SEG, YOLO_SEG_CONF, YOLO_TOP_CONF_FLOOR, predict_masks,
                                     top_instance_confidences)

HOLDOUT_PHOTOS = 1000
WINNER_TIE = 0.005            # S1: IoUs this close are a tie, and the smaller model wins
HUMAN_LABELS = "human: Open Images V7 snake masks"
PSEUDO_LABELS = "pseudo: Grounding DINO + SAM 2.1 masks (notebook 04), cleaned as load_mask cleans them"
SCORE_BATCH = 16
DETECT_CHUNK = 32             # photos read at once for a detector; it batches them itself
YOLO_WEIGHTS = "yolov8n-oiv7.pt"
YOLO_CONF = 0.25              # precision and recall
YOLO_ALL_CONF = 0.001         # keeps every prediction, for mAP@0.5
YOLO_BATCH = 16
SQUARE_LETTERBOX = False      # Ultralytics rect: False pads every call alike, as the training boxes were


def as_list(batch):
    return batch


def snake_class_index(names):
    """The index of the one class called Snake in an Ultralytics model's names."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    found = [int(index) for index, name in items if str(name).strip().lower() == "snake"]
    if len(found) != 1:
        raise ValueError(f"model.names must hold exactly one 'Snake' class; found {len(found)}")
    return found[0]


def yolo_snake_boxes(model, images, conf=YOLO_ALL_CONF, batch_size=YOLO_BATCH, normalized=False):
    """Per image, its Snake boxes as [x1, y1, x2, y2, score], in pixels or (normalized) fractions. rect=False pads
    every call to one square, as the training boxes were, where a lone photo drifts."""
    snake = snake_class_index(model.names)
    images, boxes = list(images), []
    for start in range(0, len(images), batch_size):
        for result in model.predict(images[start:start + batch_size], conf=conf, verbose=False, classes=[snake],
                                    rect=SQUARE_LETTERBOX):
            found = result.boxes
            keep = _numpy(found.cls).astype(int) == snake
            corners = _numpy(found.xyxyn if normalized else found.xyxy)[keep]
            boxes.append(np.column_stack([corners, _numpy(found.conf)[keep]]).reshape(-1, 5).tolist())
    return boxes


def _numpy(values):
    return values.cpu().numpy() if hasattr(values, "cpu") else np.asarray(values)


def open_images_ground_truth(boxes, masks):
    """(one masks-table row per Open Images photo sorted by id, its snake boxes as normalised [x1, y1, x2, y2])."""
    images = masks.drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)
    missing = sorted(set(boxes["ImageID"]) - set(images["image_id"]))
    if missing:
        raise ValueError(f"{len(missing)} boxed photos have no row in the masks table, e.g. {missing[0]}")
    by_image = {image_id: rows[["XMin", "YMin", "XMax", "YMax"]].to_numpy(dtype=float)
                for image_id, rows in boxes.groupby("ImageID")}
    return images, [by_image.get(image_id, np.zeros((0, 4))) for image_id in images["image_id"]]


def detection_baseline_scores(pred_boxes, gt_boxes, conf=YOLO_CONF):
    """Precision and recall from predictions scoring at least conf; mAP@0.5 from every prediction."""
    confident = [[box for box in boxes if box[4] >= conf] for boxes in pred_boxes]
    at_conf, everything = det_scores(confident, gt_boxes), det_scores(pred_boxes, gt_boxes)
    return {"precision": at_conf["precision"], "recall": at_conf["recall"], "map50": everything["map50"],
            "conf_threshold": conf, "images": len(gt_boxes), "gt_boxes": int(sum(len(boxes) for boxes in gt_boxes)),
            "predictions": sum(map(len, pred_boxes)), "predictions_at_conf": sum(map(len, confident))}


def oi_mask_items(table, root):
    """(one row per Open Images photo sorted by id, with its image_file and mask_files under root; how many photos
    Open Images tags with a rotation). Rotation-tagged photos are scored as stored, like their masks."""
    items = (table.sort_values(["image_id", "mask_file"]).groupby("image_id", sort=True)
             .agg(image_file=("image_file", "first"), mask_files=("mask_file", list)).reset_index())
    items["image_file"] = [str(root / file) for file in items["image_file"]]
    items["mask_files"] = [[str(root / file) for file in files] for files in items["mask_files"]]
    return items, int(table.loc[table["rotation"].fillna(0).ne(0), "image_id"].nunique())


def fits_transposed(photo_size, mask_size):
    """True when a mask's aspect ratio is nearer the photo turned a quarter than as stored; never for a square."""
    (width, height), (mask_width, mask_height) = photo_size, mask_size
    if width == height:
        return False
    aspect = mask_width / mask_height
    return abs(aspect - height / width) < abs(aspect - width / height)


def orientation_mismatches(items):
    """Ids of the photos with a mask that fits the photo turned rather than as load_rgb reads it."""
    found = []
    for image_id, image_file, mask_files in zip(items["image_id"], items["image_file"], items["mask_files"]):
        with Image.open(image_file) as image:
            photo_size = image.size
        for mask_file in mask_files:
            with Image.open(mask_file) as mask:
                if fits_transposed(photo_size, mask.size):
                    found.append(image_id)
                    break
    return found


def oi_target(masks, height, width):
    """One 0/1 uint8 target from all a photo's 0/255 masks, each resized to the photo, joined."""
    target = np.zeros((height, width), np.uint8)
    for mask in masks:
        resized = cv2.resize(np.asarray(mask, dtype=np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        target |= (resized > 127).astype(np.uint8)
    return target


def human_target(files, shape):
    masks = []
    for file in files:
        with Image.open(file) as image:
            masks.append(np.asarray(image.convert("L")))
    return oi_target(masks, *shape)


def pseudo_target(files, shape):
    """The cleaned SAM mask, which is what the segmenters were trained to give."""
    return load_mask(files[0], shape)


class ScoredPhotos(Dataset):
    """(photo, target) pairs; target_of(mask files, photo shape) builds each target."""

    def __init__(self, image_files, mask_files, target_of):
        self.image_files, self.mask_files, self.target_of = list(image_files), list(mask_files), target_of

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        rgb = load_rgb(self.image_files[index])
        return rgb, self.target_of(self.mask_files[index], rgb.shape[:2])


def holdout_frame(frame, n=HOLDOUT_PHOTOS, seed=SEED):
    """(up to n seeded rows whose valid SAM mask survives load_mask's cleaning, how many cleaning emptied)."""
    candidates = frame[frame["mask_valid"]].reset_index(drop=True)
    chosen, skipped = [], 0
    for index in np.random.default_rng(seed).permutation(len(candidates)):
        if len(chosen) == n:
            break
        file = candidates.at[index, "mask_file"]
        with Image.open(file) as image:
            shape = image.size[::-1]
        if load_mask(file, shape) is None:
            skipped += 1
        else:
            chosen.append(index)
    return candidates.iloc[chosen].reset_index(drop=True), skipped


def score_segmenter(segmenter, image_files, mask_files, target_of, batch_size=SCORE_BATCH, num_workers=0):
    """Mean and per-photo IoU and Dice against the targets, raw and cleaned as the F-runs read them (None is empty);
    for YOLO26n-seg also the top instance confidences, which tell an empty mask from a low confidence."""
    loader = DataLoader(ScoredPhotos(image_files, mask_files, target_of), batch_size=batch_size,
                        num_workers=num_workers, collate_fn=as_list)
    per_image = {key: [] for key in ("iou", "dice", "iou_cleaned", "dice_cleaned")}
    tops = []
    for batch in loader:
        photos, targets = zip(*batch)
        if segmenter.kind == YOLO_SEG:
            tops += top_instance_confidences(segmenter, list(photos)).tolist()
        for mask, target in zip(predict_masks(segmenter, list(photos)), targets):
            cleaned = clean_mask(mask)
            for suffix, prediction in (("", mask), ("_cleaned", np.zeros_like(mask) if cleaned is None else cleaned)):
                scores = seg_scores(prediction, target)
                per_image[f"iou{suffix}"].append(scores["iou"])
                per_image[f"dice{suffix}"].append(scores["dice"])
    scores = {**{key: float(np.mean(values)) for key, values in per_image.items()}, "images": len(per_image["iou"]),
              **{f"per_image_{key}": values for key, values in per_image.items()}}
    if segmenter.kind == YOLO_SEG:
        tops = np.array(tops)
        scores["yolo_instances"] = {
            "conf": YOLO_SEG_CONF, "photos": len(tops), "photos_with_instance": int((tops >= YOLO_SEG_CONF).sum()),
            **{name: float(np.quantile(tops, q)) for name, q in (("top_conf_p50", 0.5), ("top_conf_p90", 0.9),
                                                                 ("top_conf_max", 1.0))},
            "floor": YOLO_TOP_CONF_FLOOR, "per_image_top_conf": tops.tolist()}
    return scores


def score_detector(detector, image_files, truth, conf=YOLO_CONF, chunk=DETECT_CHUNK):
    """Precision and recall at conf, and mAP@0.5 over every prediction, against normalised truth boxes."""
    predictions = []
    for start in range(0, len(image_files), chunk):
        photos = [cv2.cvtColor(load_rgb(file), cv2.COLOR_RGB2BGR) for file in image_files[start:start + chunk]]
        predictions += yolo_snake_boxes(detector, photos, conf=YOLO_ALL_CONF, normalized=True)
    return detection_baseline_scores(predictions, truth, conf)


def pick_winner(candidates, tie=WINNER_TIE):
    """The winner of {name: (score, parameters)}: of those within `tie` of the best, the fewest parameters, then the
    higher score, then the name. A nan score is out."""
    scored = {name: value for name, value in candidates.items() if not math.isnan(value[0])}
    if not scored:
        raise ValueError(f"no candidate has a score: {candidates}")
    best = max(score for score, _ in scored.values())
    tied = sorted(name for name, (score, _) in scored.items() if best - score <= tie + 1e-9)  # float rounding
    return min(tied, key=lambda name: (scored[name][1], -scored[name][0]))
