"""S1 and D1 front ends: YOLO training sets, training the segmenters and the fine-tuned detector, scoring them and
picking the winners, and writing frontend_preds.csv and masks_pred/ for the front-end runs."""
import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from serpentguard.data import (ID_TYPES, MASK_MODE, MIN_BOX_SIDE, SEED, UNKNOWN_POOL, build_frame, clean_mask,
                               load_mask, load_rgb, prepare, with_files)
from serpentguard.frontend_scores import (HOLDOUT_PHOTOS, HUMAN_LABELS, PSEUDO_LABELS, WINNER_TIE, YOLO_CONF,
                                          YOLO_WEIGHTS, as_list, holdout_frame, human_target, oi_mask_items,
                                          open_images_ground_truth, orientation_mismatches, pick_winner, pseudo_target,
                                          score_detector, score_segmenter, yolo_snake_boxes)
from serpentguard.run_data import CLASSES_FILE, FRONTEND_PREDS_FILE, MASK_STATS_FILE, METADATA_FILE
from serpentguard.run_outputs import subset_hash, write_json
from serpentguard.segmenters import (MASK_THRESHOLD, RUN_FILE, SEG_AUG_STRENGTH, SEG_BATCH, SEG_EPOCHS, SEG_INPUT_SIZE,
                                     SEG_KINDS, SEGMENTER_FILE, YOLO_SEG, YOLO_SEG_CONF, YOLO_SEG_WEIGHTS,
                                     load_segmenter, parameter_count, predict_masks, read_run, train_torch_segmenter)
from serpentguard.subset import calibration_set, unknown_split
from serpentguard.train import FINAL_FILE

S1, D1 = "S1", "D1"
OFF_THE_SHELF, FINE_TUNED = "yolov8n-oiv7", "yolo26n-ft"
DET_WEIGHTS, DET_EPOCHS = "yolo26n.pt", 20
DETECTOR_FILE, WINNERS_FILE, SUMMARY_FILE = "detector.pt", "winners.json", "summary.json"
TRAIN_CHECK_FILE = "train_check.json"
DEFAULT_EXPORT_ROOT = Path(tempfile.gettempdir()) / "serpentguard_yolo_exports"  # derived photos stay out of outputs
PREDS_FILE, MASKS_PRED = FRONTEND_PREDS_FILE, "masks_pred"
BOX_FLOOR = 0.05   # boxes this confident are written with their box_conf; the app sets its own "no snake" threshold
PREDICT_BATCH = 32
YOLO_TRAIN_BATCH = 32
YOLO_VAL_PHOTOS = 64          # training photos copied to "val": YOLO validates after the last epoch whatever we ask
MIN_POLYGON_SHARE = 0.002     # a contour enclosing less of the photo than this is a speck, not a usable polygon
DATA_YAML = "data.yaml"
# notebook 06's tier-1 pipeline: flips, then affine and colour ranges scaled by the augmentation strength
TIER1_FLIPLR, TIER1_FLIPUD = 0.5, 0.2
TIER1_DEGREES, TIER1_TRANSLATE, TIER1_SCALE = 25.0, 0.05, 0.12
TIER1_HUE, TIER1_SATURATION, TIER1_VALUE = 0.04, 0.2, 0.2


# Ultralytics training sets: YOLO26n-seg reads letterboxed photos with SAM masks as polygons, YOLO26n photos shrunk to
# the input size with the Grounding DINO boxes

def yolo_box_label(box, width, height):
    """(cx, cy, w, h) as fractions of the photo for a box clipped to it; None for a non-finite or too small box."""
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
    y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
    if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
        return None
    return (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height


def _label_line(values):
    return " ".join(["0", *(f"{value:.6f}" for value in values)])


def box_label_lines(boxes, width, height):
    labels = [yolo_box_label(box, width, height) for box in boxes]
    return [_label_line(label) for label in labels if label is not None]


def mask_polygons(mask):
    """The outer contours of a 0/1 mask as (n, 2) fractions of the photo, specks dropped; YOLO polygons hold no
    holes."""
    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour.reshape(-1, 2) / np.array([width, height], dtype=float) for contour in contours
            if len(contour) >= 3 and cv2.contourArea(contour) >= MIN_POLYGON_SHARE * mask.size]


def polygon_label_lines(mask):
    return [_label_line(polygon.ravel()) for polygon in mask_polygons(mask)]


class YoloExport(Dataset):
    """Writes one photo and its label file per item under root/images/train and root/labels/train."""

    def __init__(self, frame, root, size, boxes):
        self.rows = frame[["path", "image_file", "snake_boxes" if boxes else "mask_file"]].to_dict("records")
        self.root, self.size, self.boxes = Path(root), size, boxes

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rgb = load_rgb(row["image_file"])
        height, width = rgb.shape[:2]
        if self.boxes:
            lines = box_label_lines(row["snake_boxes"], width, height)
            scale = self.size / max(height, width)
            image = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            result = {"dropped": len(row["snake_boxes"]) - len(lines)}
        else:
            mask = load_mask(row["mask_file"], rgb.shape[:2])
            image, canvas_mask = prepare(rgb, mask, MASK_MODE, self.size) if mask is not None else (None, None)
            lines = [] if mask is None else polygon_label_lines(canvas_mask)
            result = {}
        if not lines:
            return {"written": 0, "labels": 0, **result}
        stem = Path(row["path"]).with_suffix("").as_posix().replace("/", "__")  # classes never clash
        Image.fromarray(image).save(self.root / "images" / "train" / f"{stem}.jpg", quality=95)
        (self.root / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        return {"written": 1, "labels": len(lines), **result}


def _write_yolo_set(frame, root, size, boxes, val_photos=YOLO_VAL_PHOTOS, num_workers=0):
    """The YOLO set under root (emptied first); "val" copies the first training photos, as nothing is selected on it."""
    root = Path(root)
    shutil.rmtree(root, ignore_errors=True)
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
    results = list(DataLoader(YoloExport(frame, root, size, boxes), batch_size=None, num_workers=num_workers))
    for image in sorted((root / "images" / "train").glob("*.jpg"))[:val_photos]:
        shutil.copy(image, root / "images" / "val" / image.name)
        shutil.copy(root / "labels" / "train" / f"{image.stem}.txt", root / "labels" / "val" / f"{image.stem}.txt")
    (root / DATA_YAML).write_text(yaml.safe_dump({"path": str(root.resolve()), "train": "images/train",
                                                  "val": "images/val", "names": {0: "snake"}}))
    written = sum(result["written"] for result in results)
    return {"photos": written, f"skipped_no_{'box' if boxes else 'mask'}": len(results) - written,
            "boxes" if boxes else "polygons": sum(result["labels"] for result in results),
            **({"dropped_boxes": sum(result["dropped"] for result in results)} if boxes else {}),
            "val_photos": min(val_photos, written), "input_size": size}


def write_seg_dataset(frame, root, size, val_photos=YOLO_VAL_PHOTOS, num_workers=0):
    return _write_yolo_set(frame, root, size, False, val_photos, num_workers)


def write_det_dataset(frame, root, size, val_photos=YOLO_VAL_PHOTOS, num_workers=0):
    return _write_yolo_set(frame, root, size, True, val_photos, num_workers)


def yolo_train_args(data_yaml, out_dir, epochs, imgsz, batch, aug_strength, device=None, num_workers=0, seed=SEED):
    """Every argument the YOLO runs train with: notebook 06's tier-1 augmentation scaled by aug_strength, no mosaic,
    and nbs = batch, so each batch is one optimizer step as for the torch segmenters."""
    return {"data": str(data_yaml), "epochs": epochs, "imgsz": imgsz, "batch": batch, "nbs": batch,
            "optimizer": "AdamW", "lr0": 1e-3, "lrf": 0.0, "cos_lr": True, "weight_decay": 1e-4, "warmup_epochs": 1,
            "seed": seed, "deterministic": True, "workers": num_workers, "device": device,
            "project": str(Path(out_dir).resolve()), "name": "yolo", "exist_ok": True, "plots": False,
            "pretrained": True, "fliplr": TIER1_FLIPLR, "flipud": TIER1_FLIPUD, "degrees": TIER1_DEGREES * aug_strength,
            "translate": TIER1_TRANSLATE * aug_strength, "scale": TIER1_SCALE * aug_strength,
            "hsv_h": TIER1_HUE * aug_strength, "hsv_s": TIER1_SATURATION * aug_strength,
            "hsv_v": TIER1_VALUE * aug_strength, "shear": 0.0, "perspective": 0.0, "mosaic": 0.0, "close_mosaic": 0,
            "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0}


def train_yolo(weights, args):
    """Fine-tunes Ultralytics `weights`; returns the last epoch's weights and the trained model's parameter count."""
    from ultralytics import YOLO

    YOLO(weights).train(**args)
    last = Path(args["project"]) / "yolo" / "weights" / "last.pt"
    return last, parameter_count(YOLO(str(last)).model)


# Training

def read_inputs(data_root, masks_root, subset_csv):
    """metadata, classes, subset and (given masks_root) mask_stats, as the runner reads them."""
    data_root = Path(data_root)
    return {"metadata": pd.read_csv(data_root / METADATA_FILE, dtype=ID_TYPES, low_memory=False),
            "classes": pd.read_csv(data_root / CLASSES_FILE), "subset": pd.read_csv(subset_csv, dtype=ID_TYPES),
            "mask_stats": None if masks_root is None else pd.read_csv(Path(masks_root) / MASK_STATS_FILE)}


def subset_photos(metadata, subset):
    """The metadata rows of the subset's photos, which must all be training photos."""
    train = metadata[metadata["split"] == "train"]
    if not subset["path"].isin(train["path"]).all():
        raise ValueError("every subset photo must be a training photo")
    return train[train["path"].isin(subset["path"])]


def segmenter_frame(metadata, classes, subset, mask_stats, data_root, masks_root):
    """S1's training photos: every subset photo with a valid SAM mask, with image_file and mask_file."""
    frame = build_frame(subset_photos(metadata, subset), classes, ["train"], mask_stats, masks_root=masks_root)
    return with_files(frame[frame["mask_valid"]], data_root).reset_index(drop=True)


def detector_frame(metadata, subset, mask_stats, data_root):
    """D1's training photos: every subset photo with a Grounding DINO box, with image_file and the parsed boxes."""
    rows = subset_photos(metadata, subset)[["path"]].merge(mask_stats[["path", "snake_boxes"]], on="path", how="left",
                                                           validate="one_to_one")
    rows["snake_boxes"] = [json.loads(text) if isinstance(text, str) and text.strip() else []
                           for text in rows["snake_boxes"]]
    rows = rows[rows["snake_boxes"].map(len) > 0].reset_index(drop=True)
    return rows.assign(image_file=[str(Path(data_root, "data", path)) for path in rows["path"]])


def prediction_frames(metadata, classes, subset, data_root):
    """Every photo a front-end run reads: the subset uncapped, full validation, the calibration set and unknown-pool
    half A."""
    train = metadata[metadata["split"] == "train"]
    unknown_a, _ = unknown_split(metadata[metadata["split"] == UNKNOWN_POOL])
    frames = {"train": build_frame(subset_photos(metadata, subset), classes, ["train"]),
              "val": build_frame(metadata, classes, ["val"]),
              "calibration": build_frame(calibration_set(train, subset), classes, ["train"]),
              "unknown": build_frame(unknown_a, classes, [UNKNOWN_POOL])}
    return {name: with_files(frame, data_root) for name, frame in frames.items()}


def train_segmenter(kind, frame, out_dir, epochs=SEG_EPOCHS, batch_size=SEG_BATCH, size=SEG_INPUT_SIZE,
                    device=None, num_workers=0, pretrained=True, seed=SEED, record=None,
                    export_root=DEFAULT_EXPORT_ROOT):
    """Trains one S1 segmenter on the frame's SAM masks, writing segmenter.pt and run.json; returns the weights' path.
    YOLO26n-seg's training set goes to export_root/<kind>, so no derived photo lands in out_dir."""
    if kind != YOLO_SEG:
        return train_torch_segmenter(kind, frame, out_dir, epochs=epochs, batch_size=batch_size, size=size,
                                     device=device, num_workers=num_workers, pretrained=pretrained, seed=seed,
                                     record=record)
    return _train_yolo_model(kind, YOLO_SEG_WEIGHTS, SEGMENTER_FILE, frame, out_dir, epochs, batch_size, size, device,
                             num_workers, seed, record, export_root, {"kind": kind, "yolo_seg_conf": YOLO_SEG_CONF})


def train_detector(frame, out_dir, epochs=DET_EPOCHS, batch_size=YOLO_TRAIN_BATCH, size=SEG_INPUT_SIZE, device=None,
                   num_workers=0, seed=SEED, record=None, export_root=DEFAULT_EXPORT_ROOT):
    """D1: YOLO26n fine-tuned on the frame's Grounding DINO boxes, the last epoch kept, writing detector.pt and
    run.json; returns the weights' path."""
    return _train_yolo_model(FINE_TUNED, DET_WEIGHTS, DETECTOR_FILE, frame, out_dir, epochs, batch_size, size, device,
                             num_workers, seed, record, export_root, {"candidate": FINE_TUNED})


def _train_yolo_model(name, start_weights, weights_file, frame, out_dir, epochs, batch_size, size, device, num_workers,
                      seed, record, export_root, fields):
    started, out_dir, export_dir = time.perf_counter(), Path(out_dir), Path(export_root) / name
    export = _write_yolo_set(frame, export_dir, size, name == FINE_TUNED, num_workers=num_workers)
    args = yolo_train_args(export_dir / DATA_YAML, out_dir, epochs, size, batch_size, SEG_AUG_STRENGTH, device,
                           num_workers, seed)
    last, parameters = train_yolo(start_weights, args)
    weights = out_dir / weights_file
    shutil.copy(last, weights)
    write_json(out_dir / RUN_FILE, {
        **(record or {}), **fields, "weights": weights_file, "parameters": parameters, "train_photos": export["photos"],
        "export": export, "export_dir": str(export_dir), "epochs": epochs, "input_size": size,
        "seconds": time.perf_counter() - started})
    return weights


# Predictions for the front-end runs

def predict_boxes(detector, images, conf=BOX_FLOOR):
    """Per RGB photo, the detector's best snake box at conf or above as ((x1, y1, x2, y2) in pixels, confidence), or
    None. The whole photo goes in; the detector letterboxes it itself."""
    found = yolo_snake_boxes(detector, [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in images], conf=conf)
    best = [max(rows, key=lambda row: row[4]) if rows else None for rows in found]
    return [None if row is None else (tuple(float(value) for value in row[:4]), float(row[4])) for row in best]


class _Photos(Dataset):
    def __init__(self, image_files):
        self.image_files = list(image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return index, load_rgb(self.image_files[index])


def write_frontend_preds(frames, segmenter, detector, out_root, batch_size=PREDICT_BATCH, num_workers=0):
    """Writes frontend_preds.csv (path, box JSON, mask_file, box_conf; one row per photo of the frames) and the masks
    cleaning keeps; returns per frame the share with a box (at BOX_FLOOR and YOLO_CONF) and a mask."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    photos = pd.concat([frame[["path", "image_file"]].assign(frame=name) for name, frame in frames.items()])
    photos = photos.drop_duplicates("path").reset_index(drop=True)
    loader = DataLoader(_Photos(photos["image_file"]), batch_size=batch_size, num_workers=num_workers,
                        collate_fn=as_list)
    boxes, box_confs, mask_files = [""] * len(photos), [""] * len(photos), [""] * len(photos)
    confidences = np.full(len(photos), np.nan)
    for batch in loader:
        indices, images = zip(*batch)
        for index, mask, found in zip(indices, predict_masks(segmenter, list(images)), predict_boxes(detector, images)):
            if found is not None:
                box, confidences[index] = found
                boxes[index] = json.dumps([round(value, 2) for value in box])
                box_confs[index] = f"{confidences[index]:.4f}"
            if clean_mask(mask) is not None:
                relative = Path(MASKS_PRED, photos.at[index, "path"]).with_suffix(".png")
                (out_root / relative).parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(out_root / relative), mask.astype(np.uint8) * 255):
                    raise OSError(f"could not write {out_root / relative}")
                mask_files[index] = relative.as_posix()
    preds = pd.DataFrame({"path": photos["path"], "box": boxes, "mask_file": mask_files, "box_conf": box_confs})
    preds.to_csv(out_root / PREDS_FILE, index=False)
    found = photos.assign(box=preds["box"].ne(""), box_at_conf=confidences >= YOLO_CONF, mask=preds["mask_file"].ne(""))
    return {"photos": len(preds), "box_floor": BOX_FLOOR, "box_conf_threshold": YOLO_CONF, "coverage": {
        name: {"photos": len(rows), **{column: float(rows[column].mean()) for column in ("box", "box_at_conf", "mask")}}
        for name, rows in found.groupby("frame", sort=False)}}


def score_frontends(frontends, tables, data_root, masks_root, extended_root, device=None, num_workers=0, limit=None):
    """Scores every S1 candidate (Open Images human masks and a SAM-mask holdout) and both D1 detectors (Open Images
    boxes) into <candidate>/final_metrics.json, then picks the winners."""
    frontends, extended_root = Path(frontends), Path(extended_root)
    oi_boxes = pd.read_csv(extended_root / "open_images_snake_boxes.csv")
    oi_masks = pd.read_csv(extended_root / "open_images_snake_masks.csv")
    items, tagged = oi_mask_items(oi_masks, extended_root)
    metadata, subset = tables["metadata"], tables["subset"]
    calibration = with_files(build_frame(calibration_set(metadata[metadata["split"] == "train"], subset),
                                         tables["classes"], ["train"], tables["mask_stats"], masks_root=masks_root),
                             data_root)
    holdout, skipped = holdout_frame(calibration, HOLDOUT_PHOTOS, SEED)
    if limit is not None:
        items, holdout = items.head(limit), holdout.head(limit)
    mismatched = orientation_mismatches(items)
    if mismatched:
        print(f"Open Images photos whose masks fit the photo turned, not as load_rgb reads it: {mismatched}")
    oi_record = {"rotation_tagged": tagged, "orientation_mismatch": mismatched}
    common = {"subset_hash": subset_hash(subset["path"])}

    for folder in sorted((frontends / S1).glob(f"*/{RUN_FILE}")):
        started, run = time.perf_counter(), read_run(folder.parent)
        segmenter = load_segmenter(run["kind"], folder.parent / run["weights"], device)
        human = score_segmenter(segmenter, items["image_file"], items["mask_files"], human_target,
                                num_workers=num_workers)
        pseudo = score_segmenter(segmenter, holdout["image_file"], [[file] for file in holdout["mask_file"]],
                                 pseudo_target, num_workers=num_workers)
        write_json(folder.parent / FINAL_FILE, {
            "open_images": {**human, "labels": HUMAN_LABELS, **oi_record, "image_ids": items["image_id"].tolist()},
            "mask_holdout": {**pseudo, "labels": PSEUDO_LABELS, "source": "calibration set, valid SAM masks",
                             "skipped_too_small": skipped, "paths": holdout["path"].tolist()},
            "parameters": run["parameters"],
            "run": {**common, "candidate": run["kind"], "weights": str(Path(S1, folder.parent.name, run["weights"])),
                    "input_size": segmenter.size, "threshold": MASK_THRESHOLD, "yolo_seg_conf": YOLO_SEG_CONF,
                    "limit": limit, "seconds": time.perf_counter() - started}})

    images, truth = open_images_ground_truth(oi_boxes, oi_masks)
    files = [str(extended_root / file) for file in images["image_file"]]
    if limit is not None:
        files, truth = files[:limit], truth[:limit]
    detectors = {OFF_THE_SHELF: {"model": YOLO_WEIGHTS}}
    if (frontends / D1 / FINE_TUNED / DETECTOR_FILE).exists():
        detectors[FINE_TUNED] = {"weights": str(Path(D1, FINE_TUNED, DETECTOR_FILE))}
    for name, source in detectors.items():
        started, detector = time.perf_counter(), load_detector(frontends, source)
        scores = score_detector(detector, files, truth)
        (frontends / D1 / name).mkdir(parents=True, exist_ok=True)  # the off-the-shelf detector has no training folder
        write_json(frontends / D1 / name / FINAL_FILE, {
            "detection": {**scores, **oi_record}, "parameters": parameter_count(detector.model),
            "run": {**common, "candidate": name, **source, "imgsz": detector.overrides.get("imgsz"),
                    "evaluation": "open_images_snake_boxes, every photo, read as load_rgb reads them",
                    "limit": limit, "seconds": time.perf_counter() - started}})
    return pick_winners(frontends)


def train_photo_check(segmenter, tables, data_root, masks_root, n=HOLDOUT_PHOTOS, seed=SEED, num_workers=0,
                      limit=None):
    """The segmenter scored on up to n seeded photos of its own training set: the train/holdout gap in mask quality
    the F-runs inherit."""
    frame = segmenter_frame(tables["metadata"], tables["classes"], tables["subset"], tables["mask_stats"], data_root,
                            masks_root)
    photos, skipped = holdout_frame(frame, n, seed)
    photos = photos if limit is None else photos.head(limit)
    scores = score_segmenter(segmenter, photos["image_file"], [[file] for file in photos["mask_file"]], pseudo_target,
                             num_workers=num_workers)
    return {**scores, "labels": PSEUDO_LABELS, "source": "the segmenter's own training photos (subset, valid SAM masks)",
            "seed": seed, "skipped_too_small": skipped, "paths": photos["path"].tolist()}


def load_detector(frontends, source):
    """An Ultralytics model from {"weights": path under frontends} or {"model": name}."""
    from ultralytics import YOLO

    return YOLO(str(Path(frontends) / source["weights"]) if "weights" in source else source["model"])


def pick_winners(frontends):
    """winners.json: S1 by Open Images IoU of the cleaned masks the F-runs read (within WINNER_TIE the smaller model),
    D1 by Open Images mAP@0.5."""
    frontends = Path(frontends)
    finals = {stage: {path.parent.name: json.loads(path.read_text())
                      for path in sorted((frontends / stage).glob(f"*/{FINAL_FILE}"))} for stage in (S1, D1)}
    seg = {name: (final["open_images"]["iou_cleaned"], final["parameters"]) for name, final in finals[S1].items()}
    det = {name: (final["detection"]["map50"], final["parameters"]) for name, final in finals[D1].items()}
    segmenter, detector = pick_winner(seg), pick_winner(det, tie=0.0)
    det_run = finals[D1][detector]["run"]
    return write_json(frontends / WINNERS_FILE, {
        S1: {"candidate": segmenter, "kind": segmenter, "weights": finals[S1][segmenter]["run"]["weights"],
             "rule": f"best Open Images IoU of the cleaned masks; within {WINNER_TIE} the fewest parameters",
             "scores": seg, "raw_iou": {name: final["open_images"]["iou"] for name, final in finals[S1].items()}},
        D1: {"candidate": detector, **{key: det_run[key] for key in ("weights", "model") if key in det_run},
             "rule": "best Open Images mAP@0.5", "scores": det}})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m serpentguard.frontend")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train-seg", "train-det", "score", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--masks-root", type=Path, required=name != "predict")
        command.add_argument("--subset", type=Path, required=True)
        command.add_argument("--device")
        command.add_argument("--num-workers", type=int, default=4)
        command.add_argument("--limit", type=int, help="first N photos of each set, for smoke runs")
        if name in ("score", "predict"):
            command.add_argument("--frontends", type=Path, required=True)
        if name != "score":
            command.add_argument("--out", type=Path, required=True)
        if name.startswith("train"):
            command.add_argument("--epochs", type=int, default=SEG_EPOCHS if name == "train-seg" else DET_EPOCHS)
            command.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    commands.choices["train-seg"].add_argument("--kind", choices=SEG_KINDS, required=True)
    commands.choices["score"].add_argument("--extended-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tables = read_inputs(args.data_root, args.masks_root, args.subset)
    record = {"subset_hash": subset_hash(tables["subset"]["path"]), "subset": str(args.subset), "limit": args.limit}
    first = (lambda frame: frame) if args.limit is None else (lambda frame: frame.head(args.limit))
    if args.command == "train-seg":
        frame = first(segmenter_frame(tables["metadata"], tables["classes"], tables["subset"], tables["mask_stats"],
                                      args.data_root, args.masks_root))
        print(train_segmenter(args.kind, frame, args.out, epochs=args.epochs, device=args.device,
                              num_workers=args.num_workers, record=record, export_root=args.export_root))
    elif args.command == "train-det":
        frame = first(detector_frame(tables["metadata"], tables["subset"], tables["mask_stats"], args.data_root))
        print(train_detector(frame, args.out, epochs=args.epochs, device=args.device, num_workers=args.num_workers,
                             record=record, export_root=args.export_root))
    elif args.command == "score":
        winners = score_frontends(args.frontends, tables, args.data_root, args.masks_root, args.extended_root,
                                  args.device, args.num_workers, args.limit)
        print(json.dumps(winners, indent=1))
    else:
        winners = json.loads((args.frontends / WINNERS_FILE).read_text())
        frames = {name: first(frame) for name, frame in prediction_frames(
            tables["metadata"], tables["classes"], tables["subset"], args.data_root).items()}
        segmenter = load_segmenter(winners[S1]["kind"], args.frontends / winners[S1]["weights"], args.device)
        detector = load_detector(args.frontends, winners[D1])
        args.out.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        summary = write_frontend_preds(frames, segmenter, detector, args.out, num_workers=args.num_workers)
        check = None
        if args.masks_root is None:
            print("no --masks-root: the train-photo mask check is not run")
        else:
            check = write_json(args.frontends / S1 / winners[S1]["candidate"] / TRAIN_CHECK_FILE, train_photo_check(
                segmenter, tables, args.data_root, args.masks_root, num_workers=args.num_workers, limit=args.limit))
        summary = write_json(args.out / SUMMARY_FILE, {**summary, "winners": winners, **record,
                                                       "segmenter_threshold": MASK_THRESHOLD,
                                                       "yolo_seg_conf": YOLO_SEG_CONF,
                                                       "train_check": check, "seconds": time.perf_counter() - started})
        print(json.dumps(summary["coverage"], indent=1))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
