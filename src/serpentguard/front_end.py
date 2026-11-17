"""The front end: the segmenters, the detector, the helpers that train, run and score them, and the no-snake gate.

Lifted from notebook 13 (cropping) so that later notebooks can run the trained models without retraining them. The
function bodies are notebook 13's; the notebook's globals (settings, device, scratch folder) are module constants
here, and the two functions that read a notebook table (`train_segmenter`, `score_detector`) take it as an argument.
Both trainers can also take photos with no snake in them, which teach the models to find nothing (notebook 20).
"""
import json
import math
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import yaml
from albumentations import BaseCompose
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from ultralytics import YOLO

from serpentguard.data import (AUGMENT, NO_ID, TARGETS, SnakeDataset, apply_mode, clean_mask, is_missing_box,
                               load_mask, load_rgb)
from serpentguard.decide import fit_snake_threshold, photos_needed
from serpentguard.metrics import det_scores
from serpentguard.models import MaskHead, build_encoder

SEG = dict(epochs=15, batch=32, size=320, lr=1e-3, weight_decay=1e-4, aug_strength=0.25, seed=42)
DET = dict(epochs=20, batch=32, size=320, seed=42)  # the detector, same size and seed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKERS = os.cpu_count() or 2
TMP = Path("/kaggle/tmp") if Path("/kaggle/working").exists() else Path("../work/cropping/tmp")
BOX_FLOOR = 0.05           # the detector's confidence floor: the lowest box kept
PREDS_FILE, NEW_ROWS_FILE, MASKS_DIR = "frontend_preds.csv", "new_rows.csv", "masks_pred"
PREDS_COLUMNS = "path,box,mask_file,box_conf\n"
PREDICT_BATCH = 32
A = SEG["aug_strength"]
YOLO_ARGS = dict(
    optimizer="AdamW", lr0=1e-3, lrf=0.0, cos_lr=True, weight_decay=1e-4, warmup_epochs=1,
    deterministic=True, plots=False, pretrained=True, exist_ok=True, name="yolo",
    fliplr=0.5, flipud=0.2, degrees=25 * A, translate=0.05 * A, scale=0.12 * A,
    hsv_h=0.04 * A, hsv_s=0.2 * A, hsv_v=0.2 * A,
    shear=0.0, perspective=0.0, mosaic=0.0, close_mosaic=0, mixup=0.0, cutmix=0.0, copy_paste=0.0)
NEGATIVE_PREFIX = "nonsnake__"  # file stem of a no-snake photo in a YOLO training set
GATE_RULES = ("box", "mask", "both")
GATE_KEEP = 0.99         # share of calibration snakes a gate must keep
GATE_BOX_FLOOR = 0.001   # every box is kept, so a gate cut-off below notebook 13's 0.05 floor can be fitted
GATE_BATCH = 32


class MaskHeadNet(nn.Module):
    """The mask head of our own model, on its own small MobileNetV4 encoder."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder, channels = build_encoder("mobilenetv4_conv_small", pretrained, SEG["size"])
        self.head = MaskHead(channels)

    def forward(self, images):
        return self.head(self.encoder(images), images.shape[-2:])


class OutputOnly(nn.Module):
    """torchvision's segmentation models return a dict; keep the "out" logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images)["out"]


def build_net(kind, pretrained=True):
    if kind == "maskhead":
        return MaskHeadNet(pretrained)
    if kind == "lraspp":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        net = lraspp_mobilenet_v3_large(weights=None, weights_backbone=weights, num_classes=1)
        return OutputOnly(net)
    return smp.Unet("mobilenet_v2", encoder_weights="imagenet" if pretrained else None, classes=1)


def dice_bce(logits, masks, valid):
    """Cross-entropy plus soft Dice per photo, averaged over the photos whose mask is real."""
    if not valid.any():
        return logits.sum() * 0
    logits, targets = logits[valid], masks[valid].to(logits.dtype)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = bce.flatten(1).mean(1)
    probs, flat = torch.sigmoid(logits).flatten(1), targets.flatten(1)
    dice = 1 - (2 * (probs * flat).sum(1) + 1.0) / (probs.sum(1) + flat.sum(1) + 1.0)
    return (bce + dice).mean()


def seed_augmentation(dataset, seed):
    """Give each augmentation pipeline and each step its own seed."""
    parts = []

    def walk(node):
        parts.append(node)
        for child in node.transforms:
            walk(child) if isinstance(child, BaseCompose) else parts.append(child)

    for pipeline in dataset.augment.values():
        walk(pipeline)
    for part, s in zip(parts, np.random.SeedSequence(seed).generate_state(len(parts))):
        part.set_random_seed(int(s))


def init_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    random.seed(info.seed)
    np.random.seed(info.seed % 2 ** 32)
    seed_augmentation(info.dataset, info.seed)
    cv2.setNumThreads(0)


class SegmenterData(SnakeDataset):
    """SnakeDataset, plus rows marked empty_mask (photos with no snake), whose target is an empty mask."""

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        if not row["empty_mask"]:
            return super().__getitem__(index)
        rgb = load_rgb(row["image_file"])
        image, target = apply_mode(rgb, np.zeros(rgb.shape[:2], np.uint8), "full", None, self.size)
        sample = self.augment[row["tier"] if self.train else "eval"](image=image, mask=target)
        return {"image": sample["image"], "mask": sample["mask"].unsqueeze(0).float(), "mask_valid": True,
                **{name: NO_ID for name in TARGETS}}


def segmenter_frame(seg_frame, negatives=None):
    """The segmenter's photos, then the no-snake photos (path, image_file) marked empty_mask."""
    frame = seg_frame.assign(empty_mask=False)
    if negatives is None or not len(negatives):
        return frame.reset_index(drop=True)
    empty = pd.DataFrame({"path": negatives["path"].to_numpy(), "image_file": negatives["image_file"].to_numpy(),
                          "mask_file": "", "box": None, "empty_mask": True, **{name: NO_ID for name in TARGETS}})
    return pd.concat([frame, empty], ignore_index=True)


def train_segmenter(kind, out_dir, seg_frame, negatives=None):
    """AdamW, one warm-up epoch then cosine decay, mixed precision; saves the last epoch.

    negatives: optional photos with no snake (path, image_file), trained toward an empty mask.
    """
    started, out_dir = time.perf_counter(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEG["seed"]); np.random.seed(SEG["seed"]); torch.manual_seed(SEG["seed"])
    torch.backends.cudnn.benchmark = True
    frame = segmenter_frame(seg_frame, negatives)
    data = SegmenterData(frame.assign(tier="tier1"), train=True, mode="full", size=SEG["size"],
                         aug_strength=SEG["aug_strength"])
    seed_augmentation(data, SEG["seed"])
    loader = DataLoader(data, batch_size=SEG["batch"], shuffle=True, drop_last=True,
                        num_workers=WORKERS, pin_memory=True, worker_init_fn=init_worker,
                        persistent_workers=True,
                        generator=torch.Generator().manual_seed(SEG["seed"]))
    net = build_net(kind).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=SEG["lr"], weight_decay=SEG["weight_decay"])
    steps, warmup = SEG["epochs"] * len(loader), len(loader)
    cosine = lambda s: ((s + 1) / warmup if s < warmup
                        else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine)
    scaler = torch.amp.GradScaler(DEVICE)
    losses = []
    for epoch in range(1, SEG["epochs"] + 1):
        net.train()
        epoch_start, epoch_loss = time.perf_counter(), []
        for batch in loader:
            images, masks, valid = (batch[k].to(DEVICE, non_blocking=True)
                                    for k in ("image", "mask", "mask_valid"))
            with torch.autocast(DEVICE, dtype=torch.float16):
                logits = net(images)
            loss = dice_bce(logits.float(), masks, valid)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            epoch_loss.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_loss)))
        seconds = time.perf_counter() - epoch_start
        print(f"{kind} epoch {epoch}/{SEG['epochs']}: loss {losses[-1]:.4f}, {seconds:.0f} s")
    torch.save(net.state_dict(), out_dir / "segmenter.pt")
    run = dict(kind=kind, weights="segmenter.pt", train_photos=len(seg_frame),
               negative_photos=int(frame["empty_mask"].sum()), epoch_loss=losses,
               parameters=sum(p.numel() for p in net.parameters()),
               seconds=time.perf_counter() - started, **SEG)
    (out_dir / "run.json").write_text(json.dumps(run))


def label_line(values):
    return " ".join(["0"] + [f"{v:.6f}" for v in values])


def mask_lines(mask):
    """One line per outer outline of a 0/1 mask, as fractions of the photo; specks dropped."""
    h, w = mask.shape
    outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [label_line((o.reshape(-1, 2) / [w, h]).ravel()) for o in outlines
            if len(o) >= 3 and cv2.contourArea(o) >= 0.002 * mask.size]


def box_lines(boxes, w, h):
    """(centre x, centre y, width, height) as photo fractions, for finite boxes of 1 px or more."""
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2, y1, y2 = np.clip(x1, 0, w), np.clip(x2, 0, w), np.clip(y1, 0, h), np.clip(y2, 0, h)
        if np.isfinite([x1, y1, x2, y2]).all() and x2 - x1 >= 1 and y2 - y1 >= 1:
            lines.append(label_line([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                                     (x2 - x1) / w, (y2 - y1) / h]))
    return lines


def export_photo(row, root, size, boxes):
    rgb = load_rgb(row["image_file"])
    h, w = rgb.shape[:2]
    if boxes:
        lines = box_lines(row["snake_boxes"], w, h)
        scale = size / max(h, w)
        image = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    else:
        mask = load_mask(row["mask_file"], rgb.shape[:2])
        if mask is None:
            return 0
        image, canvas_mask = apply_mode(rgb, mask, "full", None, size)
        lines = mask_lines(canvas_mask)
    if not lines:
        return 0
    stem = Path(row["path"]).with_suffix("").as_posix().replace("/", "__")
    Image.fromarray(image).save(root / "images/train" / f"{stem}.jpg", quality=95)
    (root / "labels/train" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    return 1


def export_negative(row, root, size):
    """A photo with no snake, resized as the box photos are, with an empty label file (a YOLO background)."""
    rgb = load_rgb(row["image_file"])
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    image = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    stem = NEGATIVE_PREFIX + Path(row["path"]).with_suffix("").as_posix().replace("/", "__")
    Image.fromarray(image).save(root / "images/train" / f"{stem}.jpg", quality=95)
    (root / "labels/train" / f"{stem}.txt").write_text("")
    return 1


def write_yolo_set(frame, root, size, boxes, negatives=None):
    """negatives: optional photos with no snake (path, image_file), written with empty labels."""
    shutil.rmtree(root, ignore_errors=True)
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (root / kind / split).mkdir(parents=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        rows = frame.to_dict("records")
        written = sum(pool.map(lambda row: export_photo(row, root, size, boxes), rows))
        if negatives is not None:
            rows = negatives.to_dict("records")
            written += sum(pool.map(lambda row: export_negative(row, root, size), rows))
    for image in sorted((root / "images/train").glob("*.jpg"))[:64]:
        shutil.copy(image, root / "images/val" / image.name)
        label = f"{image.stem}.txt"
        shutil.copy(root / "labels/train" / label, root / "labels/val" / label)
    spec = {"path": str(root.resolve()), "train": "images/train", "val": "images/val",
            "names": {0: "snake"}}
    (root / "data.yaml").write_text(yaml.safe_dump(spec))
    return written


def train_yolo(name, start, frame, out_dir, settings, boxes, weights_file, negatives=None):
    """Export the photos, fine-tune `start`, keep the last epoch's weights and write run.json.

    negatives: optional photos with no snake (path, image_file), trained as background images.
    """
    started, out_dir = time.perf_counter(), Path(out_dir)
    root = TMP / "yolo_sets" / name
    photos = write_yolo_set(frame, root, settings["size"], boxes, negatives)
    YOLO(start).train(data=str(root / "data.yaml"), epochs=settings["epochs"],
                      imgsz=settings["size"], batch=settings["batch"], nbs=settings["batch"],
                      seed=settings["seed"], workers=WORKERS, device=0,
                      project=str(out_dir.resolve()), **YOLO_ARGS)
    last = out_dir / "yolo/weights/last.pt"
    shutil.copy(last, out_dir / weights_file)
    params = sum(p.numel() for p in YOLO(str(last)).model.parameters())
    run = dict(kind=name, weights=weights_file, parameters=params, train_photos=photos,
               negative_photos=0 if negatives is None else len(negatives),
               seconds=time.perf_counter() - started, **settings)
    (out_dir / "run.json").write_text(json.dumps(run))
    print(f"{name}: {photos:,} photos, {run['seconds'] / 60:.0f} min")


def unletterbox(canvas_map, width, height, size):
    scale = size / max(height, width)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    top, left = (size - rh) // 2, (size - rw) // 2
    window = np.asarray(canvas_map, dtype=np.float32)[top:top + rh, left:left + rw]
    return cv2.resize(window, (width, height), interpolation=cv2.INTER_LINEAR)


class Segmenter:
    """A trained segmenter: masks(photos) gives a 0/1 uint8 mask per RGB photo, at photo size."""

    def __init__(self, kind, folder):
        self.kind, self.size = kind, SEG["size"]
        if kind == "yolo26n-seg":
            self.net = YOLO(str(Path(folder) / "segmenter.pt"))
        else:
            self.net = build_net(kind, pretrained=False)
            saved = torch.load(Path(folder) / "segmenter.pt", weights_only=True, map_location=DEVICE)
            if "state_dict" in saved:  # the M3 file wraps its weights with the kind and input size
                if saved["kind"] != kind:
                    raise ValueError(f"{folder} holds a {saved['kind']} segmenter, not {kind}")
                self.size, saved = saved["input_size"], saved["state_dict"]
            self.net.load_state_dict(saved)
            self.net.to(DEVICE).eval()

    def probabilities(self, canvases):
        if self.kind == "yolo26n-seg":  # the union of the instances YOLO keeps at confidence 0.25
            bgr = [cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for c in canvases]
            found = self.net.predict(bgr, imgsz=self.size, conf=0.25, retina_masks=True,
                                     verbose=False, device=0)
            union = np.zeros((len(canvases), self.size, self.size), np.float32)
            for i, result in enumerate(found):
                if result.masks is not None and len(result.masks.data):
                    union[i] = result.masks.data.float().amax(dim=0).cpu().numpy()
            return union
        images = torch.stack([AUGMENT["eval"](image=c)["image"] for c in canvases]).to(DEVICE)
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            return torch.sigmoid(self.net(images).float())[:, 0].cpu().numpy()

    def masks(self, photos, threshold=0.5):
        canvases = [apply_mode(p, None, "full", None, self.size)[0] for p in photos]
        return [(unletterbox(prob, p.shape[1], p.shape[0], self.size) > threshold).astype(np.uint8)
                for prob, p in zip(self.probabilities(canvases), photos)]


def snake_boxes(model, bgr_photos, conf, normalized=False, batch=16):
    """Per photo, its snake boxes as [x1, y1, x2, y2, score], in pixels or photo fractions."""
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    snake = [i for i, name in names.items() if str(name).strip().lower() == "snake"]
    assert len(snake) == 1, "the detector must have exactly one snake class"
    found = []
    for start in range(0, len(bgr_photos), batch):
        chunk = bgr_photos[start:start + batch]
        # rect=False pads every call alike, as the training boxes were
        for result in model.predict(chunk, conf=conf, verbose=False, classes=snake, rect=False):
            b = result.boxes
            corners = (b.xyxyn if normalized else b.xyxy).cpu().numpy()
            found.append(np.column_stack([corners, b.conf.cpu().numpy()]).reshape(-1, 5).tolist())
    return found


def score_detector(model, oi):
    boxes = []
    for start in range(0, len(oi), 32):
        files = oi["image_file"][start:start + 32]
        photos = [cv2.cvtColor(load_rgb(f), cv2.COLOR_RGB2BGR) for f in files]
        boxes += snake_boxes(model, photos, conf=0.001, normalized=True)  # keep everything, for mAP
    sure = [[b for b in found if b[4] >= 0.25] for found in boxes]
    at_conf, everything = det_scores(sure, list(oi["boxes"])), det_scores(boxes, list(oi["boxes"]))
    return {"precision @0.25": at_conf["precision"], "recall @0.25": at_conf["recall"],
            "mAP@0.5": everything["map50"]}


def gate_parts(rgbs, detector, segmenter):
    """Per RGB photo already loaded (exposure fixed): the detector's best box and its box_conf (nan: no box), the
    segmenter's 0/1 mask, and mask_area, the share of the photo the mask paints as snake, before any cleanup."""
    found = snake_boxes(detector, [cv2.cvtColor(r, cv2.COLOR_RGB2BGR) for r in rgbs], conf=GATE_BOX_FLOOR)
    parts = []
    for mask, candidates in zip(segmenter.masks(rgbs), found):
        best = max(candidates, key=lambda b: b[4], default=None)
        parts.append({"box": None if best is None else [float(v) for v in best[:4]],
                      "box_conf": math.nan if best is None else float(best[4]),
                      "mask": mask, "mask_area": float(mask.mean())})
    return parts


def gate_scores(frame, detector, segmenter, batch=GATE_BATCH):
    """Per photo (path, is_snake, image_file): gate_parts' box, box_conf and mask_area."""
    boxes, confs, areas = [], [], []
    with ThreadPoolExecutor(WORKERS) as pool:
        for start in range(0, len(frame), batch):
            rgbs = list(pool.map(load_rgb, frame["image_file"].iloc[start:start + batch]))
            for part in gate_parts(rgbs, detector, segmenter):
                boxes.append(part["box"])
                confs.append(part["box_conf"])
                areas.append(part["mask_area"])
    return pd.DataFrame({"path": frame["path"].to_numpy(), "is_snake": frame["is_snake"].to_numpy(bool),
                         "box_conf": np.asarray(confs, float), "mask_area": np.asarray(areas, float),
                         "box": pd.Series(boxes, dtype=object).to_numpy()})


def gate_calls(scores, rule, thresholds):
    """True where the rule calls the photo a snake; a missing box or an empty mask never passes."""
    if rule not in GATE_RULES:
        raise ValueError(f"rule must be one of {GATE_RULES}, not {rule!r}")
    called = np.ones(len(scores), dtype=bool)
    if rule in ("box", "both"):
        box = scores["box_conf"].to_numpy(float)
        called &= np.where(np.isfinite(box), box, -np.inf) >= thresholds["box_conf"]
    if rule in ("mask", "both"):
        area = np.nan_to_num(scores["mask_area"].to_numpy(float), nan=0.0)
        called &= (area > 0) & (area >= thresholds["mask_area"])
    return called


def called_share(scores, members, rule, thresholds, kind):
    if not members.any():
        raise ValueError(f"no {kind} photos to count")
    return float(gate_calls(scores, rule, thresholds)[members].mean())


def snake_recall(scores, rule, thresholds):
    """Share of the snake photos that the rule calls a snake."""
    return called_share(scores, scores["is_snake"].to_numpy(bool), rule, thresholds, "snake")


def false_snake_rate(scores, rule, thresholds):
    """Share of the photos with no snake that the rule calls a snake."""
    return called_share(scores, ~scores["is_snake"].to_numpy(bool), rule, thresholds, "non-snake")


def fit_gate(calib, rule, keep=GATE_KEEP):
    """The rule's cut-offs, fitted on calibration snakes so that `keep` of them pass.

    "box" and "mask" use decide.fit_snake_threshold on their own score (no box, or an empty mask, counts as missed).
    "both" gives the two cut-offs that each keep the same number of photos, the fewest that still let `keep` of the
    photos pass both; when no pair gets there, the loosest pair, so the shortfall shows in the recall.
    """
    if rule not in GATE_RULES:
        raise ValueError(f"rule must be one of {GATE_RULES}, not {rule!r}")
    if not calib["is_snake"].to_numpy(bool).all():
        raise ValueError("gate cut-offs are fitted on snake photos only")
    box = calib["box_conf"].to_numpy(float)
    area = calib["mask_area"].to_numpy(float)
    area = np.where(area > 0, area, np.nan)
    if rule == "box":
        return {"box_conf": fit_snake_threshold(box, keep)}
    if rule == "mask":
        return {"mask_area": fit_snake_threshold(area, keep)}
    return equal_count_cutoffs(box, area, keep)


def equal_count_cutoffs(box, area, keep):
    """See fit_gate's "both"; nan in either score means the photo has none."""
    found = [np.sort(v[np.isfinite(v)])[::-1] for v in (box, area)]
    if not all(len(v) for v in found):
        raise ValueError("no calibration photo has a box, or none has a mask, so the cut-offs can't be fitted")
    box, area = (np.where(np.isfinite(v), v, -np.inf) for v in (box, area))
    needed = photos_needed(keep, len(box))
    for k in range(needed, len(box) + 1):
        cut_box, cut_area = (float(v[min(k, len(v)) - 1]) for v in found)
        if ((box >= cut_box) & (area >= cut_area)).sum() >= needed:
            break
    return {"box_conf": cut_box, "mask_area": cut_area}


def box_iou(a, b):
    """Overlap over union of two [x1, y1, x2, y2] boxes; nan when either is missing."""
    if is_missing_box(a) or is_missing_box(b):
        return math.nan
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = a[:4], b[:4]
    both = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - both
    return float(both / union) if union > 0 else 0.0


class _Photos(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return i, load_rgb(self.files[i])


def extend_predictions(full, m3_preds_dir, detector_path, segmenter_path, data_root, out_dir, limit=None):
    """M3's frontend_preds.csv plus a row for every photo of `full` it lacks, written under out_dir.

    The new photos get the detector's most confident box at BOX_FLOOR or above and the U-Net's mask (a PNG under
    masks_pred/, only if clean_mask keeps it). M3's file is copied byte for byte ahead of the new rows and its masks
    are copied beside the new ones. Each batch is appended to new_rows.csv as it finishes, so a restarted run
    carries on. Returns the joined frame; `limit` keeps only that many new photos (a smoke test).
    """
    m3_preds_dir, data_root, out_dir = Path(m3_preds_dir), Path(data_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m3_bytes = (m3_preds_dir / PREDS_FILE).read_bytes()
    m3_bytes += b"" if m3_bytes.endswith(b"\n") else b"\n"
    m3_paths = set(pd.read_csv(m3_preds_dir / PREDS_FILE, keep_default_na=False)["path"])
    missing = full[~full["path"].isin(m3_paths)][["path"]].reset_index(drop=True)
    if limit:
        missing = missing.head(limit)
    new_csv = out_dir / NEW_ROWS_FILE
    done = set(pd.read_csv(new_csv, keep_default_na=False)["path"]) if new_csv.exists() else set()
    todo = missing[~missing["path"].isin(done)].reset_index(drop=True)
    print(f"{len(missing):,} of {len(full):,} photos are missing from M3's file; {len(todo):,} still to run")
    detector, seg = YOLO(str(detector_path)), Segmenter("unet", Path(segmenter_path).parent)
    if not done:
        new_csv.write_text(PREDS_COLUMNS)
    files = [str(data_root / "data" / p) for p in todo["path"]]
    started, seen = time.perf_counter(), 0
    for batch in DataLoader(_Photos(files), batch_size=PREDICT_BATCH, num_workers=WORKERS, collate_fn=list):
        indices, rgbs = zip(*batch)
        found = snake_boxes(detector, [cv2.cvtColor(r, cv2.COLOR_RGB2BGR) for r in rgbs], conf=BOX_FLOOR)
        lines = []
        for i, mask, candidates in zip(indices, seg.masks(list(rgbs)), found):
            path, box, conf, mask_file = todo.at[i, "path"], "", "", ""
            if candidates:
                *corners, score = max(candidates, key=lambda b: b[4])
                box, conf = json.dumps([round(v, 2) for v in corners]), f"{score:.4f}"
            if clean_mask(mask) is not None:
                target = out_dir / MASKS_DIR / Path(path).with_suffix(".png")
                target.parent.mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(target), mask * 255), f"could not write {target}"
                mask_file = target.relative_to(out_dir).as_posix()
            lines.append(pd.DataFrame([[path, box, mask_file, conf]]).to_csv(header=False, index=False))
        with open(new_csv, "a") as handle:
            handle.write("".join(lines))
        seen += len(indices)
        if (seen // PREDICT_BATCH) % 100 == 0:
            print(f"{seen:,}/{len(todo):,} photos, {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    (out_dir / PREDS_FILE).write_bytes(m3_bytes + new_csv.read_bytes().split(b"\n", 1)[1])
    shutil.copytree(m3_preds_dir / MASKS_DIR, out_dir / MASKS_DIR, dirs_exist_ok=True)
    new_csv.unlink()
    return pd.read_csv(out_dir / PREDS_FILE, keep_default_na=False)
