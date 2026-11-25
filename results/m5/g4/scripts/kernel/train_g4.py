"""G4: a small snake / lookalike / other-non-snake classifier, fine-tuned end to end on a Kaggle T4.

Snakes: the dataset's training split minus every calibration photographer. Non-snakes: notebook 19's training split,
the 1,317 extra lookalikes and the new iNaturalist negatives (all training only). Optional teacher: BioCLIP's image
embedding, matched by a linear projection of the student's pooled features (feature distillation).
Writes model.pt, a config, the training log and PyTorch scores for calibration and validation snakes.
"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

VARIANT = os.environ.get("G4_VARIANT", "__VARIANT__")
SETTINGS = {
    "plain": {"backbone": "efficientnet_b0.ra_in1k", "distil_weight": 0.0},
    "distil": {"backbone": "efficientnet_b0.ra_in1k", "distil_weight": 1.0},
    "shuffled": {"backbone": "efficientnet_b0.ra_in1k", "distil_weight": 0.0, "shuffle_labels": True},
    "distil_oi": {"backbone": "efficientnet_b0.ra_in1k", "distil_weight": 1.0,
                  "class_share": {"snake": 0.45, "lookalike": 0.3, "other": 0.25}, "strict_data": True},
    "distil_oi_v2": {"backbone": "efficientnet_b0.ra_in1k", "distil_weight": 1.0, "seed": 42,
                     "class_share": {"snake": 0.45, "lookalike": 0.3, "other": 0.25}, "strict_data": True,
                     "require_origin": "negatives_v2"},
    "distil_oi_seed43": {"backbone": "efficientnet_b0.ra_in1k", "distil_weight": 1.0, "seed": 43,
                         "class_share": {"snake": 0.45, "lookalike": 0.3, "other": 0.25}, "strict_data": True},
}[VARIANT]

SEED = SETTINGS.get("seed", 42)
INPUT_SIZE = 256
PAD_RGB = (124, 116, 104)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
CLASSES = ("snake", "lookalike", "other")
EPOCHS = 12 if VARIANT != "shuffled" else 3
SAMPLES_PER_EPOCH = 48_000
CLASS_SHARE = SETTINGS.get("class_share", {"snake": 0.5, "lookalike": 0.3, "other": 0.2})
EXPOSURE = {"below": 60, "above": 200, "clip": 2.0, "tiles": (8, 8)}  # pipeline.json "exposure_fix"
BATCH = 96
LR, WEIGHT_DECAY, WARMUP_STEPS = 6e-4, 0.02, 300
LABEL_SMOOTHING = 0.05
WORKERS = 4
BIOCLIP = "hf-hub:imageomics/bioclip"
BIOCLIP_SIZE = 224

SNAKE_DATASET = "puneetbajajiitm/serpentguard-indian-snakes-clean"
OUR_DATASET = "__OUR_DATASET__"
OUT = Path("/kaggle/working")


def dataset(handle):
    import kagglehub
    return Path(kagglehub.dataset_download(handle))


SMOKE = bool(os.environ.get("G4_SMOKE"))  # a local CPU check of the whole script on a few photos
SNAKES = Path(os.environ["G4_SNAKES"]) if SMOKE else dataset(SNAKE_DATASET) / "data"
OURS = next((Path(os.environ["G4_OURS"]) if SMOKE else dataset(OUR_DATASET)).rglob("g4_data_manifest.json")).parent
OUT = Path(os.environ["G4_OUT"]) if SMOKE else OUT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if SMOKE:
    EPOCHS, SAMPLES_PER_EPOCH, BATCH, WARMUP_STEPS, WORKERS = 1, 16, 8, 1, 0


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def letterbox(rgb, size=INPUT_SIZE, scale_jitter=None):
    """The app's letterbox (reference_pipeline.letterbox); with scale_jitter the photo is placed smaller and anywhere."""
    height, width = rgb.shape[:2]
    scale = size / max(height, width)
    if scale_jitter is not None:
        scale *= scale_jitter
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if scale_jitter is None:
        top, left = (size - new_h) // 2, (size - new_w) // 2
    else:
        top, left = random.randint(0, size - new_h), random.randint(0, size - new_w)
    canvas = np.full((size, size, 3), PAD_RGB, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def app_rgb(path):
    """Scoring input as the app builds it: PIL decode, then the exposure fix (reference_pipeline.load_photo)."""
    from PIL import Image
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if EXPOSURE["below"] <= rgb.mean() <= EXPOSURE["above"]:
        return rgb
    lightness, a, b = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB))
    clahe = cv2.createCLAHE(clipLimit=EXPOSURE["clip"], tileGridSize=EXPOSURE["tiles"])
    return cv2.cvtColor(cv2.merge([clahe.apply(lightness), a, b]), cv2.COLOR_LAB2RGB)


def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"unreadable {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def augment(rgb):
    height, width = rgb.shape[:2]
    if random.random() < 0.6:  # a random crop that keeps most of the photo
        keep = random.uniform(0.6, 1.0)
        ch, cw = int(height * keep), int(width * keep)
        top, left = random.randint(0, height - ch), random.randint(0, width - cw)
        rgb = rgb[top:top + ch, left:left + cw]
    if random.random() < 0.5:
        rgb = rgb[:, ::-1]
    if random.random() < 0.15:
        rgb = np.rot90(rgb, random.choice((1, 3)))
    rgb = np.ascontiguousarray(rgb)
    canvas = letterbox(rgb, scale_jitter=random.uniform(0.75, 1.0) if random.random() < 0.3 else None)
    canvas = canvas.astype(np.float32)
    canvas = canvas * random.uniform(0.7, 1.3) + random.uniform(-25, 25)  # contrast and brightness
    grey = canvas.mean(axis=2, keepdims=True)
    canvas = grey + (canvas - grey) * random.uniform(0.6, 1.4)  # saturation
    if random.random() < 0.2:
        canvas = cv2.GaussianBlur(canvas, (0, 0), random.uniform(0.5, 1.5))
    return np.clip(canvas, 0, 255).astype(np.uint8)


def to_tensor(canvas):
    return torch.from_numpy(((canvas.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))


class Photos(Dataset):
    def __init__(self, paths, labels, train, teacher=None):
        self.paths, self.labels, self.train, self.teacher = list(paths), np.asarray(labels), train, teacher

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rgb = read_rgb(self.paths[i]) if self.train else app_rgb(self.paths[i])
        canvas = augment(rgb) if self.train else letterbox(rgb)
        target = self.teacher[i] if self.teacher is not None else np.zeros(1, np.float32)
        return to_tensor(canvas), int(self.labels[i]), torch.from_numpy(target)


def training_table():
    snakes = pd.read_csv(OURS / "snakes_train.csv")
    snakes = pd.DataFrame({"path": [str(SNAKES / p) for p in snakes["path"]], "label": 0, "group": "snake"})
    others = pd.read_csv(OURS / "nonsnake_train.csv")
    on_disk = {p.name: p for p in OURS.rglob("*.jpg")}  # Kaggle may nest uploaded folders; names are unique
    others = pd.DataFrame({"path": [str(on_disk.get(Path(f).name, OURS / f)) for f in others["path"]],
                           "label": np.where(others["group"].eq("lookalike"), 1, 2), "group": others["group"]})
    table = pd.concat([snakes, others], ignore_index=True)
    exists = np.array([Path(p).exists() for p in table["path"]])
    log("training photos", len(table), "missing", int((~exists).sum()))
    if SETTINGS.get("strict_data") and ((~exists).any() or not (OURS / "nonsnake_val.csv").exists()):
        raise SystemExit("dataset version is not the expected one: photos missing or no nonsnake_val.csv")
    required = SETTINGS.get("require_origin")
    if required and not table["path"].str.contains(f"/{required}/", regex=False).any():
        raise SystemExit(f"dataset version is not the expected one: no {required} photos")
    return table[exists].reset_index(drop=True)


def bioclip_embeddings(paths):
    if not SMOKE:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
    import open_clip
    local = os.environ.get("G4_BIOCLIP_WEIGHTS")
    model, _, preprocess = (open_clip.create_model_and_transforms("ViT-B-16", pretrained=local) if local
                            else open_clip.create_model_and_transforms(BIOCLIP))
    model = model.to(DEVICE).eval()
    half = DEVICE == "cuda"
    model = model.half() if half else model
    from PIL import Image

    class Plain(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            with Image.open(paths[i]) as image:
                return preprocess(image.convert("RGB"))

    out = []
    with torch.no_grad():
        for batch in DataLoader(Plain(), batch_size=256, num_workers=WORKERS):
            out.append(F.normalize(model.encode_image(batch.to(DEVICE).half() if half else batch).float(), dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


class Student(nn.Module):
    def __init__(self, backbone, teacher_dim):
        super().__init__()
        import timm
        self.body = timm.create_model(backbone, pretrained=not SMOKE, num_classes=0)
        self.head = nn.Linear(self.body.num_features, len(CLASSES))
        self.project = nn.Linear(self.body.num_features, teacher_dim) if teacher_dim else None

    def forward(self, x):
        features = self.body(x)
        return self.head(features), features


def sampler_for(table):
    share = table["label"].map({0: CLASS_SHARE["snake"], 1: CLASS_SHARE["lookalike"], 2: CLASS_SHARE["other"]})
    weights = share / table["label"].map(table["label"].value_counts())
    return WeightedRandomSampler(torch.tensor(weights.to_numpy(), dtype=torch.double), SAMPLES_PER_EPOCH,
                                 replacement=True, generator=torch.Generator().manual_seed(SEED))


def train(table, teacher):
    labels = table["label"].to_numpy()
    if SETTINGS.get("shuffle_labels"):
        labels = np.random.default_rng(SEED).permutation(labels)
    data = Photos(table["path"], labels, train=True, teacher=teacher)
    loader = DataLoader(data, batch_size=BATCH, sampler=sampler_for(table.assign(label=labels)),
                        num_workers=WORKERS, pin_memory=True, drop_last=True, persistent_workers=WORKERS > 0)
    model = Student(SETTINGS["backbone"], teacher.shape[1] if teacher is not None else 0).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total = EPOCHS * len(loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: min(1.0, (s + 1) / WARMUP_STEPS) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler(DEVICE, enabled=DEVICE == "cuda")
    history = []
    for epoch in range(EPOCHS):
        model.train()
        sums, count, started = np.zeros(3), 0, time.time()
        for x, y, target in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                logits, features = model(x)
                ce = F.cross_entropy(logits.float(), y, label_smoothing=LABEL_SMOOTHING)
                distil = torch.zeros((), device=x.device)
                if model.project is not None:
                    projected = F.normalize(model.project(features).float(), dim=-1)
                    distil = (1 - (projected * target.to(DEVICE).float()).sum(-1)).mean()
                loss = ce + SETTINGS["distil_weight"] * distil
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            accuracy = (logits.argmax(1) == y).float().mean()
            sums += [ce.item(), distil.item(), accuracy.item()]
            count += 1
        row = {"epoch": epoch + 1, "ce": sums[0] / count, "distil": sums[1] / count, "accuracy": sums[2] / count,
               "seconds": round(time.time() - started)}
        history.append(row)
        log(json.dumps(row))
    return model, history


def score(model, paths):
    model.eval()
    loader = DataLoader(Photos(paths, np.zeros(len(paths)), train=False), batch_size=128, num_workers=WORKERS)
    out = []
    with torch.no_grad():
        for x, _, _ in loader:
            logits, _ = model(x.to(DEVICE))
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def main():
    random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    log("variant", VARIANT, SETTINGS)
    table = training_table()
    if SMOKE:
        table = table.groupby("label").head(8).reset_index(drop=True)
    log(table["group"].value_counts().to_dict())
    teacher = None
    if SETTINGS["distil_weight"] > 0:
        teacher = bioclip_embeddings(list(table["path"]))
        np.save(OUT / "teacher_train.npy", teacher)
        log("teacher embeddings", teacher.shape)
    model, history = train(table, teacher)
    torch.save({"state_dict": model.state_dict(), "backbone": SETTINGS["backbone"], "classes": CLASSES,
                "input_size": INPUT_SIZE}, OUT / "model.pt")
    rows = pd.read_csv(OURS / "snakes_score.csv")
    nonsnake_val = OURS / "nonsnake_val.csv"
    if nonsnake_val.exists():  # validation non-snakes, scored only, never trained on
        extra = pd.read_csv(nonsnake_val)
        on_disk = {p.name: p for p in OURS.rglob("*.jpg")}
        rows = pd.concat([rows.assign(file=[str(SNAKES / p) for p in rows["path"]]),
                          extra.assign(set="nonsnake_val", file=[str(on_disk[Path(p).name]) for p in extra["path"]])])
    else:
        rows = rows.assign(file=[str(SNAKES / p) for p in rows["path"]])
    rows = rows.head(10) if SMOKE else rows.reset_index(drop=True)
    logits = score(model, list(rows["file"]))
    for i, name in enumerate(CLASSES):
        rows[f"logit_{name}"] = logits[:, i]
    rows.to_csv(OUT / "scores_torch.csv", index=False)
    config = {"variant": VARIANT, **SETTINGS, "epochs": EPOCHS, "samples_per_epoch": SAMPLES_PER_EPOCH,
              "class_share": CLASS_SHARE, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
              "label_smoothing": LABEL_SMOOTHING, "input_size": INPUT_SIZE, "pad_rgb": PAD_RGB,
              "training_photos": table["group"].value_counts().to_dict(), "history": history,
              "torch": torch.__version__}
    (OUT / "train_config.json").write_text(json.dumps(config, indent=1) + "\n")
    log("done")


if __name__ == "__main__":
    main()
