"""S1's four segmenters: building and training the three torch ones, and turning any of the four into snake masks."""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from serpentguard.data import AUGMENT, IMG_SIZE, MASK_MODE, SEED, SnakeDataset, prepare
from serpentguard.losses import dice_bce
from serpentguard.models import MaskHead, build_encoder
from serpentguard.run_data import init_worker, seed_augmentation, seed_everything
from serpentguard.run_outputs import AMP_DTYPE, write_json
from serpentguard.train import GRAD_CLIP_NORM, start_cuda, warmup_cosine

MASKHEAD, LRASPP, UNET, YOLO_SEG = "maskhead", "lraspp", "unet", "yolo26n-seg"
TORCH_KINDS = (MASKHEAD, LRASPP, UNET)
SEG_KINDS = (*TORCH_KINDS, YOLO_SEG)
YOLO_SEG_WEIGHTS = "yolo26n-seg.pt"
SEG_INPUT_SIZE = IMG_SIZE
SEG_EPOCHS, SEG_BATCH = 15, 32
SEG_LR, SEG_WEIGHT_DECAY = 1e-3, 1e-4
SEG_AUG_STRENGTH = 0.25      # notebook 06's tier-1 pipeline, its ranges scaled down: flips and light jitter
MASK_THRESHOLD = 0.5
YOLO_SEG_CONF = 0.25         # instances kept by YOLO26n-seg before their masks are joined
YOLO_TOP_CONF_FLOOR = 0.001  # the confidence diagnostic looks this low; a photo with nothing above it scores 0
SEG_PREDICT_BATCH = 32
SEGMENTER_FILE, RUN_FILE = "segmenter.pt", "run.json"


class MaskHeadSegmenter(nn.Module):
    """SnakeID's mask head on its own MobileNetV4-S encoder."""

    def __init__(self, pretrained=True, input_size=SEG_INPUT_SIZE):
        super().__init__()
        self.encoder, channels = build_encoder("mobilenetv4_conv_small", pretrained=pretrained, input_size=input_size)
        self.head = MaskHead(channels)

    def forward(self, images):
        return self.head(self.encoder(images), images.shape[-2:])


class OutputOnly(nn.Module):
    """torchvision's segmentation models return a dict; this returns its "out" logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images)["out"]


def build_segmenter(kind, pretrained=True, input_size=SEG_INPUT_SIZE):
    """A torch segmenter from (B, 3, H, W) normalised images to (B, 1, H, W) logits."""
    if kind == MASKHEAD:
        return MaskHeadSegmenter(pretrained=pretrained, input_size=input_size)
    if kind == LRASPP:
        from torchvision.models import MobileNet_V3_Large_Weights
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large

        backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        return OutputOnly(lraspp_mobilenet_v3_large(weights=None, weights_backbone=backbone, num_classes=1))
    if kind == UNET:
        import segmentation_models_pytorch as smp

        return smp.Unet("mobilenet_v2", encoder_weights="imagenet" if pretrained else None, classes=1)
    raise ValueError(f"unknown torch segmenter kind {kind!r}; expected one of {TORCH_KINDS}")


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


@dataclass
class Segmenter:
    """A trained segmenter: `net` is a torch module, or an Ultralytics YOLO for yolo26n-seg."""
    kind: str
    net: object
    size: int = SEG_INPUT_SIZE
    device: str = "cpu"

    def probabilities(self, canvases):
        """(N, size, size) float32 snake probabilities for RGB uint8 canvases of size x size."""
        if self.kind == YOLO_SEG:  # the union of the instance masks YOLO keeps at YOLO_SEG_CONF
            results = self.net.predict([cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR) for canvas in canvases],
                                       imgsz=self.size, conf=YOLO_SEG_CONF, retina_masks=True, verbose=False,
                                       device=self.device)
            union = np.zeros((len(canvases), self.size, self.size), np.float32)
            for index, result in enumerate(results):
                if result.masks is not None and len(result.masks.data):
                    union[index] = result.masks.data.float().amax(dim=0).cpu().numpy()
            return union
        device = torch.device(self.device)
        images = torch.stack([AUGMENT["eval"](image=canvas)["image"] for canvas in canvases]).to(device)
        self.net.eval()
        with torch.inference_mode(), torch.autocast(device.type, dtype=AMP_DTYPE, enabled=device.type == "cuda"):
            logits = self.net(images)
        return torch.sigmoid(logits.float())[:, 0].cpu().numpy()


def load_segmenter(kind, weights, device=None):
    """A Segmenter from train_segmenter's weights: a torch checkpoint, or YOLO's own .pt for yolo26n-seg."""
    device = str(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if kind == YOLO_SEG:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        return Segmenter(kind, model, int(model.overrides.get("imgsz", SEG_INPUT_SIZE)), device)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    net = build_segmenter(kind, pretrained=False, input_size=checkpoint["input_size"])
    net.load_state_dict(checkpoint["state_dict"])
    return Segmenter(kind, net.to(device).eval(), checkpoint["input_size"], device)


def unletterbox(canvas_map, width, height, size=IMG_SIZE):
    """A map over prepare's size x size canvas, cut to the photo's window and resized (bilinear) to the photo."""
    scale = size / max(height, width)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    top, left = (size - resized_height) // 2, (size - resized_width) // 2
    window = np.asarray(canvas_map, dtype=np.float32)[top:top + resized_height, left:left + resized_width]
    return cv2.resize(window, (width, height), interpolation=cv2.INTER_LINEAR)


def _canvas_batches(images, size, batch_size):
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        yield chunk, [prepare(rgb, None, MASK_MODE, size)[0] for rgb in chunk]


def predict_masks(segmenter, images, threshold=MASK_THRESHOLD, batch_size=SEG_PREDICT_BATCH):
    """0/1 uint8 snake masks at each RGB photo's own resolution; a photo with no snake gets an all-zero mask."""
    masks = []
    for chunk, canvases in _canvas_batches(images, segmenter.size, batch_size):
        masks += [(unletterbox(probability, rgb.shape[1], rgb.shape[0], segmenter.size) > threshold).astype(np.uint8)
                  for probability, rgb in zip(segmenter.probabilities(canvases), chunk)]
    return masks


def top_instance_confidences(segmenter, images, batch_size=SEG_PREDICT_BATCH):
    """YOLO26n-seg's top instance confidence per photo (0 below YOLO_TOP_CONF_FLOOR): it tells a confidence just
    under YOLO_SEG_CONF from a model that finds nothing."""
    tops = []
    for _, canvases in _canvas_batches(images, segmenter.size, batch_size):
        results = segmenter.net.predict([cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR) for canvas in canvases],
                                        imgsz=segmenter.size, conf=YOLO_TOP_CONF_FLOOR, max_det=1, verbose=False,
                                        device=segmenter.device)
        for result in results:
            confidences = result.boxes.conf
            confidences = np.asarray(confidences.cpu() if hasattr(confidences, "cpu") else confidences, dtype=float)
            tops.append(float(confidences.max()) if confidences.size else 0.0)
    return np.array(tops, dtype=float)


def seg_train_step(net, batch, optimizer, scaler, device):
    """One Dice+BCE step over the batch's photos with a valid mask; returns the loss."""
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    valid = batch["mask_valid"].to(device, non_blocking=True)
    with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=scaler.is_enabled()):
        logits = net(images)
    loss = dice_bce(logits.float(), masks, valid)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


def train_torch_segmenter(kind, frame, out_dir, epochs=SEG_EPOCHS, batch_size=SEG_BATCH, lr=SEG_LR,
                          weight_decay=SEG_WEIGHT_DECAY, aug_strength=SEG_AUG_STRENGTH, size=SEG_INPUT_SIZE,
                          device=None, num_workers=0, pretrained=True, seed=SEED, record=None):
    """Trains a torch segmenter (AdamW, one warm-up epoch then cosine, Dice+BCE) and saves the last epoch, as there is
    no validation to select on; writes segmenter.pt and run.json and returns the checkpoint path."""
    started = time.perf_counter()
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if amp:
        start_cuda(device)
    seed_everything(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # every training photo takes notebook 06's tier-1 pipeline, whatever its class's tier
    dataset = SnakeDataset(frame.assign(tier="tier1"), train=True, mode=MASK_MODE, size=size,
                           aug_strength=aug_strength)
    seed_augmentation(dataset, seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers,
                        pin_memory=amp, worker_init_fn=init_worker, generator=torch.Generator().manual_seed(seed),
                        persistent_workers=num_workers > 0)
    net = build_segmenter(kind, pretrained=pretrained, input_size=size).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(len(loader), epochs * len(loader)))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    epoch_loss, epoch_seconds = [], []
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_started, losses = time.perf_counter(), []
        for batch in loader:
            losses.append(seg_train_step(net, batch, optimizer, scaler, device))
            scheduler.step()
        epoch_loss.append(float(np.mean(losses)))
        epoch_seconds.append(time.perf_counter() - epoch_started)
        print(f"{kind} epoch {epoch}/{epochs}: loss {epoch_loss[-1]:.4f}, {epoch_seconds[-1]:.0f} s", flush=True)

    weights = out_dir / SEGMENTER_FILE
    torch.save({"kind": kind, "input_size": size, "state_dict": {name: value.detach().cpu() for name, value in
                                                                 net.state_dict().items()}}, weights)
    write_json(out_dir / RUN_FILE, {
        **(record or {}), "kind": kind, "weights": SEGMENTER_FILE, "parameters": parameter_count(net),
        "train_photos": len(frame), "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "aug_strength": aug_strength, "input_size": size, "pretrained": pretrained,
        "seed": seed, "device": str(device), "epoch_loss": epoch_loss, "epoch_seconds": epoch_seconds,
        "seconds": time.perf_counter() - started})
    return weights


def read_run(out_dir):
    return json.loads((Path(out_dir) / RUN_FILE).read_text())
