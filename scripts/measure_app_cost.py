"""Laptop CPU time per frame of the M3 front end that A3 needs: the detector, and the U-Net mask.

Usage: measure.py <src with front_end.py> <out json>. Imports torch before ultralytics, as notebook 16 (the app) does,
so torch keeps its default thread count; the run is repeated on one thread (milestone 3's X1 convention).
Photos: 50 val photos of the local 300-photo sample, drawn with seed 42, loaded with the app's exposure fix.
"""
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.update(YOLO_VERBOSE="False", YOLO_OFFLINE="1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

sys.path.insert(0, sys.argv[1])
from serpentguard import front_end  # noqa: E402
from serpentguard.data import load_rgb  # noqa: E402

WORK = Path(__file__).resolve().parents[4]
WEIGHTS = WORK / "kaggle/m4_runs/frontend_full/weights_ds/dl/frontends"
DETECTOR = WEIGHTS / "D1/yolo26n-ft/detector.pt"
UNET = WEIGHTS / "S1/unet"
PHOTOS = WORK / "m3-src-goldens/tiny/data/val"
N_PHOTOS, WARM_UP, SEED = 50, 5, 42
CPU = os.environ.get("CPU_NAME")  # set by hand where sysctl is not available
DETECTOR_CONF = 0.05  # the app's find_snake


def percentiles(values):
    ms = np.asarray(values) * 1000
    return {"median_ms": round(float(np.median(ms)), 2), "p90_ms": round(float(np.percentile(ms, 90)), 2),
            "mean_ms": round(float(ms.mean()), 2)}


def masks_fp32(segmenter, rgb):
    # Segmenter.masks without its float16 autocast: the same letterbox, net, unletterbox and threshold
    canvas = front_end.apply_mode(rgb, None, "full", None, segmenter.size)[0]
    image = front_end.AUGMENT["eval"](image=canvas)["image"][None]
    with torch.inference_mode():
        prob = torch.sigmoid(segmenter.net(image).float())[0, 0].numpy()
    return (front_end.unletterbox(prob, rgb.shape[1], rgb.shape[0], segmenter.size) > 0.5).astype(np.uint8)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def timed(threads, detector, snake, segmenter, photos):
    torch.set_num_threads(threads)
    det, seg, seg32, ious = [], [], [], []
    for index, rgb in enumerate(photos[:WARM_UP] + photos):
        t0 = time.perf_counter()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        detector.predict(bgr, conf=DETECTOR_CONF, classes=snake, rect=False, verbose=False)
        t1 = time.perf_counter()
        mask = segmenter.masks([rgb])[0]
        t2 = time.perf_counter()
        mask32 = masks_fp32(segmenter, rgb)
        t3 = time.perf_counter()
        assert mask.shape == mask32.shape == rgb.shape[:2]
        if index >= WARM_UP:
            det.append(t1 - t0)
            seg.append(t2 - t1)
            seg32.append(t3 - t2)
            ious.append(iou(mask, mask32))
    det, seg, seg32 = (np.asarray(v) for v in (det, seg, seg32))
    return {"torch_threads": torch.get_num_threads(), "detector": percentiles(det),
            "unet_mask_as_front_end_runs_it": percentiles(seg),
            "detector_plus_unet_as_front_end_runs_it": percentiles(det + seg),
            "unet_mask_fp32": percentiles(seg32), "detector_plus_unet_fp32": percentiles(det + seg32),
            "unet_share_fp32_median": round(float(np.median(seg32 / (det + seg32))), 3),
            "mask_iou_fp32_vs_autocast": {"median": round(float(np.median(ious)), 4),
                                          "min": round(float(np.min(ious)), 4)}}


def cpu_name():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor()


default_threads = torch.get_num_threads()
files = sorted(PHOTOS.rglob("*.jpg"))
random.Random(SEED).shuffle(files)
photos = [load_rgb(path) for path in files[:N_PHOTOS]]
detector = YOLO(str(DETECTOR))
snake = [i for i, name in detector.names.items() if str(name).strip().lower() == "snake"]
segmenter = front_end.Segmenter("unet", UNET)
runs = [timed(threads, detector, snake, segmenter, photos) for threads in (default_threads, 1)]
report = {
    "what": "CPU time per frame of the M3 front end A3 needs in the app: the detector (no-snake gate and box), "
            "and the U-Net mask for the crop to the mask's box; detector_plus_unet is one frame through both "
            "(the sum of the two timed steps of that frame)",
    "photos": len(photos), "photo_source": "local 300-photo val sample (m3-src-goldens/tiny/data/val), "
                                           f"{N_PHOTOS} drawn with seed {SEED}, app exposure fix applied",
    "photo_sizes_px": {"min_side": int(min(min(p.shape[:2]) for p in photos)),
                       "max_side": int(max(max(p.shape[:2]) for p in photos))},
    "warm_up_frames": WARM_UP, "batch": 1,
    "detector": {"weights": str(DETECTOR.relative_to(WORK)), "conf": DETECTOR_CONF,
                 "imgsz": int(detector.overrides.get("imgsz", 640)), "call": "predict(bgr, rect=False)"},
    "unet": {"weights": str((UNET / "segmenter.pt").relative_to(WORK)), "input_size": segmenter.size,
             "call": "front_end.Segmenter('unet').masks([rgb]): letterbox, U-Net, unletterbox, threshold 0.5",
             "precision": "two ways: as front_end.Segmenter runs it (torch.autocast('cpu', float16), "
                          "meant for the GPU) and plain float32, the same pre- and post-processing"},
    "machine": {"cpu": CPU or cpu_name(), "cores": os.cpu_count(), "platform": platform.platform(),
                "python": platform.python_version(), "torch": torch.__version__},
    "threads_note": "ultralytics' first predict call sets torch to min(8, cores - 1) threads (select_device), so in the app the U-Net runs on 8 threads here after the detector's first frame; the first run is that, the second is pinned to one thread (milestone 3's X1 convention)",
    "runs": runs}
Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(runs, indent=2))
