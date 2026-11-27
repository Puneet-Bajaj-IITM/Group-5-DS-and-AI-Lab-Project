"""fp16 detector and U-Net for the app: convert, then compare with fp32 on validation snake photos.

Each photo goes through the app's front end twice (fp32 and fp16 detector + U-Net); the classifier stays fp32,
so any change in the answer comes from the front end. Validation only; the test split is not read.
Control: the fp16 outputs paired with a different photo's fp32 outputs must disagree, or the check cannot fail.
Usage: python results/m5/robustness/fp16_front_end/fp16_front_end.py --photos <validation snake photos>
(the models in app/assets/models come from app/tool/m5/build_assets.py; the fp16 files are written here, not in git)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.transformers.float16 import convert_float_to_float16

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
APP = REPO / "app"
sys.path.insert(0, str(APP / "tool/m5"))
import reference_m5 as m5  # noqa: E402

ref = m5.ref
SAMPLE, SEED = 1000, 42
MODELS = ("detector", "segmenter")
LATENCY_RUNS = 50


def convert(models_dir, out_dir):
    sizes = {}
    for name in MODELS:
        src, dst = models_dir / f"{name}.onnx", out_dir / f"{name}_fp16.onnx"
        onnx.save(convert_float_to_float16(onnx.load(str(src)), keep_io_types=True), str(dst))
        sizes[name] = {"fp32_mb": round(src.stat().st_size / 1e6, 1), "fp16_mb": round(dst.stat().st_size / 1e6, 1)}
    return sizes


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads, options.log_severity_level = 1, 1, 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def front_end(pipe, rgb):
    gate, mask = pipe.gate(rgb)
    height, width = rgb.shape[:2]
    window = ref.crop_window(ref.clean_mask(mask, pipe.settings["mask_cleanup"]), height, width,
                             pipe.settings["crop"]["padding"])
    crop = rgb if window is None else rgb[window[0]:window[1], window[2]:window[3]]
    answer = ref.verdict(pipe.classify(crop), pipe.thresholds, pipe.tables)
    return gate, mask.astype(bool), window, answer


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def latency(sess, size):
    x = np.random.default_rng(SEED).standard_normal((1, 3, size, size)).astype(np.float32)
    sess.run(None, {"images": x})
    times = []
    for _ in range(LATENCY_RUNS):
        t = time.perf_counter()
        sess.run(None, {"images": x})
        times.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(times)), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", type=Path, required=True)
    args = parser.parse_args()
    models_dir = APP / "assets/models"
    sizes = convert(models_dir, HERE)
    fp32, fp16 = m5.AppPipeline(), m5.AppPipeline()
    fp16.detector, fp16.segmenter = session(HERE / "detector_fp16.onnx"), session(HERE / "segmenter_fp16.onnx")
    photos = sorted(p for p in args.photos.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    photos = random.Random(SEED).sample(photos, min(SAMPLE, len(photos)))
    exposure = fp32.settings["exposure_fix"]
    rows = []
    for i, path in enumerate(photos):
        rgb = ref.load_photo(path, exposure)
        a, b = front_end(fp32, rgb), front_end(fp16, rgb)
        rows.append({
            "photo": str(path.relative_to(args.photos)),
            "box_conf_32": a[0]["box_conf"], "box_conf_16": b[0]["box_conf"],
            "area_32": a[0]["mask_area"], "area_16": b[0]["mask_area"],
            "gate_32": a[0]["snake"], "gate_16": b[0]["snake"],
            "mask_iou": iou(a[1], b[1]), "window_same": a[2] == b[2],
            "verdict_32": a[3]["verdict"], "verdict_16": b[3]["verdict"],
            "top1_32": a[3]["top3"][0]["species"], "top1_16": b[3]["top3"][0]["species"],
            "mask_32": a[1], "mask_16": b[1],
        })
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(photos)}", flush=True)
    n = len(rows)
    shift = [rows[(i + 1) % n] for i in range(n)]
    control_iou = [iou(r["mask_16"], s["mask_32"]) if r["mask_16"].shape == s["mask_32"].shape else 0.0
                   for r, s in zip(rows, shift)]
    conf_diff = [abs(r["box_conf_32"] - r["box_conf_16"]) for r in rows
                 if r["box_conf_32"] is not None and r["box_conf_16"] is not None]
    size = fp32.settings["input_size"]
    summary = {
        "photos": n, "seed": SEED, "split": "validation snakes", "sizes": sizes,
        "box_conf_none_mismatch": sum((r["box_conf_32"] is None) != (r["box_conf_16"] is None) for r in rows),
        "box_conf_max_abs_diff": max(conf_diff), "box_conf_mean_abs_diff": float(np.mean(conf_diff)),
        "mask_area_max_abs_diff": max(abs(r["area_32"] - r["area_16"]) for r in rows),
        "mask_iou_mean": float(np.mean([r["mask_iou"] for r in rows])),
        "mask_iou_min": min(r["mask_iou"] for r in rows),
        "m4_gate_flips": sum(r["gate_32"] != r["gate_16"] for r in rows),
        "crop_window_changed": sum(not r["window_same"] for r in rows),
        "verdict_changed": sum(r["verdict_32"] != r["verdict_16"] for r in rows),
        "top1_changed": sum(r["top1_32"] != r["top1_16"] for r in rows),
        "control_shuffled_mask_iou_mean": float(np.mean(control_iou)),
        "latency_ms_laptop_1_thread": {
            name: {"fp32": latency(session(models_dir / f"{name}.onnx"), size),
                   "fp16": latency(session(HERE / f"{name}_fp16.onnx"), size)} for name in MODELS},
    }
    (HERE / "fp16_front_end.json").write_text(json.dumps(summary, indent=1))
    keep = [{k: v for k, v in r.items() if not k.startswith("mask_3") and not k.startswith("mask_1")} for r in rows]
    (HERE / "fp16_front_end_rows.json").write_text(json.dumps(keep))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
