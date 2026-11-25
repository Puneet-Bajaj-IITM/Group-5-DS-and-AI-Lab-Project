"""Laptop CPU latency and laptop-vs-kernel parity of the compressed models (section 9).

Latency: onnxruntime CPU execution provider, one thread (intra and inter op), batch 1, WARMUP runs then the median
(and 90th percentile) of RUNS timed runs on a fixed random input. Parity: the 64 validation inputs the kernel saved,
run here, against the kernel's outputs for the same file (int8 kernels can differ between x86 and ARM).

Usage: m3-final/.venv/bin/python results/m5/m5_extra/latency.py
"""
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
COMPRESSION = HERE / "kernel_out/vikas25s/out/compression"
MODELS = COMPRESSION / "models"
FROZEN_ONNX = HERE / "kernel/inputs/onnx"
OUT = HERE / "results"
WARMUP, RUNS, THREADS = 20, 200, 1
VARIANTS = ("fp32", "fp16", "int8_dynamic", "int8_static_minmax")


def model_files():
    files = {}
    for variant in VARIANTS:
        files[("classifier", variant)] = (FROZEN_ONNX / "classifier.onnx" if variant == "fp32"
                                          else MODELS / f"classifier_{variant}.onnx")
        files[("snake_check", variant)] = (FROZEN_ONNX / f"snake_check{'' if variant == 'fp32' else '_fp16'}.onnx"
                                           if variant in ("fp32", "fp16") else MODELS / f"snake_check_{variant}.onnx")
    return {k: v for k, v in files.items() if v.exists()}


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads, options.log_severity_level = THREADS, 1, 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def latency(path):
    s = session(path)
    size = s.get_inputs()[0].shape[2]
    x = np.random.default_rng(0).standard_normal((1, 3, size, size)).astype(np.float32)
    for _ in range(WARMUP):
        s.run(None, {"images": x})
    times = []
    for _ in range(RUNS):
        started = time.perf_counter()
        s.run(None, {"images": x})
        times.append((time.perf_counter() - started) * 1000)
    return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90)),
            "mb": path.stat().st_size / 1e6, "input_size": size}


def parity(files):
    out = {}
    cls = np.load(COMPRESSION / "parity_classifier.npz", allow_pickle=True)
    snake = np.load(COMPRESSION / "parity_snake.npz", allow_pickle=True)
    for (model, variant), path in files.items():
        s = session(path)
        if model == "classifier":
            mine = np.stack([s.run(["species_logits"], {"images": x[None]})[0][0] for x in cls["inputs"]])
            theirs = cls[f"{variant}_plain"]
            top_same = float((mine.argmax(1) == theirs.argmax(1)).mean())
        else:
            mine = s.run(["probs"], {"images": snake["inputs"]})[0]
            theirs = snake[variant]
            top_same = float((mine.argmax(1) == theirs.argmax(1)).mean())
        out[f"{model} {variant}"] = {"photos": len(mine), "max_abs_diff": float(np.abs(mine - theirs).max()),
                                     "same_top_class": top_same}
    return out


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return platform.processor()


def main():
    OUT.mkdir(exist_ok=True)
    files = model_files()
    report = {"machine": {"cpu": chip() or platform.machine(), "platform": platform.platform(),
                          "onnxruntime": ort.__version__, "threads": THREADS, "warmup": WARMUP, "runs": RUNS},
              "latency": {}, "parity_laptop_vs_kernel": parity(files)}
    for (model, variant), path in files.items():
        report["latency"][f"{model} {variant}"] = latency(path)
        print(model, variant, report["latency"][f"{model} {variant}"])
    (OUT / "latency.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["parity_laptop_vs_kernel"], indent=1))


if __name__ == "__main__":
    main()
