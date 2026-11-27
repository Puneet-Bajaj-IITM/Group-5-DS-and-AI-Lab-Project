"""Laptop CPU latency of the ONNX models, one photo at a time, as the app runs them (batch 1).

Usage: python -I time_onnx.py --models <onnx> [<onnx> ...] --threads 1 2 --out <json>
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

WARMUP, RUNS = 10, 100


def time_model(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads, options.log_severity_level = threads, 1, 3
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    size = session.get_inputs()[0].shape[2]
    x = np.random.default_rng(0).standard_normal((1, 3, size, size)).astype(np.float32)
    for _ in range(WARMUP):
        session.run(None, {"images": x})
    times = []
    for _ in range(RUNS):
        started = time.perf_counter()
        session.run(None, {"images": x})
        times.append((time.perf_counter() - started) * 1000)
    return {"model": Path(path).name, "threads": threads, "input_size": size,
            "median_ms": round(float(np.median(times)), 2), "p90_ms": round(float(np.percentile(times, 90)), 2)}


def main(args):
    rows = [time_model(m, t) for m in args.models for t in args.threads]
    report = {"machine": platform.platform(), "processor": platform.processor(), "onnxruntime": ort.__version__,
              "runs": RUNS, "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
