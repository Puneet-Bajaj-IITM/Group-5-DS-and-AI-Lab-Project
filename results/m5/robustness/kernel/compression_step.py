"""int8 post-training quantization of the frozen classifier and snake check, scored on CPU (onnxruntime).

Classifier: the app's fp32 ONNX export of M5_KD (g3_app_new.py; parity with best.pt checked when staged), its fp16
copy (convert_float_to_float16, keep_io_types, as the app's classifier_fp16.onnx), dynamic int8 (weights int8,
activations quantized at run time) and static int8 (QDQ, per-channel int8 weights, uint8 activations) calibrated on
500 calibration-split crops (MinMax). Inputs are the
training-path crops of the clean pass (the same tensors the locked test run's PyTorch pass saw), plain and mirrored.
Snake check: its fp32 and fp16 ONNX (frozen), dynamic int8 and static int8 calibrated on 500 calibration snakes,
inputs through the app's path (reference_pipeline.load_photo, letterbox 256, normalise). Every session runs on the
CPU execution provider. Nothing here is fitted to a test photo.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CLASSIFIER_SIZE, SNAKE_SIZE = 320, 256
SNAKE_BATCH = 32
CPU_THREADS = 4
PERCENTILE = 99.999
PARITY_PHOTOS = 64


def session(path, threads=CPU_THREADS):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads, options.log_severity_level = threads, 1, 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class Reader:
    """onnxruntime CalibrationDataReader over a list of (1, 3, S, S) float32 arrays."""

    def __init__(self, arrays):
        self.arrays, self.index = arrays, 0

    def get_next(self):
        if self.index >= len(self.arrays):
            return None
        self.index += 1
        return {"images": self.arrays[self.index - 1]}

    def rewind(self):
        self.index = 0


def quantize_all(fp32, folder, stem, calibration, log, failures):
    """{variant: path} for fp16, int8 dynamic, int8 static MinMax and Percentile."""
    import onnx
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType, quantize_dynamic,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from onnxruntime.transformers.float16 import convert_float_to_float16
    import shutil
    local = folder / f"{stem}_fp32.onnx"  # onnxruntime writes helper files beside its input; /kaggle/input is read-only
    shutil.copy2(fp32, local)
    fp32 = local
    paths = {"fp32": local}
    paths["fp16"] = folder / f"{stem}_fp16.onnx"
    if not paths["fp16"].exists():
        onnx.save(convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(paths["fp16"]))
    paths["int8_dynamic"] = folder / f"{stem}_int8_dynamic.onnx"
    quantize_dynamic(str(fp32), str(paths["int8_dynamic"]), weight_type=QuantType.QInt8)
    pre = folder / f"{stem}_pre.onnx"
    quant_pre_process(str(fp32), str(pre))
    # Percentile calibration was dropped after v3: onnxruntime keeps every activation of all 500 crops for its
    # histograms and the kernel died (no traceback) while calibrating the classifier; it also failed on the laptop.
    for name, method, extra in (("int8_static_minmax", CalibrationMethod.MinMax, {}),):
        started = time.time()
        target = folder / f"{stem}_{name}.onnx"
        try:
            quantize_static(str(pre), str(target), Reader(calibration), quant_format=QuantFormat.QDQ,
                            per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                            calibrate_method=method, extra_options=extra)
        except ValueError as error:  # onnxruntime's histogram fails on a tensor that never varies
            failures[f"{stem} {name}"] = f"{type(error).__name__}: {error}"
            log(f"{stem} {name} not produced: {error}")
            continue
        paths[name] = target
        log(f"{stem} {name}: {time.time() - started:.0f} s")
    pre.unlink()
    return paths


def classifier_inputs(ctx, frame):
    """Batches of the frame's training-path input tensors (numpy, N x 3 x 320 x 320), in frame order."""
    cfg, _, _ = ctx["load_checkpoint"](ctx["m5_dir"], "cpu")
    for batch in ctx["data"].eval_loader(cfg, frame, num_workers=ctx["loader_workers"]):
        yield batch["image"].numpy()


def run_classifier(sessions, batches, np):
    """{variant: (plain logits, mirrored logits)}; the ONNX graph takes one photo at a time."""
    plain = {k: [] for k in sessions}
    mirrored = {k: [] for k in sessions}
    for images in batches:
        for x in images:
            one, flipped = x[None], np.ascontiguousarray(x[None, :, :, ::-1])
            for name, s in sessions.items():
                plain[name].append(s.run(["species_logits"], {"images": one})[0][0])
                mirrored[name].append(s.run(["species_logits"], {"images": flipped})[0][0])
    return {k: (np.stack(plain[k]), np.stack(mirrored[k])) for k in sessions}


def snake_inputs(ctx, files):
    ref = ctx["ref"]
    settings = __import__("json").loads((ctx["inputs"] / "app_models/pipeline.json").read_text())

    def one(file):
        rgb = ref.load_photo(Path(file), settings["exposure_fix"])
        return ref.normalise(ref.letterbox(rgb, SNAKE_SIZE, settings["letterbox"]["pad_rgb"]), settings["normalise"])

    chunks = [files[i:i + SNAKE_BATCH] for i in range(0, len(files), SNAKE_BATCH)]
    with ThreadPoolExecutor(CPU_THREADS) as pool:
        for chunk in chunks:
            yield ctx["np"].concatenate(list(pool.map(one, chunk)))


def run_snake(sessions, batches, np):
    probs = {k: [] for k in sessions}
    for x in batches:
        for name, s in sessions.items():
            probs[name].append(s.run(["probs"], {"images": x})[0])
    return {k: np.concatenate(v) for k, v in probs.items()}


def run(ctx, calib_frame, parity_frame, slice_frame):
    np, pd, log, inputs = ctx["np"], ctx["pd"], ctx["log"], ctx["inputs"]
    out = ctx["out"] / "compression"
    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    report = {"photos": {"calibration_crops": len(calib_frame), "parity": len(parity_frame),
                         "g3_slice": len(slice_frame)}}

    calibration = [x[None].astype(np.float32) for batch in classifier_inputs(ctx, calib_frame) for x in batch]
    failures = report["not_produced"] = {}
    classifier = quantize_all(inputs / "onnx/classifier.onnx", models, "classifier", calibration, log, failures)
    data_root = ctx["data_root"]
    snake_calib = [str(data_root / "data" / p) for p in ctx["lists"]["snake_calib"]["path"]]
    snake_cal = [x[None] for batch in snake_inputs(ctx, snake_calib) for x in batch]
    snake_paths = quantize_all(inputs / "onnx/snake_check.onnx", models, "snake_check", snake_cal, log, failures)
    snake_paths["fp16"] = inputs / "onnx/snake_check_fp16.onnx"  # the frozen fp16 file, not a new conversion
    report["bytes"] = {f"classifier {k}": Path(p).stat().st_size for k, p in classifier.items()}
    report["bytes"].update({f"snake_check {k}": Path(p).stat().st_size for k, p in snake_paths.items()})

    sessions = {k: session(p) for k, p in classifier.items()}
    parity_x = [x for batch in classifier_inputs(ctx, parity_frame) for x in batch][:PARITY_PHOTOS]
    parity = run_classifier(sessions, [np.stack(parity_x)], np)
    np.savez_compressed(out / "parity_classifier.npz", inputs=np.stack(parity_x).astype(np.float32),
                        paths=parity_frame["path"].to_numpy()[:len(parity_x)],
                        **{f"{k}_plain": v[0] for k, v in parity.items()})
    started = time.time()
    logits = run_classifier(sessions, classifier_inputs(ctx, slice_frame), np)
    report["seconds_classifier_g3_slice"] = round(time.time() - started)
    np.savez_compressed(out / "classifier_g3_slice.npz", paths=slice_frame["path"].to_numpy(),
                        labels=slice_frame["class_id"].to_numpy(),
                        **{f"{k}_{side}": v[i].astype(np.float32) for k, v in logits.items()
                           for i, side in enumerate(("plain", "flip"))})
    log("compression: classifier variants scored on the G3 slice")

    sessions = {k: session(p) for k, p in snake_paths.items()}
    parity_files = [str(data_root / "data" / p) for p in parity_frame["path"]][:PARITY_PHOTOS]
    parity_snake_x = np.concatenate(list(snake_inputs(ctx, parity_files)))
    parity_snake = run_snake(sessions, [parity_snake_x], np)
    np.savez_compressed(out / "parity_snake.npz", inputs=parity_snake_x, **parity_snake)
    held, non = ctx["held_out"], ctx["non"]
    sets = {"held_out": (held["path"], [str(data_root / "data" / p) for p in held["path"]]),
            "nonsnake_test": (non["path"], [str(inputs / "nonsnake_test" / p) for p in non["path"]])}
    tables = []
    for name, (paths, files) in sets.items():
        probs = run_snake(sessions, snake_inputs(ctx, files), np)
        table = pd.DataFrame({"set": name, "path": list(paths)})
        for variant, p in probs.items():
            table[f"p_snake_{variant}"] = p[:, 0]
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(out / "snake_check_variants.csv", index=False)
    import onnxruntime
    report["onnxruntime"] = onnxruntime.__version__
    report["cpu"] = open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(" :\t") if Path(
        "/proc/cpuinfo").exists() else "unknown"
    ctx["status"]["checks"]["compression"] = report
    log("compression", report)
