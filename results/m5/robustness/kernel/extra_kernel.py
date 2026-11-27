"""SerpentGuard M5 extra evaluation kernel: the frozen pipeline on corrupted held-out photos, Grad-CAM, int8.

Evaluation only: every model, threshold and rule is FREEZE.md's (m5-results/test). Per condition the held-out photos
(test + test_extra, 18,533) and the 600 non-snake test photos are decoded as every pipeline loader decodes them
(PIL, RGB), corrupted (corrupt.py), and written under the same names to a scratch copy of the dataset; then the
same code paths as the locked test kernel run on that copy: the M4 front end from an empty base, M5_KD plain and
mirrored (predict / save_outputs), the snake check (the G4 kernel's own score), and the app's gate (the reference
pipeline's gate, its two ONNX models on the GPU). The clean condition goes through the same write-and-read path and
must reproduce the locked test run's outputs on a 600-photo sample before anything else runs.

The job (which conditions, Grad-CAM, int8) comes from job.json in the bundle; one kernel per account.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = json.loads(Path(os.environ["M5_EXTRA_LOCAL"]).read_text()) if os.environ.get("M5_EXTRA_LOCAL") else None
OUT = Path(LOCAL["out"]) if LOCAL else Path("/kaggle/working/out")
SCRATCH = Path(LOCAL["scratch"]) if LOCAL else Path("/kaggle/tmp/m5_extra")
REPO = Path(LOCAL["repo"]) if LOCAL else HERE
JOB = json.loads((HERE / "job.json").read_text()) if not LOCAL else LOCAL["job"]
HELD_OUT_SPLITS = ("test", "test_extra")
HELD_OUT_PHOTOS = LOCAL["held_out_photos"] if LOCAL else 18_533
CLEAN_DATASET = "puneetbajajiitm/serpentguard-indian-snakes-clean"
SNAKE_CUT = 0.20484977503483762
WRITE_WORKERS = 4
LOADER_WORKERS = 0 if LOCAL else 4
GATE_THREADS = 4
CHECK_FLOORS = {"top1_agreement": 0.98, "gate_agreement": 0.99, "presence_agreement": 0.98, "snake_flips": 2}
STATUS = {"job": JOB, "passed": [], "failed": {}, "checks": {}, "seconds": {}}
STARTED = time.time()


def log(*parts):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - STARTED) / 60:.0f} min]", *parts, flush=True)


def pip_install():
    def pip(*args):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)
    pip("timm==1.0.31", "albumentations==2.0.8", "scikit-learn==1.9.1")
    pip("ultralytics==8.4.202", "segmentation-models-pytorch==0.6.0")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "onnxruntime"])
    pip("onnx==1.23.2", "onnxruntime-gpu==1.31.1")
    os.environ.update(NO_ALBUMENTATIONS_UPDATE="1", YOLO_VERBOSE="False", TQDM_DISABLE="1")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status():
    STATUS["minutes"] = round((time.time() - STARTED) / 60, 1)
    (OUT / "status.json").write_text(json.dumps(STATUS, indent=1, default=str) + "\n")


def step(name, function, *args):
    started = time.time()
    log(f"step: {name}")
    try:
        result = function(*args)
        STATUS["passed"].append(name)
        return result
    except Exception as error:
        STATUS["failed"][name] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        return None
    finally:
        STATUS["seconds"][name] = round(time.time() - started)
        write_status()


def _write_one(item):
    import corrupt
    source, target, condition, path = item
    corrupt.write_corrupted(source, target, condition, path)


def write_condition(condition, photos, root):
    """photos: [(path, source file, relative target)] -> the corrupted copies under root."""
    items = []
    for path, source, relative in photos:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        items.append((str(source), str(target), condition, path))
    if LOCAL:
        for item in items:
            _write_one(item)
    else:
        with Pool(WRITE_WORKERS) as pool:
            pool.map(_write_one, items, chunksize=64)


def main():
    if not LOCAL:
        pip_install()
    import numpy as np
    import pandas as pd
    import torch

    sys.path[:0] = [str(REPO / "src"), str(HERE)] + ([str(HERE.parent)] if LOCAL else [])
    from serpentguard import data, decide, front_end, metrics
    from serpentguard.train import load_checkpoint, predict, save_outputs

    for folder in (OUT, SCRATCH):
        folder.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{torch.cuda.device_count()} GPU(s); job {JOB}")

    def dataset(handle):
        if LOCAL:
            return Path(LOCAL["datasets"][handle])
        import kagglehub
        return Path(kagglehub.dataset_download(handle))

    data_root = dataset(CLEAN_DATASET)
    inputs = next(dataset(JOB["inputs_dataset"]).rglob("manifest.json")).parent
    detector = inputs / "m4_front_end/detector/M4_detector.pt"
    segmenter = inputs / "m4_front_end/segmenter/segmenter.pt"
    m5_dir = inputs / "m5_kd"

    def check_inputs():
        manifest = {k: v for k, v in json.loads((inputs / "manifest.json").read_text()).items() if "__pycache__" not in k}
        bad = [name for name, digest in manifest.items() if sha256(inputs / name) != digest]
        assert not bad, f"input files differ from the staged manifest: {bad[:5]}"
        STATUS["checks"]["inputs"] = {"manifest_files": len(manifest)}

    step("inputs", check_inputs)
    if STATUS["failed"]:
        raise SystemExit("inputs failed")

    metadata, classes = data.load_tables(data_root)
    held_out = metadata[metadata["split"].isin(HELD_OUT_SPLITS)].reset_index(drop=True)
    assert len(held_out) == HELD_OUT_PHOTOS, len(held_out)
    non = pd.read_csv(inputs / "nonsnake_test.csv")
    if LOCAL:
        non = non.head(LOCAL["extra_photos"])
    lists = {name: pd.read_csv(inputs / f"lists/{name}.csv") for name in (
        "g3_slice", "quant_calib", "parity_val", "snake_calib", "gradcam_photos", "gradcam_targets")}
    if LOCAL:  # the stand-in has a few photos only
        held = set(held_out["path"])
        for name in ("g3_slice", "gradcam_photos", "gradcam_targets"):
            lists[name] = lists[name][lists[name]["path"].isin(held)].reset_index(drop=True)
        for name in ("quant_calib", "parity_val", "snake_calib"):
            on_disk = [os.path.isfile(data_root / "data" / p) for p in lists[name]["path"]]
            lists[name] = lists[name][on_disk].head(LOCAL["extra_photos"]).reset_index(drop=True)
        if lists["g3_slice"].empty:
            lists["g3_slice"] = held_out[["path"]].head(8)
        if lists["gradcam_photos"].empty:
            lists["gradcam_photos"] = held_out[["path"]].head(4).assign(kind="correct", target_class=-1)
        lists["gradcam_targets"] = held_out[["path"]].assign(target_class=-1)
    extra_paths = list(lists["quant_calib"]["path"]) + list(lists["parity_val"]["path"])

    for module in (data, decide, metrics):
        module.assert_no_test_paths = lambda frame: None
    data.TEST_SPLITS = ()
    view_meta = metadata.assign(split=metadata["split"].replace(
        {"val": "val_original", "test": "val", "test_extra": "val"}))

    sys.path.insert(0, str(inputs / "app_models"))
    import reference_pipeline as ref

    os.environ.update(G4_SMOKE="1", G4_SNAKES=str(data_root / "data"), G4_OURS=str(inputs / "snake_check"),
                      G4_OUT=str(SCRATCH / "g4"), G4_VARIANT="distil_oi")
    spec = importlib.util.spec_from_file_location("train_g4", inputs / "snake_check/train_g4.py")
    g4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g4)
    g4.WORKERS = 0 if LOCAL else 4
    checkpoint = torch.load(inputs / "snake_check/model.pt", map_location="cpu", weights_only=False)
    snake_model = g4.Student(checkpoint["backbone"], checkpoint["state_dict"]["project.weight"].shape[0])
    snake_model.load_state_dict(checkpoint["state_dict"])
    snake_model = snake_model.to(g4.DEVICE)

    import onnxruntime as ort

    def gate_pipeline(use_gpu):
        pipeline = ref.Pipeline(inputs / "app_models", threads=1)
        if use_gpu:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            pipeline.detector = ort.InferenceSession(str(inputs / "app_models/detector.onnx"), options,
                                                     providers=providers)
            pipeline.segmenter = ort.InferenceSession(str(inputs / "app_models/segmenter.onnx"), options,
                                                      providers=providers)
            assert "CUDAExecutionProvider" in pipeline.detector.get_providers(), "CUDA provider not loaded"
        return pipeline

    def gate_rows(pipeline, items):
        def one(item):
            path, file = item
            call, _ = pipeline.gate(ref.load_photo(Path(file), pipeline.settings["exposure_fix"]))
            return {"path": path, "gate_snake": bool(call["snake"]),
                    "box_conf": float("nan") if call["box_conf"] is None else float(call["box_conf"]),
                    "mask_area": float(call["mask_area"])}
        with ThreadPoolExecutor(GATE_THREADS) as pool:
            return pd.DataFrame(list(pool.map(one, items)))

    def choose_gate():
        """GPU gate only if it gives the same calls as the saved CPU gate on the 300 validation photos."""
        saved = pd.read_csv(inputs / "instrument/gate_val_sample.csv").head(LOCAL["val_sample"] if LOCAL else None)
        items = [(p, str(data_root / "data" / p)) for p in saved["path"]]
        use_gpu = device.type == "cuda"
        chosen = gate_pipeline(use_gpu)
        rows = gate_rows(chosen, items)
        agreement = float((rows["gate_snake"].to_numpy() == saved["gate_snake"].to_numpy()).mean())
        STATUS["checks"]["gate_provider"] = {"gpu": use_gpu, "val_photos": len(saved), "call_agreement": agreement,
                                             "max_abs_mask_area_diff": float(np.abs(rows["mask_area"] -
                                                                                    saved["mask_area"]).max())}
        log("gate provider check", STATUS["checks"]["gate_provider"])
        if use_gpu and agreement < 1.0:
            STATUS["checks"]["gate_provider"]["fallback"] = "CPU"
            return gate_pipeline(False)
        return chosen

    pipeline = step("gate provider", choose_gate)

    class Mirrored(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            return self.model(torch.flip(images, dims=[3]))

    def run_condition(condition):
        root = SCRATCH / "c" / condition
        view, preds_dir, out = root / "view", root / "m4_preds", OUT / condition
        out.mkdir(parents=True, exist_ok=True)
        photos = [(p, data_root / "data" / p, Path("data") / p) for p in held_out["path"]]
        photos += [(p, inputs / "nonsnake_test" / p, Path("nonsnake") / p) for p in non["path"]]
        if condition == "clean":
            photos += [(p, data_root / "data" / p, Path("data") / p) for p in extra_paths]
        started = time.time()
        write_condition(condition, photos, view)
        shutil.copy2(data_root / "classes.csv", view / "classes.csv")
        view_meta.to_csv(view / "metadata.csv", index=False)
        log(f"{condition}: {len(photos):,} photos written in {time.time() - started:.0f} s")

        base = root / "empty_base"
        (base / front_end.MASKS_DIR).mkdir(parents=True, exist_ok=True)
        (base / front_end.PREDS_FILE).write_text(front_end.PREDS_COLUMNS)
        rows = pd.DataFrame({"path": list(held_out["path"]) + (extra_paths if condition == "clean" else [])})
        front_end.extend_predictions(rows, base, detector, segmenter, view, preds_dir)
        preds = pd.read_csv(preds_dir / front_end.PREDS_FILE)
        shutil.copy2(preds_dir / front_end.PREDS_FILE, out / front_end.PREDS_FILE)

        frame = data.build_frame(data.load_tables(view)[0], classes, ["val"], frontend_preds=preds,
                                 mask_source="predicted", preds_root=preds_dir)
        frame = data.with_files(frame, view)
        assert len(frame) == HELD_OUT_PHOTOS, len(frame)
        frame[["path", "class_id", "observer_id", "mask_valid"]].to_csv(out / "held_out_frame.csv", index=False)
        cfg, model, _ = load_checkpoint(m5_dir, device)
        for name, net in (("m5_kd", model), ("m5_kd_flip", Mirrored(model).eval())):
            result = predict(net, data.eval_loader(cfg, frame, num_workers=LOADER_WORKERS), device,
                             device.type == "cuda")
            save_outputs(out / f"{name}.npz", frame, result)
        log(f"{condition}: classifier outputs saved")

        snake_sets = {"held_out": (held_out["path"], [str(view / "data" / p) for p in held_out["path"]]),
                      "nonsnake_test": (non["path"], [str(view / "nonsnake" / p) for p in non["path"]])}
        tables = []
        for name, (paths, files) in snake_sets.items():
            logits = g4.score(snake_model, files)
            table = pd.DataFrame({"set": name, "path": list(paths)})
            for i, c in enumerate(g4.CLASSES):
                table[f"logit_{c}"] = logits[:, i]
            tables.append(table)
        pd.concat(tables, ignore_index=True).to_csv(out / "snake_check_scores.csv", index=False)
        log(f"{condition}: snake check scored")

        items = [(p, str(view / "data" / p)) for p in held_out["path"]]
        gate_rows(pipeline, items).to_csv(out / "gate_heldout.csv", index=False)
        log(f"{condition}: gate called")
        if condition == "clean":
            check_clean(out)
            return frame, view, preds_dir
        shutil.rmtree(root)
        return None

    def check_clean(out):
        expected = np.load(inputs / "expected/clean_sample.npz", allow_pickle=True)
        paths = expected["paths"].astype(str)
        if LOCAL:
            return
        plain = np.load(out / "m5_kd.npz", allow_pickle=True)
        flip = np.load(out / "m5_kd_flip.npz", allow_pickle=True)
        order = pd.Series(np.arange(len(plain["paths"])), index=plain["paths"].astype(str)).loc[paths].to_numpy()
        gate = pd.read_csv(out / "gate_heldout.csv").set_index("path").loc[paths]
        front = pd.read_csv(out / front_end.PREDS_FILE, keep_default_na=False).set_index("path").loc[paths]
        snakes = pd.read_csv(out / "snake_check_scores.csv")
        snakes = snakes[snakes["set"].eq("held_out")].set_index("path").loc[paths]

        def p_snake(x):
            e = np.exp(x - x.max(1, keepdims=True))
            return e[:, 0] / e.sum(1)

        mine_snake = snakes[["logit_snake", "logit_lookalike", "logit_other"]].to_numpy()
        check = {"photos": len(paths),
                 "m5_top1_agreement": float((plain["species_logits"][order].argmax(1) ==
                                             expected["m5_plain"].argmax(1)).mean()),
                 "m5_flip_top1_agreement": float((flip["species_logits"][order].argmax(1) ==
                                                  expected["m5_flip"].argmax(1)).mean()),
                 "m5_max_abs_logit_diff": float(np.abs(plain["species_logits"][order] - expected["m5_plain"]).max()),
                 "box_presence_agreement": float((front["box"].ne("").to_numpy() == expected["box_present"]).mean()),
                 "mask_presence_agreement": float((front["mask_file"].ne("").to_numpy() ==
                                                   expected["mask_present"]).mean()),
                 "gate_call_agreement": float((gate["gate_snake"].to_numpy(bool) == expected["gate_snake"]).mean()),
                 "snake_max_abs_p_diff": float(np.abs(p_snake(mine_snake) - p_snake(expected["snake_logits"])).max()),
                 "snake_flips_at_cut": int(((p_snake(mine_snake) >= SNAKE_CUT) !=
                                            (p_snake(expected["snake_logits"]) >= SNAKE_CUT)).sum())}
        STATUS["checks"]["clean_vs_locked_test_run"] = check
        log("clean instrument check", check)
        assert min(check["m5_top1_agreement"], check["m5_flip_top1_agreement"]) >= CHECK_FLOORS["top1_agreement"], check
        assert check["gate_call_agreement"] >= CHECK_FLOORS["gate_agreement"], check
        assert min(check["box_presence_agreement"], check["mask_presence_agreement"]) >= CHECK_FLOORS[
            "presence_agreement"], check
        assert check["snake_flips_at_cut"] <= CHECK_FLOORS["snake_flips"], check

    clean = step("condition clean", run_condition, "clean")
    if clean is None:
        raise SystemExit("the clean pass failed its check: no corruption is run")
    frame, view, preds_dir = clean

    def extra_frame(meta, split, paths):
        built = data.build_frame(meta, classes, [split], frontend_preds=pd.read_csv(preds_dir / front_end.PREDS_FILE),
                                 mask_source="predicted", preds_root=preds_dir)
        built = built[built["path"].isin(set(paths))].reset_index(drop=True)
        return data.with_files(built, view)

    context = {"np": np, "pd": pd, "torch": torch, "data": data, "load_checkpoint": load_checkpoint,
               "device": device, "inputs": inputs, "out": OUT, "scratch": SCRATCH, "log": log, "status": STATUS,
               "loader_workers": LOADER_WORKERS, "m5_dir": m5_dir, "lists": lists, "local": bool(LOCAL),
               "data_root": data_root, "ref": ref, "non": non, "held_out": held_out}
    if JOB.get("gradcam"):
        import gradcam_step
        step("gradcam", gradcam_step.run, context, frame)
    if JOB.get("compression"):
        import compression_step
        calib_frame = extra_frame(metadata, "train", lists["quant_calib"]["path"])
        parity_frame = extra_frame(metadata, "val", lists["parity_val"]["path"])
        slice_frame = frame[frame["path"].isin(set(lists["g3_slice"]["path"]))].reset_index(drop=True)
        STATUS["checks"]["compression_frames"] = {"calib": len(calib_frame), "parity": len(parity_frame),
                                                  "g3_slice": len(slice_frame)}
        step("compression", compression_step.run, context, calib_frame, parity_frame, slice_frame)
    if not LOCAL:  # the laptop keeps the clean copy for a look at the maps
        shutil.rmtree(SCRATCH / "c" / "clean", ignore_errors=True)
    for condition in JOB["conditions"]:
        if condition != "clean":
            step(f"condition {condition}", run_condition, condition)
    write_status()
    log("done", STATUS["passed"], STATUS["failed"])
    if STATUS["failed"]:
        raise SystemExit(f"failed steps: {STATUS['failed']}")


if __name__ == "__main__":
    main()
