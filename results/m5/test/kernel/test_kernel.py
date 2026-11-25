"""SerpentGuard M5: the one score of the locked held-out split (test + test_extra) with the frozen pipeline.

Runs on Kaggle (Jayesh's account, two T4s). Every model and setting is the one in results/m5/test/FREEZE.md; this
kernel only produces per-photo outputs, with the validation code paths (front_end.extend_predictions, the training
code's predict / save_outputs, robustness.robustness_metrics, b1_teacher.py, the G4 kernel's own scoring functions,
reference_pipeline.Pipeline.gate). The metrics are computed on the laptop by score_test.py.

The test splits are blocked everywhere in the repo code (assert_no_test_paths, build_frame). This scorer lifts that
block, in this process only, and builds the held-out frame through a copy of metadata.csv in which the held-out
photos carry the split name "val" (and the real validation photos "val_original"), so every frame is built by the
same functions that built the validation frame. Before the held-out pass, validation photos with saved outputs are
run through each step and must agree (the instrument checks).
"""
import hashlib
import importlib.util
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = json.loads(Path(os.environ["M5_TEST_LOCAL"]).read_text()) if os.environ.get("M5_TEST_LOCAL") else None
OUT = Path(LOCAL["out"]) if LOCAL else Path("/kaggle/working/out")
SCRATCH = Path(LOCAL["scratch"]) if LOCAL else Path("/kaggle/tmp/m5_test")
REPO = Path(LOCAL["repo"]) if LOCAL else HERE  # the bundle holds the repo's src/ and scripts/
HELD_OUT_SPLITS = ("test", "test_extra")
HELD_OUT_PHOTOS = LOCAL["held_out_photos"] if LOCAL else 18_533
SHARED = "puneetbajajiitm"
INPUTS_DATASET = "jayeshbansal497/serpentguard-m5-test-inputs"
CHECKSUMS = {"detector": "29a1ad1c1b9b561c35ef4ffa1916998837c3c78525c21b51668d9d0e95607b61",
             "segmenter": "b881c338356bdf44aa0b45e0306f59d7b38252d04316da2455c40055aef656ec",
             "a3": "b19f0a5ece4f98fc08911f8781048011e14aeb84b80fde9dc1f4b9263322a02b",
             "m5_kd": "1ec2ee88b95af0d5a0e6e69feed71c9be4bbdbe3584790ffa389fa52bdb1019f",
             "snake_check": "88fd518c550a4f1502c86e7da1d8a3a8bd9c332b5377a8cc5d7f65f8144b7cfd"}
SNAKE_CUT = 0.20484977503483762
CLASSIFIER_AGREEMENT_FLOOR = 0.8 if LOCAL else 0.98  # the laptop runs fp32 on 12 photos; the kernel fp16 as the saved run
GATE_WORKERS = 4
LOADER_WORKERS = 0 if LOCAL else 4  # the laptop sandbox blocks shared-memory tensors from workers
B1_WAIT_HOURS = 10
STATUS = {"passed": [], "failed": {}, "checks": {}, "seconds": {}}
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
    os.environ.update(NO_ALBUMENTATIONS_UPDATE="1", YOLO_VERBOSE="False", HF_HUB_DISABLE_PROGRESS_BARS="1",
                      TQDM_DISABLE="1")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status():
    STATUS["minutes"] = round((time.time() - STARTED) / 60, 1)
    (OUT / "status.json").write_text(json.dumps(STATUS, indent=1, default=str) + "\n")


def step(name, function):
    started = time.time()
    log(f"step: {name}")
    try:
        result = function()
        STATUS["passed"].append(name)
        return result
    except Exception as error:  # every step runs; the run fails at the end if any failed
        STATUS["failed"][name] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        return None
    finally:
        STATUS["seconds"][name] = round(time.time() - started)
        write_status()


def main():
    if not LOCAL:
        pip_install()
    import numpy as np
    import pandas as pd
    import torch

    sys.path.insert(0, str(REPO / "src"))
    from serpentguard import data, decide, front_end, metrics, robustness
    from serpentguard.train import load_checkpoint, predict, save_outputs

    for folder in (OUT, SCRATCH):
        folder.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{torch.cuda.device_count()} GPU(s); device {device}; local stand-in: {bool(LOCAL)}")

    def dataset(handle):
        if LOCAL:
            return Path(LOCAL["datasets"][handle])
        import kagglehub
        return Path(kagglehub.dataset_download(handle))

    def found(root, pattern):
        return next(Path(root).rglob(pattern))

    # 1. inputs
    data_root = dataset(f"{SHARED}/serpentguard-indian-snakes-clean")
    subset_csv = dataset(f"{SHARED}/serpentguard-m3-subset") / "subset.csv"
    m5_inputs = dataset("jayeshbansal497/serpentguard-m5-inputs")
    inputs = found(dataset(INPUTS_DATASET), "manifest.json").parent
    b1_dir = found(dataset("jayeshbansal497/serpentguard-m3-b1-predictions"), "predictions.npz").parent
    detector, segmenter = found(m5_inputs, "M4_detector.pt"), found(m5_inputs, "segmenter/segmenter.pt")
    a3_best, m5_dir = found(m5_inputs, "a3/best.pt"), inputs / "m5_kd"

    def check_inputs():
        manifest = json.loads((inputs / "manifest.json").read_text())
        bad = [name for name, digest in manifest.items() if sha256(inputs / name) != digest]
        assert not bad, f"input files differ from the staged manifest: {bad[:5]}"
        for name, path in (("detector", detector), ("segmenter", segmenter), ("a3", a3_best),
                           ("m5_kd", m5_dir / "best.pt"), ("snake_check", inputs / "snake_check/model.pt")):
            assert sha256(path) == CHECKSUMS[name], f"{path} is not the frozen {name}"
        STATUS["checks"]["inputs"] = {"manifest_files": len(manifest), "frozen_models": sorted(CHECKSUMS)}

    step("inputs", check_inputs)
    if STATUS["failed"]:
        raise SystemExit("inputs failed: nothing is scored")

    metadata, classes = data.load_tables(data_root)
    held_out = metadata[metadata["split"].isin(HELD_OUT_SPLITS)].reset_index(drop=True)
    assert len(held_out) == HELD_OUT_PHOTOS, len(held_out)
    with ThreadPoolExecutor(32) as pool:
        on_disk = list(pool.map(os.path.isfile, [str(data_root / "data" / p) for p in held_out["path"]]))
    assert all(on_disk), f"{len(on_disk) - sum(on_disk)} held-out photos are not in the dataset"
    expected = np.load(inputs / "instrument/val_sample_expected.npz", allow_pickle=True)
    val_sample = [str(p) for p in expected["paths"]][: LOCAL["val_sample"] if LOCAL else None]
    class_of = dict(zip(classes["species"], classes["class_id"]))
    b1_classes = sorted(json.loads((b1_dir / "final_metrics.json").read_text())["coverage"]["covered_class_ids"])
    log(f"held-out photos {len(held_out):,}; validation sample {len(val_sample)}; B1 species {len(b1_classes)}")

    # 2. B1 on the held-out photos of its 62 species, in its own process on the second GPU
    b1_out = SCRATCH / "b1_heldout.npz"
    b1_photos = held_out[held_out["species"].map(class_of).isin(b1_classes)][["path"]]
    b1_photos.to_csv(SCRATCH / "b1_photos.csv", index=False)
    b1_log = open(OUT / "b1_heldout.log", "w")
    b1_process = None if LOCAL else subprocess.Popen(
        [sys.executable, str(REPO / "scripts/b1_teacher.py"), str(data_root), str(SCRATCH / "b1_photos.csv"),
         str(b1_dir / "predictions.npz"), str(SCRATCH / "downloads"), str(b1_out)],
        stdout=b1_log, stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "1" if torch.cuda.device_count() > 1 else "0"})
    log(f"B1 started for {len(b1_photos):,} held-out photos of its species")

    # 3. the M4 front end from an empty base, over the held-out photos and the validation sample
    preds_dir = SCRATCH / "m4_preds"

    def run_front_end():
        base = SCRATCH / "empty_base"
        (base / front_end.MASKS_DIR).mkdir(parents=True, exist_ok=True)
        (base / front_end.PREDS_FILE).write_text(front_end.PREDS_COLUMNS)
        rows = pd.DataFrame({"path": list(held_out["path"]) + val_sample})
        joined = front_end.extend_predictions(rows, base, detector, segmenter, data_root, preds_dir)
        assert joined["path"].is_unique and set(joined["path"]) == set(rows["path"])
        shutil.copy2(preds_dir / front_end.PREDS_FILE, OUT / front_end.PREDS_FILE)
        mine = joined.set_index("path").loc[val_sample].fillna("")
        saved = pd.read_csv(inputs / "instrument/frontend_val_sample.csv", keep_default_na=False).set_index(
            "path").loc[val_sample]
        conf = pd.to_numeric(mine["box_conf"], errors="coerce") - pd.to_numeric(saved["box_conf"], errors="coerce")
        check = {"photos": len(val_sample),
                 "box_presence_agreement": float((mine["box"].ne("") == saved["box"].ne("")).mean()),
                 "mask_presence_agreement": float((mine["mask_file"].ne("") == saved["mask_file"].ne("")).mean()),
                 "box_conf_max_abs_diff": float(np.nanmax(np.abs(conf.to_numpy(float))))}
        STATUS["checks"]["front_end_vs_saved_val"] = check
        log("front end instrument check", check)
        assert check["box_presence_agreement"] >= 0.98 and check["mask_presence_agreement"] >= 0.97, check
        held = joined[joined["path"].isin(held_out["path"])]
        STATUS["checks"]["front_end_held_out"] = {"photos": len(held), "box": float(held["box"].ne("").mean()),
                                                  "mask": float(held["mask_file"].ne("").mean())}

    step("front end", run_front_end)
    if "front end" in STATUS["failed"]:
        raise SystemExit("the front end failed")

    # 4. frames: the held-out photos through a metadata copy where they carry the split name "val"
    for module in (data, decide, metrics):
        module.assert_no_test_paths = lambda frame: None
    data.TEST_SPLITS = ()
    view = SCRATCH / "data_view"
    view.mkdir(exist_ok=True)
    if not (view / "data").exists():
        (view / "data").symlink_to(data_root / "data")
    shutil.copy2(data_root / "classes.csv", view / "classes.csv")
    view_meta = metadata.assign(split=metadata["split"].replace(
        {"val": "val_original", "test": "val", "test_extra": "val"}))
    view_meta.to_csv(view / "metadata.csv", index=False)
    preds = pd.read_csv(preds_dir / front_end.PREDS_FILE)

    def scored(meta, root, keep=None):  # make_frames' scored() for a predicted-input run, then with_files
        frame = data.build_frame(meta, classes, ["val"], frontend_preds=preds, mask_source="predicted",
                                 preds_root=preds_dir)
        if keep is not None:
            frame = frame[frame["path"].isin(keep)].reset_index(drop=True)
        return data.with_files(frame, root)

    held_frame = scored(data.load_tables(view)[0], view)
    val_frame = scored(metadata, data_root, set(val_sample))
    assert len(held_frame) == HELD_OUT_PHOTOS and len(val_frame) == len(val_sample)
    held_frame[["path", "class_id", "observer_id", "mask_valid"]].to_csv(OUT / "held_out_frame.csv", index=False)

    class Mirrored(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            return self.model(torch.flip(images, dims=[3]))

    def outputs(run_dir, frame, mirrored, file=None):
        cfg, model, _ = load_checkpoint(run_dir, device)
        net = Mirrored(model).eval() if mirrored else model
        out = predict(net, data.eval_loader(cfg, frame, num_workers=LOADER_WORKERS), device, device.type == "cuda")
        if file is not None:
            save_outputs(file, frame, out)
        return out

    # 5. classifiers: M5_KD plain and mirrored, A3 (the before), each checked on validation first
    passes = {"m5_kd": (m5_dir, False, "m5_plain"), "m5_kd_flip": (m5_dir, True, "m5_flip"),
              "a3_m4crops": (a3_best.parent, False, "a3")}

    def classifiers():
        order = pd.Series(np.arange(len(expected["paths"])), index=expected["paths"])
        for name, (run_dir, mirrored, key) in passes.items():
            mine = outputs(run_dir, val_frame, mirrored)["species_logits"]
            saved = expected[key][order.loc[val_frame["path"]].to_numpy(int)]
            check = {"photos": len(mine), "top1_agreement": float((mine.argmax(1) == saved.argmax(1)).mean()),
                     "max_abs_logit_diff": float(np.abs(mine - saved).max()),
                     "median_abs_logit_diff": float(np.median(np.abs(mine - saved)))}
            STATUS["checks"][f"{name}_vs_saved_val"] = check
            log(f"{name} instrument check", check)
            assert check["top1_agreement"] >= CLASSIFIER_AGREEMENT_FLOOR, (name, check)
        for name, (run_dir, mirrored, _) in passes.items():
            outputs(run_dir, held_frame, mirrored, OUT / f"{name}_heldout.npz")
            log(f"{name}: held-out outputs saved")

    step("classifiers", classifiers)

    # 6. G1: the scenery test on the held-out photos (M5_KD plain), through robustness_metrics
    def scenery():
        run = SCRATCH / "M5_KD_heldout"
        run.mkdir(exist_ok=True)
        for name in ("best.pt", "config.yaml", "prototypes.npz", "calib_outputs.npz", "unknown_outputs.npz"):
            shutil.copy2(m5_dir / name, run / name)
        shutil.copy2(OUT / "m5_kd_heldout.npz", run / "val_outputs.npz")
        files = [view / "data" / p for p in held_frame["path"]]
        with ThreadPoolExecutor(robustness.QUALITY_THREADS) as pool:
            flags = pd.DataFrame(list(pool.map(robustness.photo_quality, files)))
        quality = pd.DataFrame({"path": held_frame["path"], "set": "val"}).join(flags)
        (run / "r1").mkdir(exist_ok=True)
        report = robustness.robustness_metrics(run, view, preds_dir, subset_csv=subset_csv, quality=quality,
                                               out_dir=run / "r1", num_workers=LOADER_WORKERS)
        metrics.write_json(OUT / "robustness_heldout.json", report)
        for file in (run / "r1").glob("*.npz"):
            shutil.copy2(file, OUT / f"heldout_{file.name}")
        c = report["conditions"]
        log("G1 gap", c["as_is_masked"]["top1"] - c["swapped"]["top1"], report["instrument_check"])

    step("scenery test", scenery)

    # 7. the snake check (G4) with the G4 kernel's own scoring functions, PyTorch fp32, the app's input path
    def snake_check():
        os.environ.update(G4_SMOKE="1", G4_SNAKES=str(data_root / "data"), G4_OURS=str(inputs / "snake_check"),
                          G4_OUT=str(SCRATCH / "g4"), G4_VARIANT="distil_oi")
        spec = importlib.util.spec_from_file_location("train_g4", inputs / "snake_check/train_g4.py")
        g4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g4)
        g4.WORKERS = 0 if LOCAL else 4  # macOS spawns loader workers, which cannot import an exec'd module
        checkpoint = torch.load(inputs / "snake_check/model.pt", map_location="cpu", weights_only=False)
        model = g4.Student(checkpoint["backbone"], checkpoint["state_dict"]["project.weight"].shape[0])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(g4.DEVICE)
        val = pd.read_csv(inputs / "instrument/snake_check_val_sample.csv").head(LOCAL["val_sample"] if LOCAL else None)
        non = pd.read_csv(inputs / "nonsnake_test.csv")
        sets = {"val_check": (val["path"], [str(data_root / "data" / p) for p in val["path"]]),
                "held_out": (held_out["path"], [str(data_root / "data" / p) for p in held_out["path"]]),
                "nonsnake_test": (non["path"], [str(inputs / "nonsnake_test" / p) for p in non["path"]])}
        tables = []
        for name, (paths, files) in sets.items():
            logits = g4.score(model, files)
            table = pd.DataFrame({"set": name, "path": list(paths)})
            for i, c in enumerate(g4.CLASSES):
                table[f"logit_{c}"] = logits[:, i]
            tables.append(table)
            log(f"snake check: {name} {len(table):,} photos")
        scores = pd.concat(tables, ignore_index=True)
        scores.to_csv(OUT / "snake_check_scores.csv", index=False)
        mine = scores[scores["set"].eq("val_check")][[f"logit_{c}" for c in g4.CLASSES]].to_numpy()
        saved = val[[f"logit_{c}" for c in g4.CLASSES]].to_numpy()

        def p_snake(x):
            e = np.exp(x - x.max(1, keepdims=True))
            return e[:, 0] / e.sum(1)

        check = {"photos": len(mine), "max_abs_logit_diff": float(np.abs(mine - saved).max()),
                 "max_abs_p_snake_diff": float(np.abs(p_snake(mine) - p_snake(saved)).max()),
                 "decisions_flipped_at_cut": int(((p_snake(mine) >= SNAKE_CUT) != (p_snake(saved) >= SNAKE_CUT)).sum())}
        STATUS["checks"]["snake_check_vs_saved_val"] = check
        log("snake check instrument check", check)
        assert check["decisions_flipped_at_cut"] <= 1, check

    step("snake check", snake_check)

    # 8. the app's gate call per photo (ONNX on the CPU, reference_pipeline.Pipeline.gate)
    def gate():
        sys.path.insert(0, str(HERE))
        import gate_worker
        check_rows = pd.read_csv(inputs / "instrument/gate_val_sample.csv").head(LOCAL["val_sample"] if LOCAL else None)
        items = ([(p, str(data_root / "data" / p)) for p in check_rows["path"]]
                 + [(p, str(data_root / "data" / p)) for p in held_out["path"]])
        context = multiprocessing.get_context("spawn")
        with context.Pool(GATE_WORKERS, initializer=gate_worker.init,
                          initargs=(str(inputs / "app_models"),)) as pool:
            rows = pd.DataFrame(list(pool.imap(gate_worker.gate, items, chunksize=16)))
        mine = rows.iloc[:len(check_rows)].reset_index(drop=True)
        check = {"photos": len(check_rows),
                 "gate_call_agreement": float((mine["gate_snake"].to_numpy() == check_rows["gate_snake"].to_numpy()).mean()),
                 "max_abs_mask_area_diff": float(np.abs(mine["mask_area"] - check_rows["mask_area"]).max())}
        STATUS["checks"]["gate_vs_saved_val"] = check
        log("gate instrument check", check)
        rows.iloc[len(check_rows):].to_csv(OUT / "gate_heldout.csv", index=False)
        assert check["gate_call_agreement"] >= 0.99, check

    step("app gate", gate)

    # 9. B1
    def b1():
        while b1_process.poll() is None and (time.time() - STARTED) / 3600 < B1_WAIT_HOURS:
            time.sleep(60)
        assert b1_process.poll() == 0, f"B1 ended with {b1_process.poll()}: read b1_heldout.log"
        shutil.copy2(b1_out, OUT / "b1_heldout.npz")
        STATUS["checks"]["b1_own_check"] = json.loads(Path(f"{b1_out}.check.json").read_text())

    if not LOCAL:
        step("B1", b1)
    versions = {"python": sys.version, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}
    for package in ("timm", "ultralytics", "onnxruntime", "segmentation_models_pytorch", "numpy", "pandas"):
        try:
            versions[package] = __import__(package).__version__
        except Exception as error:
            versions[package] = f"not read: {error}"
    (OUT / "versions.json").write_text(json.dumps(versions, indent=1) + "\n")
    write_status()
    log("done", STATUS["passed"], STATUS["failed"])
    if STATUS["failed"]:
        raise SystemExit(f"failed steps: {STATUS['failed']}")


if __name__ == "__main__":
    main()
