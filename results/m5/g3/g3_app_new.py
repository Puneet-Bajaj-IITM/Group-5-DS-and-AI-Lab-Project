"""G3 as the app runs it, for a new model: its best.pt exported to the app's fp32 ONNX classifier, run on the app's
own crops (m4-final/for-jayesh/reference_pipeline.py: M4 detector and U-Net, crop to the cleaned mask's box) of the
11,931 slice photos, plain and with a mirrored copy averaged in. Same slice, metric, pairing and bootstrap as
m5-targets/g3_app.py; the app bundle's detector, segmenter and settings are copied unchanged, only classifier.onnx
is replaced. "Prior" adds tau * log(training count) with tau fitted on the run's calibration outputs (prior.py).

Usage: python g3_app_new.py --run out_finetune/M5_FT --name FT [--workers 8]
"""
import argparse
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUNDLE = ROOT / "m4-final/for-jayesh"
sys.path[:0] = [str(BUNDLE), str(ROOT / "m4-checks"), str(ROOT / "m5-wt/src"), str(HERE)]
import check_full_run as cf  # noqa: E402
import export_onnx  # noqa: E402
import prior  # noqa: E402
import reference_pipeline as ref  # noqa: E402
from serpentguard.models import build_model  # noqa: E402

PHOTOS = ROOT / "m5-g4-probe/photos/snakes"
METADATA = ROOT / "kaggle/clean_build/out/metadata.csv"
N_CLASSES = 184
PARITY_TOLERANCE = 1e-3
_p = None


def export(run_dir, models):
    """The app bundle with this run's classifier.onnx; checks ONNX against torch on one random input."""
    if (models / "classifier.onnx").exists():
        return
    shutil.copytree(BUNDLE / "models", models)
    (models / "classifier.onnx").unlink()
    for name in ("classifier_fp16.onnx", "SHA256SUMS"):
        (models / name).unlink(missing_ok=True)
    export_onnx.export_classifier(run_dir, models, build_model)
    import onnxruntime as ort
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    net = build_model(checkpoint["config"], pretrained=False)
    net.load_state_dict(checkpoint["state_dict"])
    sample = torch.randn(1, 3, export_onnx.INPUT_SIZE, export_onnx.INPUT_SIZE)
    with torch.no_grad():
        expected = net.eval()(sample)["species_logits"].numpy()
    got = ort.InferenceSession(str(models / "classifier.onnx")).run(["species_logits"], {"images": sample.numpy()})[0]
    gap = float(np.abs(expected - got).max())
    assert gap < PARITY_TOLERANCE, f"ONNX and torch differ by {gap}"
    print(f"exported {run_dir.name}: ONNX matches torch within {gap:.2e}", flush=True)


def _init(models):
    global _p
    _p = ref.Pipeline(models, threads=1)


def one(path):
    s = _p.settings
    rgb = ref.load_photo(PHOTOS / path, s["exposure_fix"])
    gate, mask = _p.gate(rgb)
    h, w = rgb.shape[:2]
    window = ref.crop_window(ref.clean_mask(mask, s["mask_cleanup"]), h, w, s["crop"]["padding"])
    crop = rgb if window is None else rgb[window[0]:window[1], window[2]:window[3]]
    plain = _p.classify(crop)["species_logits"].astype(float)
    flipped = _p.classify(np.ascontiguousarray(crop[:, ::-1]))["species_logits"].astype(float)
    return {"path": path, "plain": plain.round(5).tolist(), "flipped": flipped.round(5).tolist()}


def main(args):
    run_dir = (HERE / args.run).resolve()
    models = HERE / f"app_models_{args.name}"
    cache = HERE / f"g3_app_{args.name}.jsonl"
    export(run_dir, models)
    saved = cf.load(run_dir / "val_outputs.npz")
    paths, y = saved["paths"], saved["labels"].astype(int)
    b1 = cf.load(cf.B1_DIR / "predictions.npz")
    shared = np.array(sorted(json.loads((cf.B1_DIR / "final_metrics.json").read_text())["coverage"]["covered_class_ids"]))
    b1_pred = b1["probs"].argmax(axis=1)[pd.Series(np.arange(len(b1["paths"])), index=b1["paths"]).reindex(paths).to_numpy(int)]
    created = pd.DataFrame({"path": paths}).merge(pd.read_csv(METADATA, usecols=["path", "created_at"]), on="path",
                                                  how="left")["created_at"].fillna("").astype(str)
    rows = created.ge(cf.SLICE_FROM).to_numpy() & np.isin(y, shared)
    assert rows.sum() == cf.SLICE_SHARED_PHOTOS, rows.sum()
    done = {}
    if cache.exists():
        done = {r["path"]: r for r in map(json.loads, cache.read_text().splitlines()) if r}
    todo = [p for p in paths[rows] if p not in done and (PHOTOS / p).exists()]
    missing = int(sum(not (PHOTOS / p).exists() for p in paths[rows]))
    print(json.dumps({"slice_photos": int(rows.sum()), "missing": missing, "to_score": len(todo)}), flush=True)
    with Pool(args.workers, initializer=_init, initargs=(models,)) as pool, cache.open("a") as handle:
        for i, r in enumerate(pool.imap_unordered(one, todo, chunksize=8), 1):
            handle.write(json.dumps(r) + "\n")
            done[r["path"]] = r
            if i % 1000 == 0:
                print(f"{i:,}/{len(todo):,}", flush=True)
    have = np.array([p in done for p in paths]) & rows
    plain = np.array([done[p]["plain"] for p in paths[have]])
    flipped = np.array([done[p]["flipped"] for p in paths[have]])
    calib = cf.load(run_dir / "calib_outputs.npz")
    tau, _ = prior.fit_tau(calib["species_logits"], calib["labels"])
    tau_flip = None
    if (run_dir / "calib_outputs_flip.npz").exists():
        calib_flip = cf.load(run_dir / "calib_outputs_flip.npz")
        tau_flip, _ = prior.fit_tau((calib["species_logits"] + calib_flip["species_logits"]) / 2, calib["labels"])
    groups = cf.groups_of(saved["observer_id"], paths)[have]
    yy, bb = y[have], b1_pred[have]
    macro_62 = cf.shared_macro_f1(shared, N_CLASSES)
    candidates = {"training-path crops (saved outputs, same photos)": saved["species_logits"].argmax(1)[have],
                  "app crops": plain.argmax(1), "app crops + flip": (plain + flipped).argmax(1),
                  "app crops + prior": prior.corrected(plain, tau).argmax(1)}
    if tau_flip is not None:
        candidates["app crops + flip + prior"] = prior.corrected((plain + flipped) / 2, tau_flip).argmax(1)
    out = {"run": str(run_dir), "photos": int(have.sum()), "missing": missing, "tau": tau, "tau_flip": tau_flip}
    for name, pred in candidates.items():
        out[name] = {m: cf.paired(yy, pred, bb, groups, s) for m, s in (("top1", cf.top1), ("macro_f1", macro_62))}
        print(name, {m: round(v["diff"], 4) for m, v in out[name].items()}, flush=True)
    (HERE / f"g3_app_{args.name}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args())
