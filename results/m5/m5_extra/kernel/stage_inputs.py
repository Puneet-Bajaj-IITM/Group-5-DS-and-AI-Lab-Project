"""Stage the private inputs dataset of the M5 extra kernels (robustness, Grad-CAM, int8), one copy per account.

Holds only frozen files (FREEZE.md hashes checked), the photo lists of select_photos.py, and a 600-photo sample of
the locked test run's own clean outputs, which the kernel's clean pass must reproduce before any corruption runs.
No snake photo is staged: the kernel reads them from the shared clean dataset.

Usage: m3-final/.venv/bin/python -I results/m5/m5_extra/kernel/stage_inputs.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXTRA = HERE.parent
WORK = HERE.parents[2]
OUT = HERE / "inputs"
TEST_INPUTS = WORK / "results/m5/test/kernel/inputs"
KERNEL_OUT = WORK / "results/m5/test/kernel_out/out"
G3, G4 = WORK / "results/m5/g3", WORK / "results/m5/g4/runs/distil_oi"
FRONT_END = G3 / "datasets/m5_inputs/m4_front_end"
SEED, CLEAN_SAMPLE, SNAKE_CALIB = 42, 600, 500
PARITY_TOLERANCE = 1e-3
DATASET_TITLE = "serpentguard-m5-extra-inputs"
ACCOUNTS = ("jayeshbansal497", "vikas25s")
FROZEN = {
    "m5_kd/best.pt": "1ec2ee88b95af0d5a0e6e69feed71c9be4bbdbe3584790ffa389fa52bdb1019f",
    "m5_kd/prototypes.npz": "a7fa7cba23eb791e41e12187d022e1c1d7503e3e241dbec84da75e17820cade5",
    "snake_check/model.pt": "88fd518c550a4f1502c86e7da1d8a3a8bd9c332b5377a8cc5d7f65f8144b7cfd",
    "app_models/detector.onnx": "1d2da244575a39c23ebab6ee2f4f04b1c672794031a56fab9d49dc5176f4f9ba",
    "app_models/segmenter.onnx": "da5d59baca34eda5993d837673449b99ccd178c98bbd6755a6b20bbea6f99269",
    "app_models/pipeline.json": "b92bb5e27a49d10068aedea4631f2958c70f2a6b732f2d473c50bd7131798063",
    "app_models/reference_pipeline.py": "a0e0f6e6e2e9246eb12c9487c217d93071ff884d511bd4420c44827351969452",
    "m4_front_end/detector/M4_detector.pt": "29a1ad1c1b9b561c35ef4ffa1916998837c3c78525c21b51668d9d0e95607b61",
    "m4_front_end/segmenter/segmenter.pt": "b881c338356bdf44aa0b45e0306f59d7b38252d04316da2455c40055aef656ec",
    "onnx/snake_check.onnx": "8d11ea7c265fa397ced36fb2d4908bd41bad8ebf1f50efe054ad23aad626da74",
    "onnx/snake_check_fp16.onnx": "fb5b5bda5e0dbdb1895448ae1c70e56e93042ef322f9e5595cfcd6e0b8ec9fc3",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classifier_parity(onnx_path):
    """The M5_KD ONNX export (g3_app_new.py) against the frozen best.pt on a random input."""
    sys.path.insert(0, str(WORK / "m5-wt/src"))
    import onnxruntime as ort
    from serpentguard.models import build_model
    checkpoint = torch.load(OUT / "m5_kd/best.pt", map_location="cpu", weights_only=True)
    net = build_model(checkpoint["config"], pretrained=False)
    net.load_state_dict(checkpoint["state_dict"])
    sample = torch.randn(1, 3, 320, 320, generator=torch.Generator().manual_seed(SEED))
    with torch.no_grad():
        expected = net.eval()(sample)["species_logits"].numpy()
    got = ort.InferenceSession(str(onnx_path)).run(["species_logits"], {"images": sample.numpy()})[0]
    gap = float(np.abs(expected - got).max())
    assert gap < PARITY_TOLERANCE, f"classifier.onnx and best.pt differ by {gap}"
    return gap


def clean_sample():
    """600 held-out photos with the test kernel's own outputs, for the clean pass's in-kernel check."""
    plain = np.load(KERNEL_OUT / "m5_kd_heldout.npz", allow_pickle=True)
    flip = np.load(KERNEL_OUT / "m5_kd_flip_heldout.npz", allow_pickle=True)
    paths = plain["paths"].astype(str)
    assert (flip["paths"].astype(str) == paths).all()
    rows = np.sort(np.random.default_rng(SEED).choice(len(paths), CLEAN_SAMPLE, replace=False))
    pick = paths[rows]
    gate = pd.read_csv(KERNEL_OUT / "gate_heldout.csv").set_index("path").loc[pick]
    front = pd.read_csv(KERNEL_OUT / "frontend_preds.csv", keep_default_na=False).set_index("path").loc[pick]
    snake = pd.read_csv(KERNEL_OUT / "snake_check_scores.csv")
    snake = snake[snake["set"].eq("held_out")].set_index("path").loc[pick]
    np.savez_compressed(OUT / "expected/clean_sample.npz", paths=pick,
                        m5_plain=plain["species_logits"][rows], m5_flip=flip["species_logits"][rows],
                        gate_snake=gate["gate_snake"].to_numpy(bool), box_present=front["box"].ne("").to_numpy(),
                        mask_present=front["mask_file"].ne("").to_numpy(),
                        snake_logits=snake[["logit_snake", "logit_lookalike", "logit_other"]].to_numpy())


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(TEST_INPUTS, OUT, ignore=shutil.ignore_patterns("manifest.json", "dataset-metadata.json", "__pycache__"))
    for name, source in (("m4_front_end/detector/M4_detector.pt", FRONT_END / "detector/M4_detector.pt"),
                         ("m4_front_end/segmenter/segmenter.pt", FRONT_END / "segmenter/segmenter.pt"),
                         ("onnx/classifier.onnx", G3 / "app_models_KD/classifier.onnx"),
                         ("onnx/snake_check.onnx", G4 / "snake_check.onnx"),
                         ("onnx/snake_check_fp16.onnx", G4 / "snake_check_fp16.onnx")):
        (OUT / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, OUT / name)
    for name, expected in FROZEN.items():
        assert sha256(OUT / name) == expected, f"{name} is not the frozen file"
    shutil.copytree(EXTRA / "lists", OUT / "lists")
    calib = pd.read_csv(G4 / "scores_kernel/scores_calib.csv")
    assert not calib["path"].str.startswith(("test/", "test_extra/")).any()
    calib[["path"]].sample(SNAKE_CALIB, random_state=SEED).sort_values("path").to_csv(
        OUT / "lists/snake_calib.csv", index=False)
    (OUT / "expected").mkdir()
    clean_sample()
    gap = classifier_parity(OUT / "onnx/classifier.onnx")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    manifest = {p.relative_to(OUT).as_posix(): sha256(p) for p in files}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB; "
          f"classifier.onnx vs best.pt max logit diff {gap:.2e}")


def metadata_for(account):
    """dataset-metadata.json for one account's copy (written just before that upload)."""
    assert account in ACCOUNTS
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET_TITLE, "id": f"{account}/{DATASET_TITLE}", "licenses": [{"name": "other"}]}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--metadata":
        metadata_for(sys.argv[2])
    else:
        main()
