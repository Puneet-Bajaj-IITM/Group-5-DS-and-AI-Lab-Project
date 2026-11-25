"""Stage the private Kaggle dataset jayeshbansal497/serpentguard-m5-test-inputs for the one test-scoring kernel.

Holds only frozen files (FREEZE.md) and validation expectations for the in-kernel instrument checks; no snake test
photo is read here (they stay in the shared clean dataset). The 600 non-snake test photos are copied from the M4
fetch. Every staged file's SHA-256 goes into inputs/manifest.json; the frozen ones are checked against FREEZE.md.

Usage: m3-final/.venv/bin/python -I results/m5/test/kernel/stage_inputs.py
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "inputs"
G3, G4 = WORK / "results/m5/g3", WORK / "results/m5/g4/runs/distil_oi"
M5_KD = G3 / "out_kd/M5_KD"
APP = WORK / "m4-final/for-jayesh"
NON_SNAKE_PHOTOS = WORK / "kaggle/m4_runs/gate/non_snake_test_photos"
NB19 = WORK / "kaggle/m4_runs/non_snake/out_full/manifest.csv"
SEED, VAL_SAMPLE, G4_SAMPLE, GATE_SAMPLE = 42, 600, 500, 300
DATASET = "jayeshbansal497/serpentguard-m5-test-inputs"
FROZEN = {
    "m5_kd/best.pt": "1ec2ee88b95af0d5a0e6e69feed71c9be4bbdbe3584790ffa389fa52bdb1019f",
    "m5_kd/prototypes.npz": "a7fa7cba23eb791e41e12187d022e1c1d7503e3e241dbec84da75e17820cade5",
    "m5_kd/calib_outputs.npz": "cf5554a26e9030cdbc91e775e46be4bfdfaa04e93b9a0829fe56e8179f3c877a",
    "m5_kd/calib_outputs_flip.npz": "830a59d915f541063ff48580c822943ea3e2c82ff14d86f2f9f349e2a6fd6493",
    "snake_check/model.pt": "88fd518c550a4f1502c86e7da1d8a3a8bd9c332b5377a8cc5d7f65f8144b7cfd",
    "app_models/detector.onnx": "1d2da244575a39c23ebab6ee2f4f04b1c672794031a56fab9d49dc5176f4f9ba",
    "app_models/segmenter.onnx": "da5d59baca34eda5993d837673449b99ccd178c98bbd6755a6b20bbea6f99269",
    "app_models/pipeline.json": "b92bb5e27a49d10068aedea4631f2958c70f2a6b732f2d473c50bd7131798063",
    "app_models/reference_pipeline.py": "a0e0f6e6e2e9246eb12c9487c217d93071ff884d511bd4420c44827351969452",
}
COPIES = {
    **{f"m5_kd/{name}": M5_KD / name for name in (
        "best.pt", "config.yaml", "prototypes.npz", "decision_metrics.json", "calib_outputs.npz",
        "calib_outputs_flip.npz", "unknown_outputs.npz", "unknown_outputs_flip.npz")},
    "snake_check/model.pt": G4 / "model.pt",
    "snake_check/train_g4.py": G4 / "pushed_code/sg-m5-g4-distil-oi.py",
    **{f"app_models/{name}": APP / "models" / name for name in (
        "detector.onnx", "segmenter.onnx", "classifier.onnx", "pipeline.json", "tables.json")},
    "app_models/reference_pipeline.py": APP / "reference_pipeline.py",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_of(saved, paths):
    index = pd.Series(np.arange(len(saved["paths"])), index=saved["paths"]).reindex(paths)
    assert index.notna().all()
    return index.to_numpy(int)


def instrument_expectations():
    """Validation photos whose saved outputs the kernel must reproduce before it scores the held-out set."""
    rng = np.random.default_rng(SEED)
    plain = np.load(M5_KD / "val_outputs.npz", allow_pickle=True)
    flip = np.load(M5_KD / "val_outputs_flip.npz", allow_pickle=True)
    a3 = np.load(G3 / "out_kd/A3_m4crops/val_outputs.npz", allow_pickle=True)
    paths = np.sort(rng.choice(plain["paths"], VAL_SAMPLE, replace=False))
    np.savez_compressed(OUT / "instrument/val_sample_expected.npz", paths=paths,
                        m5_plain=plain["species_logits"][rows_of(plain, paths)],
                        m5_flip=flip["species_logits"][rows_of(flip, paths)],
                        a3=a3["species_logits"][rows_of(a3, paths)])
    preds = pd.read_csv(G3 / "out_kd/full/frontend_preds.csv", keep_default_na=False)
    preds[preds["path"].isin(paths)].to_csv(OUT / "instrument/frontend_val_sample.csv", index=False)
    torch_scores = pd.read_csv(G4 / "scores_torch.csv")
    val_snakes = torch_scores[torch_scores["set"].eq("val")]
    val_snakes = val_snakes.iloc[np.sort(rng.choice(len(val_snakes), G4_SAMPLE, replace=False))]
    val_snakes[["path", "logit_snake", "logit_lookalike", "logit_other"]].to_csv(
        OUT / "instrument/snake_check_val_sample.csv", index=False)
    app = np.load(WORK / "m5-targets/app_outputs/val_outputs.npz", allow_pickle=True)
    pick = np.sort(rng.choice(len(app["paths"]), GATE_SAMPLE, replace=False))
    pd.DataFrame({"path": app["paths"][pick], "gate_snake": app["gate_snake"][pick],
                  "box_conf": app["box_conf"][pick], "mask_area": app["mask_area"][pick]}).to_csv(
        OUT / "instrument/gate_val_sample.csv", index=False)


def non_snake_test():
    manifest = pd.read_csv(NB19)
    test = manifest[manifest["split"].eq("test")]
    assert len(test) == 600
    target = OUT / "nonsnake_test"
    for file in test["file"]:
        (target / file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NON_SNAKE_PHOTOS / file, target / file)
    test[["file", "category", "owner", "source"]].rename(columns={"file": "path"}).to_csv(
        OUT / "nonsnake_test.csv", index=False)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "instrument").mkdir(parents=True)
    for name, source in COPIES.items():
        (OUT / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, OUT / name)
    (OUT / "snake_check/g4_data_manifest.json").write_text("{}\n")  # train_g4's smoke import looks for it
    for name, expected in FROZEN.items():
        assert sha256(OUT / name) == expected, f"{name} is not the frozen file"
    instrument_expectations()
    non_snake_test()
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    manifest = {p.relative_to(OUT).as_posix(): sha256(p) for p in files}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": "serpentguard-m5-test-inputs", "id": DATASET, "licenses": [{"name": "other"}]}, indent=1))
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB in {OUT}")


if __name__ == "__main__":
    main()
