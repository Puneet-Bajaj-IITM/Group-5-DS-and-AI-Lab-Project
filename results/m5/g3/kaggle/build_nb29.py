"""Writes notebooks/29_m5_kd_pilot.ipynb into the repo given as the first argument.

It reuses notebook 28's setup, input and B1-check cells (28 must be built first) and adds the pilot cells.
"""
import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
SOURCE = REPO / "notebooks" / "28_m5_finetune.ipynb"
NOTEBOOK = REPO / "notebooks" / "29_m5_kd_pilot.ipynb"
REUSED = ("c01", "c02", "c03", "c04", "c05")  # setup, inputs, B1 check

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md("""
# A small pilot for the distillation settings

Notebook 28's kd run distils from B1 with a weight of 1.0, a temperature of 2 and a learning rate of 2e-4,
settings that were not chosen on a pilot. This pilot compares them, on the 16% experiment subset only,
with A3's own losses (no distillation) and with one other distillation setting: each starts from A3's
`best.pt`, trains 2 epochs on the subset's training photos cut by the M4 front end, and is scored on every
validation photo. The pick is by the G3 slice score (ours minus B1, top-1 and macro-F1 over B1's 62
species, photos from 2022) and by validation macro-F1 over all species. Calibration photos are not used.
B1 runs on the subset's training photos of its 62 species on the second GPU while the first builds the
front end. About 1.5 hours on two T4 GPUs.
""")

PILOT_CELLS = [
    ('code', '''
SETTINGS = {  # name: (losses added to A3's, kd_weight, kd_temperature)
    "no kd": ([], None, None),
    "kd w1 T2 (the full run)": (["kd"], 1.0, 2.0),
    "kd w3 T4": (["kd"], 3.0, 4.0),
}
PILOT_EPOCHS = 2
SUBSET_TRAIN = data.cap_train(metadata[(metadata["split"] == "train") & metadata["path"].isin(subset["path"])])
calibration = data.calibration_set(metadata[metadata["split"] == "train"], subset)
unknown = data.unknown_split(metadata[metadata["split"] == "unknown_pool"])[0]
val_rows = metadata[metadata["split"] == "val"]
front_end_rows = pd.concat([SUBSET_TRAIN[["path"]], calibration[["path"]], val_rows[["path"]],
                            unknown[["path"]]]).drop_duplicates("path").reset_index(drop=True)
assert_no_test_paths(front_end_rows)
print(f"pilot: {len(SUBSET_TRAIN):,} subset training photos; the front end covers {len(front_end_rows):,}")

TEACHER_DIR = SCRATCH / "teacher"
TEACHER_RUN = TEACHER_DIR / "b1_subset.npz"
TEACHER_DIR.mkdir(parents=True, exist_ok=True)
labelled = SUBSET_TRAIN.merge(classes[["species", "class_id"]], on="species", how="left", validate="many_to_one")
labelled[labelled["class_id"].isin(B1_CLASSES)][["path"]].to_csv(TEACHER_DIR / "photos.csv", index=False)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "onnxruntime"])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx==1.23.2", "onnxruntime-gpu==1.31.1"], check=True)
log = open(WORK / "b1_teacher.log", "w")
teacher_process = subprocess.Popen(
    [sys.executable, str(REPO / "scripts/b1_teacher.py"), str(DATA), str(TEACHER_DIR / "photos.csv"),
     str(B1_DIR / "predictions.npz"), str(SCRATCH / "downloads"), str(TEACHER_RUN)],
    stdout=log, stderr=subprocess.STDOUT,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "1" if torch.cuda.device_count() > 1 else "0"})

BANK = SCRATCH / "scenery_bank"
BANK_FROM = {"run_id": "scenery_bank", "front_end": "F1", "scenery_swap": 0.5,
             "mask_source": "predicted", "train_list": None, "scenery_bank": None}
BANK_SHA = "2eed027132bfacf956e85895b29131fa09f845c5b60bfcfa58272a687544fa5e"
a3 = data.run_config(yaml.safe_load((REPO / "results/m4/full/A3/config.yaml").read_text()))
bank = data.build_scenery_bank(data.make_frames({**a3, **BANK_FROM}, DATA, MASKS, SUBSET, M3_PREDS)["train"], BANK)
assert sha256(BANK / data.BANK_FILE) == BANK_SHA, "the bank differs from A3's"

EMPTY_BASE, PREDS = SCRATCH / "empty_base", SCRATCH / "m4_preds"
(EMPTY_BASE / front_end.MASKS_DIR).mkdir(parents=True, exist_ok=True)
(EMPTY_BASE / front_end.PREDS_FILE).write_text(front_end.PREDS_COLUMNS)
joined = front_end.extend_predictions(front_end_rows, EMPTY_BASE, DETECTOR, SEGMENTER, DATA, PREDS)
assert set(joined["path"]) == set(front_end_rows["path"]) and joined["path"].is_unique
assert_no_test_paths(joined)
print(f"M4 front end: {len(joined):,} rows, {joined['box'].ne('').mean():.1%} boxed, "
      f"{joined['mask_file'].ne('').mean():.1%} masked")
'''),
    ('code', '''
while teacher_process.poll() is None:
    time.sleep(30)
assert teacher_process.returncode == 0, "B1 failed: read b1_teacher.log"
done = np.load(TEACHER_RUN)
val_known = np.isin(b1["labels"], B1_CLASSES)
TEACHER = TEACHER_DIR / "pilot_teacher.npz"
np.savez(TEACHER, paths=np.concatenate([done["paths"], b1["paths"][val_known]]).astype(str),
         probs=np.concatenate([done["probs"], b1["probs"][val_known]]).astype(np.float32))
print(f"teacher: {len(done['paths']):,} training photos, {val_known.sum():,} validation photos; "
      f"check {Path(f'{TEACHER_RUN}.check.json').read_text()}")
'''),
    ('code', '''
def g3_slice(run_dir):
    ours = load_outputs(run_dir / VAL_OUTPUTS)
    labels = ours["labels"]
    b1_on = b1_pred[rows_for_b1(ours["paths"])]
    groups = photographer_groups(pd.DataFrame({"observer_id": ours["observer_id"], "path": ours["paths"]}))
    rows = uploaded_2022(ours["paths"]) & np.isin(labels, B1_CLASSES)
    assert rows.sum() == 11_931
    pred = ours["species_logits"].argmax(1)
    result = {}
    for metric, score in (("top1", top1), ("macro_f1", macro_f1)):
        diff, low, high = paired_difference_ci(labels[rows], pred[rows], b1_on[rows], groups[rows], score)
        result[metric] = {"diff": diff, "low": low, "high": high}
    shuffled = np.random.default_rng(0).permutation(pred)
    control = paired_difference_ci(labels[rows], shuffled[rows], b1_on[rows], groups[rows], top1)
    assert control[2] < -0.3, "the shuffled control must lose clearly"
    result["control_top1_shuffled"] = control
    return result


def rows_for_b1(paths):
    position = pd.Series(np.arange(len(b1["paths"])), index=b1["paths"]).reindex(paths)
    assert position.notna().all()
    return position.to_numpy(dtype=int)


table = {}
for name, (extra, weight, temperature) in SETTINGS.items():
    run_id = "PILOT_" + "".join(ch if ch.isalnum() else "_" for ch in name)
    cfg = {**a3, "run_id": run_id, "train_list": None, "epochs": PILOT_EPOCHS, "eval_every": 1, "lr": LR,
           "warmup_epochs": 1, "init_from": str(A3_BEST), "scenery_bank": str(BANK),
           "losses": [*a3["losses"], *extra]}
    if extra:
        cfg.update(teacher_probs=str(TEACHER), kd_weight=weight, kd_temperature=temperature)
    run_dir = WORK / "pilot" / run_id
    final = train(cfg, run_dir, DATA, MASKS, SUBSET, preds_root=PREDS)
    history = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    table[name] = {"run": run_id, "train_photos": final["run"]["train_photos"],
                   "best_epoch": final["run"]["best_epoch"], "val_macro_f1": final["head"]["macro_f1"],
                   "val_top1": final["head"]["top1"], "history": history, "g3_slice": g3_slice(run_dir)}
    write_json(WORK / "pilot" / "pilot.json", table)
    g = table[name]["g3_slice"]
    print(f"{name}: val macro-F1 {final['head']['macro_f1']:.4f}, top-1 {final['head']['top1']:.4f}; G3 top-1 "
          f"{g['top1']['diff']:+.4f} ({g['top1']['low']:+.4f} to {g['top1']['high']:+.4f}), macro-F1 "
          f"{g['macro_f1']['diff']:+.4f} ({g['macro_f1']['low']:+.4f} to {g['macro_f1']['high']:+.4f})", flush=True)
'''),
]

source = json.loads(SOURCE.read_text())
reused = [(c["cell_type"], "".join(c["source"])) for c in source["cells"] if c["id"] in REUSED]
setup = reused[0][1].replace('SCRATCH = Path("/kaggle/tmp/m5_finetune")', 'SCRATCH = Path("/kaggle/tmp/m5_pilot")')
reused[0] = ("code", setup.replace('VARIANT = "finetune"', 'VARIANT = "kd"'))
CELLS += reused + PILOT_CELLS


def cell(index, kind, text):
    base = {"cell_type": kind, "id": f"c{index:02d}", "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


notebook = {**source, "cells": [cell(i, kind, text) for i, (kind, text) in enumerate(CELLS)]}
NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"{NOTEBOOK}: {len(CELLS)} cells")
