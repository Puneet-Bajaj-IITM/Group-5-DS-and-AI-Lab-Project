"""Writes notebooks/28_m5_finetune.ipynb (no outputs) into the repo given as the first argument."""
import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
NOTEBOOK = REPO / "notebooks" / "28_m5_finetune.ipynb"

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md("""
# Fine-tuning A3 on the app's own crops

A3, the model the app ships, was trained on crops cut by Milestone 3's front end (detector and U-Net),
but the app cuts its crops with Milestone 4's front end, which costs A3 about 3 points of top-1. This
notebook fine-tunes A3 on crops from the M4 front end, for a few epochs at a lower learning rate, with
everything else in A3's recipe unchanged (notebook 21: the same 49,782 training photos, scenery swap
0.5 with predicted masks, the weighted sampler). It logs the validation loss at every evaluation and
saves the full training state (`last.pt`) next to the best epoch (`best.pt`).

`VARIANT` picks one of two runs:

- **finetune**: A3's own losses;
- **kd**: the same plus a distillation term from B1, the published SnakeCLEF 2021 model (Bolon et al.
  2022), on the training photos of the 62 species B1 knows. B1 runs on those photos on the second GPU
  while the first builds the front end and runs the smoke pass (`scripts/b1_teacher.py`).

Everything happens in one session: the inputs and their checksums; the M4 front end over every photo
the run reads (training list, calibration, validation, unknown species), starting from an empty file;
a smoke pass (300 photos, one epoch, every step); then the full pass. The full pass also scores A3
itself on the same M4 crops (the "before"), and saves the new model's outputs with each photo mirrored
(flip test-time augmentation). The test splits are never read.

Run on Kaggle with two T4 GPUs and internet on: about 3 hours.
""")

code('''
import time
STARTED = time.time()

%pip install -q timm==1.0.31 albumentations==2.0.8 scikit-learn==1.9.1
%pip install -q ultralytics==8.4.202 segmentation-models-pytorch==0.6.0
import hashlib, json, os, shutil, subprocess, sys, traceback, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

os.environ.update(NO_ALBUMENTATIONS_UPDATE="1", YOLO_VERBOSE="False",
                  HF_HUB_DISABLE_PROGRESS_BARS="1", TQDM_DISABLE="1")
warnings.filterwarnings("ignore")

import kagglehub, torch, yaml
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ON_KAGGLE = Path("/kaggle/working").exists()
if ON_KAGGLE:
    REPO = next(p for p in (Path("/kaggle/working/repo"), Path("/kaggle/input/serpentguard-code"))
                if (p / "src").exists())
else:
    REPO = Path("..").resolve()
sys.path.insert(0, str(REPO / "src"))
WORK = Path("/kaggle/working") if ON_KAGGLE else Path("work_m5")
SCRATCH = Path("/kaggle/tmp/m5_finetune") if ON_KAGGLE else WORK / "tmp"

from serpentguard import data, front_end, robustness
from serpentguard.decide import decision_report, load_outputs
from serpentguard.metrics import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, assert_no_test_paths,
                                  class_tables, paired_difference_ci, photographer_groups, softmax,
                                  species_report, write_json)
from serpentguard.train import (CALIBRATION_OUTPUTS, UNKNOWN_OUTPUTS, VAL_OUTPUTS, load_checkpoint,
                                predict, save_outputs, train)

ACCOUNT = "jayesh"     # the Kaggle account this session runs on
VARIANT = "finetune"   # finetune, or kd (adds distillation from B1)
EPOCHS = 5             # fine-tuning epochs, each evaluated
LR = 2e-4              # A3 trained at 9.5e-4; the encoder learns at a tenth of this
KD_WEIGHT, KD_TEMPERATURE = 1.0, 2.0
SMOKE_PHOTOS, SMOKE_EPOCHS = 300, 1
TIME_LIMIT_HOURS = 11
TEACHER_WAIT_HOURS = 2.5  # the session time by which the full pass starts with whatever B1 finished
assert VARIANT in ("finetune", "kd"), VARIANT
RUN_ID = {"finetune": "M5_FT", "kd": "M5_KD"}[VARIANT]
PASSED = set()
SMOKE_PASSED = FULL_PASSED = False
for folder in (WORK, SCRATCH):
    folder.mkdir(parents=True, exist_ok=True)
print(f"{torch.cuda.device_count()} GPU(s), account {ACCOUNT}, variant {VARIANT}; "
      f"results go to {WORK}")
''')

md("""
## 1. The inputs

The cleaned photos, the experiment subset and the SAM masks are shared with the team. The M4 front end
(the detector and U-Net of the gate notebook, `puneetbajajiitm/sg-m4-gate`) and A3's `best.pt`
(`jayeshbansal497/sg-m4-full-a3`) are copied into one private dataset; M3's front-end file is read only
to rebuild A3's scenery bank exactly; B1's saved validation predictions and M3's final run come from
notebook 21's private copies. Every file is checked by its SHA-256.
""")

code('''
ACCOUNTS = {"puneet": "puneetbajajiitm", "jayesh": "jayeshbansal497", "vikas": "vikas25s"}
SHARED = "puneetbajajiitm"
CHECKSUMS = {
    "detector": "29a1ad1c1b9b561c35ef4ffa1916998837c3c78525c21b51668d9d0e95607b61",
    "segmenter": "b881c338356bdf44aa0b45e0306f59d7b38252d04316da2455c40055aef656ec",
    "a3": "b19f0a5ece4f98fc08911f8781048011e14aeb84b80fde9dc1f4b9263322a02b",
    "m3_preds": "9dcc0576eb1c60d3e767ba9ca3c34a6a1b92b6b01946e377042dee1b1287b9e3",
    "b1": "f701e21261d690d9e3d7cb61b44ace3dc05660f82d4d977db2b92149f3e93251"}
OWNER = ACCOUNTS[ACCOUNT]


def dataset(handle):
    return Path(kagglehub.dataset_download(handle))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def found(root, pattern):
    return next(Path(root).rglob(pattern))


DATA = dataset(f"{SHARED}/serpentguard-indian-snakes-clean")
SUBSET = dataset(f"{SHARED}/serpentguard-m3-subset") / "subset.csv"
MASKS = dataset(f"{SHARED}/serpentguard-sam-masks")
M5_INPUTS = dataset(f"{OWNER}/serpentguard-m5-inputs")
DETECTOR = found(M5_INPUTS, "M4_detector.pt")
SEGMENTER = found(M5_INPUTS, "segmenter/segmenter.pt")  # extend_predictions reads <folder>/segmenter.pt
A3_BEST = found(M5_INPUTS, "a3/best.pt")
M3_PREDS = found(dataset(f"{OWNER}/serpentguard-m4-frontend-masks"), front_end.PREDS_FILE).parent
B1_DIR = found(dataset(f"{OWNER}/serpentguard-m3-b1-predictions"), "predictions.npz").parent
M3_RUN = found(dataset(f"{OWNER}/serpentguard-m3-run-final"), VAL_OUTPUTS).parent
for name, path in (("detector", DETECTOR), ("segmenter", SEGMENTER), ("a3", A3_BEST),
                   ("m3_preds", M3_PREDS / front_end.PREDS_FILE), ("b1", B1_DIR / "predictions.npz")):
    assert sha256(path) == CHECKSUMS[name], f"{path} is not the {name} file this notebook expects"

ids = {"obs_id": str, "photo_id": str, "observer_id": str}
metadata = pd.read_csv(DATA / "metadata.csv", dtype=ids, low_memory=False)
classes = pd.read_csv(DATA / "classes.csv")
subset = pd.read_csv(SUBSET, dtype=ids)
taxa = class_tables(classes)
print(f"{len(metadata):,} photos of {len(classes)} species; every checksum matches")
PASSED.add("inputs")
''')

md("""
### B1's saved predictions

As in notebook 21: B1's saved predictions must still give M3's recorded B1 scores before any GPU time.
The G3 headline macro-F1 is over the 62 species, on the photos of those species only.
""")

code('''
SLICE_FROM = "2022-01-01"
SLICE_PHOTOS, B1_SPECIES = 13_947, 62

b1 = load_outputs(B1_DIR / "predictions.npz")
b1_metrics = json.loads((B1_DIR / "final_metrics.json").read_text())
m3_comparison = json.loads((B1_DIR / "compare_b1_shared.json").read_text())
B1_CLASSES = np.array(sorted(b1_metrics["coverage"]["covered_class_ids"]))
assert len(B1_CLASSES) == B1_SPECIES == m3_comparison["shared_classes"]


def uploaded_2022(paths):
    created = pd.DataFrame({"path": paths}).merge(metadata[["path", "created_at"]], on="path",
                                                  how="left", validate="many_to_one")
    assert created["created_at"].notna().all(), "a photo has no upload date"
    return created["created_at"].astype(str).ge(SLICE_FROM).to_numpy()


def top1(y_true, pred):
    return float(np.mean(y_true == pred))


def macro_f1(y_true, pred):
    present = np.intersect1d(B1_CLASSES, y_true)
    return float(f1_score(y_true, pred, labels=present, average="macro", zero_division=0))


b1_pred = b1["probs"].argmax(axis=1)
keep = uploaded_2022(b1["paths"])
assert keep.sum() == SLICE_PHOTOS
shared_rows = keep & np.isin(b1["labels"], B1_CLASSES)
recorded = m3_comparison["upload_2022_slice"]["b1"]
assert shared_rows.sum() == recorded["photos"]
assert abs(top1(b1["labels"][shared_rows], b1_pred[shared_rows]) - recorded["top1"]) < 1e-9
PASSED.add("B1 check")
print(f"B1 on the 2022 slice: {shared_rows.sum():,} photos of its species, top-1 "
      f"{recorded['top1']:.4f} as recorded")
''')

md("""
## 2. The training list and the photos the front end must cover

The training list is notebook 21's: 49,782 photos of all 184 species, no calibration photographer. The
M4 front end must give a row to every photo the run reads: this list, the calibration photos, the
validation photos and the unknown-species photos.
""")

code('''
FULL_PHOTOS, FULL_SPECIES, SUBSET_FINGERPRINT = 49_782, 184, "40196d19a68e"

fingerprint = hashlib.sha256("\\n".join(sorted(subset["path"])).encode()).hexdigest()
assert fingerprint.startswith(SUBSET_FINGERPRINT), "this is not the experiment subset"
full = data.cap_train(data.full_train_rows(metadata, subset))
train_split = metadata[metadata["split"] == "train"]
calibration = data.calibration_set(train_split, subset)
assert len(full) == FULL_PHOTOS and full["species"].nunique() == FULL_SPECIES, "not the full list"
assert not set(full["observer_id"]) & set(calibration["observer_id"]), "a calibration photographer"
unknown = data.unknown_split(metadata[metadata["split"] == "unknown_pool"])[0]
val_rows = metadata[metadata["split"] == "val"]
SCORED_PHOTOS = {"val": len(val_rows), "calibration": len(calibration), "unknown": len(unknown)}
LISTS = {"smoke": WORK / "smoke" / "smoke_train.csv", "full": WORK / "full" / "full_train.csv"}
for name, rows in (("smoke", full.head(SMOKE_PHOTOS)), ("full", full)):
    LISTS[name].parent.mkdir(parents=True, exist_ok=True)
    rows[["path"]].to_csv(LISTS[name], index=False)
front_end_rows = pd.concat([full[["path"]], calibration[["path"]], val_rows[["path"]],
                            unknown[["path"]]]).drop_duplicates("path").reset_index(drop=True)
assert_no_test_paths(front_end_rows)
print(f"{FULL_PHOTOS:,} training photos; scored sets {SCORED_PHOTOS}; the front end covers "
      f"{len(front_end_rows):,} photos")
PASSED.add("training list")
''')

md("""
## 3. B1 as the teacher (kd only)

B1 runs on the training photos of its 62 species on the second GPU, in its own process, while this one
builds the front end and runs the smoke pass. The script first scores 64 validation photos whose
notebook-11 probabilities are saved: they must come back within 0.002, and with the colour channels
swapped they must not (the control). The smoke's training photos come first in its list, so the smoke
can distil from them; the full pass waits for B1 to finish, or until `TEACHER_WAIT_HOURS` into the
session, and then distils from the photos B1 has done.

The teacher file also holds B1's saved probabilities for the validation photos of its 62 species, so
the validation loss has its kd term too (logged only; nothing is fitted on validation).
""")

code('''
TEACHER_DIR = SCRATCH / "teacher"
TEACHER_RUN = TEACHER_DIR / "b1_train.npz"
teacher_process = None


def teacher_photos():
    """The training photos of B1's species, the smoke's first."""
    labelled = full.merge(classes[["species", "class_id"]], on="species", how="left",
                          validate="many_to_one")
    known = labelled[labelled["class_id"].isin(B1_CLASSES)]
    smoke_first = known["path"].isin(full.head(SMOKE_PHOTOS)["path"])
    return pd.concat([known[smoke_first], known[~smoke_first]])[["path"]]


def teacher_file(name):
    """An npz of B1's probabilities for the training photos done so far and the validation photos."""
    parts = sorted(Path(f"{TEACHER_RUN}.parts").glob("*.npz"))
    done = [np.load(p) for p in parts]
    val_known = np.isin(b1["labels"], B1_CLASSES)
    paths = np.concatenate([*(d["paths"] for d in done), b1["paths"][val_known]]).astype(str)
    probs = np.concatenate([*(d["probs"] for d in done), b1["probs"][val_known]]).astype(np.float32)
    assert len(set(paths)) == len(paths), "a photo has two teacher rows"
    out = TEACHER_DIR / f"{name}_teacher.npz"
    np.savez(out, paths=paths, probs=probs)
    trained = sum(len(d["paths"]) for d in done)
    print(f"{name} teacher: {trained:,} training photos and {val_known.sum():,} validation photos")
    return out, trained


if VARIANT == "kd":
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "onnxruntime"])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx==1.23.2",
                    "onnxruntime-gpu==1.31.1"], check=True)
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    photos = teacher_photos()
    photos.to_csv(TEACHER_DIR / "photos.csv", index=False)
    gpu = "1" if torch.cuda.device_count() > 1 else "0"
    log = open(WORK / "b1_teacher.log", "w")
    teacher_process = subprocess.Popen(
        [sys.executable, str(REPO / "scripts/b1_teacher.py"), str(DATA), str(TEACHER_DIR / "photos.csv"),
         str(B1_DIR / "predictions.npz"), str(SCRATCH / "downloads"), str(TEACHER_RUN)],
        stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
    print(f"B1 started on GPU {gpu} for {len(photos):,} training photos")
PASSED.add("teacher")
''')

md("""
## 4. The recipe

A3's own `config.yaml` (results/m4/full/A3) with these changes: the run's name, the training list, the
epochs, an evaluation after every epoch, the lower learning rate (one warm-up epoch, then cosine to 0)
and `init_from`, A3's `best.pt`. The kd run adds the `kd` loss and its teacher file. The scenery bank
is rebuilt exactly as notebook 21 built it (from the subset's training photos with M3's masks), and
its `bank.csv` must have A3's fingerprint.
""")

code('''
BANK = SCRATCH / "scenery_bank"
BANK_FROM = {"run_id": "scenery_bank", "front_end": "F1", "scenery_swap": 0.5,
             "mask_source": "predicted", "train_list": None, "scenery_bank": None}
BANK_SHA = "2eed027132bfacf956e85895b29131fa09f845c5b60bfcfa58272a687544fa5e"

a3 = data.run_config(yaml.safe_load((REPO / "results/m4/full/A3/config.yaml").read_text()))
assert a3["scenery_swap"] == 0.5 and a3["mask_source"] == "predicted" and a3["sampler"] == "weighted"
bank_frames = data.make_frames({**a3, **BANK_FROM}, DATA, MASKS, SUBSET, M3_PREDS)
bank = data.build_scenery_bank(bank_frames["train"], BANK)
assert sha256(BANK / data.BANK_FILE) == BANK_SHA, "the bank differs from A3's"


def recipe(run_id, train_list, epochs, teacher=None):
    cfg = {**a3, "run_id": run_id, "train_list": str(train_list), "epochs": epochs, "eval_every": 1,
           "lr": LR, "warmup_epochs": 1, "init_from": str(A3_BEST), "scenery_bank": str(BANK)}
    if VARIANT == "kd":
        cfg.update(losses=[*a3["losses"], "kd"], teacher_probs=str(teacher), kd_weight=KD_WEIGHT,
                   kd_temperature=KD_TEMPERATURE)
    return cfg


print(f"scenery bank: {len(bank):,} sceneries, A3's fingerprint; A3 trained with lr {a3['lr']:.2e}, "
      f"this run {LR:.2e} for {EPOCHS} epochs from {A3_BEST.name}")
PASSED.add("recipe")
''')

md("""
## 5. The M4 front end

The M4 detector's most confident box at 0.05 or above and the M4 U-Net's mask (kept only if the cleanup
keeps it) for every photo above, with `front_end.extend_predictions` started from an empty file. It
runs in two stages, so that the smoke pass can start early: first the scored sets and the smoke's 300
training photos, then (before the full pass) the rest of the training list, appended to the first
file. Each file is read back: one row per photo, every photo present, no test photo, a confidence
with every box, and every listed mask on disk.
""")

code('''
EMPTY_BASE = SCRATCH / "empty_base"
SCORED_PREDS, ALL_PREDS = SCRATCH / "m4_preds_scored", SCRATCH / "m4_preds_all"


def run_front_end(rows, base, out):
    """The M4 front end's file for `rows`, appended to the one in `base`, written to `out`."""
    started = time.time()
    joined = front_end.extend_predictions(rows, base, DETECTOR, SEGMENTER, DATA, out)
    assert joined["path"].is_unique, "a photo has two rows"
    assert set(rows["path"]) == set(joined["path"]), "a photo has no row, or an extra one"
    assert_no_test_paths(joined)
    assert joined["box"].ne("").equals(joined["box_conf"].ne("")), "a box without its confidence"
    masks = [out / name for name in joined.loc[joined["mask_file"].ne(""), "mask_file"]]
    with ThreadPoolExecutor(16) as pool:
        assert all(pool.map(os.path.isfile, masks)), "a listed mask is missing"
    split = joined.merge(metadata[["path", "split"]], on="path", how="left")["split"]
    cover = joined.assign(split=split.to_numpy()).groupby("split").agg(
        photos=("path", "size"), box=("box", lambda b: b.ne("").mean()),
        mask=("mask_file", lambda m: m.ne("").mean()))
    print(f"M4 front end: {len(joined):,} photos, this stage {(time.time() - started) / 60:.0f} min")
    print(cover.round(3).to_string())
    return out


assert {"inputs", "B1 check", "training list", "recipe", "teacher"} <= PASSED
(EMPTY_BASE / front_end.MASKS_DIR).mkdir(parents=True, exist_ok=True)
(EMPTY_BASE / front_end.PREDS_FILE).write_text(front_end.PREDS_COLUMNS)
first_rows = front_end_rows[~front_end_rows["path"].isin(full["path"].iloc[SMOKE_PHOTOS:])]
PREDS = run_front_end(first_rows, EMPTY_BASE, SCORED_PREDS)  # the smoke reads this one
PASSED.add("front end")
''')

md("""
## 6. The steps of one pass

As notebook 21: training with a time check, reading the run back (with a shuffled-label control),
verdicts with thresholds fitted on calibration only, species counts, the comparison with B1 and the
scenery test. Two more steps here: the new model's outputs with every photo mirrored (flip test-time
augmentation, the average is taken later), and, in the full pass, A3 itself on the same M4 crops.
""")

code('''
HOUR, SCORING_MINUTES = 3600, 40
WHO_RECALL_FLOOR = 0.95
F1_LEVELS, MIN_VAL_PHOTOS, MEASURABLE = (0.3, 0.5, 0.7), 5, 158
NO_COUNTS = np.zeros(taxa.n_classes, dtype=int)
CONTROL_BELOW = -0.3
FLIP_OUTPUTS = {"val": "val_outputs_flip.npz", "calibration": "calib_outputs_flip.npz",
                "unknown": "unknown_outputs_flip.npz"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def time_check(cfg, train_photos):
    def check(epoch, line):
        left = ((cfg["epochs"] - epoch) * (line["seconds"] + line["eval_seconds"])
                + (train_photos + sum(SCORED_PHOTOS.values()) * 3) / (SCORED_PHOTOS["val"]
                                                                      / line["eval_seconds"])
                + SCORING_MINUTES * 60)
        hours = (time.time() - STARTED + left) / HOUR
        print(f"projected session length {hours:.1f} h (limit {TIME_LIMIT_HOURS} h)", flush=True)
        if epoch == 1 and hours > TIME_LIMIT_HOURS:
            raise RuntimeError(f"the session would take {hours:.1f} h, over {TIME_LIMIT_HOURS} h")
    return check


def read_back(run_dir, final, cfg, limit):
    history = pd.DataFrame([json.loads(line)
                            for line in (run_dir / "metrics.jsonl").read_text().splitlines()])
    assert history["epoch"].tolist() == list(range(1, cfg["epochs"] + 1)), "an evaluation is missing"
    assert {"val_loss_total", "val_loss_species"} <= set(history), "no validation loss logged"
    if VARIANT == "kd":
        assert "loss_kd" in history and "val_loss_kd" in history, "no kd term logged"
    assert (run_dir / "last.pt").exists() and (run_dir / "best.pt").exists()
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert last["epoch"] == cfg["epochs"] and {"optimizer", "scheduler", "scaler", "rng"} <= set(last)
    assert final["run"]["train_photos"] == (FULL_PHOTOS if limit is None else SMOKE_PHOTOS)
    val_out = load_outputs(run_dir / VAL_OUTPUTS)
    val_counts = np.bincount(val_out["labels"], minlength=taxa.n_classes)

    def head_macro_f1(labels):
        return species_report(labels, softmax(val_out["species_logits"]), NO_COUNTS, val_counts,
                              taxa.tier)["macro_f1"]

    recomputed = head_macro_f1(val_out["labels"])
    control = head_macro_f1(np.random.default_rng(0).permutation(val_out["labels"]))
    assert np.isclose(recomputed, final["head"]["macro_f1"], rtol=0, atol=1e-9,
                      equal_nan=limit is not None), "the saved outputs disagree"
    if limit is None:
        assert control < 0.02, "with shuffled labels the score must fall to chance"
    print(f"{cfg['run_id']}: best epoch {final['run']['best_epoch']} of {cfg['epochs']}; macro-F1 "
          f"{recomputed:.4f} from {VAL_OUTPUTS}, {final['head']['macro_f1']:.4f} recorded, "
          f"{control:.4f} with shuffled labels")
    return history


def score_full_calibration(run_dir, cfg):
    _, model, _ = load_checkpoint(run_dir, DEVICE)
    frame = data.make_frames(cfg, DATA, MASKS, SUBSET, PREDS)["calibration"]
    save_outputs(run_dir / CALIBRATION_OUTPUTS, frame,
                 predict(model, data.eval_loader(cfg, frame), DEVICE, DEVICE.type == "cuda"))
    return len(frame)


def verdicts(run_dir, cfg, out, limit):
    preds = pd.read_csv(PREDS / front_end.PREDS_FILE)
    box_conf = dict(zip(preds["path"], preds["box_conf"]))
    quality = robustness.quality_table(run_dir, DATA)
    try:
        decision = decision_report(run_dir, DATA, quality=quality, box_conf=box_conf,
                                   recall_bound="lower")
    except ValueError as error:
        if limit is None:
            raise
        print(f"smoke: the fit failed ({error}); scoring all calibration photos, "
              f"{score_full_calibration(run_dir, cfg):,}")
        quality = robustness.quality_table(run_dir, DATA)
        decision = decision_report(run_dir, DATA, quality=quality, box_conf=box_conf,
                                   recall_bound="lower")
    shutil.copy2(run_dir / "decision_metrics.json", out / "decision_metrics.json")
    print(f"{cfg['run_id']}: thresholds fitted on {decision['fit']['photos']:,} calibration photos, "
          f"WHO recall lower bound {decision['recall_bound_fit']['lower_bound']:.3f}; WHO recall "
          f"{decision['verdict_metrics']['who_dangerous_recall']:.3f} on validation")
    return decision, quality


def measured_f1(final_metrics, counts):
    return np.array([final_metrics["head"]["per_class"][str(c)]["f1"]
                     for c in np.flatnonzero(counts >= MIN_VAL_PHOTOS)])


def class_counts(f1):
    return {"measurable": len(f1),
            **{f"f1_at_least_{level}": int((f1 >= level).sum()) for level in F1_LEVELS},
            "median_f1": float(np.median(f1)), "f1_zero": int((f1 == 0).sum())}


def species_counts(final, run_dir, out, limit):
    labels = load_outputs(run_dir / VAL_OUTPUTS)["labels"]
    counts = class_counts(measured_f1(final, np.bincount(labels, minlength=taxa.n_classes)))
    if limit is None:
        assert counts["measurable"] == MEASURABLE, "not 158 species"
    write_json(out / "per_class_counts.json", {"levels": F1_LEVELS, "min_val_photos": MIN_VAL_PHOTOS,
                                               final["run"]["run_id"]: counts})
    return counts


def rows_for(saved, paths):
    position = pd.Series(np.arange(len(saved["paths"])), index=saved["paths"]).reindex(paths)
    assert position.notna().all(), "a validation photo is missing from a saved set"
    return position.to_numpy(dtype=int)


def standing(interval):
    if interval["diff"] > 0:
        return "higher"
    return "tied" if interval["low"] <= 0 <= interval["high"] else "lower"


def compare_with_b1(run_dir, out, limit, extra=None):
    """Ours minus B1 on B1's 62 species, 2022 slice, photographer bootstrap (seed 42)."""
    ours = load_outputs(run_dir / VAL_OUTPUTS)
    labels = ours["labels"]
    b1_rows = rows_for(b1, ours["paths"])
    assert (b1["labels"][b1_rows] == labels).all()
    groups = photographer_groups(pd.DataFrame({"observer_id": ours["observer_id"],
                                               "path": ours["paths"]}))
    predicted = {"ours": ours["species_logits"].argmax(axis=1), **(extra or {})}
    predicted["ours, predictions shuffled"] = np.random.default_rng(0).permutation(predicted["ours"])
    b1_on = b1_pred[b1_rows]
    keep = uploaded_2022(ours["paths"])
    rows = keep & np.isin(labels, B1_CLASSES)
    result = {"photos": int(rows.sum()), "photographers": int(np.unique(groups[rows]).size)}
    for model, pred in predicted.items():
        result[model] = {}
        for metric, score in (("top1", top1), ("macro_f1", macro_f1)):
            diff, low, high = paired_difference_ci(labels[rows], pred[rows], b1_on[rows], groups[rows],
                                                   score)
            result[model][metric] = {"diff": diff, "low": low, "high": high}
    control = result["ours, predictions shuffled"]
    if limit is None:
        assert result["photos"] == m3_comparison["upload_2022_slice"]["b1"]["photos"]
        assert max(control[m]["high"] for m in ("top1", "macro_f1")) < CONTROL_BELOW, "control"
    g3 = {metric: standing(result["ours"][metric]) for metric in ("top1", "macro_f1")}
    g3["passed"] = "lower" not in g3.values()
    write_json(out / "b1_62.json", {"run": run_dir.name, "resamples": BOOTSTRAP_RESAMPLES,
                                    "seed": BOOTSTRAP_SEED, "upload_2022_slice": result, "g3": g3})
    return result, g3


def scenery_test(run_dir, quality, out, limit):
    (run_dir / "r1").mkdir(exist_ok=True)
    report = robustness.robustness_metrics(run_dir, DATA, PREDS, subset_csv=SUBSET, quality=quality,
                                           out_dir=run_dir / "r1", limit=limit)
    return write_json(out / "robustness_metrics.json", report)


class Mirrored(torch.nn.Module):
    """The model on each photo mirrored left to right."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(torch.flip(images, dims=[3]))


def flip_outputs(run_dir, cfg, limit):
    _, model, _ = load_checkpoint(run_dir, DEVICE)
    frames = data.make_frames(cfg, DATA, MASKS, SUBSET, PREDS, limit)
    mirrored = Mirrored(model).eval()
    for name, file in FLIP_OUTPUTS.items():
        outputs = predict(mirrored, data.eval_loader(cfg, frames[name]), DEVICE, DEVICE.type == "cuda")
        save_outputs(run_dir / file, frames[name], outputs)
    flipped, plain = load_outputs(run_dir / FLIP_OUTPUTS["val"]), load_outputs(run_dir / VAL_OUTPUTS)
    assert (flipped["paths"] == plain["paths"]).all(), "the mirrored pass is in another order"
    agree = float((flipped["species_logits"].argmax(1) == plain["species_logits"].argmax(1)).mean())
    print(f"{cfg['run_id']}: mirrored outputs saved; top species agree on {agree:.1%} of val photos")
    return agree


def a3_on_m4_crops(cfg):
    """A3 itself on the same M4 crops: the 'before' of this fine-tune."""
    out_dir = WORK / "A3_m4crops"
    out_dir.mkdir(exist_ok=True)
    a3_cfg, model, epoch = load_checkpoint(A3_BEST.parent, DEVICE)
    frames = data.make_frames(cfg, DATA, MASKS, SUBSET, PREDS)
    for name, file in (("val", VAL_OUTPUTS), ("calibration", CALIBRATION_OUTPUTS),
                       ("unknown", UNKNOWN_OUTPUTS)):
        outputs = predict(model, data.eval_loader(a3_cfg, frames[name]), DEVICE, DEVICE.type == "cuda")
        save_outputs(out_dir / file, frames[name], outputs)
    saved = load_outputs(out_dir / VAL_OUTPUTS)
    print(f"A3 (epoch {epoch}) on M4 crops: top-1 "
          f"{(saved['species_logits'].argmax(1) == saved['labels']).mean():.4f} on validation")
    return out_dir


def run_pass(name, epochs, limit=None, timed=False, teacher=None):
    started = time.time()
    out = WORK / name
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(LISTS[name])
    cfg = recipe(RUN_ID if name == "full" else f"{RUN_ID}_{name}", LISTS[name], epochs, teacher)
    run_dir = WORK / cfg["run_id"]
    final = train(cfg, run_dir, DATA, MASKS, SUBSET, preds_root=PREDS, limit=limit,
                  on_eval=time_check(cfg, len(rows)) if timed else None)
    done, failed = {}, {}

    def quality():
        return done["verdicts"][1] if "verdicts" in done else None

    steps = {"read back": lambda: read_back(run_dir, final, cfg, limit),
             "species counts": lambda: species_counts(final, run_dir, out, limit),
             "against B1": lambda: compare_with_b1(run_dir, out, limit),
             "verdicts": lambda: verdicts(run_dir, cfg, out, limit),
             "flip outputs": lambda: flip_outputs(run_dir, cfg, limit),
             "scenery test": lambda: scenery_test(run_dir, quality(), out, limit)}
    if limit is None:
        steps["A3 on M4 crops"] = lambda: a3_on_m4_crops(cfg)
    for step, run_step in steps.items():
        try:
            done[step] = run_step()
        except Exception as error:
            failed[step] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
    if failed:
        raise RuntimeError(f"{name} pass: {len(failed)} step(s) failed: {failed}")
    return SimpleNamespace(name=name, cfg=cfg, run_dir=run_dir, final=final, history=done["read back"],
                           decision=done["verdicts"][0], counts=done["species counts"],
                           b1=done["against B1"], r1=done["scenery test"],
                           hours=(time.time() - started) / HOUR)
''')

md("""
## 7. The smoke pass

Every step on the first 300 training photos for one epoch, with 300 photos of each scored set drawn
across species. The kd smoke distils from the B1 photos finished so far, which come from these 300
first.
""")

code('''
assert "front end" in PASSED, "the front end did not finish"
smoke_teacher = None
if VARIANT == "kd":
    while not list(Path(f"{TEACHER_RUN}.parts").glob("*.npz")):
        assert teacher_process.poll() is None, "B1 stopped: read b1_teacher.log"
        time.sleep(30)
    smoke_teacher, _ = teacher_file("smoke")
smoke = run_pass("smoke", SMOKE_EPOCHS, limit=SMOKE_PHOTOS, teacher=smoke_teacher)
SMOKE_PASSED = True
print(f"smoke pass: every step ran in {smoke.hours * 60:.0f} min")
smoke.history.T
''')

md("""
## 8. The full pass
""")

code('''
assert SMOKE_PASSED, "the smoke pass did not finish, so the full pass does not start"
PREDS = run_front_end(front_end_rows, SCORED_PREDS, ALL_PREDS)
shutil.copy2(PREDS / front_end.PREDS_FILE, WORK / "full" / front_end.PREDS_FILE)
full_teacher, teacher_check = None, None
if VARIANT == "kd":
    while teacher_process.poll() is None and (time.time() - STARTED) / HOUR < TEACHER_WAIT_HOURS:
        time.sleep(60)
    if teacher_process.poll() is None:
        teacher_process.terminate()
        print("B1 had not finished by the wait limit: the full pass distils from the photos it did")
    teacher_check = json.loads(Path(f"{TEACHER_RUN}.check.json").read_text())
    full_teacher, trained = teacher_file("full")
    write_json(WORK / "full" / "teacher.json", {"check": teacher_check, "training_photos": trained,
                                                "of": len(teacher_photos())})
full_run = run_pass("full", EPOCHS, timed=True, teacher=full_teacher)
FULL_PASSED = True
print(f"full pass: {full_run.hours:.1f} h; the session so far {(time.time() - STARTED) / HOUR:.1f} h")
''')

md("""
## 9. What the full pass gave
""")

code('''
columns = [c for c in ("epoch", "macro_f1", "top1", "top3", "loss_total", "val_loss_total",
                       "loss_species", "val_loss_species", "loss_kd", "val_loss_kd", "lr", "seconds",
                       "eval_seconds") if c in full_run.history]
print(full_run.history[columns].round(4).to_string(index=False))
d = full_run.decision
summary = {
    "best epoch": full_run.final["run"]["best_epoch"],
    "macro-F1, validation": full_run.final["head"]["macro_f1"],
    "top-1, validation": full_run.final["head"]["top1"],
    "WHO recall lower bound, calibration": d["recall_bound_fit"]["lower_bound"],
    "WHO recall, validation": d["verdict_metrics"]["who_dangerous_recall"],
    "false alarms, validation": d["verdict_metrics"]["false_alarm"],
    "species at F1 >= 0.5 (of 158)": full_run.counts["f1_at_least_0.5"],
    "scenery gap (G1)": (full_run.r1["conditions"]["as_is_masked"]["top1"]
                         - full_run.r1["conditions"]["swapped"]["top1"]),
    "G3 top-1, ours minus B1": full_run.b1[0]["ours"]["top1"]["diff"],
    "G3 macro-F1, ours minus B1": full_run.b1[0]["ours"]["macro_f1"]["diff"],
    "G3 passed": full_run.b1[1]["passed"]}
write_json(WORK / "full" / "summary.json", summary)
pd.Series(summary).to_frame(RUN_ID)
''')


def cell(index, kind, text):
    lines = text.splitlines(keepends=True)
    base = {"cell_type": kind, "id": f"c{index:02d}", "metadata": {}, "source": lines}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


notebook = {
    "cells": [cell(i, kind, text) for i, (kind, text) in enumerate(CELLS)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"codemirror_mode": {"name": "ipython", "version": 3},
                                   "file_extension": ".py", "mimetype": "text/x-python",
                                   "name": "python", "nbconvert_exporter": "python",
                                   "pygments_lexer": "ipython3", "version": "3.13.16"}},
    "nbformat": 4, "nbformat_minor": 5}
NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"{NOTEBOOK}: {len(CELLS)} cells")
