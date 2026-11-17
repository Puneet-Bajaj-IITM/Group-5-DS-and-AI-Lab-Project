"""The robustness test (notebook 15's R1): does the model name the snake or the scenery around it?

Every validation photo with a usable predicted mask is scored three ways: as it is, with the snake
blanked (its pixels painted the ImageNet mean grey), and with the snake pasted onto the scenery of
another validation photo of a different species (that photo's own snake blanked first).
`pick_arm` is M4's rule for choosing a background arm from those scores.
"""
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from serpentguard import data
from serpentguard.decide import decide, fit_thresholds, load_outputs, repo_tables
from serpentguard.metrics import (VERDICTS, class_tables, grouped_bootstrap, photographer_groups,
                                  share, softmax, species_report)
from serpentguard.train import library_versions, load_checkpoint, subset_hash

FILL = data.GREY_RGB  # the ImageNet mean pixel: zero after the model's normalisation
OUTPUT_KEYS = ("embedding", "species_logits", "who_logits", "genus_logits", "family_logits")
SAVED_SETS = {"val": "val_outputs", "calibration": "calib_outputs", "unknown": "unknown_outputs"}
CONDITIONS = ("as_is", "blanked", "swapped")
SLICES = ("india", "dark", "blurry", "few_photographers")
NAN2 = [float("nan"), float("nan")]
DARK_BELOW, BLUR_BELOW = 60, 100.0      # mean brightness; variance of the Laplacian
SWAP_SEED = 42
FEW_PHOTOGRAPHERS = 10                  # a species with at most this many is a slice
INSTRUMENT_FLOOR = 0.99                 # as-is top-1 agreement with the run's saved outputs
PROTOTYPE_SET = "plain"
QUALITY_THREADS = 8
PICK_COLUMNS = ("macro_f1", "who_recall_dr0", "top1", "top1_swapped")
SKIPPED = "no photo can be shown this way (e.g. a set of one species has no donor), so skipped"


@dataclass
class Validation:
    """The validation photos as the test reads them, and what every report needs about them."""
    frame: pd.DataFrame
    status: np.ndarray        # per photo: usable, none, size_mismatch, unreadable, ...
    usable: np.ndarray
    donors: np.ndarray        # per photo: its swap donor's row, -1 for none
    y_true: np.ndarray
    groups: np.ndarray        # photographer per photo
    taxa: SimpleNamespace
    train_counts: np.ndarray
    val_counts: np.ndarray
    few: np.ndarray           # per class: at most FEW_PHOTOGRAPHERS photographers in training
    tables: SimpleNamespace
    quality: pd.DataFrame
    mask_warm: dict
    subset_hash: str
    preds_root: Path


def blank_snake(rgb, mask):
    blanked = rgb.copy()
    blanked[mask.astype(bool)] = FILL
    return blanked


def swap_background(rgb, mask, scenery):
    # the snake's own pixels on another photo's scenery, resized to this photo
    scenery = cv2.resize(np.ascontiguousarray(scenery), rgb.shape[1::-1],
                         interpolation=cv2.INTER_LINEAR)
    return np.where(mask.astype(bool)[..., None], rgb, scenery).astype(np.uint8)


def draw_donors(class_ids, eligible, seed=SWAP_SEED):
    # per photo, a seeded eligible photo of another species; -1 when the photo is not eligible
    pool = np.flatnonzero(eligible)
    order = pool[np.argsort(class_ids[pool], kind="stable")]
    start, end = (np.searchsorted(class_ids[order], class_ids[pool], side=side)
                  for side in ("left", "right"))
    others = len(pool) - (end - start)
    draw = np.floor(np.random.default_rng(seed).random(len(pool)) * others).astype(int)
    position = np.where(draw < start, draw, draw + end - start)  # skip the photo's own species
    donors = np.full(len(class_ids), -1)
    donors[pool[others > 0]] = order[position[others > 0]]
    return donors


def load_raw_rgb(path):
    with Image.open(path) as image:  # as stored, before the exposure fix
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def photo_quality(path):
    rgb = load_raw_rgb(path)
    brightness = float(rgb.mean())
    sharpness = float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    return {"dark": brightness < DARK_BELOW, "blurry": sharpness < BLUR_BELOW,
            "brightness": brightness, "laplacian_var": sharpness}


def quality_table(run_dir, data_root):
    """Dark and blurry flags of every photo in the run's saved val, calibration and unknown sets."""
    quality = pd.concat([pd.DataFrame({"path": load_outputs(Path(run_dir, f"{file}.npz"))["paths"],
                                       "set": name})
                         for name, file in SAVED_SETS.items()], ignore_index=True)
    assert not quality["path"].str.contains(r"(?:^|/)test(?:_extra)?/").any(), "a test photo"
    files = [Path(data_root, "data", path) for path in quality["path"]]
    data.warm_cache(files)
    with ThreadPoolExecutor(QUALITY_THREADS) as pool:
        return quality.join(pd.DataFrame(list(pool.map(photo_quality, files))))


def mask_status(row):
    if not row.mask_file:
        return "none"
    try:
        with Image.open(row.mask_file) as image:
            if (image.height, image.width) != (row.height, row.width):
                return "size_mismatch"
    except (UnidentifiedImageError, ImportError):  # offline, ultralytics' open raises ImportError
        return "unreadable"
    cleaned = data.load_mask(row.mask_file, (row.height, row.width))
    return "usable" if cleaned is not None else "dropped_by_cleaning"


def validation(data_root, preds_root, subset_csv, quality, limit=None):
    """The validation photos with predicted masks and boxes, slice flags and swap donors."""
    metadata, classes = data.load_tables(data_root)
    preds = pd.read_csv(Path(preds_root, "frontend_preds.csv"))
    subset_paths = pd.read_csv(subset_csv)["path"]
    frame = data.with_files(data.build_frame(metadata, classes, ["val"], frontend_preds=preds,
                                             mask_source="predicted", preds_root=preds_root),
                            data_root)
    frame = data.smoke_sample(frame, limit).reset_index(drop=True)  # make_frames' photos
    train_rows = metadata[(metadata["split"] == "train") & metadata["path"].isin(subset_paths)]
    train_frame = data.cap_train(data.build_frame(train_rows, classes, ["train"]))  # as in training
    taxa = class_tables(classes)
    per_class = (pd.Series(photographer_groups(train_frame))
                 .groupby(train_frame["class_id"].to_numpy()).nunique())
    photographer_count = np.zeros(taxa.n_classes, dtype=int)
    photographer_count[per_class.index.to_numpy(dtype=int)] = per_class.to_numpy()
    few = photographer_count <= FEW_PHOTOGRAPHERS
    flags = quality.set_index("path").loc[frame["path"]]
    frame = frame.assign(india=frame["in_india"].astype(bool), dark=flags["dark"].to_numpy(bool),
                         blurry=flags["blurry"].to_numpy(bool),
                         few_photographers=few[frame["class_id"].to_numpy()])
    y_true = frame["class_id"].to_numpy()
    mask_warm = data.warm_cache(list(frame.loc[frame["mask_valid"], "mask_file"]))
    with ThreadPoolExecutor(QUALITY_THREADS) as pool:
        status = np.array(list(pool.map(mask_status, frame.itertuples(index=False))), dtype=object)
    usable = status == "usable"
    assert usable.any(), "no validation photo has a usable predicted mask"
    return Validation(frame=frame, status=status, usable=usable, donors=draw_donors(y_true, usable),
                      y_true=y_true, groups=photographer_groups(frame), taxa=taxa,
                      train_counts=np.bincount(train_frame["class_id"], minlength=taxa.n_classes),
                      val_counts=np.bincount(y_true, minlength=taxa.n_classes), few=few,
                      tables=repo_tables(), quality=quality, mask_warm=mask_warm,
                      subset_hash=subset_hash(subset_paths), preds_root=Path(preds_root))


class ConditionPhotos(Dataset):
    # validation photos `rows` as one condition shows them to the model

    def __init__(self, frame, donors, condition, mode, size, rows):
        self.frame, self.donors, self.condition = frame, donors, condition
        self.mode, self.size, self.rows = mode, size, rows
        self.mask_input = mode in data.MASK_INPUT_MODES  # the input itself is cut from the mask

    def __len__(self):
        return len(self.rows)

    def photo(self, position):
        row = self.frame.iloc[position]
        rgb = data.load_rgb(row["image_file"])
        needs_mask = self.condition != "as_is" or self.mask_input
        return row, rgb, data.load_mask(row["mask_file"], rgb.shape[:2]) if needs_mask else None

    def __getitem__(self, index):
        position = int(self.rows[index])
        row, rgb, mask = self.photo(position)
        if self.condition == "blanked":
            rgb = blank_snake(rgb, mask)
        elif self.condition == "swapped":
            _, donor_rgb, donor_mask = self.photo(int(self.donors[position]))
            rgb = swap_background(rgb, mask, blank_snake(donor_rgb, donor_mask))
        image, _ = data.apply_mode(rgb, mask if self.mask_input else None, self.mode, row["box"],
                                   self.size)
        return data.AUGMENT["eval"](image=image)["image"]


def one_opencv_thread(_):
    cv2.setNumThreads(0)


@torch.inference_mode()
def predict(model, photos, batch_size, device, num_workers=None):
    workers = os.cpu_count() if num_workers is None else num_workers
    loader = DataLoader(photos, batch_size=batch_size, num_workers=workers,
                        pin_memory=device.type == "cuda", worker_init_fn=one_opencv_thread)
    collected = {name: [] for name in OUTPUT_KEYS}
    for images in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(images.to(device, non_blocking=True))
        for name in OUTPUT_KEYS:
            if out.get(name) is not None:
                collected[name].append(out[name].float())
    return {name: torch.cat(parts).cpu().numpy() for name, parts in collected.items() if parts}


def instrument_check(v, run, outputs):
    saved = load_outputs(Path(run, "val_outputs.npz"))
    position = {path: index for index, path in enumerate(saved["paths"])}
    pairs = np.array([(row, position[path]) for row, path in enumerate(v.frame["path"])
                      if path in position], dtype=int).reshape(-1, 2)
    mine, theirs = outputs["species_logits"][pairs[:, 0]], saved["species_logits"][pairs[:, 1]]
    return {"photos": len(pairs),
            "top1_agreement": float((mine.argmax(1) == theirs.argmax(1)).mean()),
            "max_abs_logit_diff": float(np.abs(mine - theirs).max())}


def save_condition(v, file, rows, outputs, with_donor):
    kept = v.frame.iloc[rows]
    extra = {}
    if with_donor:
        extra = {"donor_path": v.frame["path"].to_numpy(str)[v.donors[rows]],
                 "donor_class_id": v.y_true[v.donors[rows]]}
    observers = kept["observer_id"].astype("string").fillna("").to_numpy(str)
    np.savez_compressed(file, paths=kept["path"].to_numpy(str), labels=kept["class_id"].to_numpy(),
                        observer_id=observers, **extra,
                        **{**outputs, "embedding": outputs["embedding"].astype(np.float16)})


def score_run(v, run, device, out_dir=None, num_workers=None):
    """The run's outputs and verdicts in each condition; out_dir, when given, gets them as .npz."""
    cfg, model, epoch = load_checkpoint(run, device)
    mode, size, batch = data.input_mode(cfg), cfg["input_size"], 2 * cfg["batch_size"]
    prototypes = np.load(Path(run, "prototypes.npz"))[PROTOTYPE_SET]
    thresholds = fit_thresholds(load_outputs(Path(run, "calib_outputs.npz")), prototypes, v.tables,
                                recall_bound="lower")
    rows = {"as_is": np.arange(len(v.frame)), "blanked": np.flatnonzero(v.usable),
            "swapped": np.flatnonzero(v.donors >= 0)}
    outputs = {}
    for condition in CONDITIONS:
        if not len(rows[condition]):
            print(f"{condition}: {SKIPPED}", flush=True)
            outputs[condition] = None
            continue
        photos = ConditionPhotos(v.frame, v.donors, condition, mode, size, rows[condition])
        outputs[condition] = predict(model, photos, batch, device, num_workers)
        outputs[condition]["verdict"] = decide(outputs[condition], thresholds, prototypes,
                                               v.tables).label
        if out_dir is not None:
            save_condition(v, Path(out_dir, f"robustness_{condition}.npz"), rows[condition],
                           outputs[condition], with_donor=condition == "swapped")
        if condition == "as_is":
            check = instrument_check(v, run, outputs["as_is"])
            assert check["top1_agreement"] >= INSTRUMENT_FLOOR, f"not the run's own model: {check}"
    return cfg, epoch, thresholds, rows, outputs, check


def ci(fn, groups):
    return list(grouped_bootstrap(fn, groups)) if len(groups) else NAN2


def species_scores(v, labels, probs, groups):
    if not len(labels):
        return {"photos": 0, "photographers": 0, "top1": np.nan, "top3": np.nan,
                "macro_f1": np.nan, "balanced_acc": np.nan, "top1_ci": NAN2}
    report = species_report(labels, probs, v.train_counts, v.val_counts, v.taxa.tier)
    right = probs.argmax(axis=1) == labels
    return {"photos": len(labels), "photographers": int(np.unique(groups).size),
            **{key: report[key] for key in ("top1", "top3", "macro_f1", "balanced_acc")},
            "top1_ci": ci(lambda rows: share(right[rows]), groups)}


def shortcut_report(labels, as_is, blanked, groups):
    right, still = as_is == labels, blanked == labels
    return {"right_as_is": int(right.sum()), "shortcut_rate": share(still[right]),
            "shortcut_rate_ci": ci(lambda rows: share(still[rows][right[rows]]), groups),
            "blanked_top1": share(still), "changed_prediction": share(blanked != as_is)}


def donor_report(labels, donor_labels, as_is, swapped, groups, donor_blanked):
    # donor_blanked: the top species of each donor's own blanked photo (-1 when it has none)
    keeps, follows = swapped == labels, swapped == donor_labels
    known, silhouette = donor_blanked != -1, donor_blanked == donor_labels
    clean = known & ~silhouette
    return {"keeps_own_class": share(keeps),
            "keeps_own_class_ci": ci(lambda rows: share(keeps[rows]), groups),
            "follows_donor": share(follows),
            "follows_donor_ci": ci(lambda rows: share(follows[rows]), groups),
            "follows_donor_as_is": share(as_is == donor_labels),
            "keeps_own_among_right": share(keeps[as_is == labels]),
            "changed_prediction": share(swapped != as_is),
            "donor_silhouette_rate": share(silhouette[known]),
            "swaps_without_silhouette": int(clean.sum()),
            "follows_donor_without_silhouette": share(follows[clean])}


def verdict_counts(verdicts):
    return {label: int((np.asarray(verdicts) == label).sum()) for label in VERDICTS}


def verdict_change(before, after):
    return {"photos": len(before), "changed": share(before != after),
            "transitions": {f"{old}->{new}": int(((before == old) & (after == new)).sum())
                            for old in VERDICTS for new in VERDICTS if old != new},
            "before": verdict_counts(before), "after": verdict_counts(after)}


def slice_metrics(v, probs):
    # top-1 inside each slice against outside it; the gap's interval resamples photographers
    right = probs.argmax(axis=1) == v.y_true
    report = {}
    for name in SLICES:
        inside = v.frame[name].to_numpy(bool)
        sides = {side: species_scores(v, v.y_true[members], probs[members], v.groups[members])
                 for side, members in (("in", inside), ("out", ~inside))}

        def gap(rows, inside=inside):
            return share(right[rows][inside[rows]]) - share(right[rows][~inside[rows]])

        report[name] = {**sides, "top1_gap": sides["in"]["top1"] - sides["out"]["top1"],
                        "top1_gap_ci": ci(gap, v.groups)}
    return report


def conditions_report(v, rows, outputs):
    # a condition with no photo (outputs None) is reported as skipped
    probs = {condition: softmax(o["species_logits"]) for condition, o in outputs.items()
             if o is not None}
    top1 = {condition: p.argmax(axis=1) for condition, p in probs.items()}
    blanked_top1 = np.full(len(v.frame), -1)
    if "blanked" in top1:
        blanked_top1[rows["blanked"]] = top1["blanked"]
    india = v.frame["india"].to_numpy(bool)

    def scored(condition, members=None):
        members = np.arange(len(rows[condition])) if members is None else members
        picked = rows[condition][members]
        return species_scores(v, v.y_true[picked], probs[condition][members], v.groups[picked])

    def shortcut(m):
        blanked = top1["blanked"][np.searchsorted(rows["blanked"], m)]
        return shortcut_report(v.y_true[m], top1["as_is"][m], blanked, v.groups[m])

    def donor(m):
        swapped = top1["swapped"][np.searchsorted(rows["swapped"], m)]
        return donor_report(v.y_true[m], v.y_true[v.donors[m]], top1["as_is"][m], swapped,
                            v.groups[m], blanked_top1[v.donors[m]])

    def by_india(report, m):
        return {"india": report(m[india[m]]), "not_india": report(m[~india[m]])}

    blanked, swapped = rows["blanked"], rows["swapped"]
    skipped = {"photos": 0, "skipped": SKIPPED}
    return {"as_is": scored("as_is"), "as_is_masked": scored("as_is", np.flatnonzero(v.usable)),
            "blanked": skipped if outputs["blanked"] is None else {
                **scored("blanked"), "shortcut": shortcut(blanked),
                "by_india": by_india(shortcut, blanked)},
            "swapped": skipped if outputs["swapped"] is None else {
                **scored("swapped"), "as_is_same_photos": scored("as_is", swapped),
                "donor": donor(swapped), "by_india": by_india(donor, swapped)}}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_counts(v, rows):
    misfit = v.status == "size_mismatch"
    return {"val": len(v.frame), "scored": len(v.frame), "with_mask": int(v.usable.sum()),
            "without_mask": int((~v.usable).sum()),
            "mask_dropped_by_cleaning": int((v.status == "dropped_by_cleaning").sum()),
            "mask_size_mismatch": int(misfit.sum()),
            "mask_size_mismatch_examples": v.frame.loc[misfit, "path"].head(5).tolist(),
            "mask_unreadable": int((v.status == "unreadable").sum()),
            "blanked": len(rows["blanked"]), "swapped": len(rows["swapped"]),
            "without_donor": int(v.usable.sum()) - len(rows["swapped"])}


def provenance(v, run, epoch):
    winners = v.preds_root / "summary.json"
    segmenter = (json.loads(winners.read_text()).get("winners", {}).get("S1", "not available")
                 if winners.exists() else "not available")
    return {"run_id": run.name, "run_dir": str(run), "checkpoint_epoch": epoch,
            "checkpoint_sha256": sha256(run / "best.pt"), "preds_root": str(v.preds_root),
            "frontend_preds_sha256": sha256(v.preds_root / "frontend_preds.csv"),
            "segmenter": segmenter}


def robustness_metrics(run_dir, data_root, preds_root, subset_csv=None, quality=None,
                       device=None, out_dir=None, limit=None, num_workers=None):
    """Notebook 15's robustness_metrics.json for one run folder; limit keeps N val photos."""
    # subset_csv defaults to the run's own; quality to the dark/blurry flags of its saved photos;
    # num_workers to one loader worker per CPU
    started = time.perf_counter()
    run = Path(run_dir)
    run_cfg = data.run_config(yaml.safe_load((run / "config.yaml").read_text()))
    subset_csv = run_cfg["subset_path"] if subset_csv is None else subset_csv
    quality = quality_table(run, data_root) if quality is None else quality
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    v = validation(data_root, preds_root, subset_csv, quality, limit)
    cfg, epoch, thresholds, rows, outputs, check = score_run(v, run, device, out_dir, num_workers)
    as_is = outputs["as_is"]["verdict"]
    return {
        "run": run.name, "front_end": cfg["front_end"], "input_mode": data.input_mode(cfg),
        "checkpoint_epoch": epoch, "provenance": provenance(v, run, epoch),
        "photos": mask_counts(v, rows), "instrument_check": check,
        "conditions": conditions_report(v, rows, outputs),
        "verdicts": {"recall_bound": "lower", "thresholds": asdict(thresholds),
                     "as_is": verdict_counts(as_is),
                     **{c: {"photos": 0, "skipped": SKIPPED} if outputs[c] is None
                        else verdict_change(as_is[rows[c]], outputs[c]["verdict"])
                        for c in ("blanked", "swapped")}},
        "slices": slice_metrics(v, softmax(outputs["as_is"]["species_logits"])),
        "quality": {name: {"photos": len(g), "dark": float(g["dark"].mean()),
                           "blurry": float(g["blurry"].mean())}
                    for name, g in quality.groupby("set", sort=False)},
        "settings": {"fill_rgb": FILL, "dark_below": DARK_BELOW, "blur_below": BLUR_BELOW,
                     "swap_seed": SWAP_SEED, "few_photographers_at_most": FEW_PHOTOGRAPHERS,
                     "few_photographer_classes": int(v.few.sum()),
                     "prototype_set": PROTOTYPE_SET, "instrument_floor": INSTRUMENT_FLOOR,
                     "limit": limit},
        "warm_cache": v.mask_warm, "subset_hash": v.subset_hash,
        "versions": library_versions(device), "seconds": time.perf_counter() - started}


def arm_row(arm, final_metrics, decision, robustness):
    """One row of arms.csv from a run's final, decision and robustness metrics."""
    swapped = robustness["conditions"]["swapped"]
    return {"arm": arm, "run": robustness["run"], "front_end": robustness["front_end"],
            "macro_f1": final_metrics["head"]["macro_f1"],
            "who_recall_dr0": decision["verdict_metrics"]["who_dangerous_recall"],
            "top1": swapped["as_is_same_photos"]["top1"], "top1_swapped": swapped["top1"],
            "top1_blanked": robustness["conditions"]["blanked"]["top1"],
            "swap_photos": swapped["photos"]}


def pick_arm(rows, baseline="A0", f1_floor=0.005, who_floor=0.006):
    """The arm to train in full: of the baseline and the arms within both of its floors, the
    smallest gap between normal and swapped-scenery top-1 (a tie goes to the higher macro-F1).
    """
    missing = rows[list(PICK_COLUMNS)].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"arms {rows.loc[missing, 'arm'].tolist()} have missing values")
    base = rows[rows["arm"] == baseline]
    if len(base) != 1:
        raise ValueError(f"rows hold {len(base)} {baseline} rows; the rule needs exactly one")
    base = base.iloc[0]
    # the baseline is within its own floors, so an arm wins only with a smaller gap than its
    qualified = rows[(rows["macro_f1"] >= base["macro_f1"] - f1_floor)
                     & (rows["who_recall_dr0"] >= base["who_recall_dr0"] - who_floor)]
    ranked = (qualified.assign(gap=qualified["top1"] - qualified["top1_swapped"])
              .sort_values(["gap", "macro_f1"], ascending=[True, False], kind="stable"))
    return str(ranked["arm"].iloc[0])
