"""Species, safety, calibration, open-set, taxonomy, segmentation and detection metrics, the photographer-grouped
bootstrap, and the guard that keeps the test splits out. Classes are ids 0..C-1, probs is (N, C), and a rate with an
empty denominator is nan, never 0."""
import math

import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, recall_score, roc_auc_score, roc_curve,
                             top_k_accuracy_score)

MIN_VAL_PHOTOS = 5                    # a class needs this many validation photos to enter a macro average
TOP_K = 3                             # the screen shows three species
TIERS = ("tier1", "tier2")
SIZE_BANDS = {"<30": (0, 30), "30-99": (30, 100), "100+": (100, math.inf)}  # training photos, [low, high)
WHO_NOT_LISTED = 0                    # notebook 06: 0 = not listed, 1 = WHO Category 2, 2 = WHO Category 1
WHO_CODES = (0, 1, 2)
DANGEROUS, NOT_LISTED, UNKNOWN = "dangerous", "not_listed", "unknown"
VERDICTS = (DANGEROUS, NOT_LISTED, UNKNOWN)
ECE_BINS = 15
COVERAGE = 0.80                       # O4: error among the most confident 80%
TPR_LEVEL = 0.95                      # FPR is read where 95% of unknowns are flagged
DET_IOU = 0.5
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 1000, 42
CI_PERCENTILES = (2.5, 97.5)
TEST_SPLITS = ("test", "test_extra")
TEST_FOLDER = rf"(?:^|/)(?:{'|'.join(TEST_SPLITS)})/"


def species_report(y_true, probs, train_counts, val_counts, tier_of_class, min_val=MIN_VAL_PHOTOS):
    """Species metrics; a class under min_val validation photos is listed, never averaged in as 0."""
    y_true, probs = np.asarray(y_true, dtype=np.int64), np.asarray(probs, dtype=float)
    train_counts, val_counts = np.asarray(train_counts), np.asarray(val_counts)
    tier_of_class = np.asarray(tier_of_class)
    classes = probs.shape[1]
    y_pred = probs.argmax(axis=1)
    measurable = np.flatnonzero(val_counts >= min_val)
    few = np.flatnonzero((val_counts > 0) & (val_counts < min_val))
    f1 = np.full(len(measurable), math.nan)  # stays nan for a class with no photo among those passed
    keep = np.isin(y_true, measurable)
    if keep.any():
        _, _, f1, measured = precision_recall_fscore_support(y_true[keep], y_pred[keep], labels=measurable,
                                                             zero_division=0)
        f1[measured == 0] = math.nan

    def macro(selected):
        scored = f1[selected & ~np.isnan(f1)]
        return float(scored.mean()) if len(scored) else math.nan

    measured_train, measured_tier = train_counts[measurable], tier_of_class[measurable]
    rare = np.isin(y_true, few)
    precision, recall, per_f1, support = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(classes),
                                                                         zero_division=np.nan)
    recall[support == 0] = per_f1[support == 0] = math.nan  # no photos: nothing to recall, so no F1 either
    return {
        "macro_f1": macro(np.ones(len(measurable), dtype=bool)),
        **{f"macro_f1_{tier}": macro(measured_tier == tier) for tier in TIERS},
        "macro_f1_band": {band: macro((measured_train >= low) & (measured_train < high))
                          for band, (low, high) in SIZE_BANDS.items()},
        "pooled_recall_rare": _share(y_pred[rare] == y_true[rare]),
        "top1": float(accuracy_score(y_true, y_pred)),
        "top3": 1.0 if TOP_K >= classes else float(top_k_accuracy_score(y_true, probs, k=TOP_K,
                                                                         labels=np.arange(classes))),
        "balanced_acc": float(recall_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)),
        "per_class": {int(label): {"precision": float(precision[label]), "recall": float(recall[label]),
                                   "f1": float(per_f1[label]), "support": int(support[label])}
                      for label in range(classes)},
        "not_measurable": {"none": np.flatnonzero(val_counts == 0).tolist(), "few": few.tolist()},
    }


def safety_report(y_true, probs, who_of_class, dangerous_of_class, verdicts=None):
    """WHO recall and raw O3 from top-1; with verdicts, O3, abstention, false alarms, venomous recall."""
    y_true, probs = np.asarray(y_true, dtype=np.int64), np.asarray(probs, dtype=float)
    who_of_class, dangerous_of_class = np.asarray(who_of_class), np.asarray(dangerous_of_class, dtype=bool)
    if not np.isin(who_of_class, WHO_CODES).all():
        raise ValueError(f"who_of_class must hold notebook 06's integer codes {WHO_CODES}")
    listed = who_of_class != WHO_NOT_LISTED
    on_list = listed[y_true]
    top1_listed = listed[probs.argmax(axis=1)[on_list]]
    report = {"who_recall_raw": _share(top1_listed), "top1_error_raw": _share(~top1_listed)}
    if verdicts is None:
        return report
    verdicts = np.asarray(verdicts)
    if not np.isin(verdicts, VERDICTS).all():
        raise ValueError(f"verdicts must be one of {VERDICTS}")
    answered = verdicts != UNKNOWN
    venomous = dangerous_of_class[y_true]
    report.update({
        "dangerous_as_not_listed_all": _share(verdicts[on_list] == NOT_LISTED),
        "dangerous_as_not_listed_answered": _share(verdicts[on_list & answered] == NOT_LISTED),
        "coverage_listed": _share(answered[on_list]),
        "abstain_rate": _share(~answered),
        "false_alarm": _share(verdicts[~venomous] == DANGEROUS),
        "venomous_recall": _share(verdicts[venomous] == DANGEROUS),
        "venomous_as_not_listed": _share(verdicts[venomous] == NOT_LISTED),
    })
    return report


def calibration(probs, y_true, bins=ECE_BINS):
    """ECE, area under the risk-coverage curve, and error at COVERAGE; tied photos share their mean error."""
    y_true, probs = np.asarray(y_true, dtype=np.int64), np.asarray(probs, dtype=float)
    confidence = probs.max(axis=1)
    errors = (probs.argmax(axis=1) != y_true).astype(float)
    bin_of = np.searchsorted(np.linspace(0, 1, bins + 1)[1:-1], confidence, side="left")  # bins are (low, high]
    gaps = np.bincount(bin_of, weights=1 - errors, minlength=bins) - np.bincount(bin_of, weights=confidence, minlength=bins)
    order = np.argsort(-confidence, kind="stable")
    _, group, size = np.unique(confidence[order], return_inverse=True, return_counts=True)
    tie_averaged = (np.bincount(group, weights=errors[order]) / size)[group]
    risk = np.cumsum(tie_averaged) / np.arange(1, len(errors) + 1)
    return {"ece": float(np.abs(gaps).sum() / len(errors)), "aurc": float(risk.mean()),
            "error_at_80": float(risk[math.ceil(COVERAGE * len(errors)) - 1])}


def open_set(scores_known, scores_unknown):
    """Higher score = more likely unknown; fpr95 is the known photos' flag rate where TPR_LEVEL of unknowns are."""
    known, unknown = np.ravel(scores_known).astype(float), np.ravel(scores_unknown).astype(float)
    is_unknown = np.r_[np.zeros(len(known)), np.ones(len(unknown))]
    scores = np.r_[known, unknown]
    fpr, tpr, _ = roc_curve(is_unknown, scores, drop_intermediate=False)
    return {"auroc": float(roc_auc_score(is_unknown, scores)), "fpr95": float(fpr[np.argmax(tpr >= TPR_LEVEL)])}


def taxonomy_report(species_probs, genus_head_probs, family_head_probs, genus_of_class, family_of_class, y_genus, y_family):
    """Genus and family accuracy from summed species probabilities and from each head given."""
    species_probs = np.asarray(species_probs, dtype=float)
    report = {}
    for level, head, of_class, truth in (("genus", genus_head_probs, genus_of_class, y_genus),
                                         ("family", family_head_probs, family_of_class, y_family)):
        of_class, truth = np.asarray(of_class, dtype=np.int64), np.asarray(truth, dtype=np.int64)
        width = np.shape(head)[1] if head is not None else int(max(of_class.max(), truth.max())) + 1
        from_species = (species_probs @ np.eye(width)[of_class]).argmax(axis=1)
        report[f"{level}_acc_summed"] = float(accuracy_score(truth, from_species))
        if head is not None:
            from_head = np.asarray(head, dtype=float).argmax(axis=1)
            report[f"{level}_acc_head"] = float(accuracy_score(truth, from_head))
            report[f"disagreement_{level}"] = _share(from_species != from_head)
    return report


def seg_scores(pred, gt):
    """Batch-mean IoU and Dice of binary masks, (B, H, W) or one (H, W) mask; an image where both are empty scores 1."""
    pred, gt = np.asarray(pred), np.asarray(gt)
    if not (np.isin(pred, (0, 1)).all() and np.isin(gt, (0, 1)).all()):
        raise ValueError("masks must be binary (0/1 or bool)")
    pred, gt = (masks.astype(bool).reshape(1 if masks.ndim == 2 else len(masks), -1) for masks in (pred, gt))
    overlap = (pred & gt).sum(axis=1)
    areas = pred.sum(axis=1) + gt.sum(axis=1)
    union = areas - overlap
    iou = np.divide(overlap, union, out=np.ones(len(union)), where=union > 0)
    dice = np.divide(2 * overlap, areas, out=np.ones(len(areas)), where=areas > 0)
    return {"iou": float(iou.mean()), "dice": float(dice.mean())}


def det_scores(pred_boxes, gt_boxes, iou_threshold=DET_IOU):
    """Single-class precision, recall and all-point AP; boxes [x1, y1, x2, y2(, score)] matched greedily."""
    preds = [np.asarray(boxes, dtype=float).reshape(-1, 5) for boxes in pred_boxes]
    truths = [np.asarray(boxes, dtype=float).reshape(-1, 4) for boxes in gt_boxes]
    image_of = np.concatenate([np.full(len(boxes), image) for image, boxes in enumerate(preds)]).astype(int)
    pooled = np.concatenate(preds)
    claimed = [np.zeros(len(boxes), dtype=bool) for boxes in truths]
    hits = []
    for index in np.argsort(-pooled[:, 4], kind="stable"):
        box, boxes = pooled[index, :4], truths[image_of[index]]
        corner_low, corner_high = np.maximum(box[:2], boxes[:, :2]), np.minimum(box[2:], boxes[:, 2:])
        overlap = np.prod(np.clip(corner_high - corner_low, 0, None), axis=1)
        union = np.prod(box[2:] - box[:2]) + np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) - overlap
        iou = np.divide(overlap, union, out=np.zeros(len(boxes)), where=union > 0)
        overlaps = np.where(claimed[image_of[index]], -1.0, iou)
        hit = overlaps.size > 0 and overlaps.max() >= iou_threshold
        if hit:
            claimed[image_of[index]][overlaps.argmax()] = True
        hits.append(hit)
    hits = np.array(hits, dtype=bool)
    truth_count = sum(len(boxes) for boxes in truths)
    if not truth_count:
        return {"precision": _share(hits), "recall": math.nan, "map50": math.nan}
    true_positives = np.cumsum(hits)
    recall = np.r_[0.0, true_positives / truth_count, 1.0]
    precision = np.r_[0.0, true_positives / np.arange(1, len(hits) + 1), 0.0]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.flatnonzero(recall[1:] != recall[:-1])
    return {"precision": _share(hits), "recall": float(hits.sum() / truth_count),
            "map50": float(((recall[steps + 1] - recall[steps]) * precision[steps + 1]).sum())}


def grouped_bootstrap(fn, groups, n=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """95% interval of fn(indices) over n resamples that draw whole groups (photographers) with replacement."""
    low, high = np.percentile([fn(rows) for rows in grouped_resamples(groups, n, seed)], CI_PERCENTILES)
    return float(low), float(high)


def grouped_resamples(groups, n=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """The row indices of n resamples, each drawing as many whole groups as there are, with replacement."""
    _, group_of = np.unique(np.asarray(groups), return_inverse=True)
    members = np.split(np.argsort(group_of, kind="stable"), np.cumsum(np.bincount(group_of))[:-1])
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield np.concatenate([members[group] for group in rng.integers(0, len(members), len(members))])


def assert_no_test_paths(frame):
    """Raises ValueError if any row comes from the test or test_extra split, by its path folder or split column."""
    if "path" not in frame.columns and "split" not in frame.columns:
        raise ValueError("frame has neither a 'path' nor a 'split' column, so it can't be checked for test photos")
    leaks = np.zeros(len(frame), dtype=bool)
    if "path" in frame.columns:
        paths = frame["path"].astype(str).str.replace("\\", "/", regex=False)
        leaks |= paths.str.contains(TEST_FOLDER, regex=True, na=False).to_numpy(dtype=bool)
    if "split" in frame.columns:
        leaks |= frame["split"].isin(TEST_SPLITS).to_numpy(dtype=bool)
    if leaks.any():
        raise ValueError(f"{int(leaks.sum())} rows come from the test splits {TEST_SPLITS}")


def _share(flags):
    return float(np.mean(flags)) if len(flags) else math.nan
