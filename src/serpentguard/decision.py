"""The verdict a photo gets (dangerous, not on the WHO list, or we don't know), its thresholds fitted on calibration
photos, and the decision report on a run's saved outputs. WHO codes: 0 not listed, 1 Category 2, 2 Category 1."""
import math
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from serpentguard.config import load_config
from serpentguard.data import NO_ID, WHO_RISK_ID, taxonomy_index
from serpentguard.fewshot import masked_cosine
from serpentguard.medical_importance import (BASES, CLASSES_CSV, GUIDELINE_NAMED, GUIDELINE_NAMED_PAGE, MEDICAL_CSV,
                                             NO_BASIS, VENOMOUS_FAMILY, WHO_CATEGORY_1, WHO_CATEGORY_2, pages)
from serpentguard.metrics import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, COVERAGE, DANGEROUS, NOT_LISTED, TOP_K, UNKNOWN,
                                  VERDICTS, _share, assert_no_test_paths, calibration, grouped_bootstrap,
                                  grouped_resamples, open_set, safety_report, taxonomy_report)
from serpentguard.run_files import CALIBRATION_OUTPUTS, CONFIG_FILE, PROTOTYPES_FILE, UNKNOWN_OUTPUTS, VAL_OUTPUTS
from serpentguard.run_outputs import photographer_groups, softmax, write_json
from serpentguard.subset import DANGEROUS_FAMILIES, DANGEROUS_GENERA_EXTRA

WHO_RECALL_TARGET = 0.95      # O2: share of WHO-listed calibration photos that must get the dangerous verdict
KNOWN_ACCEPT = 0.95           # share of known calibration photos left above tau_unknown
THETA_V, THETA_GENUS = 0.5, 0.5
DANGER_GRID_POINTS = 101      # quantiles per WHO signal in the joint (theta_d, theta_who) search, +inf added
SINGLE_SIGNALS = {"species_mass_only": (True, False), "who_head_only": (False, True)}
O4_UNKNOWN_TARGET = 0.70      # O4: share of unknown-pool photos flagged
SNAKE_KEEP = 0.99             # share of calibration photos (all real snakes) that keep a snake at tau_snake
TEMPERATURE_RANGE, TEMPERATURE_XTOL = (0.05, 20.0), 1e-6  # search bounds and tolerance on log temperature
SHARE_SLACK = 1e-9            # slack before rounding a target share times a count up to whole photos
NOT_WHO, WHO_CAT2, WHO_CAT1 = WHO_RISK_ID["not_listed"], WHO_RISK_ID["2"], WHO_RISK_ID["1"]
PLAIN, HEAD, PROTOTYPE = "plain", "head", "prototype"
SCORERS = (HEAD, PROTOTYPE)   # what ranks the species: the species head, or the nearest prototype
POINT, LOWER = "point", "lower"
RECALL_BOUNDS = (POINT, LOWER)  # theta_d and theta_who fit the calibration WHO recall, or its lower bound
BOUND_TARGET_STEP, BOUND_TARGET_MAX = 0.005, 0.995  # LOWER tries targets from WHO_RECALL_TARGET up to this
BOUND_PERCENTILE = 5          # one-sided 95% lower bound of the photographer-bootstrap WHO recall
QUALITY_FLAGS = ("dark", "blurry")
DECISION_FILE = "decision_metrics.json"

WHO_MASS, WHO_HEAD, TOP_SPECIES, FAMILY_MASS = "who_listed_mass", "who_head", "top_species_dangerous", "venomous_family_mass"
DANGEROUS_GENUS, NO_SNAKE, UNCLEAR, UNFAMILIAR, POOR_IMAGE = ("dangerous_genus", "no_snake", "unclear_between_species",
                                                              "unfamiliar", "poor_image")
REASONS = (WHO_MASS, WHO_HEAD, TOP_SPECIES, FAMILY_MASS, DANGEROUS_GENUS, NO_SNAKE, UNCLEAR, UNFAMILIAR, POOR_IMAGE)
NO_MAJORITY = "no_majority"  # a clip's own reason, when no label holds 4 of the last 6 frames; no photo rule
OPEN_SET_SCORES = ("msp", "energy", "prototype_distance", "cosine_margin")

NOT_LISTED_TEXT = "not on the WHO list, still keep away"
VENOMOUS_NOTE = "venomous family, not on WHO's priority list"
CATEGORY_NOTE = "WHO Category {}"
LISTED_SPECIES_NOTE = "could be a WHO-listed species: {} (Category {})"
LISTED_MASS_NOTE = "could be a WHO-listed species: {} combined probability"  # none in the top 3
WHO_HEAD_NOTE = f"the photo looks like a {CATEGORY_NOTE} snake"
GENUS_LISTED_NOTE = "genus {} has species on the WHO list"
GENUS_NAMED_NOTE = "genus {} has species named as medically important in the WHO guideline"
DANGEROUS_MESSAGE, UNKNOWN_MESSAGE = "dangerous: {}", "we don't know: {}"
REASON_TEXT = {NO_SNAKE: "no snake found", UNCLEAR: "unclear between species", UNFAMILIAR: "unlike any species we know",
               POOR_IMAGE: "the photo is dark or blurry", NO_MAJORITY: "the frames do not agree"}
BASIS_NOTE = {WHO_CATEGORY_1: CATEGORY_NOTE.format(1), WHO_CATEGORY_2: CATEGORY_NOTE.format(2),
              VENOMOUS_FAMILY: VENOMOUS_NOTE, NO_BASIS: "",
              GUIDELINE_NAMED: f"named as medically important in the WHO guideline, {pages(GUIDELINE_NAMED_PAGE)}"}
NO_SNAKE_NOT_AVAILABLE = "not available: no front-end result, so every photo was scored as showing a snake"
SNAKE_THRESHOLD_SOURCE = ("front end: box_conf >= tau_snake, fitted so 99% of calibration photos keep a snake; every "
                          "calibration photo is a real snake and there is no licensed non-snake set, so this is the "
                          "only fit we can defend and no false-snake rate is measured; a photo with no box counts as "
                          "no snake")
NON_SNAKE_SKIPPED = "skipped (plan section 14.8): the unknown pool holds no non-snake photos"
PROTOTYPE_SCORER_NOTE = ("species probabilities, top 3, msp and energy read softmax(prototype_scale x cosine to each "
                         "prototype), not the species head; a class with no prototype scores cosine 0")
SCORE_DIRECTION = ("higher means more likely unknown: msp = -max softmax (temperature-scaled), energy = -logsumexp of "
                   "the raw species logits, prototype_distance = 1 - best prototype cosine, cosine_margin = -(best - "
                   "second-best prototype cosine)")


@dataclass(frozen=True)
class Thresholds:
    temperature: float
    temperature_who: float
    theta_d: float
    theta_who: float
    tau_unknown: float
    theta_conf: float
    theta_v: float = THETA_V
    theta_genus: float = THETA_GENUS


@dataclass(frozen=True)
class DangerTables:
    species: np.ndarray
    genus_names: tuple
    genus_of_class: np.ndarray
    family_of_class: np.ndarray
    who: np.ndarray
    dangerous: np.ndarray
    note: np.ndarray
    venomous_class: np.ndarray
    guarded_genus: np.ndarray    # holds a WHO-listed class, or is a keelback genus
    listed_genus: np.ndarray     # holds a WHO-listed class
    venomous_family: np.ndarray


@dataclass(frozen=True)
class Signals:
    probs: np.ndarray            # temperature-scaled species probabilities from raw_logits
    listed_mass: np.ndarray      # Category 1 mass + Category 2 mass
    who_head: np.ndarray | None
    genus_head: np.ndarray | None
    family_head: np.ndarray | None
    genus_mass: np.ndarray       # species probability summed per genus
    best_cosine: np.ndarray
    second_cosine: np.ndarray
    raw_logits: np.ndarray       # the species head's logits, or scale x prototype cosine

    @property
    def who_listed(self):
        return None if self.who_head is None else 1 - self.who_head[:, NOT_WHO]


@dataclass(frozen=True)
class Decisions:
    label: np.ndarray
    reasons: dict
    top1: np.ndarray
    top1_prob: np.ndarray
    margin: np.ndarray
    guard_genus: np.ndarray      # the guarded genus the dangerous-genus rule read, or NO_ID
    signals: Signals


@dataclass(frozen=True)
class GenusFallback:
    genus: str
    probability: float
    who_listed: bool


@dataclass(frozen=True)
class Verdict:
    label: str
    top3: tuple
    genus_fallback: GenusFallback | None
    reason: tuple
    note: str
    message: str
    margin: float | None
    emergency_guidance_available: bool = True


@dataclass(frozen=True)
class DangerFit:
    thresholds: Thresholds
    who_recall: float
    false_alarm: float


def dangerous_genera(classes):
    """The genera holding a WHO-listed class, sorted."""
    listed = classes["who_category"].astype(str).map(WHO_RISK_ID).ne(NOT_WHO)
    return tuple(sorted(classes.loc[listed, "genus"].unique()))


DANGEROUS_GENERA = dangerous_genera(pd.read_csv(CLASSES_CSV))


def danger_tables(classes, medical):
    ordered = classes.sort_values("class_id")
    genus_names, family_names, genus_of_class, family_of_class = taxonomy_index(classes)
    medical = medical.reset_index(drop=True)
    if medical["species"].tolist() != ordered["species"].tolist():
        raise ValueError("the danger table must list classes.csv's species in class_id order")
    basis, dangerous = medical["basis"].astype(str).to_numpy(), medical["dangerous"].to_numpy()
    who = ordered["who_category"].astype(str).map(WHO_RISK_ID).to_numpy(dtype=int)
    venomous_class = ordered["family"].isin(DANGEROUS_FAMILIES).to_numpy()
    category = np.where(who == WHO_CAT1, WHO_CATEGORY_1, WHO_CATEGORY_2)
    if (not np.isin(basis, BASES).all() or dangerous.dtype != bool or (dangerous != (basis != NO_BASIS)).any()
            or (basis != category)[who != NOT_WHO].any() or not dangerous[venomous_class].all()):
        raise ValueError("the danger table must mark dangerous exactly the rows with a basis, give every WHO-listed "
                         "class its category as basis and keep every viper or elapid dangerous")
    listed_genus = np.isin(genus_names, dangerous_genera(classes))
    return DangerTables(ordered["species"].to_numpy(dtype=str), tuple(genus_names), genus_of_class, family_of_class,
                        who, dangerous, np.array([BASIS_NOTE[name] for name in basis]), venomous_class,
                        listed_genus | np.isin(genus_names, DANGEROUS_GENERA_EXTRA), listed_genus,
                        np.isin(family_names, DANGEROUS_FAMILIES))


@cache
def repo_tables():
    return danger_tables(pd.read_csv(CLASSES_CSV), pd.read_csv(MEDICAL_CSV, keep_default_na=False))


def check_scorer(scorer, scale=None):
    if scorer not in SCORERS or (scorer == PROTOTYPE and (scale is None or not scale > 0)):
        raise ValueError(f"scorer must be one of {SCORERS}, and {PROTOTYPE} needs the run's cosine scale (arcface_s); "
                         f"got {scorer!r} with scale {scale!r}")


def _cosine_and_logits(outputs, prototypes, scorer, scale):
    """Cosine to each prototype (NaN across a non-finite embedding) and the species logits the scorer ranks by."""
    check_scorer(scorer, scale)
    embedding = np.atleast_2d(outputs["embedding"])
    cosine = masked_cosine(embedding, prototypes)
    cosine[~np.isfinite(embedding).all(axis=1)] = math.nan
    if scorer == HEAD:
        return cosine, np.atleast_2d(np.asarray(outputs["species_logits"], dtype=float))
    return cosine, scale * np.where(np.isneginf(cosine), 0.0, cosine)


def signals(outputs, temperature, temperature_who, prototypes, tables, scorer=HEAD, scale=None):
    """What the rules read, with the species ranked by `scorer` at `temperature`."""
    cosine, raw = _cosine_and_logits(outputs, prototypes, scorer, scale)
    temperatures = {"who_logits": temperature_who, "genus_logits": 1.0, "family_logits": 1.0}
    with np.errstate(invalid="ignore"):  # a non-finite row comes out NaN, and _decide treats it as such
        probs = softmax(raw / temperature)
        heads = {name: softmax(np.atleast_2d(np.asarray(outputs[name], dtype=float)) / temperatures[name])
                 if name in outputs else None for name in temperatures}
    cosine = np.sort(cosine, axis=1)
    listed_mass = probs[:, tables.who == WHO_CAT1].sum(axis=1) + probs[:, tables.who == WHO_CAT2].sum(axis=1)
    return Signals(probs, listed_mass, heads["who_logits"], heads["genus_logits"], heads["family_logits"],
                   probs @ np.eye(len(tables.genus_names))[tables.genus_of_class], cosine[:, -1], cosine[:, -2], raw)


def decide(outputs, thresholds, prototypes, tables=None, snake_found=None, poor_image=None, scorer=HEAD, scale=None):
    """The verdicts of a batch; scorer and scale must be those the thresholds were fitted with."""
    tables = repo_tables() if tables is None else tables
    found = signals(outputs, thresholds.temperature, thresholds.temperature_who, prototypes, tables, scorer, scale)
    return _decide(found, thresholds, tables, snake_found, poor_image)


def _decide(found, thresholds, tables, snake_found=None, poor_image=None):
    """Danger evidence first; a row with non-finite species probabilities or best cosine is unclear, never safe."""
    n = len(found.probs)
    snake = np.ones(n, dtype=bool) if snake_found is None else np.asarray(snake_found, dtype=bool)
    poor = np.zeros(n, dtype=bool) if poor_image is None else np.asarray(poor_image, dtype=bool)
    species_ok = np.isfinite(found.probs).all(axis=1)
    broken = ~species_ok | ~np.isfinite(found.best_cosine)
    ranked = -np.sort(-found.probs, axis=1)
    top1, top1_prob, margin = found.probs.argmax(axis=1), ranked[:, 0], ranked[:, 0] - ranked[:, 1]
    tied_at_top = found.probs == ranked[:, :1]
    absent = np.zeros(n, dtype=bool)
    triggers = {
        WHO_MASS: species_ok & (found.listed_mass >= thresholds.theta_d),
        WHO_HEAD: absent if found.who_head is None else found.who_listed >= thresholds.theta_who,
        TOP_SPECIES: species_ok & (tied_at_top & tables.dangerous).any(axis=1),
        FAMILY_MASS: absent if found.family_head is None else
        found.family_head[:, tables.venomous_family].sum(axis=1) >= thresholds.theta_v,
    }
    evidence = snake & np.logical_or.reduce(list(triggers.values()))
    open_ = snake & ~evidence
    unclear = broken | (top1_prob < thresholds.theta_conf)
    unfamiliar = ~broken & (found.best_cosine < thresholds.tau_unknown)
    guard_genus, guard_prob = _guarded_genus(found, tables, top1)
    guarded = open_ & ~(unclear | unfamiliar) & (guard_genus != NO_ID)
    by_genus = guarded & (guard_prob >= thresholds.theta_genus)
    genus_unclear = guarded & ~by_genus
    unknown = ~snake | (open_ & (unclear | unfamiliar)) | genus_unclear
    label = np.where(evidence | by_genus, DANGEROUS, np.where(unknown, UNKNOWN, NOT_LISTED))
    reasons = {**{name: evidence & fired for name, fired in triggers.items()}, DANGEROUS_GENUS: by_genus,
               NO_SNAKE: ~snake, UNCLEAR: (open_ & unclear) | genus_unclear, UNFAMILIAR: open_ & unfamiliar,
               POOR_IMAGE: snake & unknown & poor}
    return Decisions(label, reasons, top1, top1_prob, margin, guard_genus, found)


def _guarded_genus(found, tables, top1):
    """The likelier guarded genus of the top species' and the genus head's, and its probability."""
    species_genus = tables.genus_of_class[top1]
    candidates = [species_genus]
    if found.genus_head is not None:
        head_ok = np.isfinite(found.genus_head).all(axis=1)
        candidates.append(np.where(head_ok, found.genus_head.argmax(axis=1), species_genus))
    rows = np.arange(len(top1))
    best_genus, best_prob = np.full(len(top1), NO_ID), np.full(len(top1), -np.inf)
    for genus in candidates:
        prob = found.genus_mass[rows, genus]
        if found.genus_head is not None:
            prob = np.fmax(prob, found.genus_head[rows, genus])
        better = tables.guarded_genus[genus] & (prob > best_prob)
        best_genus, best_prob = np.where(better, genus, best_genus), np.where(better, prob, best_prob)
    return best_genus, best_prob


def expected_verdict(class_id, tables):
    return DANGEROUS if tables.dangerous[class_id] else NOT_LISTED


def verdict(outputs, thresholds, prototypes, tables=None, snake_found=True, poor_image=False, scorer=HEAD,
            scale=None) -> Verdict:
    """One photo's verdict. With no snake found, nothing else is read and outputs may be None."""
    if not snake_found:
        return Verdict(UNKNOWN, (), None, (NO_SNAKE,), "", UNKNOWN_MESSAGE.format(REASON_TEXT[NO_SNAKE]), None)
    tables = repo_tables() if tables is None else tables
    decided = decide(outputs, thresholds, prototypes, tables, poor_image=[poor_image], scorer=scorer, scale=scale)
    found, label = decided.signals, str(decided.label[0])
    reason = tuple(name for name in REASONS if decided.reasons[name][0])
    order = np.argsort(-found.probs[0], kind="stable")[:TOP_K]
    top3 = tuple((str(tables.species[k]), float(found.probs[0, k])) for k in order)
    note, fallback, message = "", None, NOT_LISTED_TEXT
    if label == DANGEROUS:
        note = _danger_note(decided, tables, order)
        message = DANGEROUS_MESSAGE.format(note)
    elif label == UNKNOWN:
        message = UNKNOWN_MESSAGE.format(", ".join(REASON_TEXT[name] for name in reason))
        head = found.genus_head
        if head is not None and head[0].max() >= thresholds.theta_genus:  # a NaN row shows nothing
            genus = int(head[0].argmax())
            fallback = GenusFallback(tables.genus_names[genus], float(head[0, genus]), bool(tables.listed_genus[genus]))
    return Verdict(label, top3, fallback, reason, note, message, float(decided.margin[0]))


def _danger_note(decided, tables, shown):
    """The note names what the screen shows: the top species, a listed species in the top 3, then the heads."""
    found, fired = decided.signals, {name: flags[0] for name, flags in decided.reasons.items()}
    probs = found.probs[0]
    if fired[TOP_SPECIES]:
        return str(tables.note[np.flatnonzero((probs == probs.max()) & tables.dangerous)[0]])
    if fired[WHO_MASS]:
        named = [class_id for class_id in shown if tables.who[class_id] != NOT_WHO]
        if named:
            return LISTED_SPECIES_NOTE.format(tables.species[named[0]], 1 if tables.who[named[0]] == WHO_CAT1 else 2)
        mass = found.listed_mass[0]
        return LISTED_MASS_NOTE.format("under 1%" if mass < 0.01 else f"{mass:.0%}")  # never "0%" on a danger note
    if fired[WHO_HEAD]:
        return WHO_HEAD_NOTE.format(1 if found.who_head[0, WHO_CAT1] >= found.who_head[0, WHO_CAT2] else 2)
    if fired[FAMILY_MASS]:
        return VENOMOUS_NOTE
    genus = decided.guard_genus[0]
    return (GENUS_LISTED_NOTE if tables.listed_genus[genus] else GENUS_NAMED_NOTE).format(tables.genus_names[genus])


def fit_temperature(logits, y):
    """The temperature that minimises the negative log-likelihood of y under softmax(logits / T)."""
    logits, y = np.asarray(logits, dtype=float), np.asarray(y, dtype=int)
    rows = np.arange(len(y))

    def nll(log_temperature):
        scaled = logits / math.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[rows, y]))

    result = minimize_scalar(nll, bounds=np.log(TEMPERATURE_RANGE), method="bounded",
                             options={"xatol": TEMPERATURE_XTOL})
    return float(math.exp(result.x))


def fit_thresholds(calib_outputs, prototypes, tables=None, who_recall_target=WHO_RECALL_TARGET, scorer=HEAD, scale=None,
                   recall_bound=POINT) -> Thresholds:
    """Thresholds from known calibration photos; theta_d and theta_who are fitted jointly (see _joint_fit)."""
    tables = repo_tables() if tables is None else tables
    found, base = _calibrated(calib_outputs, prototypes, tables, scorer, scale)
    return _joint_fit(found, base, tables, calib_outputs, who_recall_target, recall_bound)[0].thresholds


def check_recall_bound(recall_bound):
    if recall_bound not in RECALL_BOUNDS:
        raise ValueError(f"recall_bound must be one of {RECALL_BOUNDS}, got {recall_bound!r}")


def _joint_fit(found, base, tables, calib_outputs, target, recall_bound):
    """(fit, None) at the target itself for POINT; (fit, summary) at its bootstrap lower bound for LOWER."""
    check_recall_bound(recall_bound)
    if recall_bound == POINT:
        return _reached(danger_fit(found, base, tables, calib_outputs["labels"], target), target), None
    return bounded_fit(found, base, tables, calib_outputs, target)


def _reached(fit, target):
    if fit is None:
        raise ValueError(f"no (theta_d, theta_who) pair gives the dangerous verdict to {target:.0%} of the WHO-listed "
                         "calibration photos, so the thresholds can't be fitted")
    return fit


def _calibrated(calib_outputs, prototypes, tables, scorer=HEAD, scale=None):
    """Calibration signals and every threshold but the two WHO ones (left off)."""
    labels = np.asarray(calib_outputs["labels"], dtype=int)
    if (labels < 0).any():
        raise ValueError("the thresholds are fitted on known calibration photos only, never the unknown pool")
    temperature = fit_temperature(_cosine_and_logits(calib_outputs, prototypes, scorer, scale)[1], labels)
    temperature_who = (fit_temperature(calib_outputs["who_logits"], calib_outputs["who_risk_id"])
                       if "who_logits" in calib_outputs else 1.0)
    found = signals(calib_outputs, temperature, temperature_who, prototypes, tables, scorer, scale)
    tau = _kth_largest(found.best_cosine, _photos_needed(KNOWN_ACCEPT, len(labels)))
    passing = found.probs.max(axis=1)[found.best_cosine >= tau]
    theta_conf = _kth_largest(passing, _photos_needed(COVERAGE, len(labels)), default=0.0)
    return found, Thresholds(temperature, temperature_who, math.inf, math.inf, tau, theta_conf)


def _photos_needed(share, n):
    return math.ceil(share * n - SHARE_SLACK)


def _kth_largest(values, k, default=None):
    return default if k > len(values) else float(np.sort(values)[::-1][k - 1])


def listed_and_outside(labels, tables):
    """Per photo: WHO-listed, and outside the WHO list and the viper and elapid families."""
    listed_class = tables.who != NOT_WHO
    return listed_class[labels], ~(listed_class | tables.venomous_class)[labels]


def threshold_grid(values):
    """Quantiles of one WHO signal, ascending, then +inf (the signal off)."""
    return np.append(np.unique(np.quantile(values, np.linspace(0, 1, DANGER_GRID_POINTS))), math.inf)


def danger_fit(found, base, tables, labels, who_recall_target=WHO_RECALL_TARGET, use_mass=True, use_head=True,
               at_most=None):
    """The grid pair with the fewest false alarms reaching the target, ties to larger thresholds; None if none does."""
    labels = np.asarray(labels, dtype=int)
    listed, outside = listed_and_outside(labels, tables)
    if not listed.any():
        raise ValueError("the calibration set has no WHO-listed photo, so theta_d and theta_who can't be fitted")
    off = np.array([math.inf])
    use_head = use_head and found.who_head is not None
    mass_grid = threshold_grid(found.listed_mass) if use_mass else off
    head_grid = threshold_grid(found.who_listed) if use_head else off
    fixed = _decide(found, replace(base, theta_d=math.inf, theta_who=math.inf), tables).label == DANGEROUS
    head_values = found.who_listed if use_head else np.zeros(len(labels))

    def hits(members):
        by_head = head_values[members][None, :] >= head_grid[:, None]
        return np.array([((fixed[members] | (found.listed_mass[members] >= theta_d))[None, :] | by_head).sum(axis=1)
                         for theta_d in mass_grid])

    recalled, alarms = hits(listed), hits(outside)
    reaching = recalled >= _photos_needed(who_recall_target, int(listed.sum()))
    if at_most is not None:
        reaching &= (mass_grid[:, None] <= at_most.theta_d) & (head_grid[None, :] <= at_most.theta_who)
    mass_at, head_at = np.nonzero(reaching)
    if not len(mass_at):
        return None
    best = np.lexsort((-mass_at, -(mass_at + head_at), alarms[mass_at, head_at]))[0]
    mass_at, head_at = mass_at[best], head_at[best]
    return DangerFit(replace(base, theta_d=float(mass_grid[mass_at]), theta_who=float(head_grid[head_at])),
                     float(recalled[mass_at, head_at] / listed.sum()),
                     float(alarms[mass_at, head_at] / outside.sum()) if outside.any() else math.nan)


def recall_lower_bound(dangerous, groups, n=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """The 5th percentile of the dangerous share over resamples of whole photographers."""
    dangerous = np.asarray(dangerous, dtype=bool)
    return float(np.percentile([dangerous[rows].mean() for rows in grouped_resamples(groups, n, seed)],
                               BOUND_PERCENTILE))


def bounded_fit(found, base, tables, calib_outputs, required=WHO_RECALL_TARGET):
    """(fit, summary) at the first target whose lower bound reaches `required`, within the point fit's pair."""
    labels = np.asarray(calib_outputs["labels"], dtype=int)
    cap = _reached(danger_fit(found, base, tables, labels, required), required).thresholds
    listed = (tables.who != NOT_WHO)[labels]
    groups = photographers(calib_outputs)[listed]
    steps = math.floor((BOUND_TARGET_MAX - required) / BOUND_TARGET_STEP + SHARE_SLACK)
    tried = []
    for target in [round(required + BOUND_TARGET_STEP * step, 6) for step in range(steps + 1)]:
        fit = danger_fit(found, base, tables, labels, target, at_most=cap)
        if fit is None:
            break
        bound = recall_lower_bound(_decide(found, fit.thresholds, tables).label[listed] == DANGEROUS, groups)
        tried.append((target, bound))
        if bound >= required:
            return fit, {"required": required, "percentile": BOUND_PERCENTILE, "resamples": BOOTSTRAP_RESAMPLES,
                         "seed": BOOTSTRAP_SEED, "listed_photos": int(listed.sum()),
                         "photographers": len(np.unique(groups)), "target": target, "lower_bound": bound,
                         "point_recall": fit.who_recall, "false_alarm": fit.false_alarm,
                         "point_thresholds": {"theta_d": cap.theta_d, "theta_who": cap.theta_who},
                         "tried": [{"target": t, "lower_bound": b} for t, b in tried]}
    raise ValueError(f"no WHO recall target from {required} to {BOUND_TARGET_MAX} gives a photographer-bootstrap lower "
                     f"bound of {required} with theta_d <= {cap.theta_d} and theta_who <= {cap.theta_who} (the point "
                     "fit's); lower bound per target tried: " + ", ".join(f"{t:.3f}: {b:.4f}" for t, b in tried))


def fit_snake_threshold(box_conf_calibration, keep=SNAKE_KEEP):
    """The box confidence keeping `keep` of the calibration photos (all snakes); no box (NaN) is never kept."""
    conf = np.asarray(box_conf_calibration, dtype=float)
    detected = conf[np.isfinite(conf)]
    if not len(detected):
        raise ValueError("no calibration photo has a box, so tau_snake can't be fitted")
    return _kth_largest(detected, _photos_needed(keep, len(conf)), default=float(detected.min()))


def decision_report(run_dir, classes, metadata=None, medical=None, prototype_set=PLAIN, quality=None, box_conf=None,
                    scorer=HEAD, recall_bound=POINT):
    """Fits on calibration, scores validation and the unknown pool, and writes decision_metrics.json."""
    run_dir = Path(run_dir)
    scale = load_config(run_dir / CONFIG_FILE).arcface_s if scorer == PROTOTYPE else None
    scoring = {"scorer": scorer, "scale": scale}
    tables = danger_tables(classes, pd.read_csv(MEDICAL_CSV, keep_default_na=False) if medical is None else medical)
    outputs = {name: _load(run_dir / file) for name, file in
               (("calibration", CALIBRATION_OUTPUTS), ("val", VAL_OUTPUTS), ("unknown", UNKNOWN_OUTPUTS))}
    if metadata is not None:
        _check_splits(metadata, outputs)
    prototypes = np.load(run_dir / PROTOTYPES_FILE)[prototype_set]
    _check_finite(outputs, prototype_set, prototypes)
    calib_outputs, val_outputs, unknown_outputs = outputs["calibration"], outputs["val"], outputs["unknown"]
    calib_labels, labels = calib_outputs["labels"], val_outputs["labels"]
    found, base = _calibrated(calib_outputs, prototypes, tables, **scoring)
    fit, bounded = _joint_fit(found, base, tables, calib_outputs, WHO_RECALL_TARGET, recall_bound)
    target = WHO_RECALL_TARGET if bounded is None else bounded["target"]
    fits = {**{name: danger_fit(found, base, tables, calib_labels, target, use_mass=mass, use_head=head)
               for name, (mass, head) in SINGLE_SIGNALS.items()}, "joint": fit}
    thresholds = fit.thresholds
    tau_snake = None if box_conf is None else fit_snake_threshold(_box_conf_values(box_conf, calib_outputs["paths"]))

    def snake_in(paths):
        return None if tau_snake is None else _box_conf_values(box_conf, paths) >= tau_snake

    def decided_on(frame, flagged=True, at=thresholds):
        return decide(frame, at, prototypes, tables, snake_in(frame["paths"]) if flagged else None,
                      _poor_flags(quality, frame["paths"]) if flagged else None, **scoring)

    calib, val, unknown = decided_on(calib_outputs, flagged=False), decided_on(val_outputs), decided_on(unknown_outputs)
    if bounded is not None:
        point = replace(thresholds, **bounded["point_thresholds"])
        bound_fit = {**bounded, "against_point": {
            "calibration": _against_point(calib_labels, calib.label, decided_on(calib_outputs, False, point).label,
                                          tables),
            "validation": _against_point(labels, val.label, decided_on(val_outputs, at=point).label, tables)}}
    scored = {name: outputs[name]["paths"] for name in ("val", "unknown")}
    report = {
        "run": run_dir.name,
        "prototype_set": prototype_set,
        "scorer": scorer,
        **({"prototype_scale": scale, "scorer_note": PROTOTYPE_SCORER_NOTE} if scorer == PROTOTYPE else {}),
        "recall_bound": recall_bound,
        **({"recall_bound_fit": bound_fit} if bounded is not None else {}),
        "photos": {name: len(frame["labels"]) for name, frame in outputs.items()},
        "thresholds": asdict(thresholds),
        "fit": _fit_summary(calib_labels, calib, thresholds, tables),
        "danger_signals": _danger_signals(fits, val_outputs, prototypes, tables, scoring),
        "calibration": {"before": calibration(softmax(val.signals.raw_logits), labels),
                        "after": calibration(val.signals.probs, labels)},
        "o4_point": _o4_point(val, labels, thresholds),
        "verdicts": {"val": _counts(val), "unknown": _counts(unknown)},
        "verdict_metrics": _verdict_metrics(labels, val, tables),
        "intervals": _intervals(val_outputs, val, tables),
        "fallback": _fallback(val_outputs, val, thresholds, tables),
        "u1": _u1(val, unknown_outputs, unknown, thresholds),
        "poor_image": "not available" if quality is None else {
            "flags": list(QUALITY_FLAGS),
            "flagged": {name: int(_poor_flags(quality, paths).sum()) for name, paths in scored.items()},
            "covered": {name: _share(np.isin(paths, quality["path"])) for name, paths in scored.items()}},
        "no_snake": {"val": int(val.reasons[NO_SNAKE].sum()), "unknown": int(unknown.reasons[NO_SNAKE].sum()),
                     "source": NO_SNAKE_NOT_AVAILABLE},
    }
    if tau_snake is not None:
        report["no_snake"].update(source=SNAKE_THRESHOLD_SOURCE, tau_snake=tau_snake, keep_target=SNAKE_KEEP,
                                  calibration_kept=_share(snake_in(calib_outputs["paths"])))
    return write_json(run_dir / DECISION_FILE, report)


def _load(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def _check_splits(metadata, outputs):
    split_of = metadata.set_index("path")["split"]
    for name, expected in (("calibration", "train"), ("val", "val"), ("unknown", "unknown_pool")):
        paths = outputs[name]["paths"]
        splits = split_of.reindex(paths).to_numpy()
        assert_no_test_paths(pd.DataFrame({"path": paths, "split": splits}))
        if (splits != expected).any():
            raise ValueError(f"{name} photos outside metadata's {expected} split, e.g. {paths[splits != expected][0]}")


def _check_finite(outputs, prototype_set, prototypes):
    """Refuses any non-finite saved value, so a broken run fails loudly instead of quietly lowering O3."""
    arrays = {**outputs, "prototypes": {prototype_set: prototypes}}
    problems = [f"{name} {key}: {bad} rows" for name, frame in arrays.items() for key, values in frame.items()
                if values.dtype.kind == "f"
                for bad in [int((~np.isfinite(values.reshape(len(values), -1))).any(axis=1).sum())] if bad]
    if problems:
        raise ValueError(f"non-finite values in the saved outputs: {', '.join(problems)}")


def photographers(outputs):
    return photographer_groups(pd.DataFrame({"observer_id": outputs["observer_id"], "path": outputs["paths"]}))


def _box_conf_values(box_conf, paths):
    return np.array([box_conf.get(path, math.nan) for path in paths], dtype=float)


def _poor_flags(quality, paths):
    if quality is None:
        return None
    flags = quality.set_index("path")[list(QUALITY_FLAGS)].astype(bool).any(axis=1)
    return flags.reindex(paths, fill_value=False).to_numpy(dtype=bool)


def _counts(decided, members=None):
    members = np.ones(len(decided.label), dtype=bool) if members is None else members
    labels = decided.label[members]
    return {"total": int(labels.size), **{label: int((labels == label).sum()) for label in VERDICTS},
            "reasons": {name: int(flags[members].sum()) for name, flags in decided.reasons.items()}}


def _o4_point(decided, labels, thresholds):
    answered = (decided.top1_prob >= thresholds.theta_conf) & (decided.signals.best_cosine >= thresholds.tau_unknown)
    return {"answered_share": _share(answered), "error": _share(decided.top1[answered] != labels[answered]),
            "verdict_answered_share": _share(decided.label != UNKNOWN)}


def _fit_summary(labels, decided, thresholds, tables):
    present = set(np.unique(labels).tolist())
    caught = _recall_and_alarm(labels, decided.label, tables)
    return {"photos": len(labels), "classes": len(present),
            "not_in_calibration": [str(name) for class_id, name in enumerate(tables.species) if class_id not in present],
            "who_listed_photos": int((tables.who[labels] != NOT_WHO).sum()),
            "who_dangerous_recall": caught["who_recall"], "false_alarm": caught["false_alarm"],
            "tau_pass_share": _share(decided.signals.best_cosine >= thresholds.tau_unknown),
            **{f"o4_{key}": value for key, value in _o4_point(decided, labels, thresholds).items()}}


def _against_point(labels, lower, point, tables):
    lower, point = lower == DANGEROUS, point == DANGEROUS
    extra, not_listed = lower & ~point, tables.who[np.asarray(labels, dtype=int)] == NOT_WHO
    return {"point_dangerous_not_lower": int((point & ~lower).sum()), "extra_dangerous": int(extra.sum()),
            "extra_dangerous_not_who_listed": int((extra & not_listed).sum())}


def _recall_and_alarm(labels, verdicts, tables):
    listed, outside = listed_and_outside(labels, tables)
    return {"who_recall": _share(verdicts[listed] == DANGEROUS), "false_alarm": _share(verdicts[outside] == DANGEROUS)}


def _danger_signals(fits, val_outputs, prototypes, tables, scoring):
    result = dict.fromkeys(fits)
    for name, fit in fits.items():
        if fit is not None:
            verdicts = decide(val_outputs, fit.thresholds, prototypes, tables, **scoring).label
            result[name] = {"theta_d": fit.thresholds.theta_d, "theta_who": fit.thresholds.theta_who,
                            "calibration": {"who_recall": fit.who_recall, "false_alarm": fit.false_alarm},
                            "validation": _recall_and_alarm(val_outputs["labels"], verdicts, tables)}
    return result


def _verdict_metrics(labels, decided, tables):
    """false_alarm leaves out WHO-listed, viper and elapid photos; false_alarm_outside_table every dangerous class."""
    probs, verdicts = decided.signals.probs, decided.label
    listed = tables.who != NOT_WHO
    by_table, by_family, by_list_or_family = (safety_report(labels, probs, tables.who, flags, verdicts) for flags in
                                              (tables.dangerous, tables.venomous_class, listed | tables.venomous_class))
    o3_keys = ("dangerous_as_not_listed_all", "dangerous_as_not_listed_answered", "coverage_listed", "top1_error_raw")
    return {"who_dangerous_recall": _share(verdicts[listed[labels]] == DANGEROUS),
            "false_alarm": by_list_or_family["false_alarm"], "false_alarm_outside_table": by_table["false_alarm"],
            "venomous_recall": by_family["venomous_recall"], "venomous_as_not_listed": by_family["venomous_as_not_listed"],
            "table_dangerous_recall": by_table["venomous_recall"], "who_recall_raw": by_table["who_recall_raw"],
            "abstain_rate": by_table["abstain_rate"], "o3": {key: by_table[key] for key in o3_keys}}


def _intervals(val_outputs, decided, tables):
    labels, verdicts = val_outputs["labels"], decided.label
    listed, venomous = (tables.who != NOT_WHO)[labels], tables.venomous_class[labels]
    dangerous, not_listed = verdicts == DANGEROUS, verdicts == NOT_LISTED
    shares = {"who_dangerous_recall": (listed, dangerous), "false_alarm": (~(listed | venomous), dangerous),
              "venomous_recall": (venomous, dangerous), "dangerous_as_not_listed_all": (listed, not_listed)}
    groups = photographers(val_outputs)
    return {name: list(grouped_bootstrap(lambda rows, among=among, hit=hit: _share(hit[rows][among[rows]]), groups))
            for name, (among, hit) in shares.items()}


def _fallback(val_outputs, decided, thresholds, tables):
    found = decided.signals
    open_unknown = (decided.label == UNKNOWN) & ~decided.reasons[NO_SNAKE]
    result = {"genus": None, "family": None}
    for level, head, truth in (("genus", found.genus_head, val_outputs["genus_id"]),
                               ("family", found.family_head, val_outputs["family_id"])):
        if head is not None:
            shown = open_unknown & (head.max(axis=1) >= thresholds.theta_genus)
            result[level] = {"unknown_photos": int(open_unknown.sum()), "shown": int(shown.sum()),
                             "shown_share": _share(shown[open_unknown]),
                             "correct": _share(head.argmax(axis=1)[shown] == truth[shown])}
    result["taxonomy"] = taxonomy_report(found.probs, found.genus_head, found.family_head, tables.genus_of_class,
                                         tables.family_of_class, val_outputs["genus_id"], val_outputs["family_id"])
    return result


def _u1(val, unknown_outputs, unknown, thresholds):
    """U1: the unknown pool by known and unknown genus, against validation, and each WHO-listed species."""
    def scores(found):
        return {"msp": -found.probs.max(axis=1), "energy": -logsumexp(found.raw_logits, axis=1),
                "prototype_distance": 1 - found.best_cosine, "cosine_margin": found.second_cosine - found.best_cosine}

    known_scores, unknown_scores = scores(val.signals), scores(unknown.signals)
    genus_known = unknown_outputs["genus_id"] != NO_ID
    species = unknown_outputs["species"]
    flagged = unknown.signals.best_cosine < thresholds.tau_unknown
    groups = {}
    for name, members in (("all", np.ones_like(genus_known)), ("known_genus", genus_known),
                          ("unknown_genus", ~genus_known)):
        groups[name] = {"photos": int(members.sum()), "species": int(np.unique(species[members]).size)}
        if members.any():
            groups[name].update({"open_set": {score: open_set(known_scores[score], unknown_scores[score][members])
                                              for score in OPEN_SET_SCORES},
                                 "flagged_at_tau": _share(flagged[members]), "verdicts": _counts(unknown, members)})
    who_listed = {}
    for name in np.unique(species[unknown_outputs["who_risk_id"] != NOT_WHO]):
        members = species == name
        who_listed[str(name)] = {"photos": int(members.sum()),
                                 "who_risk_id": int(unknown_outputs["who_risk_id"][members][0]),
                                 "genus_known": bool(genus_known[members].all()), "flagged_at_tau": _share(flagged[members]),
                                 "verdicts": _counts(unknown, members)}
    return {"scores": SCORE_DIRECTION, "o4_target_flagged": O4_UNKNOWN_TARGET, "groups": groups,
            "who_listed_species": who_listed, "non_snake_objects": NON_SNAKE_SKIPPED}
