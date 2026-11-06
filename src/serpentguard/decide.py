"""Verdicts (dangerous, not on the WHO list, we don't know), their thresholds and the report.

Also the danger table the verdicts read. WHO codes (who_risk_id): 0 not listed, 1 WHO Category 2,
2 WHO Category 1.
"""
import math
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from serpentguard.data import NO_ID
from serpentguard.metrics import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, DANGEROUS_FAMILIES, VERDICTS,
                                  WHO_RISK_ID, assert_no_test_paths, calibration, grouped_bootstrap,
                                  grouped_resamples, open_set, photographer_groups, safety_report,
                                  share, softmax, taxonomy_index, taxonomy_report, write_json)
from serpentguard.models import masked_cosine
from serpentguard.train import (CALIBRATION_OUTPUTS, CONFIG_FILE, PROTOTYPES_FILE, UNKNOWN_OUTPUTS,
                                VAL_OUTPUTS)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES_CSV = REPO_ROOT / "data" / "classes.csv"
MEDICAL_CSV = REPO_ROOT / "data" / "m3" / "medical_importance.csv"
DECISION_FILE = "decision_metrics.json"
PRINTED_PAGE_OFFSET = 3  # the guideline's printed page numbers run 3 behind its PDF pages
DANGEROUS_GENERA_EXTRA = ("Rhabdophis",)  # keelbacks: medically important outside the two families
WHO_CAT2, WHO_CAT1 = 1, 2
WHO_RECALL_TARGET = 0.95  # share of WHO-listed calibration photos that must be called dangerous
REASONS = ("who_listed_mass", "who_head", "top_species_dangerous", "venomous_family_mass",
           "dangerous_genus", "no_snake", "unclear_between_species", "unfamiliar", "poor_image")
# no_majority is a clip's own reason: no label holds 4 of its last 6 frames
REASON_TEXT = {"no_snake": "no snake found", "unclear_between_species": "unclear between species",
               "unfamiliar": "unlike any species we know",
               "poor_image": "the photo is dark or blurry",
               "no_majority": "the frames do not agree"}


@dataclass(frozen=True)
class Thresholds:
    temperature: float
    temperature_who: float
    theta_d: float
    theta_who: float
    tau_unknown: float
    theta_conf: float
    theta_v: float = 0.5
    theta_genus: float = 0.5


@dataclass(frozen=True)
class Verdict:
    """One photo's verdict, as the app shows it; asdict(verdict) gives a JSON-ready dict."""
    label: str
    top3: tuple
    genus_fallback: dict | None
    reason: tuple
    note: str
    message: str
    margin: float | None
    emergency_guidance_available: bool = True


def pages(*pdf_pages):
    """'PDF p. N (printed p. M)', or the 'pp.' form for several pages."""
    prefix = "p." if len(pdf_pages) == 1 else "pp."
    printed = (page - PRINTED_PAGE_OFFSET for page in pdf_pages)
    return (f"PDF {prefix} {', '.join(map(str, pdf_pages))} "
            f"(printed {prefix} {', '.join(map(str, printed))})")


BASIS_NOTE = {"who_category_1": "WHO Category 1", "who_category_2": "WHO Category 2",
              "venomous_family": "venomous family, not on WHO's priority list",
              "who_guideline_named": "named as medically important in the WHO guideline, "
                                     f"{pages(65)}",
              "none": ""}


def build_medical_importance(classes):
    """The danger table (data/m3/medical_importance.csv): each class's danger and its basis."""
    # cites pages of the WHO South-East Asia snakebite guideline (2016), never its wording
    category_source = f"WHO SEARO 2016, {pages(87)}, India lists; categories defined {pages(84)}"
    venomous_source = f"WHO SEARO 2016, {pages(30)}, venomous families; family from classes.csv"
    named_source = f"WHO SEARO 2016, {pages(65, 30)}"  # medically important colubrids; keelbacks
    listed_as = {  # the guideline's older names for our classes
        "Ophiophagus kaalinga": "listed as Ophiophagus hannah, which O. kaalinga was later split "
                                "from",
        "Macrovipera lebetinus": "listed as Macrovipera lebetina",
        "Craspedocephalus malabaricus": "listed as Trimeresurus (Craspedocephalus) malabaricus",
        "Craspedocephalus gramineus": "listed as Trimeresurus (Craspedocephalus) gramineus",
    }
    named_as = {"Rhabdophis helleri": "named as Rhabdophis subminiatus, which R. helleri was split "
                                      "from"}
    rows = []
    for record in classes.sort_values("class_id").itertuples(index=False):
        who = str(record.who_category)
        if who != "not_listed":
            basis, source = {"1": "who_category_1", "2": "who_category_2"}[who], category_source
            note = "; ".join(filter(None, (f"India Category {who}", listed_as.get(record.species))))
        elif record.family in DANGEROUS_FAMILIES:
            basis, source = "venomous_family", venomous_source
            note = f"{record.family}, not in the WHO category lists"
        elif record.genus in DANGEROUS_GENERA_EXTRA:
            basis, source = "who_guideline_named", named_source
            note = named_as.get(record.species, "genus named through R. subminiatus and "
                                                "R. tigrinus; this species not named")
        else:
            basis, source, note = "none", "classes.csv", ""
        rows.append({"species": record.species, "family": record.family, "genus": record.genus,
                     "who_category": who, "dangerous": basis != "none", "basis": basis,
                     "source": source, "note": note})
    return pd.DataFrame(rows, columns=["species", "family", "genus", "who_category", "dangerous",
                                       "basis", "source", "note"])


def dangerous_genera(classes):
    """The genera holding a WHO-listed class, sorted."""
    listed = classes["who_category"].astype(str).map(WHO_RISK_ID).ne(0)
    return tuple(sorted(classes.loc[listed, "genus"].unique()))


def danger_tables(classes, medical):
    """The per-class and per-genus arrays the verdict rules read, checked against the table."""
    ordered = classes.sort_values("class_id")
    genus_names, family_names, genus_of_class, family_of_class = taxonomy_index(classes)
    medical = medical.reset_index(drop=True)
    if medical["species"].tolist() != ordered["species"].tolist():
        raise ValueError("the danger table must list classes.csv's species in class_id order")
    basis, dangerous = medical["basis"].astype(str).to_numpy(), medical["dangerous"].to_numpy()
    who = ordered["who_category"].astype(str).map(WHO_RISK_ID).to_numpy(dtype=int)
    venomous_class = ordered["family"].isin(DANGEROUS_FAMILIES).to_numpy()
    category = np.where(who == WHO_CAT1, "who_category_1", "who_category_2")
    if (not np.isin(basis, list(BASIS_NOTE)).all() or dangerous.dtype != bool
            or (dangerous != (basis != "none")).any() or (basis != category)[who != 0].any()
            or not dangerous[venomous_class].all()):
        raise ValueError("the danger table must mark dangerous exactly the rows with a basis, give "
                         "every WHO-listed class its category as basis and keep every viper or "
                         "elapid dangerous")
    listed_genus = np.isin(genus_names, dangerous_genera(classes))
    return SimpleNamespace(
        species=ordered["species"].to_numpy(dtype=str), genus_names=tuple(genus_names),
        genus_of_class=genus_of_class, family_of_class=family_of_class, who=who,
        dangerous=dangerous, note=np.array([BASIS_NOTE[name] for name in basis]),
        venomous_class=venomous_class,
        # a genus is guarded when it holds a WHO-listed class, or is a keelback genus
        guarded_genus=listed_genus | np.isin(genus_names, DANGEROUS_GENERA_EXTRA),
        listed_genus=listed_genus, venomous_family=np.isin(family_names, DANGEROUS_FAMILIES))


@cache
def repo_tables():
    """danger_tables for the repo's classes.csv and danger table."""
    return danger_tables(pd.read_csv(CLASSES_CSV), pd.read_csv(MEDICAL_CSV, keep_default_na=False))


def expected_verdict(class_id, tables):
    return "dangerous" if tables.dangerous[class_id] else "not_listed"


def cosine_and_logits(outputs, prototypes, scorer, scale):
    """Cosine to each prototype and the logits that rank the species ("head" or "prototype")."""
    # "prototype" ranks by scale x cosine, scale being the run's arcface_s
    if scorer not in ("head", "prototype") or (scorer == "prototype"
                                               and (scale is None or not scale > 0)):
        raise ValueError(f"scorer must be 'head' or 'prototype', and 'prototype' needs the run's "
                         f"cosine scale (arcface_s); got {scorer!r} with scale {scale!r}")
    embedding = np.atleast_2d(outputs["embedding"])
    cosine = masked_cosine(embedding, prototypes)
    cosine[~np.isfinite(embedding).all(axis=1)] = math.nan  # nan across a non-finite embedding
    if scorer == "head":
        return cosine, np.atleast_2d(np.asarray(outputs["species_logits"], dtype=float))
    return cosine, scale * np.where(np.isneginf(cosine), 0.0, cosine)


def signals(outputs, temperature, temperature_who, prototypes, tables, scorer="head", scale=None):
    """What the rules read, with the species ranked by `scorer` at `temperature`."""
    cosine, raw = cosine_and_logits(outputs, prototypes, scorer, scale)
    temperatures = {"who_logits": temperature_who, "genus_logits": 1.0, "family_logits": 1.0}
    with np.errstate(invalid="ignore"):  # a non-finite row comes out nan; the rules handle it
        probs = softmax(raw / temperature)
        heads = {name: softmax(np.atleast_2d(np.asarray(outputs[name], dtype=float))
                               / temperatures[name])
                 if name in outputs else None for name in temperatures}
    cosine = np.sort(cosine, axis=1)
    who, who_head = tables.who, heads["who_logits"]
    return SimpleNamespace(
        probs=probs,  # temperature-scaled species probabilities from raw_logits
        listed_mass=probs[:, who == WHO_CAT1].sum(axis=1) + probs[:, who == WHO_CAT2].sum(axis=1),
        who_head=who_head, who_listed=None if who_head is None else 1 - who_head[:, 0],
        genus_head=heads["genus_logits"], family_head=heads["family_logits"],
        genus_mass=probs @ np.eye(len(tables.genus_names))[tables.genus_of_class],
        best_cosine=cosine[:, -1], second_cosine=cosine[:, -2], raw_logits=raw)


def decide(outputs, thresholds, prototypes, tables=None, snake_found=None, poor_image=None,
           scorer="head", scale=None):
    """The verdicts of a batch; scorer and scale must be those the thresholds were fitted with."""
    tables = repo_tables() if tables is None else tables
    found = signals(outputs, thresholds.temperature, thresholds.temperature_who, prototypes,
                    tables, scorer, scale)
    return decide_signals(found, thresholds, tables, snake_found, poor_image)


def decide_signals(found, thresholds, tables, snake_found=None, poor_image=None):
    """Danger evidence first; a row with non-finite scores is unclear, never safe."""
    probs = found.probs
    n = len(probs)
    snake = np.ones(n, dtype=bool) if snake_found is None else np.asarray(snake_found, dtype=bool)
    poor = np.zeros(n, dtype=bool) if poor_image is None else np.asarray(poor_image, dtype=bool)
    species_ok = np.isfinite(probs).all(axis=1)
    broken = ~species_ok | ~np.isfinite(found.best_cosine)
    ranked = -np.sort(-probs, axis=1)
    top1, top1_prob, margin = probs.argmax(axis=1), ranked[:, 0], ranked[:, 0] - ranked[:, 1]
    tied_at_top = probs == ranked[:, :1]
    absent = np.zeros(n, dtype=bool)
    family_head = found.family_head
    triggers = {
        "who_listed_mass": species_ok & (found.listed_mass >= thresholds.theta_d),
        "who_head": (absent if found.who_head is None
                     else found.who_listed >= thresholds.theta_who),
        "top_species_dangerous": species_ok & (tied_at_top & tables.dangerous).any(axis=1),
        "venomous_family_mass": absent if family_head is None else
        family_head[:, tables.venomous_family].sum(axis=1) >= thresholds.theta_v,
    }
    evidence = snake & np.logical_or.reduce(list(triggers.values()))
    open_ = snake & ~evidence
    unclear = broken | (top1_prob < thresholds.theta_conf)
    unfamiliar = ~broken & (found.best_cosine < thresholds.tau_unknown)
    guard_genus, guard_prob = guarded_genus(found, tables, top1)
    guarded = open_ & ~(unclear | unfamiliar) & (guard_genus != NO_ID)
    by_genus = guarded & (guard_prob >= thresholds.theta_genus)
    genus_unclear = guarded & ~by_genus
    unknown = ~snake | (open_ & (unclear | unfamiliar)) | genus_unclear
    label = np.where(evidence | by_genus, "dangerous",
                     np.where(unknown, "unknown", "not_listed"))
    reasons = {**{name: evidence & fired for name, fired in triggers.items()},
               "dangerous_genus": by_genus, "no_snake": ~snake,
               "unclear_between_species": (open_ & unclear) | genus_unclear,
               "unfamiliar": open_ & unfamiliar, "poor_image": snake & unknown & poor}
    return SimpleNamespace(label=label, reasons=reasons, top1=top1, top1_prob=top1_prob,
                           margin=margin, guard_genus=guard_genus,  # guarded genus read, or -1
                           signals=found)


def guarded_genus(found, tables, top1):
    """The likelier guarded genus of the top species' and the genus head's, and its probability."""
    genus_head = found.genus_head
    species_genus = tables.genus_of_class[top1]
    candidates = [species_genus]
    if genus_head is not None:
        head_ok = np.isfinite(genus_head).all(axis=1)
        candidates.append(np.where(head_ok, genus_head.argmax(axis=1), species_genus))
    rows = np.arange(len(top1))
    best_genus, best_prob = np.full(len(top1), NO_ID), np.full(len(top1), -np.inf)
    for genus in candidates:
        prob = found.genus_mass[rows, genus]
        if genus_head is not None:
            prob = np.fmax(prob, genus_head[rows, genus])
        better = tables.guarded_genus[genus] & (prob > best_prob)
        best_genus, best_prob = (np.where(better, genus, best_genus),
                                 np.where(better, prob, best_prob))
    return best_genus, best_prob


def verdict(outputs, thresholds, prototypes, tables=None, snake_found=True, poor_image=False,
            scorer="head", scale=None):
    """One photo's verdict: label, top3, genus_fallback, reason, note, message and margin."""
    if not snake_found:  # nothing else is read, so outputs may be None
        return Verdict("unknown", (), None, ("no_snake",), "", "we don't know: no snake found",
                       None)
    tables = repo_tables() if tables is None else tables
    decided = decide(outputs, thresholds, prototypes, tables, poor_image=[poor_image],
                     scorer=scorer, scale=scale)
    found, label = decided.signals, str(decided.label[0])
    reason = tuple(name for name in REASONS if decided.reasons[name][0])
    order = np.argsort(-found.probs[0], kind="stable")[:3]
    top3 = tuple((str(tables.species[k]), float(found.probs[0, k])) for k in order)
    note, fallback, message = "", None, "not on the WHO list, still keep away"
    if label == "dangerous":
        note = danger_note(decided, tables, order)
        message = f"dangerous: {note}"
    elif label == "unknown":
        message = "we don't know: " + ", ".join(REASON_TEXT[name] for name in reason)
        head = found.genus_head
        if head is not None and head[0].max() >= thresholds.theta_genus:  # nan shows nothing
            genus = int(head[0].argmax())
            fallback = {"genus": tables.genus_names[genus],
                        "probability": float(head[0, genus]),
                        "who_listed": bool(tables.listed_genus[genus])}
    return Verdict(label, top3, fallback, reason, note, message, float(decided.margin[0]))


def danger_note(decided, tables, shown):
    """The note names what the screen shows: the top species, a listed one in the top 3, a head."""
    found, fired = (decided.signals,
                    {name: flags[0] for name, flags in decided.reasons.items()})
    probs = found.probs[0]
    if fired["top_species_dangerous"]:
        return str(tables.note[np.flatnonzero((probs == probs.max()) & tables.dangerous)[0]])
    if fired["who_listed_mass"]:
        named = [class_id for class_id in shown if tables.who[class_id] != 0]
        if named:
            category = 1 if tables.who[named[0]] == WHO_CAT1 else 2
            return (f"could be a WHO-listed species: {tables.species[named[0]]} "
                    f"(Category {category})")
        mass = found.listed_mass[0]  # no listed species in the top 3; never "0%" here
        return ("could be a WHO-listed species: "
                f"{'under 1%' if mass < 0.01 else f'{mass:.0%}'} combined probability")
    if fired["who_head"]:
        category = 1 if found.who_head[0, WHO_CAT1] >= found.who_head[0, WHO_CAT2] else 2
        return f"the photo looks like a WHO Category {category} snake"
    if fired["venomous_family_mass"]:
        return BASIS_NOTE["venomous_family"]
    genus = decided.guard_genus[0]
    if tables.listed_genus[genus]:
        return f"genus {tables.genus_names[genus]} has species on the WHO list"
    return (f"genus {tables.genus_names[genus]} has species named as medically important in "
            "the WHO guideline")


def fit_temperature(logits, y):
    """The temperature (0.05..20) minimising the negative log-likelihood of softmax(logits / T)."""
    logits, y = np.asarray(logits, dtype=float), np.asarray(y, dtype=int)
    rows = np.arange(len(y))

    def nll(log_temperature):
        scaled = logits / math.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[rows, y]))

    result = minimize_scalar(nll, bounds=np.log((0.05, 20.0)), method="bounded",
                             options={"xatol": 1e-6})
    return float(math.exp(result.x))


def photos_needed(target_share, n):
    return math.ceil(target_share * n - 1e-9)  # the slack keeps 0.95 x 20 at 19, not 20


def kth_largest(values, k, default=None):
    return default if k > len(values) else float(np.sort(values)[::-1][k - 1])


def calibrated(calib_outputs, prototypes, tables, scorer="head", scale=None):
    """Calibration signals and every threshold but the two WHO ones, which are left off (inf)."""
    labels = np.asarray(calib_outputs["labels"], dtype=int)
    if (labels < 0).any():
        raise ValueError("the thresholds are fitted on known calibration photos only, never the "
                         "unknown pool")
    temperature = fit_temperature(cosine_and_logits(calib_outputs, prototypes, scorer, scale)[1],
                                  labels)
    temperature_who = (fit_temperature(calib_outputs["who_logits"], calib_outputs["who_risk_id"])
                       if "who_logits" in calib_outputs else 1.0)
    found = signals(calib_outputs, temperature, temperature_who, prototypes, tables, scorer, scale)
    tau = kth_largest(found.best_cosine, photos_needed(0.95, len(labels)))  # 95% pass tau
    passing = found.probs.max(axis=1)[found.best_cosine >= tau]
    theta_conf = kth_largest(passing, photos_needed(0.8, len(labels)), default=0.0)  # 80% answered
    return found, Thresholds(temperature, temperature_who, math.inf, math.inf, tau, theta_conf)


def listed_and_outside(labels, tables):
    """Per photo: WHO-listed, and outside both the WHO list and the viper and elapid families."""
    listed_class = tables.who != 0
    return listed_class[labels], ~(listed_class | tables.venomous_class)[labels]


def threshold_grid(values):
    """101 quantiles of one WHO signal, ascending, then +inf (the signal off)."""
    return np.append(np.unique(np.quantile(values, np.linspace(0, 1, 101))), math.inf)


def danger_fit(found, base, tables, labels, target=WHO_RECALL_TARGET, use_mass=True,
               use_head=True, at_most=None):
    """The (theta_d, theta_who) pair with the fewest false alarms reaching the target, or None."""
    # ties go to the larger thresholds; returns {thresholds, who_recall, false_alarm}
    labels = np.asarray(labels, dtype=int)
    listed, outside = listed_and_outside(labels, tables)
    if not listed.any():
        raise ValueError("the calibration set has no WHO-listed photo, so theta_d and theta_who "
                         "can't be fitted")
    off = np.array([math.inf])
    use_head = use_head and found.who_head is not None
    mass_grid = threshold_grid(found.listed_mass) if use_mass else off
    head_grid = threshold_grid(found.who_listed) if use_head else off
    signals_off = replace(base, theta_d=math.inf, theta_who=math.inf)
    fixed = decide_signals(found, signals_off, tables).label == "dangerous"
    head_values = found.who_listed if use_head else np.zeros(len(labels))

    def hits(members):
        by_head = head_values[members][None, :] >= head_grid[:, None]
        return np.array([((fixed[members] | (found.listed_mass[members] >= theta_d))[None, :]
                          | by_head).sum(axis=1) for theta_d in mass_grid])

    recalled, alarms = hits(listed), hits(outside)
    reaching = recalled >= photos_needed(target, int(listed.sum()))
    if at_most is not None:
        reaching &= ((mass_grid[:, None] <= at_most.theta_d)
                     & (head_grid[None, :] <= at_most.theta_who))
    mass_at, head_at = np.nonzero(reaching)
    if not len(mass_at):
        return None
    best = np.lexsort((-mass_at, -(mass_at + head_at), alarms[mass_at, head_at]))[0]
    mass_at, head_at = mass_at[best], head_at[best]
    return SimpleNamespace(
        thresholds=replace(base, theta_d=float(mass_grid[mass_at]),
                           theta_who=float(head_grid[head_at])),
        who_recall=float(recalled[mass_at, head_at] / listed.sum()),
        false_alarm=(float(alarms[mass_at, head_at] / outside.sum()) if outside.any()
                     else math.nan))


def reached(fit, target):
    if fit is None:
        raise ValueError("no (theta_d, theta_who) pair gives the dangerous verdict to "
                         f"{target:.0%} of the WHO-listed calibration photos, so the thresholds "
                         "can't be fitted")
    return fit


def recall_lower_bound(dangerous, groups, n=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """The 5th percentile of the dangerous share over resamples of whole photographers."""
    dangerous = np.asarray(dangerous, dtype=bool)
    return float(np.percentile([dangerous[rows].mean()
                                for rows in grouped_resamples(groups, n, seed)], 5))


def bounded_fit(found, base, tables, calib_outputs, required=WHO_RECALL_TARGET):
    """(fit, summary) at the first target whose bootstrap lower bound reaches `required`."""
    # targets step by 0.005 up to 0.995, each within the point fit's pair
    labels = np.asarray(calib_outputs["labels"], dtype=int)
    cap = reached(danger_fit(found, base, tables, labels, required), required).thresholds
    listed = (tables.who != 0)[labels]
    groups = photographers(calib_outputs)[listed]
    steps = math.floor((0.995 - required) / 0.005 + 1e-9)
    tried = []
    for target in [round(required + 0.005 * step, 6) for step in range(steps + 1)]:
        fit = danger_fit(found, base, tables, labels, target, at_most=cap)
        if fit is None:
            break
        bound = recall_lower_bound(
            decide_signals(found, fit.thresholds, tables).label[listed] == "dangerous",
            groups)
        tried.append((target, bound))
        if bound >= required:
            return fit, {"required": required, "percentile": 5, "resamples": BOOTSTRAP_RESAMPLES,
                         "seed": BOOTSTRAP_SEED, "listed_photos": int(listed.sum()),
                         "photographers": len(np.unique(groups)), "target": target,
                         "lower_bound": bound, "point_recall": fit.who_recall,
                         "false_alarm": fit.false_alarm,
                         "point_thresholds": {"theta_d": cap.theta_d,
                                              "theta_who": cap.theta_who},
                         "tried": [{"target": t, "lower_bound": b} for t, b in tried]}
    raise ValueError(f"no WHO recall target from {required} to 0.995 gives a photographer-"
                     f"bootstrap lower bound of {required} with theta_d <= {cap.theta_d} and "
                     f"theta_who <= {cap.theta_who} (the point fit's); lower bound per target "
                     "tried: " + ", ".join(f"{t:.3f}: {b:.4f}" for t, b in tried))


def joint_fit(found, base, tables, calib_outputs, target, recall_bound):
    """(fit, None) at the target for "point"; (fit, summary) at its lower bound for "lower"."""
    if recall_bound not in ("point", "lower"):
        raise ValueError(f"recall_bound must be 'point' or 'lower', got {recall_bound!r}")
    if recall_bound == "point":
        return reached(danger_fit(found, base, tables, calib_outputs["labels"], target),
                       target), None
    return bounded_fit(found, base, tables, calib_outputs, target)


def fit_thresholds(calib_outputs, prototypes, tables=None, who_recall_target=WHO_RECALL_TARGET,
                   scorer="head", scale=None, recall_bound="lower"):
    """Thresholds from known calibration photos; theta_d and theta_who are fitted jointly."""
    tables = repo_tables() if tables is None else tables
    found, base = calibrated(calib_outputs, prototypes, tables, scorer, scale)
    return joint_fit(found, base, tables, calib_outputs, who_recall_target,
                     recall_bound)[0].thresholds


def fit_snake_threshold(box_conf_calibration, keep=0.99):
    """The box confidence keeping `keep` of the calibration photos (all snakes); nan: no box."""
    conf = np.asarray(box_conf_calibration, dtype=float)
    detected = conf[np.isfinite(conf)]
    if not len(detected):
        raise ValueError("no calibration photo has a box, so tau_snake can't be fitted")
    return kth_largest(detected, photos_needed(keep, len(conf)), default=float(detected.min()))


def load_outputs(path):
    """A saved *_outputs.npz as a dict of arrays."""
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def check_splits(metadata, outputs):
    """Raises unless calibration, val and unknown photos come from train, val and unknown_pool."""
    split_of = metadata.set_index("path")["split"]
    for name, expected in (("calibration", "train"), ("val", "val"),
                           ("unknown", "unknown_pool")):
        paths = outputs[name]["paths"]
        splits = split_of.reindex(paths).to_numpy()
        assert_no_test_paths(pd.DataFrame({"path": paths, "split": splits}))
        if (splits != expected).any():
            raise ValueError(f"{name} photos outside metadata's {expected} split, e.g. "
                             f"{paths[splits != expected][0]}")


def check_finite(outputs, prototype_set, prototypes):
    """Refuses any non-finite saved value, so a broken run fails loudly, not quietly."""
    arrays = {**outputs, "prototypes": {prototype_set: prototypes}}
    problems = [f"{name} {key}: {bad} rows" for name, frame in arrays.items()
                for key, values in frame.items() if values.dtype.kind == "f"
                for bad in [int((~np.isfinite(values.reshape(len(values), -1))).any(axis=1).sum())]
                if bad]
    if problems:
        raise ValueError(f"non-finite values in the saved outputs: {', '.join(problems)}")


def photographers(outputs):
    return photographer_groups(pd.DataFrame({"observer_id": outputs["observer_id"],
                                             "path": outputs["paths"]}))


def box_conf_values(box_conf, paths):
    return np.array([box_conf.get(path, math.nan) for path in paths], dtype=float)


def poor_flags(quality, paths):
    """Per path: flagged dark or blurry in the quality table (a missing path is not flagged)."""
    if quality is None:
        return None
    flags = quality.set_index("path")[["dark", "blurry"]].astype(bool).any(axis=1)
    return flags.reindex(paths, fill_value=False).to_numpy(dtype=bool)


def decision_report(run_dir, data_root, quality=None, box_conf=None, prototype_set="plain",
                    medical=None, scorer="head", recall_bound="lower"):
    """Fits thresholds on a run's calibration outputs, scores val and unknown, writes the report."""
    # quality: dark/blurry flags by path; box_conf: {path: front-end box confidence}, which
    # turns on the no-snake rule. Writes decision_metrics.json into run_dir.
    run_dir = Path(run_dir)
    classes = pd.read_csv(Path(data_root, "classes.csv"))
    metadata = pd.read_csv(Path(data_root, "metadata.csv"), usecols=["path", "split"])
    scale = None
    if scorer == "prototype":
        scale = float(yaml.safe_load((run_dir / CONFIG_FILE).read_text())["arcface_s"])
    scoring = {"scorer": scorer, "scale": scale}
    tables = danger_tables(classes, pd.read_csv(MEDICAL_CSV, keep_default_na=False)
                           if medical is None else medical)
    outputs = {name: load_outputs(run_dir / file) for name, file in
               (("calibration", CALIBRATION_OUTPUTS), ("val", VAL_OUTPUTS),
                ("unknown", UNKNOWN_OUTPUTS))}
    check_splits(metadata, outputs)
    with np.load(run_dir / PROTOTYPES_FILE) as saved:
        prototypes = saved[prototype_set]
    check_finite(outputs, prototype_set, prototypes)
    calib_outputs, val_outputs, unknown_outputs = (outputs["calibration"], outputs["val"],
                                                   outputs["unknown"])
    calib_labels, labels = calib_outputs["labels"], val_outputs["labels"]
    found, base = calibrated(calib_outputs, prototypes, tables, **scoring)
    fit, bounded = joint_fit(found, base, tables, calib_outputs, WHO_RECALL_TARGET, recall_bound)
    target = WHO_RECALL_TARGET if bounded is None else bounded["target"]
    single_signals = {"species_mass_only": (True, False), "who_head_only": (False, True)}
    fits = {**{name: danger_fit(found, base, tables, calib_labels, target, use_mass=mass,
                                use_head=head)
               for name, (mass, head) in single_signals.items()}, "joint": fit}
    thresholds = fit.thresholds
    tau_snake = (None if box_conf is None
                 else fit_snake_threshold(box_conf_values(box_conf, calib_outputs["paths"])))

    def snake_in(paths):
        return None if tau_snake is None else box_conf_values(box_conf, paths) >= tau_snake

    def decided_on(frame, flagged=True, at=thresholds):
        return decide(frame, at, prototypes, tables, snake_in(frame["paths"]) if flagged else None,
                      poor_flags(quality, frame["paths"]) if flagged else None, **scoring)

    calib, val, unknown = (decided_on(calib_outputs, flagged=False), decided_on(val_outputs),
                           decided_on(unknown_outputs))
    if bounded is not None:
        point = replace(thresholds, **bounded["point_thresholds"])
        bound_fit = {**bounded, "against_point": {
            "calibration": against_point(calib_labels, calib.label,
                                         decided_on(calib_outputs, False, point).label, tables),
            "validation": against_point(labels, val.label,
                                        decided_on(val_outputs, at=point).label, tables)}}
    scored = {name: outputs[name]["paths"] for name in ("val", "unknown")}
    report = {
        "run": run_dir.name,
        "prototype_set": prototype_set,
        "scorer": scorer,
        **({"prototype_scale": scale,
            "scorer_note": "species probabilities, top 3, msp and energy read "
                           "softmax(prototype_scale x cosine to each prototype), not the species "
                           "head; a class with no prototype scores cosine 0"}
           if scorer == "prototype" else {}),
        "recall_bound": recall_bound,
        **({"recall_bound_fit": bound_fit} if bounded is not None else {}),
        "photos": {name: len(frame["labels"]) for name, frame in outputs.items()},
        "thresholds": asdict(thresholds),
        "fit": fit_summary(calib_labels, calib, thresholds, tables),
        "danger_signals": danger_signals_report(fits, val_outputs, prototypes, tables, scoring),
        "calibration": {"before": calibration(softmax(val.signals.raw_logits), labels),
                        "after": calibration(val.signals.probs, labels)},
        "o4_point": o4_point(val, labels, thresholds),
        "verdicts": {"val": verdict_counts(val), "unknown": verdict_counts(unknown)},
        "verdict_metrics": verdict_metrics(labels, val, tables),
        "intervals": intervals(val_outputs, val, tables),
        "fallback": fallback_report(val_outputs, val, thresholds, tables),
        "u1": unknown_pool_report(val, unknown_outputs, unknown, thresholds),
        "poor_image": "not available" if quality is None else {
            "flags": ["dark", "blurry"],
            "flagged": {name: int(poor_flags(quality, paths).sum())
                        for name, paths in scored.items()},
            "covered": {name: share(np.isin(paths, quality["path"]))
                        for name, paths in scored.items()}},
        "no_snake": {"val": int(val.reasons["no_snake"].sum()),
                     "unknown": int(unknown.reasons["no_snake"].sum()),
                     "source": "not available: no front-end result, so every photo was scored as "
                               "showing a snake"},
    }
    if tau_snake is not None:
        report["no_snake"].update(
            source="front end: box_conf >= tau_snake, fitted so 99% of calibration photos keep a "
                   "snake; every calibration photo is a real snake and there is no licensed "
                   "non-snake set, so this is the only fit we can defend and no false-snake rate "
                   "is measured; a photo with no box counts as no snake",
            tau_snake=tau_snake, keep_target=0.99,
            calibration_kept=share(snake_in(calib_outputs["paths"])))
    return write_json(run_dir / DECISION_FILE, report)


def verdict_counts(decided, members=None):
    members = np.ones(len(decided.label), dtype=bool) if members is None else members
    labels = decided.label[members]
    return {"total": int(labels.size),
            **{label: int((labels == label).sum()) for label in VERDICTS},
            "reasons": {name: int(flags[members].sum())
                        for name, flags in decided.reasons.items()}}


def o4_point(decided, labels, thresholds):
    """The share answered (confident and familiar) and the species error among them."""
    answered = ((decided.top1_prob >= thresholds.theta_conf)
                & (decided.signals.best_cosine >= thresholds.tau_unknown))
    return {"answered_share": share(answered),
            "error": share(decided.top1[answered] != labels[answered]),
            "verdict_answered_share": share(decided.label != "unknown")}


def fit_summary(labels, decided, thresholds, tables):
    present = set(np.unique(labels).tolist())
    caught = recall_and_alarm(labels, decided.label, tables)
    return {"photos": len(labels), "classes": len(present),
            "not_in_calibration": [str(name) for class_id, name in enumerate(tables.species)
                                   if class_id not in present],
            "who_listed_photos": int((tables.who[labels] != 0).sum()),
            "who_dangerous_recall": caught["who_recall"], "false_alarm": caught["false_alarm"],
            "tau_pass_share": share(decided.signals.best_cosine >= thresholds.tau_unknown),
            **{f"o4_{key}": value for key, value in o4_point(decided, labels, thresholds).items()}}


def against_point(labels, lower, point, tables):
    """How the lower-bound verdicts differ from the point fit's."""
    lower, point = lower == "dangerous", point == "dangerous"
    extra, not_listed = lower & ~point, tables.who[np.asarray(labels, dtype=int)] == 0
    return {"point_dangerous_not_lower": int((point & ~lower).sum()),
            "extra_dangerous": int(extra.sum()),
            "extra_dangerous_not_who_listed": int((extra & not_listed).sum())}


def recall_and_alarm(labels, verdicts, tables):
    listed, outside = listed_and_outside(labels, tables)
    return {"who_recall": share(verdicts[listed] == "dangerous"),
            "false_alarm": share(verdicts[outside] == "dangerous")}


def danger_signals_report(fits, val_outputs, prototypes, tables, scoring):
    """Each WHO signal alone and both jointly: thresholds, recall and false alarms."""
    result = dict.fromkeys(fits)
    for name, fit in fits.items():
        if fit is not None:
            verdicts = decide(val_outputs, fit.thresholds, prototypes, tables,
                              **scoring).label
            result[name] = {"theta_d": fit.thresholds.theta_d,
                            "theta_who": fit.thresholds.theta_who,
                            "calibration": {"who_recall": fit.who_recall,
                                            "false_alarm": fit.false_alarm},
                            "validation": recall_and_alarm(val_outputs["labels"], verdicts, tables)}
    return result


def verdict_metrics(labels, decided, tables):
    """Validation verdict rates; false_alarm leaves out WHO-listed, viper and elapid photos."""
    # false_alarm_outside_table leaves out every class the danger table marks dangerous
    probs, verdicts = decided.signals.probs, decided.label
    listed = tables.who != 0
    by_table, by_family, by_list_or_family = (
        safety_report(labels, probs, tables.who, flags, verdicts)
        for flags in (tables.dangerous, tables.venomous_class,
                      listed | tables.venomous_class))
    o3_keys = ("dangerous_as_not_listed_all", "dangerous_as_not_listed_answered",
               "coverage_listed", "top1_error_raw")
    return {"who_dangerous_recall": share(verdicts[listed[labels]] == "dangerous"),
            "false_alarm": by_list_or_family["false_alarm"],
            "false_alarm_outside_table": by_table["false_alarm"],
            "venomous_recall": by_family["venomous_recall"],
            "venomous_as_not_listed": by_family["venomous_as_not_listed"],
            "table_dangerous_recall": by_table["venomous_recall"],
            "who_recall_raw": by_table["who_recall_raw"], "abstain_rate": by_table["abstain_rate"],
            "o3": {key: by_table[key] for key in o3_keys}}


def intervals(val_outputs, decided, tables):
    """Photographer-bootstrap 95% intervals of the headline validation shares."""
    labels, verdicts = val_outputs["labels"], decided.label
    listed, venomous = (tables.who != 0)[labels], tables.venomous_class[labels]
    dangerous, not_listed = verdicts == "dangerous", verdicts == "not_listed"
    shares = {"who_dangerous_recall": (listed, dangerous),
              "false_alarm": (~(listed | venomous), dangerous),
              "venomous_recall": (venomous, dangerous),
              "dangerous_as_not_listed_all": (listed, not_listed)}
    groups = photographers(val_outputs)
    return {name: list(grouped_bootstrap(lambda rows, among=among, hit=hit:
                                         share(hit[rows][among[rows]]), groups))
            for name, (among, hit) in shares.items()}


def fallback_report(val_outputs, decided, thresholds, tables):
    """How often a "we don't know" photo can show a genus or family instead, and how often right."""
    found = decided.signals
    open_unknown = (decided.label == "unknown") & ~decided.reasons["no_snake"]
    result = {"genus": None, "family": None}
    for level, head, truth in (("genus", found.genus_head, val_outputs["genus_id"]),
                               ("family", found.family_head, val_outputs["family_id"])):
        if head is not None:
            shown = open_unknown & (head.max(axis=1) >= thresholds.theta_genus)
            result[level] = {"unknown_photos": int(open_unknown.sum()), "shown": int(shown.sum()),
                             "shown_share": share(shown[open_unknown]),
                             "correct": share(head.argmax(axis=1)[shown] == truth[shown])}
    result["taxonomy"] = taxonomy_report(found.probs, found.genus_head, found.family_head,
                                         tables.genus_of_class, tables.family_of_class,
                                         val_outputs["genus_id"], val_outputs["family_id"])
    return result


def unknown_pool_report(val, unknown_outputs, unknown, thresholds):
    """The unknown pool by known and unknown genus, against validation, and per listed species."""
    def scores(found):
        return {"msp": -found.probs.max(axis=1),
                "energy": -logsumexp(found.raw_logits, axis=1),
                "prototype_distance": 1 - found.best_cosine,
                "cosine_margin": found.second_cosine - found.best_cosine}

    known_scores, unknown_scores = scores(val.signals), scores(unknown.signals)
    genus_known = unknown_outputs["genus_id"] != NO_ID
    species = unknown_outputs["species"]
    flagged = unknown.signals.best_cosine < thresholds.tau_unknown
    groups = {}
    for name, members in (("all", np.ones_like(genus_known)), ("known_genus", genus_known),
                          ("unknown_genus", ~genus_known)):
        groups[name] = {"photos": int(members.sum()),
                        "species": int(np.unique(species[members]).size)}
        if members.any():
            groups[name].update({
                "open_set": {score: open_set(known_scores[score], unknown_scores[score][members])
                             for score in known_scores},
                "flagged_at_tau": share(flagged[members]),
                "verdicts": verdict_counts(unknown, members)})
    who_listed = {}
    for name in np.unique(species[unknown_outputs["who_risk_id"] != 0]):
        members = species == name
        who_listed[str(name)] = {"photos": int(members.sum()),
                                 "who_risk_id": int(unknown_outputs["who_risk_id"][members][0]),
                                 "genus_known": bool(genus_known[members].all()),
                                 "flagged_at_tau": share(flagged[members]),
                                 "verdicts": verdict_counts(unknown, members)}
    return {"scores": "higher means more likely unknown: msp = -max softmax (temperature-scaled), "
                      "energy = -logsumexp of the raw species logits, prototype_distance = 1 - "
                      "best prototype cosine, cosine_margin = -(best - second-best prototype "
                      "cosine)",
            "o4_target_flagged": 0.70, "groups": groups, "who_listed_species": who_listed,
            "non_snake_objects": "skipped: the unknown pool holds no non-snake photos"}


if __name__ == "__main__":
    build_medical_importance(pd.read_csv(CLASSES_CSV)).to_csv(MEDICAL_CSV, index=False)
