"""The end-to-end pipeline (P1): preprocess -> front end -> SnakeID -> decision -> response, on a photo or a clip.
Every node logs a short summary and its milliseconds; passages are quoted with their page, never generated."""
import json
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from serpentguard.data import FRONT_END_MODES, IMG_SIZE, apply_mode, fix_exposure, load_rgb, prepare
from serpentguard.decision import (DECISION_FILE, HEAD, NO_MAJORITY, NOT_WHO, PLAIN, PROTOTYPE, REASON_TEXT,
                                   UNKNOWN_MESSAGE, DangerTables, Thresholds, Verdict, decide, expected_verdict,
                                   repo_tables, verdict)
from serpentguard.frontend import D1, WINNERS_FILE, load_detector, predict_boxes
from serpentguard.metrics import DANGEROUS, NOT_LISTED, UNKNOWN, VERDICTS
from serpentguard.respond import FIRST_AID_IDS, TOP_K, LookupRetriever, species_card
from serpentguard.run_data import BOX_CROP
from serpentguard.run_files import PROTOTYPES_FILE, VAL_OUTPUTS
from serpentguard.run_outputs import PREDICTION_KEYS, load_checkpoint, softmax
from serpentguard.video import CLIP_FPS, TARGET_FPS, VOTE, normalised, schedule, video_metrics, vote_k_of_n

NODES = PREPROCESS, FRONT_END, SNAKEID, DECISION, RESPONSE = ("preprocess", "front_end", "snakeid", "decision",
                                                             "response")
PER_FRAME_NODES = NODES[:4]
CLIP_SCOPE = "clip"             # the "frame" of a clip's own decision and response entries
FULL = "full"
NO_DETECTOR = "no detector: every frame is treated as showing a snake"
LOOKUP_K = len(FIRST_AID_IDS) + 2   # every R0 rank: the note, the species card and every first-aid passage
MIN_FRAME_SECONDS = 1 / TARGET_FPS
CAP_SKIPPING = f"the {TARGET_FPS} frames-per-second cap"
NO_MAJORITY_VERDICT = Verdict(UNKNOWN, (), None, (NO_MAJORITY,), "", UNKNOWN_MESSAGE.format(REASON_TEXT[NO_MAJORITY]),
                              None)
MS_PER_S = 1000.0
MS_DECIMALS, PROB_DECIMALS = 3, 4
MISSED_DANGER, FALSE_ALARM = "missed_danger", "false_alarm"
FAILURE_KINDS = (MISSED_DANGER, FALSE_ALARM)
FAILURE = "failure"


@dataclass(frozen=True)
class PipelineModels:
    """What run_pipeline reads: F0 (the whole photo) or F1 (the D1 box; no snake under tau_snake)."""
    snakeid: Callable
    prototypes: np.ndarray
    front_end: str = "F0"
    input_size: int = IMG_SIZE
    tables: DangerTables | None = None
    detector: object = None
    tau_snake: float | None = None
    backend: str = "torch"
    scorer: str = HEAD
    prototype_scale: float | None = None

    def __post_init__(self):
        mode = FRONT_END_MODES[self.front_end]
        if mode not in (FULL, BOX_CROP):
            raise ValueError(f"P1 reads F0 or F1 input, not {self.front_end} ({mode})")
        if mode == BOX_CROP and self.detector is None:
            raise ValueError(f"{self.front_end} crops to the detector's box, so it needs a detector")


@dataclass(frozen=True)
class ClipFrame:
    """One processed clip frame: when its result shows, its own verdict and what the screen shows after smoothing."""
    index: int
    shown_at: float
    verdict: Verdict
    shown: Verdict


@dataclass(frozen=True)
class PipelineResult:
    """The screen (verdict, card, passages, typed-question answers), the node log and a clip's summary and frames."""
    verdict: Verdict
    card: object
    passages: tuple
    question_passages: tuple
    log: tuple
    total_ms: float
    snake_found: bool
    clip: dict | None = None
    frames: tuple = ()

    @property
    def label(self):
        return self.verdict.label

    @property
    def top3(self):
        return self.verdict.top3


def run_pipeline(image_or_frames, models, thresholds, kb, question=None, retriever=None, fps=CLIP_FPS):
    """A photo (path, PIL image or uint8 array) or a clip (a list of frames) through every node; `retriever` (R1 or
    R2) answers a typed question."""
    started, log = time.perf_counter(), []
    lookup = kb if isinstance(kb, LookupRetriever) else LookupRetriever(kb)
    clip, scope, frames = None, None, ()
    if isinstance(image_or_frames, (list, tuple)) or getattr(image_or_frames, "ndim", 0) == 4:
        shown, clip, snake, frames = _run_clip(image_or_frames, models, thresholds, log, fps)
        scope = CLIP_SCOPE
    else:
        shown, snake = _run_photo(image_or_frames, models, thresholds, log)
    with _logged(log, RESPONSE, scope) as summary:
        card = species_card(shown, cards=lookup.cards, dangerous=lookup.dangerous)
        passages = tuple(lookup.search(shown, k=LOOKUP_K))
        answers = tuple(retriever.search(question, k=TOP_K)) if question is not None else ()
        summary.update(retriever="R0 lookup", card=None if card is None else card.id,
                       passages=[p.id for p in passages], pages=[p.page for p in passages])
        if question is not None:
            summary.update(question_retriever=type(retriever).__name__, question_passages=[p.id for p in answers])
    total = (time.perf_counter() - started) * MS_PER_S
    return PipelineResult(shown, card, passages, answers, tuple(log), total, snake, clip, frames)


@contextmanager
def _logged(log, node, frame=None):
    summary, started = {}, time.perf_counter()
    yield summary
    entry = {"node": node, "ms": (time.perf_counter() - started) * MS_PER_S, "summary": summary}
    if frame is not None:
        entry["frame"] = frame
    log.append(entry)


def _run_photo(image, models, thresholds, log, frame=None):
    tables = models.tables or repo_tables()
    with _logged(log, PREPROCESS, frame) as summary:
        rgb, whole = _preprocess(image, models.input_size, summary)
    with _logged(log, FRONT_END, frame) as summary:
        snake, tensor = _front_end(rgb, whole, models, summary)
    with _logged(log, SNAKEID, frame) as summary:
        outputs = _snakeid(tensor, models, tables, summary)
    with _logged(log, DECISION, frame) as summary:
        shown = verdict(outputs, thresholds, models.prototypes, tables, snake_found=snake, scorer=models.scorer,
                        scale=models.prototype_scale)
        top1 = [shown.top3[0][0], round(shown.top3[0][1], PROB_DECIMALS)] if shown.top3 else None
        summary.update(verdict=shown.label, reason=list(shown.reason), top1=top1, message=shown.message)
    return shown, snake


def _preprocess(image, size, summary):
    """The exposure-fixed RGB, and the whole photo letterboxed and normalised: F0's input."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            summary["mode"] = opened.mode
        rgb = load_rgb(image)
    else:
        picture = image if isinstance(image, Image.Image) else Image.fromarray(image)
        summary["mode"] = picture.mode
        rgb = fix_exposure(np.asarray(picture.convert("RGB"), dtype=np.uint8))
    summary.update(size=[int(rgb.shape[1]), int(rgb.shape[0])], brightness=round(float(rgb.mean()), 1),
                   input=[size, size])
    return rgb, normalised(prepare(rgb, None, FULL, size)[0])


def _front_end(rgb, whole, models, summary):
    """(snake found, the model's input or None)."""
    mode = FRONT_END_MODES[models.front_end]
    summary["mode"] = f"{models.front_end} ({mode})"
    box = None
    if models.detector is None:
        summary["detector"], snake = NO_DETECTOR, True
    else:
        found = predict_boxes(models.detector, [rgb])[0]
        box, conf = found if found is not None else (None, None)
        snake = conf is not None and conf >= models.tau_snake
        summary.update(box=None if box is None else [round(value, 1) for value in box], box_conf=conf,
                       tau_snake=models.tau_snake, snake_found=snake)
    if not snake:
        return False, None
    if mode == FULL:
        summary["input"] = "whole photo"
        return True, whole
    summary["input"] = mode
    return True, normalised(apply_mode(rgb, None, mode, box, models.input_size)[0])


def _snakeid(tensor, models, tables, summary):
    summary["backend"] = models.backend
    if tensor is None:
        summary["skipped"] = "no snake found"
        return None
    outputs = models.snakeid(tensor)
    probs = softmax(np.atleast_2d(np.asarray(outputs["species_logits"], dtype=float)))[0]
    top = int(probs.argmax())
    summary.update(top1=str(tables.species[top]), top1_raw_prob=round(float(probs[top]), PROB_DECIMALS),
                   outputs=sorted(outputs))
    return outputs


def _run_clip(frames, models, thresholds, log, fps):
    """The per-frame nodes on the frames kept at TARGET_FPS, then 4-of-6 voting; no majority shows "we don't know"."""
    processed, shown_at = schedule(len(frames), fps, MIN_FRAME_SECONDS)
    verdicts, snakes = [], []
    for index in processed:
        shown, snake = _run_photo(frames[index], models, thresholds, log, frame=int(index))
        verdicts.append(shown)
        snakes.append(snake)
    with _logged(log, DECISION, CLIP_SCOPE) as summary:
        voted = vote_k_of_n([shown.label for shown in verdicts])
        labels = [UNKNOWN if label is None else label for label in voted]
        newest, screens = {}, []  # per frame: the newest frame Verdict with the voted label, or no majority
        for shown, label in zip(verdicts, voted):
            newest[shown.label] = shown
            screens.append(NO_MAJORITY_VERDICT if label is None else newest[label])
        first = next((index for index, label in enumerate(voted) if label is not None), None)
        final, end = screens[-1], len(frames) / fps
        stability = video_metrics(labels, None, shown_at, end)
        summary.update(method=VOTE, frames=len(frames), fps=fps, frame_seconds=MIN_FRAME_SECONDS,
                       skipping=CAP_SKIPPING, processed=len(processed), skipped_share=1 - len(processed) / len(frames),
                       effective_fps=len(processed) / max(end, float(shown_at[-1])),
                       per_frame=dict(Counter(shown.label for shown in verdicts)),
                       shown=dict(Counter(labels)), majority=voted[-1] is not None, final=final.label,
                       final_reason=list(final.reason), flips=stability["flips"],
                       time_to_stable=stability["time_to_stable"], final_stable=stability["final_stable"],
                       first_majority_at=None if first is None else float(shown_at[first]), screen=labels)
    clip_frames = tuple(ClipFrame(int(index), float(at), own, screen)
                        for index, at, own, screen in zip(processed, shown_at, verdicts, screens))
    return final, dict(summary), any(snakes), clip_frames


def torch_snakeid(model, device="cpu"):
    """`PipelineModels.snakeid` for a SnakeID module."""
    model = model.to(device).eval()

    @torch.inference_mode()
    def predict(tensor):
        outputs = model(tensor.to(device))
        return {name: outputs[name].float().cpu().numpy() for name in PREDICTION_KEYS if outputs.get(name) is not None}

    return predict


def load_models(run_dir, decision_file=None, frontend_dir=None, onnx_path=None, tables=None, device="cpu",
                scorer=HEAD):
    """(PipelineModels, Thresholds) for a run, its checked decision file and the D1 detector. onnx_path is kept for
    notebook 15 and not read: every P1 run used best.pt."""
    run_dir = Path(run_dir)
    report = json.loads(Path(decision_file or run_dir / DECISION_FILE).read_text())
    if report.get("scorer", HEAD) != scorer:
        raise ValueError(f"the decision file was fitted with the {report.get('scorer', HEAD)} scorer, not {scorer}")
    with np.load(run_dir / PROTOTYPES_FILE) as saved:
        prototypes = saved[report.get("prototype_set", PLAIN)]
    check_decision_file(run_dir, report, prototypes, tables)
    cfg, model, _ = load_checkpoint(run_dir)
    scale = report.get("prototype_scale")
    if scorer == PROTOTYPE and scale != cfg.arcface_s:
        raise ValueError(f"the decision file scores prototypes at scale {scale}, but the run has arcface_s "
                         f"{cfg.arcface_s}")
    detector = None
    if frontend_dir is not None:
        detector = load_detector(Path(frontend_dir), json.loads((Path(frontend_dir) / WINNERS_FILE).read_text())[D1])
    models = PipelineModels(torch_snakeid(model, device), prototypes, cfg.front_end, cfg.input_size, tables, detector,
                            report.get("no_snake", {}).get("tau_snake"), "torch", scorer, scale)
    return models, Thresholds(**report["thresholds"])


def check_decision_file(run_dir, report, prototypes, tables=None):
    """Raises unless decide() on the run's val_outputs.npz gives the file's verdict counts, within the photos it
    flagged as no snake or poor image (not saved per photo)."""
    with np.load(Path(run_dir) / VAL_OUTPUTS) as saved:
        outputs = {key: saved[key] for key in PREDICTION_KEYS if key in saved.files}
    labels = decide(outputs, Thresholds(**report["thresholds"]), prototypes, tables, scorer=report.get("scorer", HEAD),
                    scale=report.get("prototype_scale")).label
    found = {"total": int(labels.size), **{label: int((labels == label).sum()) for label in VERDICTS}}
    expected, poor = report["verdicts"]["val"], report.get("poor_image")
    flagged = int(report.get("no_snake", {}).get("val", 0))
    flagged += int(poor["flagged"]["val"]) if isinstance(poor, dict) else 0
    off = {key: abs(found[key] - int(expected[key])) for key in found}
    if off["total"] or max(off.values()) > flagged:
        raise ValueError(f"the decision file does not reproduce {Path(run_dir).name}'s validation verdicts: it says "
                         f"{expected}, the run's outputs give {found} (allowed difference {flagged})")
    return found


def result_record(result, with_text=False):
    """The result as JSON-ready values; passage text only with with_text, which committed files never use."""
    shown = result.verdict
    return {"verdict": shown.label, "message": shown.message, "note": shown.note, "reason": list(shown.reason),
            "top3": [[name, round(prob, PROB_DECIMALS)] for name, prob in shown.top3],
            "genus_fallback": None if shown.genus_fallback is None else asdict(shown.genus_fallback),
            "margin": shown.margin, "snake_found": result.snake_found,
            "card": None if result.card is None else _passage_record(result.card, with_text),
            "passages": [_passage_record(p, with_text) for p in result.passages],
            "question_passages": [_passage_record(p, with_text) for p in result.question_passages],
            "total_ms": round(result.total_ms, MS_DECIMALS), "clip": result.clip,
            "log": [{**entry, "ms": round(entry["ms"], MS_DECIMALS)} for entry in result.log]}


def _passage_record(passage, with_text):
    record = {"id": passage.id, "source": passage.source, "page": passage.page, "citation": passage.citation}
    return {**record, "text": passage.text} if with_text else record


def failure_kind(label, class_id, tables):
    """MISSED_DANGER: a danger-table snake not called dangerous; FALSE_ALARM: any other snake called dangerous."""
    if tables.dangerous[class_id] and label != DANGEROUS:
        return MISSED_DANGER
    if not tables.dangerous[class_id] and label == DANGEROUS:
        return FALSE_ALARM
    return None


def p1_record(photo, result, tables, with_text=False):
    """One P1 log line: the photo's manifest fields, the expected verdict, any failure, then the result."""
    class_id = int(photo["class_id"])
    record = result_record(result, with_text)
    return {"path": photo["path"], "species": photo["species"], "class_id": class_id, "license": photo["license"],
            "attribution": photo["attribution"], "expected": expected_verdict(class_id, tables),
            "who_listed": bool(tables.who[class_id] != NOT_WHO), "venomous_family": bool(tables.venomous_class[class_id]),
            "failure": failure_kind(record["verdict"], class_id, tables), **record}


def pick_examples(records, per_verdict=2, failures=2):
    """The first per_verdict non-failures of each verdict, then the worst failures (missed danger before false alarm,
    a viper or elapid called "not listed" first), one of each kind first; each copy gains "example"."""
    picked = []
    for label in VERDICTS:
        chosen = [record for record in records if record["verdict"] == label and record["failure"] is None]
        picked += [{**record, "example": label} for record in chosen[:per_verdict]]
    failing = sorted((record for record in records if record["failure"] is not None),
                     key=lambda record: (FAILURE_KINDS.index(record["failure"]), record["verdict"] != NOT_LISTED,
                                         not record.get("venomous_family", False)))
    firsts = [next(r for r in failing if r["failure"] == kind) for kind in FAILURE_KINDS
              if any(r["failure"] == kind for r in failing)]
    ordered = firsts + [record for record in failing if record not in firsts]
    return picked + [{**record, "example": FAILURE} for record in ordered[:failures]]
