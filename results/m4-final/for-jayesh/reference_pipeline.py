"""The app's whole path from one photo to a verdict, with ONNX Runtime, numpy and OpenCV only.

Usage: python reference_pipeline.py --models <folder from export_onnx.py> [--fp16] photo.jpg [photo.jpg ...]

This is the version to port to Dart: one function per step, every number read from pipeline.json and
tables.json. It gives the same answers as the repo's scripts/verify_inference.py (see check_parity.py).
Steps: exposure fix -> detector (box_conf) and U-Net (mask, mask_area) -> no-snake gate -> mask cleanup
-> crop to the mask's box -> classifier -> decision rules (DR0 with A3's thresholds).
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

DANGEROUS, UNKNOWN, NOT_LISTED = "dangerous", "unknown", "not_listed"
NOT_LISTED_MESSAGE = "not on the WHO list, still keep away"
NO_SNAKE_MESSAGE = "we don't know: no snake found"
WHO_CAT2, WHO_CAT1 = 1, 2  # tables.json "who" codes; 0 is not listed
UNDER_ONE_PERCENT = 0.01
NO_GENUS = -1


def py_round(value):
    """Python's round: halves go to the even neighbour (2.5 -> 2). Dart's round() sends them up."""
    return round(value)


def load_photo(path, exposure):
    """The photo as RGB uint8, exposure fixed. Decode it upright: the model never saw a rotated photo."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return fix_exposure(rgb, exposure)


def fix_exposure(rgb, exposure):
    """CLAHE on the LAB lightness of a photo whose mean brightness (all pixels, all channels) is out of range."""
    if exposure["when_mean_below"] <= rgb.mean() <= exposure["when_mean_above"]:
        return rgb
    lightness, a, b = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB))
    clahe = cv2.createCLAHE(clipLimit=exposure["clahe_clip_limit"], tileGridSize=tuple(exposure["clahe_tile_grid"]))
    return cv2.cvtColor(cv2.merge([clahe.apply(lightness), a, b]), cv2.COLOR_LAB2RGB)


def placed_size(height, width, size):
    """Width and height of the photo inside the size x size letterbox."""
    scale = size / max(height, width)
    return max(1, py_round(width * scale)), max(1, py_round(height * scale)), scale


def letterbox(rgb, size, pad_rgb):
    """The U-Net's and classifier's input: the photo resized to fit, centred on a grey square."""
    height, width = rgb.shape[:2]
    new_w, new_h, scale = placed_size(height, width, size)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas = np.full((size, size, 3), pad_rgb, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def normalise(canvas, settings):
    """uint8 HWC RGB -> float32 NCHW with ImageNet mean and std."""
    scaled = canvas.astype(np.float32) / np.float32(settings["scale"])
    mean, std = np.array(settings["mean"], np.float32), np.array(settings["std"], np.float32)
    return ((scaled - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)


def detector_input(rgb, size, pad_value):
    """YOLO's letterbox: resize with INTER_LINEAR, centre with 114 padding, RGB / 255."""
    height, width = rgb.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = max(py_round(width * ratio), 1), max(py_round(height * ratio), 1)
    if (width, height) != (new_w, new_h):
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom, left, right = (py_round(dh - 0.1), py_round(dh + 0.1), py_round(dw - 0.1), py_round(dw + 0.1))
    canvas = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    return (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def box_conf(detector_output, snake_class, floor):
    """The highest snake score, or nan when no anchor scores above the floor."""
    best = float(detector_output[0, 4 + snake_class].max())
    return best if best > floor else math.nan


def unletterbox(probability, height, width, size):
    """The U-Net's output for the placed window, resized back to the photo."""
    new_w, new_h, _ = placed_size(height, width, size)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    window = probability[top:top + new_h, left:left + new_w].astype(np.float32)
    return cv2.resize(window, (width, height), interpolation=cv2.INTER_LINEAR)


def clean_mask(mask, cleanup):
    """Closed, small parts dropped; None when the largest part is under 1.5% of the photo."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    parts, labels, part_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=cleanup["connectivity"])
    areas = part_stats[1:, cv2.CC_STAT_AREA]
    if parts <= 1 or areas.max() < cleanup["min_largest_share"] * mask.size:
        return None
    floor = max(cleanup["keep_part_share_of_largest"] * areas.max(), cleanup["keep_part_share_of_photo"] * mask.size)
    keep = 1 + np.flatnonzero(areas >= floor)
    return np.isin(labels, keep).astype(np.uint8)


def crop_window(mask, height, width, padding):
    """(top, bottom, left, right) of the mask's box plus padding, or None for the whole photo."""
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    pad_x, pad_y = int((x2 - x1) * padding), int((y2 - y1) * padding)
    return max(0, y1 - pad_y), min(height, y2 + pad_y + 1), max(0, x1 - pad_x), min(width, x2 + pad_x + 1)


def softmax(values):
    shifted = np.exp(values - values.max())
    return shifted / shifted.sum()


def unit(vector):
    return vector / np.linalg.norm(vector)


def best_cosine(embedding, prototypes):
    """Cosine to the nearest species prototype; an all-zero prototype never wins."""
    norms = np.linalg.norm(prototypes, axis=1)
    active = norms > 0
    return float(((prototypes[active] / norms[active, None]) @ unit(embedding)).max())


def guarded_genus(probs, genus_head, top1, tables):
    """The likelier guarded genus of the top species' genus and the genus head's pick, and its probability."""
    genus_of_class = tables["genus_of_class"]
    genus_mass = np.bincount(genus_of_class, weights=probs, minlength=len(tables["genus_names"]))
    candidates = [int(genus_of_class[top1]), int(genus_head.argmax())]
    best, best_prob = NO_GENUS, -math.inf
    for genus in candidates:
        prob = max(genus_mass[genus], genus_head[genus])
        if tables["guarded_genus"][genus] and prob > best_prob:
            best, best_prob = genus, prob
    return best, best_prob


def decide(outputs, thresholds, tables):
    """DR0 for one photo with a snake: the label, the reasons that fired, and what the screen shows."""
    probs = softmax(outputs["species_logits"].astype(np.float64) / thresholds["temperature"])
    who_head = softmax(outputs["who_logits"].astype(np.float64) / thresholds["temperature_who"])
    genus_head = softmax(outputs["genus_logits"].astype(np.float64))
    family_head = softmax(outputs["family_logits"].astype(np.float64))
    cosine = best_cosine(outputs["embedding"].astype(np.float64), tables["prototypes"])

    who, dangerous = tables["who"], tables["dangerous"]
    top1 = int(probs.argmax())
    ranked = np.sort(probs)[::-1]
    listed_mass = probs[who != 0].sum()
    fired = {"who_listed_mass": listed_mass >= thresholds["theta_d"],
             "who_head": 1 - who_head[0] >= thresholds["theta_who"],
             "top_species_dangerous": bool(((probs == ranked[0]) & dangerous).any()),
             "venomous_family_mass": family_head[tables["venomous_family"]].sum() >= thresholds["theta_v"]}
    evidence = any(fired.values())
    unclear = ranked[0] < thresholds["theta_conf"]
    unfamiliar = cosine < thresholds["tau_unknown"]
    genus, genus_prob = guarded_genus(probs, genus_head, top1, tables)
    guarded = not evidence and not (unclear or unfamiliar) and genus != NO_GENUS
    by_genus = guarded and genus_prob >= thresholds["theta_genus"]
    genus_unclear = guarded and not by_genus
    unknown = (not evidence and (unclear or unfamiliar)) or genus_unclear
    label = DANGEROUS if evidence or by_genus else UNKNOWN if unknown else NOT_LISTED
    reasons = {**{name: evidence and flag for name, flag in fired.items()}, "dangerous_genus": by_genus,
               "no_snake": False, "unclear_between_species": (not evidence and unclear) or genus_unclear,
               "unfamiliar": not evidence and unfamiliar, "poor_image": False}
    shown = [int(k) for k in np.argsort(-probs, kind="stable")[:3]]
    found = dict(probs=probs, who_head=who_head, genus_head=genus_head, listed_mass=listed_mass, genus=genus,
                 fired=fired, shown=shown, margin=float(ranked[0] - ranked[1]))
    return label, [name for name in tables["reasons"] if reasons[name]], found


def danger_note(found, tables):
    """What the dangerous screen says, naming the evidence the screen shows."""
    probs, fired, species, who = found["probs"], found["fired"], tables["species"], tables["who"]
    if fired["top_species_dangerous"]:
        return tables["note"][int(np.flatnonzero((probs == probs.max()) & tables["dangerous"])[0])]
    if fired["who_listed_mass"]:
        named = [k for k in found["shown"] if who[k] != 0]
        if named:
            category = 1 if who[named[0]] == WHO_CAT1 else 2
            return f"could be a WHO-listed species: {species[named[0]]} (Category {category})"
        mass = found["listed_mass"]
        share = "under 1%" if mass < UNDER_ONE_PERCENT else f"{mass:.0%}"
        return f"could be a WHO-listed species: {share} combined probability"
    if fired["who_head"]:
        category = 1 if found["who_head"][WHO_CAT1] >= found["who_head"][WHO_CAT2] else 2
        return f"the photo looks like a WHO Category {category} snake"
    if fired["venomous_family_mass"]:
        return tables["venomous_family_note"]
    name = tables["genus_names"][found["genus"]]
    if tables["listed_genus"][found["genus"]]:
        return f"genus {name} has species on the WHO list"
    return f"genus {name} has species named as medically important in the WHO guideline"


def verdict(outputs, thresholds, tables):
    """The record the app shows: verdict, reason, message, top 3 and the genus fallback."""
    label, reasons, found = decide(outputs, thresholds, tables)
    message, fallback = NOT_LISTED_MESSAGE, None
    if label == DANGEROUS:
        message = f"dangerous: {danger_note(found, tables)}"
    elif label == UNKNOWN:
        message = "we don't know: " + ", ".join(tables["reason_text"][name] for name in reasons)
        head = found["genus_head"]
        if head.max() >= thresholds["theta_genus"]:
            genus = int(head.argmax())
            fallback = {"genus": tables["genus_names"][genus], "probability": float(head[genus]),
                        "who_listed": bool(tables["listed_genus"][genus])}
    top3 = [{"species": tables["species"][k], "probability": float(found["probs"][k]),
             "who_category": tables["who_category"][k]} for k in found["shown"]]
    return {"top3": top3, "who_category": top3[0]["who_category"], "verdict": label, "reason": reasons,
            "message": message, "genus_fallback": fallback, "margin": found["margin"]}


def no_snake_verdict():
    return {"top3": [], "who_category": None, "verdict": UNKNOWN, "reason": ["no_snake"],
            "message": NO_SNAKE_MESSAGE, "genus_fallback": None, "margin": None}


def read_tables(path):
    tables = json.loads(Path(path).read_text())
    for name in ("genus_of_class", "who"):
        tables[name] = np.asarray(tables[name], dtype=int)
    for name in ("dangerous", "guarded_genus", "listed_genus", "venomous_family"):
        tables[name] = np.asarray(tables[name], dtype=bool)
    tables["prototypes"] = np.asarray(tables["prototypes"], dtype=np.float64)
    return tables


class Pipeline:
    """The three ONNX models and the two settings files, run on one photo at a time."""

    def __init__(self, models, fp16=False, threads=1):
        models = Path(models)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        options.log_severity_level = 3

        def session(name):
            return ort.InferenceSession(str(models / name), options, providers=["CPUExecutionProvider"])

        suffix = "_fp16" if fp16 else ""
        self.classifier = session(f"classifier{suffix}.onnx")
        self.segmenter = session(f"segmenter{suffix}.onnx")
        self.detector = session("detector.onnx")
        self.settings = json.loads((models / "pipeline.json").read_text())
        self.tables = read_tables(models / "tables.json")
        self.output_names = [o.name for o in self.classifier.get_outputs()]

    def gate(self, rgb):
        s = self.settings
        size, height, width = s["input_size"], *rgb.shape[:2]
        detected = self.detector.run(None, {"images": detector_input(rgb, size, s["detector"]["pad_value"])})[0]
        conf = box_conf(detected, s["detector"]["snake_class"], s["front_end_box_floor"])
        canvas = letterbox(rgb, size, s["letterbox"]["pad_rgb"])
        probability = self.segmenter.run(None, {"images": normalise(canvas, s["normalise"])})[0][0, 0]
        mask = (unletterbox(probability, height, width, size) > s["segmenter"]["threshold"]).astype(np.uint8)
        area = float(mask.mean())
        limits = s["gate"]["thresholds"]
        snake = math.isfinite(conf) and conf >= limits["box_conf"] and area > 0 and area >= limits["mask_area"]
        return {"rule": s["gate"]["rule"], "snake": snake, "box_conf": None if math.isnan(conf) else conf,
                "mask_area": area}, mask

    def classify(self, crop):
        s = self.settings
        image = normalise(letterbox(crop, s["input_size"], s["letterbox"]["pad_rgb"]), s["normalise"])
        return {name: value[0] for name, value in zip(self.output_names, self.classifier.run(None, {"images": image}))}

    def answer_rgb(self, rgb):
        """One photo, already loaded and exposure-fixed, to the app's record."""
        gate, mask = self.gate(rgb)
        if not gate["snake"]:
            return {"gate": gate, "crop": None, **no_snake_verdict()}
        height, width = rgb.shape[:2]
        window = crop_window(clean_mask(mask, self.settings["mask_cleanup"]), height, width,
                             self.settings["crop"]["padding"])
        crop = rgb if window is None else rgb[window[0]:window[1], window[2]:window[3]]
        outputs = self.classify(crop)
        return {"gate": gate, "crop": "whole photo" if window is None else "mask box", "crop_window": window,
                **verdict(outputs, self.settings["decision"]["thresholds"], self.tables)}

    def answer(self, path):
        return self.answer_rgb(load_photo(path, self.settings["exposure_fix"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true", help="the fp16 classifier and segmenter")
    parser.add_argument("photos", nargs="+", type=Path)
    args = parser.parse_args()
    pipeline = Pipeline(args.models, fp16=args.fp16)
    for path in args.photos:
        print(json.dumps({"photo": path.name, **pipeline.answer(path)}))


if __name__ == "__main__":
    main()
