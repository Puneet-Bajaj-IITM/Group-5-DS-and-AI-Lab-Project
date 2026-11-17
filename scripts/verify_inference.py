"""Run the shipped model on a few credited validation photos and compare with the saved answers.

One command, CPU only:

    python scripts/verify_inference.py --models <folder>

<folder> holds A3's best.pt, prototypes.npz and decision_metrics.json and the M4 front end's
detector.pt and segmenter.pt (see results/m4/artifacts.md). Each photo in verify_photos.csv is
downloaded from its source url and kept only if its SHA-1 matches; no photo is written into the repo.
Each photo takes the app's path, the one notebook 24 takes per frame: the exposure fix, the M4
detector and segmenter, the no-snake gate of results/m4/gate/gate.json, A3's crop to the mask's box,
the model, and the decision rules with A3's thresholds. One JSON line is printed per photo. The script
exits 1 when, against verify_inference_expected.json, the gate's result, the top species or the verdict
differs, or a score or top-3 probability moves by more than FLOAT_TOLERANCE.
"""
import os

os.environ.update(CUDA_VISIBLE_DEVICES="", YOLO_VERBOSE="False", YOLO_OFFLINE="1",
                  NO_ALBUMENTATIONS_UPDATE="1")

import argparse
import hashlib
import http.client
import json
import sys
import tempfile
import urllib.request
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from serpentguard import decide, front_end
from serpentguard.data import AUGMENT, apply_mode, clean_mask, input_mode, load_rgb
from serpentguard.models import build_model

PHOTOS_CSV = REPO / "scripts/verify_photos.csv"
EXPECTED = REPO / "scripts/verify_inference_expected.json"
RUN_FOLDER = REPO / "results/m4/full/A3"
GATE_FILE = REPO / "results/m4/gate/gate.json"
FRONT_END_HASHES = REPO / "results/m4/gate/non_snake_test.json"  # weights_sha256 of the M4 front end
CLASSES = REPO / "data/classes.csv"
PHOTO_DIR = Path(tempfile.gettempdir()) / "serpentguard_verify_photos"
RUN_ID = "M4_FULL"  # the run_id of every full run; A3's files are listed under it
MODEL_FILES = ("best.pt", "prototypes.npz")
FRONT_END_FILES = {"detector.pt": "M4 detector", "segmenter.pt": "M4 segmenter"}
SEGMENTER_KIND = "unet"
FLOAT_TOLERANCE = 0.01  # scores move between machines; the Kaggle GPU differed by up to 0.009
USER_AGENT = "SerpentGuardStudentProject/1.0"
TIMEOUT_SECONDS = 60
FETCH_ATTEMPTS = 5
OUTPUT_KEYS = ("embedding", "species_logits", "who_logits", "genus_logits", "family_logits")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_files(models):
    """Refuses model files that are not the shipped run's and the gate's."""
    shipped = json.loads((RUN_FOLDER / "manifest.json").read_text())["not_in_repo"]
    for name in MODEL_FILES:
        if sha256(models / name) != shipped[f"{RUN_ID}/{name}"]["sha256"]:
            sys.exit(f"{models / name} is not A3's {name} (results/m4/full/A3/manifest.json)")
    decision = decide.DECISION_FILE
    if (models / decision).read_bytes() != (RUN_FOLDER / decision).read_bytes():
        sys.exit(f"{models / decision} differs from results/m4/full/A3/{decision}")
    recorded = json.loads(FRONT_END_HASHES.read_text())["weights_sha256"]
    for name, label in FRONT_END_FILES.items():
        if sha256(models / name) != recorded[label]:
            sys.exit(f"{models / name} is not the {label} (results/m4/gate/non_snake_test.json)")


def download(row):
    """The photo's bytes from its source url; a cut-off or failed download is tried again."""
    request = urllib.request.Request(row.url, headers={"User-Agent": USER_AGENT})
    for _ in range(FETCH_ATTEMPTS):
        try:
            body = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS).read()
        except (OSError, http.client.HTTPException):
            continue
        if hashlib.sha1(body).hexdigest() == row.sha1:
            return body
    sys.exit(f"{row.url}: no download in {FETCH_ATTEMPTS} attempts had the dataset's SHA-1 {row.sha1}")


def fetch(row, folder):
    """The photo's local copy, downloaded once and kept only if its SHA-1 is the dataset's."""
    target = folder / f"{row.photo_id}.jpg"
    if not target.exists():
        target.write_bytes(download(row))
    if hashlib.sha1(target.read_bytes()).hexdigest() != row.sha1:
        sys.exit(f"{target} is not the dataset's photo {row.photo_id}")
    return target


class App:
    """The shipped model behind the M4 front end, as the app runs one photo."""

    def __init__(self, models):
        checkpoint = torch.load(models / "best.pt", map_location="cpu", weights_only=True)
        self.cfg = checkpoint["config"]
        self.model = build_model(self.cfg, pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.mode = input_mode(self.cfg)
        decision = json.loads((models / decide.DECISION_FILE).read_text())
        self.thresholds = decide.Thresholds(**decision["thresholds"])
        self.scorer, self.scale = decision["scorer"], self.cfg.get("arcface_s")
        with np.load(models / "prototypes.npz") as saved:
            self.prototypes = saved[decision["prototype_set"]]
        self.tables = decide.repo_tables()
        self.who = pd.read_csv(CLASSES).set_index("species")["who_category"].astype(str)
        self.detector = YOLO(str(models / "detector.pt"))
        self.segmenter = front_end.Segmenter(SEGMENTER_KIND, models)
        self.gate = json.loads(GATE_FILE.read_text())

    def outputs(self, crop):
        image = AUGMENT["eval"](image=crop)["image"][None]
        with torch.inference_mode():
            out = self.model(image)
        return {key: out[key].float().numpy() for key in OUTPUT_KEYS}

    def answer(self, path):
        rgb = load_rgb(path)  # the exposure fix included
        part = front_end.gate_parts([rgb], self.detector, self.segmenter)[0]
        scores = pd.DataFrame({"box_conf": [part["box_conf"]], "mask_area": [part["mask_area"]]})
        snake = bool(front_end.gate_calls(scores, self.gate["rule"], self.gate["thresholds"])[0])
        gate = {"rule": self.gate["rule"], "snake": snake,
                "box_conf": None if np.isnan(part["box_conf"]) else part["box_conf"],
                "mask_area": part["mask_area"]}
        if not snake:
            verdict = decide.verdict(None, self.thresholds, self.prototypes, self.tables, snake_found=False)
            return {"gate": gate, "crop": None, **self.as_record(verdict)}
        mask = clean_mask(part["mask"])
        crop = apply_mode(rgb, mask, self.mode, None, self.cfg["input_size"])[0]
        verdict = decide.verdict(self.outputs(crop), self.thresholds, self.prototypes, self.tables,
                                 scorer=self.scorer, scale=self.scale)
        return {"gate": gate, "crop": "whole photo" if mask is None else "mask box", **self.as_record(verdict)}

    def as_record(self, verdict):
        record = asdict(verdict)
        top3 = [{"species": species, "probability": probability, "who_category": self.who[species]}
                for species, probability in record["top3"]]
        return {"top3": top3, "who_category": top3[0]["who_category"] if top3 else None,
                "verdict": record["label"], "reason": list(record["reason"]), "message": record["message"],
                "genus_fallback": record["genus_fallback"]}


def close(got, want):
    """Two scores agree within FLOAT_TOLERANCE; a missing score agrees only with a missing one."""
    if got is None or want is None:
        return got is want
    return abs(got - want) <= FLOAT_TOLERANCE


def top3_differences(got, want, where):
    """The same species with close probabilities; two may swap places when their probabilities are close."""
    expected = {entry["species"]: entry["probability"] for entry in want}
    if {entry["species"] for entry in got} != set(expected) or len(got) != len(want):
        return [f"{where}: species {[e['species'] for e in got]} != {[e['species'] for e in want]}"]
    found = [f"{where}: {entry['species']} {entry['probability']} != {expected[entry['species']]}"
             for entry in got if not close(entry["probability"], expected[entry["species"]])]
    found += [f"{where}[{rank}]: {g['species']} in place of {w['species']}, probabilities not close"
              for rank, (g, w) in enumerate(zip(got, want))
              if g["species"] != w["species"] and not close(expected[g["species"]], w["probability"])]
    return found


def differences(got, want):
    """Where one photo's answer differs: the gate's result, the top species and the verdict exactly,
    the gate's scores and the top 3 within FLOAT_TOLERANCE."""
    where = f"photo {want['photo_id']}"
    if got["photo_id"] != want["photo_id"]:
        return [f"{where}: answered photo {got['photo_id']}"]

    def top_species(answer):
        return answer["top3"][0]["species"] if answer["top3"] else None

    exact = {"gate result": (got["gate"]["snake"], want["gate"]["snake"]),
             "top species": (top_species(got), top_species(want)), "verdict": (got["verdict"], want["verdict"])}
    found = [f"{where} {name}: {g!r} != {w!r}" for name, (g, w) in exact.items() if g != w]
    found += [f"{where} {score}: {got['gate'][score]} != {want['gate'][score]}"
              for score in ("box_conf", "mask_area") if not close(got["gate"][score], want["gate"][score])]
    return found + top3_differences(got["top3"], want["top3"], f"{where} top 3")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, required=True, help="folder with the model files")
    parser.add_argument("--expected", type=Path, default=EXPECTED, help="the saved answers")
    parser.add_argument("--photos", type=Path, default=PHOTO_DIR, help="where downloaded photos go")
    parser.add_argument("--write-expected", action="store_true", help="save this run's answers instead")
    args = parser.parse_args()
    check_files(args.models)
    args.photos.mkdir(parents=True, exist_ok=True)
    app = App(args.models)
    photos = pd.read_csv(PHOTOS_CSV, dtype=str)
    answers = []
    for row in photos.itertuples():
        answer = {"photo_id": row.photo_id, "true_species": row.species, **app.answer(fetch(row, args.photos))}
        answers.append(json.loads(json.dumps(answer)))  # as the expected file stores it
        print(json.dumps(answer))
    if args.write_expected:
        args.expected.write_text(json.dumps(answers, indent=1) + "\n")
        print(f"wrote {len(answers)} answers to {args.expected}")
        return 0
    expected = json.loads(args.expected.read_text())
    found = ([f"{len(answers)} photos answered, {len(expected)} expected"] if len(answers) != len(expected) else
             [line for got, want in zip(answers, expected) for line in differences(got, want)])
    for line in found:
        print("MISMATCH", line)
    print(f"{len(answers)} photos: {'all match' if not found else f'{len(found)} value(s) differ'} "
          f"({args.expected.name}, tolerance {FLOAT_TOLERANCE})")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
