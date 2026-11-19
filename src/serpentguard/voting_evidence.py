"""Validate the evidence shared for one uncertain photo or live encounter."""

import re
from io import BytesIO

from PIL import Image, ImageOps


MAX_EVIDENCE_BYTES = 1024 * 1024
MAX_IMAGE_SIDE = 2048
PREDICTION_FIELDS = (
    "model_confidence", "top_predictions", "genus_prediction", "family_prediction",
)
CAPTURE_FIELDS = (
    "source", "snake_found", "reason", "coarse_location", "evidence",
    "top_predictions", "genus_prediction", "family_prediction", "training_consent",
)


def is_confidence(value):
    return type(value) in (int, float) and 0 <= value <= 1


def validate_prediction(prediction):
    if (
        not isinstance(prediction, dict) or set(prediction) != {"label", "confidence"}
        or not isinstance(prediction["label"], str) or not prediction["label"].strip()
        or len(prediction["label"]) > 100 or not is_confidence(prediction["confidence"])
    ):
        raise ValueError("Each prediction needs a taxon label and confidence between 0 and 1")


def validate_capture(data):
    source = data.get("source")
    if source not in ("photo", "live_video"):
        raise ValueError("source must be photo or live_video")
    if data.get("snake_found") is not True:
        raise ValueError("Community review requires a detected snake")
    if "training_consent" in data and type(data["training_consent"]) is not bool:
        raise ValueError("training_consent must be true or false")
    reasons = ("unclear_between_species", "unfamiliar", "poor_image")
    if source == "live_video":
        reasons += ("no_majority",)
    if "reason" in data and data["reason"] not in reasons:
        raise ValueError("Unknown review reason for this source")

    location = data.get("coarse_location")
    if location is not None and (
        not isinstance(location, str) or not re.fullmatch(r"[A-Z]{2}-[A-Z0-9]{1,3}", location)
    ):
        raise ValueError("coarse_location must be a region code such as IN-TN")

    evidence = data.get("evidence")
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= (1 if source == "photo" else 3):
        raise ValueError("Send one JPEG for a photo or one to three JPEGs for a live encounter")
    expected_fields = {"sha256"} if source == "photo" else {"sha256", "offset_ms"}
    hashes, offsets = set(), set()
    for item in evidence:
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError("Evidence needs sha256 and, for live video, offset_ms")
        digest = item["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Evidence sha256 must be 64 lowercase hexadecimal characters")
        if digest in hashes:
            raise ValueError("Do not include the same JPEG more than once")
        hashes.add(digest)
        if source == "live_video":
            offset = item["offset_ms"]
            if type(offset) is not int or offset < 0 or offset in offsets:
                raise ValueError("Live frame offsets must be distinct nonnegative milliseconds")
            offsets.add(offset)

    top = data.get("top_predictions", [])
    if not isinstance(top, list) or len(top) > 3:
        raise ValueError("top_predictions must contain at most three species predictions")
    for prediction in top:
        validate_prediction(prediction)
    if len({prediction["label"] for prediction in top}) != len(top):
        raise ValueError("Species predictions must have distinct labels")
    for field in ("genus_prediction", "family_prediction"):
        if data.get(field) is not None:
            validate_prediction(data[field])


def sanitized_jpeg(raw):
    """Decode a bounded JPEG, apply its orientation and retain only pixels."""
    try:
        with Image.open(BytesIO(raw), formats=("JPEG",)) as image:
            if not (1 <= min(image.size) <= max(image.size) <= MAX_IMAGE_SIDE):
                raise ValueError(f"Use a JPEG with each side at most {MAX_IMAGE_SIDE} pixels")
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            normalized.info.clear()
            output = BytesIO()
            normalized.save(output, format="JPEG", quality=90)
    except (OSError, SyntaxError, Image.DecompressionBombError):
        raise ValueError("Evidence must be a readable JPEG") from None
    return output.getvalue()
