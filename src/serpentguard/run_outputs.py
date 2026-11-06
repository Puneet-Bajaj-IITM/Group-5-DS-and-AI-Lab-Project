"""What a run measures and saves: predictions, prototypes, the final metrics, the saved outputs and the provenance
(library versions, hashes)."""
import hashlib
import json
import platform
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import albumentations
import cv2
import numpy as np
import pandas as pd
import sklearn
import timm
import torch
from scipy import special

from serpentguard.config import RunConfig
from serpentguard.data import WHO_RISK_ID, taxonomy_index
from serpentguard.metrics import calibration, safety_report, species_report, taxonomy_report
from serpentguard.models import SnakeID, prototypes_from
from serpentguard.run_files import CHECKPOINT_FILE
from serpentguard.subset import DANGEROUS_FAMILIES

PREDICTION_KEYS = ("embedding", "species_logits", "who_logits", "genus_logits", "family_logits")
FRAME_KEYS = {"paths": "path", "labels": "class_id", "species": "species", "genus_id": "genus_id",
              "family_id": "family_id", "who_risk_id": "who_risk_id", "observer_id": "observer_id"}
HELDOUT_KEYS = ("split", "obs_id")
AMP_DTYPE = torch.float16     # autocast's dtype on CUDA; the T4 has no bfloat16 tensor cores
MEBIBYTE = 2 ** 20


@dataclass(frozen=True)
class ClassTables:
    """Per-class arrays in class_id order, from classes.csv."""
    species: np.ndarray
    tier: np.ndarray
    who: np.ndarray
    listed_or_venomous: np.ndarray  # WHO-listed, or a viper or elapid
    genus_of_class: np.ndarray
    family_of_class: np.ndarray
    n_genus: int
    n_family: int

    @property
    def n_classes(self):
        return len(self.species)


def class_tables(classes):
    genus_names, family_names, genus_of_class, family_of_class = taxonomy_index(classes)
    ordered = classes.sort_values("class_id")
    who = ordered["who_category"].astype(str).map(WHO_RISK_ID).to_numpy(dtype=int)
    listed_or_venomous = (who != WHO_RISK_ID["not_listed"]) | ordered["family"].isin(DANGEROUS_FAMILIES).to_numpy()
    return ClassTables(ordered["species"].to_numpy(dtype=str), ordered["tier"].to_numpy(dtype=str), who,
                       listed_or_venomous, genus_of_class, family_of_class, len(genus_names), len(family_names))


def load_checkpoint(run_dir, device="cpu"):
    """(RunConfig, SnakeID in eval mode, best epoch) from best.pt, built from the config the checkpoint carries."""
    checkpoint = torch.load(Path(run_dir) / CHECKPOINT_FILE, map_location="cpu", weights_only=True)
    cfg = RunConfig(**checkpoint["config"])
    model = SnakeID.from_config(cfg, checkpoint["n_classes"], checkpoint["n_genus"], checkpoint["n_family"],
                                pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return cfg, model.to(device).eval(), checkpoint.get("epoch")


@torch.inference_mode()
def predict(model, loader, device, amp):
    """The model's outputs on every photo of the loader, in order, as float32 arrays (missing heads left out)."""
    model.eval()
    collected = defaultdict(list)
    for batch in loader:
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=amp):
            out = model(batch["image"].to(device, non_blocking=True))
        for name in PREDICTION_KEYS:
            if out[name] is not None:
                collected[name].append(out[name].float())  # kept on the device until the end
    return {name: torch.cat(parts).cpu().numpy() for name, parts in collected.items()}


def softmax(logits):
    return special.softmax(np.asarray(logits, dtype=np.float64), axis=1)


def photographer_groups(frame):
    """observer_id per photo; a photo without one is its own photographer."""
    observers = frame["observer_id"].astype("string")
    return observers.where(observers.notna() & observers.str.strip().ne(""), frame["path"]).to_numpy(dtype=str)


def build_prototypes(embedding, frame, n_classes):
    """Plain and photographer-balanced class prototypes from the training frame's embeddings, as (C, D) arrays."""
    embedding, labels = torch.from_numpy(embedding), frame["class_id"].to_numpy(dtype=np.int64, copy=True)
    plain = prototypes_from(embedding, labels, n_classes)
    balanced = prototypes_from(embedding, labels, n_classes, groups=photographer_groups(frame))
    return plain.numpy(), balanced.numpy()


def prototype_probs(embedding, prototypes, scale):
    """softmax(scale x cosine) of each embedding against each prototype."""
    cosine = SnakeID.cosine_to(torch.from_numpy(embedding), torch.from_numpy(prototypes)).numpy()
    return softmax(scale * cosine)


def species_scores(frame, probs, train_counts, taxa, val_counts=None):
    """species_report on a frame; val_counts default to the frame's own, so a slice must pass the full set's."""
    y_true = frame["class_id"].to_numpy()
    if val_counts is None:
        val_counts = np.bincount(y_true, minlength=taxa.n_classes)
    return species_report(y_true, probs, train_counts, val_counts, taxa.tier)


def final_metrics(val, outputs, prototypes, train_counts, taxa, scale, val_counts=None):
    """The head, plain-prototype and balanced-prototype species reports, raw safety, calibration and taxonomy."""
    y_true = val["class_id"].to_numpy()
    head = softmax(outputs["species_logits"])
    plain, balanced = prototypes
    genus_head, family_head = (softmax(outputs[name]) if name in outputs else None
                               for name in ("genus_logits", "family_logits"))
    return {
        "head": species_scores(val, head, train_counts, taxa, val_counts),
        "prototype": species_scores(val, prototype_probs(outputs["embedding"], plain, scale), train_counts, taxa,
                                    val_counts),
        "prototype_balanced": species_scores(val, prototype_probs(outputs["embedding"], balanced, scale),
                                             train_counts, taxa, val_counts),
        "safety": safety_report(y_true, head, taxa.who, taxa.listed_or_venomous),
        "calibration": calibration(head, y_true),
        "taxonomy": taxonomy_report(head, genus_head, family_head, taxa.genus_of_class, taxa.family_of_class,
                                    val["genus_id"].to_numpy(), val["family_id"].to_numpy()),
    }


def save_outputs(path, frame, outputs, extra_columns=()):
    """The frame's identities and labels next to its predictions, one row per photo, without pickled objects."""
    def plain(column):
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
            return column.to_numpy()
        return column.astype("string").fillna("").to_numpy(dtype=str)

    columns = {**FRAME_KEYS, **{name: name for name in extra_columns}}
    np.savez_compressed(path, **{key: plain(frame[column]) for key, column in columns.items()}, **outputs)


def subset_hash(paths):
    """SHA-256 of the sorted subset paths joined by newlines."""
    return hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(MEBIBYTE), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload):
    """SHA-256 of payload as canonical JSON, numpy values as plain numbers and lists."""
    text = json.dumps(payload, sort_keys=True, default=lambda value: np.asarray(value).tolist())
    return hashlib.sha256(text.encode()).hexdigest()


def device_name(device):
    return torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or device.type


def peak_gpu_memory_mb(device):
    return torch.cuda.max_memory_allocated(device) / MEBIBYTE if device.type == "cuda" else None


def library_versions(device=None):
    """Library versions, with cuDNN and the GPU name on CUDA: the run's device, or the first GPU if none is given."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    on_gpu = device.type == "cuda"
    return {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
            "albumentations": albumentations.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "opencv": cv2.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if on_gpu else None, "gpu": device_name(device) if on_gpu else None}


def write_json(path, payload):
    """Writes payload and returns it as read back, so callers hold what the file holds (string keys, plain types)."""
    def plain(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"can't write {type(value).__name__} to JSON")

    text = json.dumps(payload, indent=2, default=plain)
    Path(path).write_text(text)
    return json.loads(text)
