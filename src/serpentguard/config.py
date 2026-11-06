"""One run's settings, read from and written to YAML."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass(kw_only=True)
class RunConfig:
    run_id: str
    encoder: str
    head: str
    losses: list[str]
    front_end: str
    input_size: int = 320
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.05
    batch_size: int = 64
    sampler: str
    seed: int = 42
    subset_path: str
    embedding_dim: int = 256
    arcface_s: float = 30.0
    arcface_m: float = 0.3
    triplet_margin: float = 0.2
    triplet_weight: float = 1.0
    mask_weight: float = 1.0
    hierarchy: bool = True
    mask_head: bool = False
    mask_keep: float = 0.2
    pooling: str = "gap"           # "gap", or "mask" for mask-guided pooling (needs mask_head)
    danger_weight: float = 5.0
    aug_strength: float = 1.0
    mask_source: str = "sam"
    box_source: str = "none"
    eval_every: int = 2
    warmup_epochs: int = 1
    exclude_species: list[str] = field(default_factory=list)  # K1: kept out of training by the runner


FLOAT_FIELDS = {entry.name for entry in fields(RunConfig) if entry.type is float}


def load_config(path) -> RunConfig:
    values = yaml.safe_load(Path(path).read_text())
    for name in FLOAT_FIELDS & values.keys():
        values[name] = float(values[name])  # PyYAML reads 3e-4, with no dot, as a string
    return RunConfig(**values)


def save_config(config: RunConfig, path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
