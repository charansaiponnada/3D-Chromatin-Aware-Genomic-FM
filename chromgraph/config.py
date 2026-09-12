"""Config loading and machine-independent paths.

Two machines run this code: a Windows laptop (Phases 0-2, no CUDA) and the
college L40S cluster (Phases 3-6). Only git moves between them, so no path may
be hardcoded and no data file may be assumed present. Everything resolves from
CHROMGRAPH_DATA / CHROMGRAPH_RESULTS, defaulting to ./data and ./results.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID_ARMS = (
    "b0_dna_only",
    "b1_random_graph",
    "b2_distance_only",
    "b3_shuffled_hic",
    "b4_late_fusion",
    "b5_contrastive_only",
    "full",
)


def data_dir() -> Path:
    return Path(os.environ.get("CHROMGRAPH_DATA", REPO_ROOT / "data"))


def results_dir() -> Path:
    return Path(os.environ.get("CHROMGRAPH_RESULTS", REPO_ROOT / "results"))


@dataclass
class DataConfig:
    assembly: str
    bin_size: int
    nodes_per_sample: int
    cell_lines: list[str]
    chroms_train: list[str]
    chroms_val: list[str]
    chroms_test: list[str]
    pilot_chroms: list[str]
    top_k_edges: int
    local_radius: int
    min_separation: int
    held_out_edge_frac: float
    max_n_frac: float

    @property
    def context_bp(self) -> int:
        """Genomic span of one sample."""
        return self.bin_size * self.nodes_per_sample


@dataclass
class ModelConfig:
    d_model: int
    conv_depth: int
    encoder_layers: int
    d_state: int
    d_conv: int
    expand: int
    block_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    structure_dropout: float


@dataclass
class TrainConfig:
    arm: str
    steps: int
    batch_size: int
    lr: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    grad_accum_steps: int
    precision: str
    grad_checkpoint: bool
    mask_frac: float
    temperature: float
    lambda_dna: float
    lambda_contrast: float
    lambda_contact: float
    ckpt_every: int
    eval_every: int
    log_every: int


@dataclass
class Config:
    seed: int
    control_seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write the resolved config beside a run's outputs.

        A result without the exact config that produced it is not reproducible,
        so every run writes this before its first step, not after its last.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def _build(cls, section: dict[str, Any], name: str):
    """Instantiate a config dataclass, rejecting unknown and missing keys.

    A silently ignored typo in a YAML key is a config that lies about the run
    it describes, which is worse than a crash.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"unknown key(s) in '{name}': {sorted(unknown)}")
    missing = known - set(section)
    if missing:
        raise ValueError(f"missing key(s) in '{name}': {sorted(missing)}")
    return cls(**section)


def load_config(path: str | Path = REPO_ROOT / "configs" / "base.yaml",
                **overrides: Any) -> Config:
    """Load a YAML config. Overrides use dotted keys, e.g. train.arm='b0_dna_only'."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if not key:
            raw[section] = value
        elif section not in raw:
            raise ValueError(f"no such config section: '{section}'")
        else:
            raw[section][key] = value

    cfg = Config(
        seed=raw["seed"],
        control_seed=raw["control_seed"],
        data=_build(DataConfig, raw["data"], "data"),
        model=_build(ModelConfig, raw["model"], "model"),
        train=_build(TrainConfig, raw["train"], "train"),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.train.arm not in VALID_ARMS:
        raise ValueError(f"train.arm must be one of {VALID_ARMS}, got '{cfg.train.arm}'")

    d = cfg.data
    splits = {"train": set(d.chroms_train), "val": set(d.chroms_val), "test": set(d.chroms_test)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = splits[a] & splits[b]
        if shared:
            raise ValueError(f"chromosome leak between {a} and {b}: {sorted(shared)}")

    if cfg.model.d_model % cfg.model.n_heads:
        raise ValueError(
            f"d_model ({cfg.model.d_model}) must divide evenly by n_heads ({cfg.model.n_heads})"
        )
    if not 0.0 <= cfg.model.structure_dropout < 1.0:
        raise ValueError("model.structure_dropout must be in [0, 1)")
    if not 0.0 < d.held_out_edge_frac < 1.0:
        raise ValueError("data.held_out_edge_frac must be in (0, 1)")
    if d.min_separation <= d.local_radius:
        raise ValueError(
            f"data.min_separation ({d.min_separation}) must exceed local_radius "
            f"({d.local_radius}), otherwise local and distal edges overlap and "
            f"the distance-only control (B2) is not distinguishable from the full model"
        )
