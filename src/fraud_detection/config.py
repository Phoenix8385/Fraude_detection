"""Load configs/config.yaml into immutable, typed settings objects.

Every module reads settings from here instead of hardcoding paths, seeds, or costs.
Paths in the YAML are relative to the project root and are resolved to absolute paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# src/fraud_detection/config.py -> parents[0]=fraud_detection, [1]=src, [2]=project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    """Absolute filesystem locations used by the pipeline."""

    raw_data: Path
    processed_dir: Path
    models_dir: Path
    figures_dir: Path
    metrics_dir: Path
    mlflow_db: Path
    mlflow_artifacts_dir: Path


@dataclass(frozen=True)
class SplitConfig:
    """Fractions of rows assigned to train / valid / test."""

    train: float
    valid: float
    test: float


@dataclass(frozen=True)
class CostConfig:
    """Business costs used for threshold selection."""

    review_cost_per_alert: float
    missed_fraud_cost: str


@dataclass(frozen=True)
class Config:
    """Top-level project configuration."""

    paths: PathsConfig
    seed: int
    split: SplitConfig
    costs: CostConfig
    target_recall: float


def _resolve(relative: str, root: Path) -> Path:
    """Turn a project-relative path from the YAML into an absolute path."""
    return (root / relative).resolve()


def load_config(path: Path | str | None = None, root: Path = PROJECT_ROOT) -> Config:
    """Read the YAML config and return a validated, frozen Config.

    Args:
        path: Config file to read. Defaults to configs/config.yaml.
        root: Directory that relative paths in the YAML are resolved against.

    Raises:
        ValueError: If split fractions do not sum to 1.0 or values are out of range.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    paths = PathsConfig(**{key: _resolve(value, root) for key, value in raw["paths"].items()})
    split = SplitConfig(**{key: float(value) for key, value in raw["split"].items()})
    costs = CostConfig(
        review_cost_per_alert=float(raw["costs"]["review_cost_per_alert"]),
        missed_fraud_cost=str(raw["costs"]["missed_fraud_cost"]),
    )
    config = Config(
        paths=paths,
        seed=int(raw["seed"]),
        split=split,
        costs=costs,
        target_recall=float(raw["target_recall"]),
    )
    _validate(config)
    return config


def _validate(config: Config) -> None:
    """Fail fast on settings that would silently corrupt later phases."""
    total = config.split.train + config.split.valid + config.split.test
    if not math.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    if not 0.0 < config.target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {config.target_recall}")
    if config.costs.review_cost_per_alert < 0:
        raise ValueError("review_cost_per_alert must be non-negative")
