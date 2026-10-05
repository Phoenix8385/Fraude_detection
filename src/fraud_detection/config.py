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
class TuningConfig:
    """Optuna search budget (Phase 7)."""

    n_trials: int
    cv_folds: int


@dataclass(frozen=True)
class BootstrapConfig:
    """Bootstrap settings for confidence intervals; the resampling seed is Config.seed."""

    n_resamples: int
    ci_level: float


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration selection settings (Phase 8)."""

    none_tolerance: float
    min_half_positives: int


@dataclass(frozen=True)
class PolicyConfig:
    """Two-threshold decision policy settings (Phase 8)."""

    block_precision: float
    block_min_tp: int
    stability_resamples: int


@dataclass(frozen=True)
class Config:
    """Top-level project configuration."""

    paths: PathsConfig
    seed: int
    split: SplitConfig
    costs: CostConfig
    target_recall: float
    tuning: TuningConfig
    bootstrap: BootstrapConfig
    stability_seeds: tuple[int, ...]
    calibration: CalibrationConfig
    policy: PolicyConfig


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
        tuning=TuningConfig(
            n_trials=int(raw["tuning"]["n_trials"]), cv_folds=int(raw["tuning"]["cv_folds"])
        ),
        bootstrap=BootstrapConfig(
            n_resamples=int(raw["bootstrap"]["n_resamples"]),
            ci_level=float(raw["bootstrap"]["ci_level"]),
        ),
        stability_seeds=tuple(int(s) for s in raw["stability_seeds"]),
        calibration=CalibrationConfig(
            none_tolerance=float(raw["calibration"]["none_tolerance"]),
            min_half_positives=int(raw["calibration"]["min_half_positives"]),
        ),
        policy=PolicyConfig(
            block_precision=float(raw["policy"]["block_precision"]),
            block_min_tp=int(raw["policy"]["block_min_tp"]),
            stability_resamples=int(raw["policy"]["stability_resamples"]),
        ),
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
    if config.tuning.n_trials < 1 or config.tuning.cv_folds < 2:
        raise ValueError("tuning needs n_trials >= 1 and cv_folds >= 2")
    if config.bootstrap.n_resamples < 1 or not 0.0 < config.bootstrap.ci_level < 1.0:
        raise ValueError("bootstrap needs n_resamples >= 1 and ci_level in (0, 1)")
    seeds = config.stability_seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("stability_seeds must be a non-empty list of distinct integers")
    if not 0.0 <= config.calibration.none_tolerance < 1.0:
        raise ValueError("calibration.none_tolerance must be in [0, 1)")
    if not 0.0 < config.policy.block_precision <= 1.0 or config.policy.block_min_tp < 1:
        raise ValueError("policy needs block_precision in (0, 1] and block_min_tp >= 1")
    if config.policy.stability_resamples < 1:
        raise ValueError("policy.stability_resamples must be >= 1")
