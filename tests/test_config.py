"""Tests for fraud_detection.config."""

import dataclasses
import math
from pathlib import Path

import pytest

from fraud_detection.config import PROJECT_ROOT, load_config


def test_config_loads_with_expected_values() -> None:
    config = load_config()
    assert config.seed == 42
    assert config.target_recall == 0.85
    assert config.costs.review_cost_per_alert == 5.0
    assert config.costs.missed_fraud_cost == "amount"
    assert config.tuning.n_trials == 50 and config.tuning.cv_folds == 5
    assert config.bootstrap.n_resamples == 1000 and config.bootstrap.ci_level == 0.95
    assert config.stability_seeds == (42, 43, 44, 45, 46)
    assert config.calibration.none_tolerance == 0.05
    assert config.calibration.min_half_positives == 30
    assert config.policy.block_precision == 0.90 and config.policy.block_min_tp == 5
    assert config.policy.stability_resamples == 200


def test_duplicate_stability_seeds_raise(tmp_path: Path) -> None:
    text = (PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("[42, 43, 44, 45, 46]", "[42, 42]"), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct"):
        load_config(bad)


def test_split_ratios_sum_to_one() -> None:
    split = load_config().split
    assert math.isclose(split.train + split.valid + split.test, 1.0)


def test_paths_are_absolute_and_inside_project() -> None:
    raw_data = load_config().paths.raw_data
    assert raw_data.is_absolute()
    assert raw_data.is_relative_to(PROJECT_ROOT)


def test_config_is_frozen() -> None:
    config = load_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 7  # type: ignore[misc]


def test_bad_split_raises(tmp_path: Path) -> None:
    text = (PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("test: 0.2", "test: 0.3"), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_config(bad)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("target_recall: 0.85", "target_recall: 1.5", "target_recall"),
        ("review_cost_per_alert: 5.0", "review_cost_per_alert: -1.0", "non-negative"),
        ("n_trials: 50", "n_trials: 0", "tuning"),
        ("n_resamples: 1000", "n_resamples: 0", "bootstrap"),
        ("none_tolerance: 0.05", "none_tolerance: 1.5", "none_tolerance"),
        ("block_precision: 0.90", "block_precision: 0.0", "block_precision"),
        ("stability_resamples: 200", "stability_resamples: 0", "stability_resamples"),
    ],
)
def test_invalid_settings_fail_fast(tmp_path: Path, old: str, new: str, match: str) -> None:
    text = (PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    assert old in text
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(bad)
