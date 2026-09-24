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
