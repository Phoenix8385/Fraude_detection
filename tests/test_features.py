"""Tests for fraud_detection.features."""

import math

import pandas as pd
import pytest

from fraud_detection.features import ENGINEERED_FEATURES, MODEL_FEATURES, add_features
from fraud_detection.schema import FEATURE_COLUMNS


@pytest.mark.parametrize(
    ("time", "expected_hour"),
    [(0.0, 0), (3599.0, 0), (3600.0, 1), (86399.0, 23), (86400.0, 0), (90000.0, 1)],
)
def test_hour_of_day(time: float, expected_hour: int) -> None:
    out = add_features(pd.DataFrame({"Time": [time], "Amount": [1.0]}))
    assert out.loc[0, "hour_of_day"] == expected_hour


@pytest.mark.parametrize(
    ("amount", "expected"),
    [(0.0, 0.0), (math.e - 1, 1.0), (99.0, math.log(100.0))],
)
def test_log_amount(amount: float, expected: float) -> None:
    out = add_features(pd.DataFrame({"Time": [0.0], "Amount": [amount]}))
    assert out.loc[0, "log_amount"] == pytest.approx(expected)


def test_input_not_mutated(raw_df: pd.DataFrame) -> None:
    before = raw_df.copy()
    out = add_features(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)
    assert out is not raw_df
    assert "log_amount" not in raw_df.columns


def test_all_model_features_present(raw_df: pd.DataFrame) -> None:
    out = add_features(raw_df)
    assert set(MODEL_FEATURES) <= set(out.columns)
    assert len(out) == len(raw_df)


def test_model_features_contract() -> None:
    assert MODEL_FEATURES == FEATURE_COLUMNS + ENGINEERED_FEATURES
    assert len(MODEL_FEATURES) == 32
    assert len(set(MODEL_FEATURES)) == 32
