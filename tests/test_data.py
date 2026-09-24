"""Tests for fraud_detection.schema and fraud_detection.data (synthetic data only)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from fraud_detection.data import drop_exact_duplicates, load_clean, load_raw, save_summary, validate
from fraud_detection.schema import FEATURE_COLUMNS, TARGET


def test_feature_columns_contract() -> None:
    assert len(FEATURE_COLUMNS) == 30
    assert FEATURE_COLUMNS[0] == "Time" and FEATURE_COLUMNS[-1] == "Amount"
    assert TARGET not in FEATURE_COLUMNS


def test_valid_frame_passes(raw_df: pd.DataFrame) -> None:
    out = validate(raw_df)
    assert out.shape == raw_df.shape


def test_missing_column_fails(raw_df: pd.DataFrame) -> None:
    with pytest.raises(pa.errors.SchemaErrors):
        validate(raw_df.drop(columns=["V7"]))


def test_extra_column_fails(raw_df: pd.DataFrame) -> None:
    with pytest.raises(pa.errors.SchemaErrors):
        validate(raw_df.assign(extra=1.0))


def test_negative_amount_fails(raw_df: pd.DataFrame) -> None:
    raw_df.loc[0, "Amount"] = -1.0
    with pytest.raises(pa.errors.SchemaErrors):
        validate(raw_df)


def test_nan_fails(raw_df: pd.DataFrame) -> None:
    raw_df.loc[3, "V4"] = np.nan
    with pytest.raises(pa.errors.SchemaErrors):
        validate(raw_df)


def test_class_two_fails(raw_df: pd.DataFrame) -> None:
    raw_df.loc[0, TARGET] = 2
    with pytest.raises(pa.errors.SchemaErrors):
        validate(raw_df)


def test_duplicates_removed_and_counted(raw_df: pd.DataFrame) -> None:
    # rows 0 and 3 (fraud) copied once, row 3 copied twice -> 3 duplicates, 2 of them fraud
    df = pd.concat([raw_df, raw_df.iloc[[0, 3, 3]]], ignore_index=True)
    clean, n_dropped = drop_exact_duplicates(df)
    assert n_dropped == 3
    assert len(clean) == len(raw_df)
    assert not clean.duplicated().any()
    assert list(clean.index) == list(range(len(clean)))


def test_same_features_different_label_is_not_duplicate(raw_df: pd.DataFrame) -> None:
    flipped = raw_df.iloc[[0]].assign(**{TARGET: 1})
    _, n_dropped = drop_exact_duplicates(pd.concat([raw_df, flipped], ignore_index=True))
    assert n_dropped == 0


def test_load_raw_casts_int_time_to_float(raw_df: pd.DataFrame, tmp_path: Path) -> None:
    csv = tmp_path / "raw.csv"
    raw_df.assign(Time=raw_df["Time"].astype("int64")).to_csv(csv, index=False)
    df = load_raw(csv)
    assert df["Time"].dtype == "float64"
    assert df[TARGET].dtype == "int64"


def test_load_clean_summary(raw_df: pd.DataFrame, tmp_path: Path) -> None:
    csv = tmp_path / "raw.csv"
    pd.concat([raw_df, raw_df.iloc[[3]]], ignore_index=True).to_csv(csv, index=False)
    clean, summary = load_clean(csv)
    assert summary == {
        "rows_raw": 11,
        "rows": 10,
        "fraud_count": 2,
        "fraud_rate": 0.2,
        "duplicates_dropped": 1,
        "duplicate_fraud_dropped": 1,
    }
    assert len(clean) == 10


def test_save_summary_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "summary.json"
    save_summary({"rows": 1}, out)
    assert json.loads(out.read_text(encoding="utf-8")) == {"rows": 1}
