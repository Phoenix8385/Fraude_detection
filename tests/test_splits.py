"""Tests for fraud_detection.splits (synthetic data only)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import SplitConfig
from fraud_detection.schema import TARGET
from fraud_detection.splits import PARTS, save_splits, stratified_split, time_split

RATIOS = SplitConfig(train=0.6, valid=0.2, test=0.2)


@pytest.fixture
def split_df() -> pd.DataFrame:
    """1,000 rows, 50 fraud (5%), Time shuffled and with ties, non-default index."""
    rng = np.random.default_rng(1)
    n = 1000
    return pd.DataFrame(
        {
            "Time": rng.integers(0, 500, size=n).astype("float64"),  # many equal Times
            "V1": rng.normal(size=n),
            "Amount": rng.uniform(0, 100, size=n),
            TARGET: rng.permutation([1] * 50 + [0] * (n - 50)),
        },
        index=np.arange(n) + 10_000,
    )


@pytest.fixture(params=["stratified", "time"])
def splits(request: pytest.FixtureRequest, split_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if request.param == "stratified":
        return stratified_split(split_df, RATIOS, seed=42)
    return time_split(split_df, RATIOS)


def test_no_row_in_two_parts_and_nothing_lost(
    splits: dict[str, pd.DataFrame], split_df: pd.DataFrame
) -> None:
    indices = [set(splits[p].index) for p in PARTS]
    assert not indices[0] & indices[1]
    assert not indices[0] & indices[2]
    assert not indices[1] & indices[2]
    assert set().union(*indices) == set(split_df.index)


def test_ratios_within_one_percent(splits: dict[str, pd.DataFrame], split_df: pd.DataFrame) -> None:
    n = len(split_df)
    for part, expected in zip(PARTS, (RATIOS.train, RATIOS.valid, RATIOS.test), strict=True):
        assert abs(len(splits[part]) / n - expected) <= 0.01


def test_stratified_keeps_fraud_rate(split_df: pd.DataFrame) -> None:
    splits = stratified_split(split_df, RATIOS, seed=42)
    overall = split_df[TARGET].mean()
    for part in PARTS:
        assert splits[part][TARGET].mean() == pytest.approx(overall, abs=0.005)


def test_time_split_is_chronological(split_df: pd.DataFrame) -> None:
    splits = time_split(split_df, RATIOS)
    assert splits["train"]["Time"].max() <= splits["valid"]["Time"].min()
    assert splits["valid"]["Time"].max() <= splits["test"]["Time"].min()


def test_stratified_deterministic_with_same_seed(split_df: pd.DataFrame) -> None:
    a = stratified_split(split_df, RATIOS, seed=42)
    b = stratified_split(split_df, RATIOS, seed=42)
    c = stratified_split(split_df, RATIOS, seed=7)
    for part in PARTS:
        pd.testing.assert_frame_equal(a[part], b[part])
    assert not a["test"].index.equals(c["test"].index)


def test_time_split_deterministic(split_df: pd.DataFrame) -> None:
    a = time_split(split_df, RATIOS)
    b = time_split(split_df.sample(frac=1.0, random_state=3).sort_index(), RATIOS)
    for part in PARTS:
        pd.testing.assert_frame_equal(a[part], b[part])


def test_save_splits_writes_parquet_and_summary(split_df: pd.DataFrame, tmp_path: Path) -> None:
    splits = stratified_split(split_df, RATIOS, seed=42)
    summary = save_splits(splits, "demo", processed_dir=tmp_path / "p", metrics_dir=tmp_path / "m")

    for part in PARTS:
        back = pd.read_parquet(tmp_path / "p" / f"demo_{part}.parquet")
        pd.testing.assert_frame_equal(back, splits[part])

    on_disk = json.loads((tmp_path / "m" / "splits_demo.json").read_text(encoding="utf-8"))
    assert on_disk == summary
    assert sum(on_disk[p]["rows"] for p in PARTS) == len(split_df)
    assert sum(on_disk[p]["fraud_count"] for p in PARTS) == 50
    assert set(on_disk["train"]) == {"rows", "fraud_count", "fraud_rate", "time_min", "time_max"}
