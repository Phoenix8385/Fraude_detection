"""Tests for fraud_detection.train helpers (no MLflow, no real data)."""

from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.train import append_experiment_row, load_part


def test_load_part_refuses_test_split(tmp_path: Path) -> None:
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "time_test.parquet")
    with pytest.raises(PermissionError, match="evaluate.py"):
        load_part("time", "test", tmp_path)


def test_load_part_reads_train(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2]})
    df.to_parquet(tmp_path / "time_train.parquet")
    pd.testing.assert_frame_equal(load_part("time", "train", tmp_path), df)


def test_append_experiment_row_writes_header_once(tmp_path: Path) -> None:
    csv = tmp_path / "sub" / "experiments.csv"
    append_experiment_row({"model": "dummy", "pr_auc": 0.1}, csv)
    append_experiment_row({"model": "logreg", "pr_auc": 0.7}, csv)
    back = pd.read_csv(csv)
    assert list(back["model"]) == ["dummy", "logreg"]
    assert list(back.columns) == ["model", "pr_auc"]


def test_train_scale_pos_weight() -> None:
    from fraud_detection.train import train_scale_pos_weight

    assert train_scale_pos_weight(pd.Series([0] * 90 + [1] * 10)) == 9.0
    with pytest.raises(ValueError, match="no fraud"):
        train_scale_pos_weight(pd.Series([0, 0, 0]))
