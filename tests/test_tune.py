"""Tests for fraud_detection.tune (tiny synthetic data, no MLflow)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from fraud_detection.features import MODEL_FEATURES
from fraud_detection.models import build_model
from fraud_detection.schema import TARGET
from fraud_detection.tune import (
    SEARCH_SPACE,
    best_params_unprefixed,
    make_cv,
    pipeline_space,
    search,
    summary_markdown,
    untuned_valid_pr_auc,
)

TINY_SPACE = {"n_estimators": [5, 10], "max_depth": [2, 3]}


@pytest.fixture
def tune_df() -> pd.DataFrame:
    """400 rows with all MODEL_FEATURES, 40 fraud spread over time, Time shuffled."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(rng.normal(size=(n, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    df["Time"] = rng.permutation(np.arange(n, dtype="float64"))
    y = np.zeros(n, dtype=int)
    y[::10] = 1
    df[TARGET] = y
    df.loc[df[TARGET] == 1, "V14"] -= 3  # make fraud learnable
    return df


def test_pipeline_space_prefixes_and_matches_pipeline() -> None:
    space = pipeline_space()
    assert set(space) == {f"model__{k}" for k in SEARCH_SPACE}
    # every key must be a real parameter of the pipeline (set_params raises otherwise)
    pipe = build_model("xgb", {"random_state": 0})
    pipe.set_params(**{k: v[0] for k, v in space.items()})


def test_make_cv_types() -> None:
    assert isinstance(make_cv("time", 42), TimeSeriesSplit)
    cv = make_cv("stratified", 42)
    assert isinstance(cv, StratifiedKFold) and cv.shuffle and cv.random_state == 42


def test_search_returns_params_from_space(tune_df: pd.DataFrame) -> None:
    s = search("xgb", tune_df, "stratified", seed=0, n_iter=3, n_splits=3, space=TINY_SPACE)
    best = best_params_unprefixed(s)
    assert set(best) == set(TINY_SPACE)
    for key, value in best.items():
        assert value in TINY_SPACE[key]
    assert s.scoring == "average_precision"
    assert s.best_estimator_.predict_proba(tune_df[MODEL_FEATURES]).shape == (len(tune_df), 2)


def test_time_search_sorts_rows_by_time(tune_df: pd.DataFrame) -> None:
    s = search("xgb_weighted", tune_df, "time", seed=0, n_iter=2, n_splits=3, space=TINY_SPACE)
    assert isinstance(s.cv, TimeSeriesSplit)
    # the weight was computed from the train labels: 360 legit / 40 fraud
    assert s.best_estimator_.named_steps["model"].get_params()["scale_pos_weight"] == 9.0


def test_search_is_deterministic(tune_df: pd.DataFrame) -> None:
    a = search("xgb", tune_df, "stratified", seed=1, n_iter=3, n_splits=3, space=TINY_SPACE)
    b = search("xgb", tune_df, "stratified", seed=1, n_iter=3, n_splits=3, space=TINY_SPACE)
    assert a.best_params_ == b.best_params_
    scores_a, scores_b = a.cv_results_["mean_test_score"], b.cv_results_["mean_test_score"]
    np.testing.assert_array_equal(scores_a, scores_b)


def test_search_rejects_non_xgb(tune_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="XGBoost models only"):
        search("logreg", tune_df, "stratified", seed=0)


def test_untuned_lookup_uses_latest_row(tmp_path: Path) -> None:
    csv = tmp_path / "experiments.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-02"],
            "model": ["xgb", "xgb", "xgb_tuned"],
            "split": ["time", "time", "time"],
            "pr_auc": [0.5, 0.78, 0.99],
        }
    ).to_csv(csv, index=False)
    assert untuned_valid_pr_auc("xgb", "time", csv) == 0.78
    assert untuned_valid_pr_auc("xgb", "stratified", csv) is None
    assert untuned_valid_pr_auc("xgb", "time", tmp_path / "missing.csv") is None


def test_summary_markdown_change_column() -> None:
    md = summary_markdown(
        [
            {
                "model": "xgb", "split": "time", "cv": "TimeSeriesSplit",
                "cv_pr_auc_mean": 0.7, "cv_pr_auc_std": 0.05,
                "valid_pr_auc_untuned": 0.78, "valid_pr_auc_tuned": 0.80,
            }
        ]
    )  # fmt: skip
    assert "| xgb | time | TimeSeriesSplit | 0.7000 ± 0.0500 | 0.7800 | 0.8000 | +0.0200 |" in md
