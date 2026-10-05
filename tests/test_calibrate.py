"""Tests for fraud_detection.calibrate (synthetic data, no MLflow, no real model)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fraud_detection.calibrate import (
    CALIBRATION_API,
    fit_calibrator,
    grid_diagnostics,
    half_summary,
    plot_reliability,
    select_calibration,
    split_validation,
    summary_line,
)
from fraud_detection.schema import TARGET
from fraud_detection.threshold import threshold_table


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """400 rows, 40 fraud, Time shuffled so that row order != time order."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x": rng.normal(size=n), "Amount": rng.uniform(1, 100, n)})
    df["Time"] = rng.permutation(np.arange(n, dtype="float64"))
    df[TARGET] = (np.arange(n) % 10 == 0).astype(int)
    df.loc[df[TARGET] == 1, "x"] += 2
    return df


def test_time_halves_are_chronological(valid_df: pd.DataFrame) -> None:
    cal, thr = split_validation(valid_df, "time", seed=0)
    assert len(cal) == len(thr) == 200
    assert cal["Time"].max() < thr["Time"].min()  # earlier = valid_cal, later = valid_thr
    assert cal.index.intersection(thr.index).empty


def test_stratified_halves_keep_fraud_rate(valid_df: pd.DataFrame) -> None:
    cal, thr = split_validation(valid_df, "stratified", seed=0)
    assert int(cal[TARGET].sum()) == int(thr[TARGET].sum()) == 20
    assert cal.index.intersection(thr.index).empty
    again, _ = split_validation(valid_df, "stratified", seed=0)
    assert again.index.equals(cal.index)  # seeded


def test_half_summary_warns_below_min_positives(valid_df: pd.DataFrame) -> None:
    cal, thr = split_validation(valid_df, "stratified", seed=0)
    summary, warnings = half_summary(cal, thr, min_positives=30)
    assert summary["valid_cal"]["fraud"] == 20 and len(warnings) == 2
    assert half_summary(cal, thr, min_positives=20)[1] == []


@pytest.mark.parametrize(
    ("briers", "selected"),
    [
        ({"none": 0.0100, "sigmoid": 0.0090, "isotonic": 0.0094}, "sigmoid"),  # 10% better
        ({"none": 0.0100, "sigmoid": 0.0096, "isotonic": 0.0097}, "none"),  # 4%: within 5%
        ({"none": 0.0100, "sigmoid": 0.0095, "isotonic": 0.0099}, "none"),  # exactly 5%
        ({"none": 0.0100, "sigmoid": 0.0120, "isotonic": 0.0110}, "none"),  # none is lowest
        ({"none": 0.0100, "sigmoid": 0.0098, "isotonic": 0.0080}, "isotonic"),  # 20% better
    ],
)
def test_select_calibration_rule(briers: dict, selected: str) -> None:
    s = select_calibration(briers, tolerance=0.05)
    assert s["selected"] == selected
    assert s["lowest_brier"] == min(briers, key=briers.get)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_does_not_refit_the_model(valid_df: pd.DataFrame, method: str) -> None:
    X, y = valid_df[["x"]], valid_df[TARGET]
    model = LogisticRegression().fit(X.iloc[:200], y.iloc[:200])
    coef = model.coef_.copy()
    calibrated = fit_calibrator(model, X.iloc[200:], y.iloc[200:], method)
    np.testing.assert_array_equal(model.coef_, coef)  # frozen: the model was not refitted
    p = calibrated.predict_proba(X)[:, 1]
    assert p.shape == (len(X),) and (p >= 0).all() and (p <= 1).all()


def test_grid_diagnostics_counts_fraud_below_grid() -> None:
    y = pd.Series([1, 1, 1, 0, 0])
    proba = np.array([0.005, 0.5, 0.9, 0.001, 0.2])
    table = threshold_table(y, proba, [1.0] * 5, review_cost=1.0)
    d = grid_diagnostics(table, y, proba)
    assert d["fraud_scored_below_grid_min"] == 1 and d["fraud_total"] == 3
    assert d["max_recall_on_grid"] == pytest.approx(2 / 3)


def test_reliability_plot_writes_file(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    y = valid_df[TARGET].to_numpy()
    probas = {"none": np.linspace(0, 1, len(y)), "sigmoid": np.full(len(y), 0.1)}
    out = tmp_path / "rel.png"
    plot_reliability(y, probas, {"none": 0.2, "sigmoid": 0.1}, "t", out)
    assert out.exists() and out.stat().st_size > 0


def test_select_calibration_explains_how_rule_was_applied() -> None:
    within = select_calibration({"none": 0.0100, "sigmoid": 0.0099, "isotonic": 0.0097}, 0.05)
    assert "within 5%" in within["how_applied"] and within["lowest_brier"] == "isotonic"
    beats = select_calibration({"none": 0.0100, "sigmoid": 0.0050, "isotonic": 0.0097}, 0.05)
    assert "more than 5%" in beats["how_applied"]
    assert select_calibration({"none": 0.001, "sigmoid": 0.002}, 0.05)["how_applied"] == (
        "'none' has the lowest Brier"
    )


def test_calibration_api_path_is_recorded() -> None:
    assert "FrozenEstimator" in CALIBRATION_API or "prefit" in CALIBRATION_API
    assert "sklearn" in CALIBRATION_API


def test_summary_line_is_console_safe_and_reports_missing_t_block() -> None:
    result = {
        "calibration": {"selected": "none", "candidates_valid_thr": {"none": {"brier": 0.001}}},
        "policy": {
            "split": "time", "model": "logreg", "warnings": [],
            "t_review_detail": {"threshold": 0.01, "recall": 0.8, "precision": 0.5,
                                "expected_cost": 10.0, "fallback": True},
            "t_block_detail": {"exists": False, "threshold": None},
        },
    }  # fmt: skip
    line = summary_line(result)
    line.encode("cp1252")  # the CLI prints this on a Windows console
    assert "t_block DOES NOT EXIST" in line and "fallback True" in line
