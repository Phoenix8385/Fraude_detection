"""Tests for fraud_detection.explain — a tiny synthetic LR + isotonic model with the same
structure as the production model (never the real artifact or the real CSV)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_detection.explain import (
    explain_row,
    global_importance,
    linear_contributions,
    linear_pipeline,
)
from fraud_detection.features import MODEL_FEATURES


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    y = pd.Series((np.arange(n) % 10 == 0).astype(int))
    X.loc[y == 1, "V14"] -= 3
    return X, y


@pytest.fixture(scope="module")
def pipe(data) -> Pipeline:  # type: ignore[no-untyped-def]
    X, y = data
    return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(
        X.iloc[:300], y.iloc[:300]
    )


@pytest.fixture(scope="module")
def calibrated(pipe, data):  # type: ignore[no-untyped-def]
    X, y = data
    return CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(
        X.iloc[300:], y.iloc[300:]
    )


def test_unwraps_calibrated_and_plain_pipeline(pipe, calibrated) -> None:  # type: ignore[no-untyped-def]
    assert linear_pipeline(calibrated) is not None
    assert linear_pipeline(calibrated).named_steps["model"] is pipe.named_steps["model"]
    assert linear_pipeline(pipe) is pipe


def test_rejects_non_linear_model(data) -> None:  # type: ignore[no-untyped-def]
    X, y = data
    xgb = Pipeline([("model", XGBClassifier(n_estimators=5))]).fit(X, y)
    with pytest.raises(TypeError, match="linear"):
        linear_pipeline(xgb)


def test_contributions_sum_exactly_to_uncalibrated_log_odds(pipe, calibrated, data) -> None:  # type: ignore[no-untyped-def]
    X, _ = data
    contrib, base = linear_contributions(calibrated, X)
    np.testing.assert_allclose(base + contrib.sum(axis=1), pipe.decision_function(X), rtol=1e-9)
    assert list(contrib.columns) == MODEL_FEATURES
    assert base == pipe.named_steps["model"].intercept_[0]


def test_contribution_is_coef_times_scaled_value(pipe, data) -> None:  # type: ignore[no-untyped-def]
    X, _ = data
    contrib, _ = linear_contributions(pipe, X.iloc[:1])
    scaler, lr = pipe.named_steps["scaler"], pipe.named_steps["model"]
    j = MODEL_FEATURES.index("V14")
    z = (X.iloc[0]["V14"] - scaler.mean_[j]) / scaler.scale_[j]
    assert contrib.iloc[0]["V14"] == pytest.approx(lr.coef_[0][j] * z)


def test_explain_row_top_k_sorted_by_absolute_contribution(calibrated, data) -> None:  # type: ignore[no-untyped-def]
    X, _ = data
    out = explain_row(calibrated, X.iloc[[0]], top_k=5)
    mags = [abs(c["contribution"]) for c in out["contributions"]]
    assert len(mags) == 5 and mags == sorted(mags, reverse=True)
    assert out["units"] == "log-odds, uncalibrated model"
    full, _ = linear_contributions(calibrated, X.iloc[[0]])
    assert mags[0] == pytest.approx(full.iloc[0].abs().max())
    with pytest.raises(ValueError, match="exactly one row"):
        explain_row(calibrated, X.iloc[:2])


def test_global_importance_ranks_features(calibrated, data) -> None:  # type: ignore[no-untyped-def]
    X, _ = data
    contrib, _ = linear_contributions(calibrated, X)
    imp = global_importance(contrib, calibrated)
    assert len(imp) == len(MODEL_FEATURES)
    assert imp["mean_abs_contribution"].is_monotonic_decreasing
    assert imp.iloc[0]["feature"] == "V14"  # the only feature that separates the classes
