"""Tests for fraud_detection.models (synthetic data only)."""

import numpy as np
import pytest

from fraud_detection.models import MODEL_NAMES, build_model


@pytest.fixture
def xy() -> tuple[np.ndarray, np.ndarray]:
    """300 normal points around 0 plus 10 far-away 'fraud' points around 6."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, size=(300, 4)), rng.normal(6, 1, size=(10, 4))])
    y = np.array([0] * 300 + [1] * 10)
    return X, y


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predict_proba_shape_and_range(name: str, xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = xy
    params = {"random_state": 0} if name in ("logreg", "iforest") else {}
    proba = build_model(name, params).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_dummy_never_flags_fraud(xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = xy
    proba = build_model("dummy").fit(X, y).predict_proba(X)[:, 1]
    assert np.all(proba == 0.0)


def test_iforest_ignores_labels(xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = xy
    a = build_model("iforest", {"random_state": 0}).fit(X, y).predict_proba(X)
    b = build_model("iforest", {"random_state": 0}).fit(X, np.zeros_like(y)).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_iforest_scores_outliers_higher(xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = xy
    score = build_model("iforest", {"random_state": 0}).fit(X, y).predict_proba(X)[:, 1]
    assert score[y == 1].mean() > score[y == 0].mean()


def test_iforest_scaling_uses_train_range(xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = xy
    model = build_model("iforest", {"random_state": 0}).fit(X, y)
    train_score = model.predict_proba(X)[:, 1]
    assert train_score.min() == pytest.approx(0.0) and train_score.max() == pytest.approx(1.0)
    # a point far beyond anything seen in training is clipped to 1, not > 1
    assert model.predict_proba(np.full((1, 4), 100.0))[0, 1] == 1.0


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("xgboost_typo")


def test_scale_pos_weight_rejected_for_baselines() -> None:
    with pytest.raises(ValueError, match="scale_pos_weight"):
        build_model("logreg", scale_pos_weight=10.0)
