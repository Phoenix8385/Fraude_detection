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
    params = {} if name == "dummy" else {"random_state": 0}
    spw = 30.0 if name == "xgb_weighted" else None
    proba = build_model(name, params, scale_pos_weight=spw).fit(X, y).predict_proba(X)
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


# ---------------------------------------------------------------------------
# Phase 6: XGBoost variants
# ---------------------------------------------------------------------------
FAST = {"random_state": 0, "n_estimators": 10}


def test_xgb_smote_resamples_train_only(
    xy: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SMOTE sits inside the pipeline: the model trains on MORE rows than train,
    but predict_proba returns exactly one row per validation row (no resampling)."""
    from xgboost import XGBClassifier

    X_train, y_train = xy  # 310 rows, 10 fraud (SMOTE needs >= 6 fraud for 5 neighbours)
    X_valid = np.random.default_rng(1).normal(size=(37, 4))  # different size on purpose

    rows_seen: list[int] = []
    original_fit = XGBClassifier.fit

    def spy_fit(self, X_fit, y_fit, *args, **kwargs):  # type: ignore[no-untyped-def]
        rows_seen.append(len(X_fit))
        return original_fit(self, X_fit, y_fit, *args, **kwargs)

    monkeypatch.setattr(XGBClassifier, "fit", spy_fit)

    model = build_model("xgb_smote", FAST).fit(X_train, y_train)
    proba = model.predict_proba(X_valid)

    assert rows_seen[0] > len(X_train)
    n_legit = int((y_train == 0).sum())
    assert rows_seen[0] == 2 * n_legit  # "auto" = oversample fraud up to the legit count
    assert proba.shape == (len(X_valid), 2)


def test_xgb_smote_10_ratio(
    xy: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    from xgboost import XGBClassifier

    X, y = xy
    seen_y: list[np.ndarray] = []
    original_fit = XGBClassifier.fit

    def spy_fit(self, X_fit, y_fit, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen_y.append(np.asarray(y_fit))
        return original_fit(self, X_fit, y_fit, *args, **kwargs)

    monkeypatch.setattr(XGBClassifier, "fit", spy_fit)
    build_model("xgb_smote_10", FAST).fit(X, y)  # 300 legit, 10 fraud
    assert int(seen_y[0].sum()) == 30  # fraud raised to 10% of 300 legit


def test_xgb_variants_share_base_params() -> None:
    from fraud_detection.models import XGB_BASE_PARAMS, XGB_NAMES

    for name in XGB_NAMES:
        spw = 5.0 if name == "xgb_weighted" else None
        params = build_model(name, {"random_state": 0}, scale_pos_weight=spw)
        xgb_params = params.named_steps["model"].get_params()
        for key, value in XGB_BASE_PARAMS.items():
            assert xgb_params[key] == value, (name, key)
        assert xgb_params["random_state"] == 0


def test_xgb_weighted_requires_and_uses_weight() -> None:
    with pytest.raises(ValueError, match="requires scale_pos_weight"):
        build_model("xgb_weighted", FAST)
    model = build_model("xgb_weighted", FAST, scale_pos_weight=12.5)
    assert model.named_steps["model"].get_params()["scale_pos_weight"] == 12.5
    assert build_model("xgb", FAST).named_steps["model"].get_params()["scale_pos_weight"] is None


@pytest.mark.parametrize("name", ["xgb", "xgb_weighted", "xgb_smote", "xgb_smote_10"])
def test_xgb_variants_deterministic(name: str, xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = xy
    spw = 30.0 if name == "xgb_weighted" else None
    a = build_model(name, FAST, scale_pos_weight=spw).fit(X, y).predict_proba(X)
    b = build_model(name, FAST, scale_pos_weight=spw).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
