"""Tests for fraud_detection.tune (tiny synthetic data, no MLflow)."""

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from fraud_detection.features import MODEL_FEATURES
from fraud_detection.schema import TARGET
from fraud_detection.tune import (
    BASELINE_TRIALS,
    TUNED_MODELS,
    build_tuned,
    cv_pr_auc,
    make_cv,
    run_study,
    suggest_params,
)

TINY_XGB = {
    "n_estimators": 10, "max_depth": 2, "learning_rate": 0.1, "subsample": 1.0,
    "colsample_bytree": 1.0, "min_child_weight": 1.0, "reg_lambda": 1.0,
}  # fmt: skip


@pytest.fixture
def tune_df() -> pd.DataFrame:
    """400 rows with all MODEL_FEATURES, 40 fraud spread over time, Time shuffled."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(rng.normal(size=(n, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    df["Time"] = rng.permutation(np.arange(n, dtype="float64"))
    df["Amount"] = rng.uniform(1, 100, size=n)
    y = np.zeros(n, dtype=int)
    y[::10] = 1
    df[TARGET] = y
    df.loc[df[TARGET] == 1, "V14"] -= 3  # make fraud learnable
    return df


@pytest.fixture(autouse=True)
def quiet_optuna() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.mark.parametrize("model_key", TUNED_MODELS)
def test_suggested_params_build_a_valid_pipeline(model_key: str) -> None:
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
    params = suggest_params(trial, model_key)
    y = pd.Series([0] * 9 + [1])
    build_tuned(model_key, params, y, seed=0)  # set_params raises on unknown names


def test_baseline_trials_match_search_space() -> None:
    for model_key, trials in BASELINE_TRIALS.items():
        trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
        assert all(set(t) == set(suggest_params(trial, model_key)) for t in trials)


@pytest.mark.parametrize(("u", "expected"), [(0.0, 1.0), (1.0, 9.0), (0.5, 3.0)])
def test_xgb_u_weight_is_ratio_to_the_power_u(u: float, expected: float) -> None:
    y = pd.Series([0] * 9 + [1])  # 9 legit / 1 fraud
    model = build_tuned("xgb_u", {**TINY_XGB, "u": u}, y, seed=0)
    xgb = model.named_steps["model"]
    assert xgb.get_params()["scale_pos_weight"] == pytest.approx(expected)
    assert "u" not in xgb.get_params()


def test_logreg_class_weight_can_be_none() -> None:
    model = build_tuned("logreg", {"C": 0.5, "class_weight": None}, pd.Series([0, 1]), seed=0)
    lr = model.named_steps["model"]
    assert lr.C == 0.5 and lr.class_weight is None


def test_unknown_model_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tuned model"):
        build_tuned("iforest", {}, pd.Series([0, 1]), seed=0)


def test_make_cv_types() -> None:
    assert isinstance(make_cv("time", 42, 5), TimeSeriesSplit)
    cv = make_cv("stratified", 42, 5)
    assert isinstance(cv, StratifiedKFold) and cv.shuffle and cv.random_state == 42


def test_time_cv_never_trains_on_later_rows(tune_df: pd.DataFrame) -> None:
    ordered = tune_df.sort_values("Time", kind="stable")
    for fit_idx, hold_idx in make_cv("time", 0, 3).split(ordered):
        assert ordered["Time"].iloc[fit_idx].max() < ordered["Time"].iloc[hold_idx].min()


def test_xgb_u_weight_uses_fold_labels_only(
    tune_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each fold's weight comes from that fold's training rows, never the whole train part."""
    import fraud_detection.tune as tune

    seen: list[int] = []
    original = tune.build_tuned

    def spy(model_key, params, y_fit, seed):  # noqa: ANN001, ANN202
        seen.append(len(y_fit))
        return original(model_key, params, y_fit, seed)

    monkeypatch.setattr(tune, "build_tuned", spy)
    cv_pr_auc("xgb_u", {**TINY_XGB, "u": 1.0}, tune_df, "stratified", seed=0, n_splits=4)
    assert seen == [300, 300, 300, 300]  # 3/4 of 400 rows per fold, not 400


def test_cv_scores_are_pr_auc_per_fold(tune_df: pd.DataFrame) -> None:
    scores = cv_pr_auc("logreg", {"C": 1.0, "class_weight": "balanced"}, tune_df,
                       "stratified", seed=0, n_splits=3)  # fmt: skip
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_study_enqueues_baselines_and_is_deterministic(tune_df: pd.DataFrame) -> None:
    a = run_study("xgb_u", tune_df, "time", seed=1, n_trials=3, n_splits=3)
    b = run_study("xgb_u", tune_df, "time", seed=1, n_trials=3, n_splits=3)
    # first two trials are the Phase 6 configs: xgb_weighted (u=1) then plain xgb (u=0)
    assert [t.params["u"] for t in a.trials[:2]] == [1.0, 0.0]
    assert a.best_params == b.best_params
    assert [t.value for t in a.trials] == [t.value for t in b.trials]
    assert len(a.best_trial.user_attrs["cv_scores"]) == 3
