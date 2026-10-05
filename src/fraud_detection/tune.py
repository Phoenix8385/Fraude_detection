"""Hyperparameter tuning with Optuna: cross-validation on the TRAIN part only, scored by PR-AUC.

Two models are tuned (docs/evaluation.md, "Champion rule"):
- "xgb_u": XGBoost whose class weight is scale_pos_weight = (n_legit / n_fraud) ** u.
  u is searched in [0, 1]: u = 0 is plain XGBoost, u = 1 is the Phase 6 "xgb_weighted".
  The ratio is recomputed from each CV fold's own training rows, so held-out fold labels
  never influence the weight.
- "logreg": scaled Logistic Regression; C and class_weight are searched.

The Phase 5/6 untuned configurations are enqueued as the first trials, so the search
always evaluates them. VALIDATION is used once afterwards, to score the refitted best
model. The test split is never read.

    python -m fraud_detection.tune --models xgb_u logreg --splits stratified time
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, TimeSeriesSplit

from fraud_detection.config import Config, load_config
from fraud_detection.features import MODEL_FEATURES
from fraud_detection.metrics import compute_metrics
from fraud_detection.models import XGB_BASE_PARAMS, build_model
from fraud_detection.schema import TARGET
from fraud_detection.train import (
    SPLIT_NAMES,
    load_part,
    score_and_log,
    setup_mlflow,
    train_scale_pos_weight,
)

logger = logging.getLogger(__name__)

TUNED_MODELS: tuple[str, ...] = ("xgb_u", "logreg")

# Untuned Phase 5/6 configurations, evaluated first in every study.
BASELINE_TRIALS: dict[str, list[dict[str, Any]]] = {
    "xgb_u": [
        {
            **{k: XGB_BASE_PARAMS[k] for k in
               ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree")},
            "min_child_weight": 1.0,  # XGBoost default
            "reg_lambda": 1.0,  # XGBoost default
            "u": u,
        }
        for u in (1.0, 0.0)  # Phase 6 xgb_weighted, Phase 6 plain xgb
    ],
    "logreg": [{"C": 1.0, "class_weight": "balanced"}],  # Phase 5 logreg
}  # fmt: skip


def suggest_params(trial: optuna.Trial, model_key: str) -> dict[str, Any]:
    """Search space for one model (unprefixed parameter names, plus "u" for xgb_u)."""
    if model_key == "xgb_u":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "u": trial.suggest_float("u", 0.0, 1.0),
        }
    if model_key == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
    raise ValueError(f"no search space for '{model_key}'; expected one of {TUNED_MODELS}")


def build_tuned(model_key: str, params: dict[str, Any], y_fit: pd.Series, seed: int) -> Pipeline:
    """Unfitted pipeline for a tuned configuration.

    For xgb_u the class ratio comes from y_fit, the labels of the rows it will be fitted on.
    """
    if model_key == "xgb_u":
        xgb_params = {k: v for k, v in params.items() if k != "u"}
        spw = train_scale_pos_weight(y_fit) ** params["u"]
        return build_model(
            "xgb_weighted", {**xgb_params, "random_state": seed}, scale_pos_weight=spw
        )
    if model_key == "logreg":
        model = build_model("logreg", {"random_state": seed, "C": params["C"]})
        return model.set_params(model__class_weight=params["class_weight"])
    raise ValueError(f"unknown tuned model '{model_key}'; expected one of {TUNED_MODELS}")


def make_cv(split_name: str, seed: int, n_splits: int) -> BaseCrossValidator:
    """Stratified split -> shuffled StratifiedKFold; time split -> TimeSeriesSplit.

    TimeSeriesSplit always validates on rows LATER than the ones it trains on, which
    mirrors how the time split itself was built. It requires time-ordered rows.
    """
    if split_name == "time":
        return TimeSeriesSplit(n_splits=n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cv_pr_auc(
    model_key: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    split_name: str,
    seed: int,
    n_splits: int,
) -> list[float]:
    """PR-AUC on each CV fold of TRAIN for one configuration."""
    if split_name == "time":
        train = train.sort_values("Time", kind="stable")  # TimeSeriesSplit needs time order
    X, y, amounts = train[MODEL_FEATURES], train[TARGET], train["Amount"]
    scores = []
    for fit_idx, hold_idx in make_cv(split_name, seed, n_splits).split(X, y):
        model = build_tuned(model_key, params, y.iloc[fit_idx], seed)
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        proba = model.predict_proba(X.iloc[hold_idx])[:, 1]
        # pr_auc is threshold-free; threshold and review cost do not affect it
        m = compute_metrics(y.iloc[hold_idx], proba, 0.5, amounts.iloc[hold_idx], 0.0)
        scores.append(m["pr_auc"])
    return scores


def run_study(
    model_key: str,
    train: pd.DataFrame,
    split_name: str,
    seed: int,
    n_trials: int,
    n_splits: int,
) -> optuna.Study:
    """Maximise mean CV PR-AUC on TRAIN with a seeded TPE sampler."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"{model_key}_{split_name}",
    )
    for params in BASELINE_TRIALS[model_key][:n_trials]:
        study.enqueue_trial(params)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_key)
        scores = cv_pr_auc(model_key, params, train, split_name, seed, n_splits)
        trial.set_user_attr("cv_scores", scores)
        trial.set_user_attr("cv_std", float(np.std(scores)))
        return float(np.mean(scores))

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        logger.info(
            "%s/%s trial %d/%d: cv PR-AUC %.4f (best %.4f)",
            model_key, split_name, trial.number + 1, n_trials, trial.value, study.best_value,
        )  # fmt: skip

    study.optimize(objective, n_trials=n_trials, callbacks=[log_trial])
    return study


def run_tuning(
    model_key: str, split_name: str, config: Config, n_trials: int | None = None
) -> dict[str, Any]:
    """Tune one model on one split, refit on all of TRAIN, score VALIDATION once, save JSON."""
    n_trials = n_trials or config.tuning.n_trials
    n_folds = config.tuning.cv_folds
    train = load_part(split_name, "train", config.paths.processed_dir)
    valid = load_part(split_name, "valid", config.paths.processed_dir)

    logger.info("Tuning %s on %s: %d trials x %d folds", model_key, split_name, n_trials, n_folds)
    start = time.perf_counter()
    study = run_study(model_key, train, split_name, config.seed, n_trials, n_folds)
    search_seconds = time.perf_counter() - start

    best = study.best_trial
    params = dict(best.params)
    cv_std = float(best.user_attrs["cv_std"])
    model = build_tuned(model_key, params, train[TARGET], config.seed)
    fit_start = time.perf_counter()
    model.fit(train[MODEL_FEATURES], train[TARGET])
    fit_seconds = time.perf_counter() - fit_start

    with tempfile.TemporaryDirectory() as tmp:
        trials_csv = Path(tmp) / f"optuna_trials_{model_key}_{split_name}.csv"
        study.trials_dataframe().to_csv(trials_csv, index=False)
        row = score_and_log(
            model,
            f"{model_key}_tuned",
            split_name,
            train,
            valid,
            fit_seconds=fit_seconds,
            config=config,
            tags={"phase": "7", "stage": "tuned", "tuner": "optuna", "base_model": model_key},
            extra_params={
                "cv": type(make_cv(split_name, config.seed, n_folds)).__name__,
                "cv_folds": n_folds,
                "n_trials": n_trials,
                "best_params": json.dumps(params),
            },
            extra_metrics={
                "cv_pr_auc_mean": float(best.value),
                "cv_pr_auc_std": cv_std,
                "search_seconds": search_seconds,
            },
            artifacts=[trials_csv],
        )
        (config.paths.metrics_dir / trials_csv.name).write_bytes(trials_csv.read_bytes())

    result = {
        "model": model_key,
        "split": split_name,
        "best_params": params,
        "best_trial": best.number,
        "cv": type(make_cv(split_name, config.seed, n_folds)).__name__,
        "cv_folds": n_folds,
        "n_trials": n_trials,
        "sampler": f"TPESampler(seed={config.seed})",
        "cv_pr_auc_mean": float(best.value),
        "cv_pr_auc_std": cv_std,
        "cv_pr_auc_folds": best.user_attrs["cv_scores"],
        "valid_pr_auc": row["pr_auc"],
        "search_seconds": round(search_seconds, 1),
        "mlflow_run_id": row["run_id"],
    }
    if model_key == "xgb_u":
        result["scale_pos_weight"] = train_scale_pos_weight(train[TARGET]) ** params["u"]
    out = config.paths.metrics_dir / f"best_params_{model_key}_{split_name}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %s", out)
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Tune models with Optuna (train CV, PR-AUC).")
    parser.add_argument("--models", nargs="+", choices=TUNED_MODELS, default=list(TUNED_MODELS))
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES))
    parser.add_argument("--n-trials", type=int, default=None, help="default: config tuning")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # our callback logs each trial
    config = load_config()
    setup_mlflow(config)

    for split in args.splits:
        for model_key in args.models:
            r = run_tuning(model_key, split, config, args.n_trials)
            print(
                f"{r['model']:7s} {r['split']:10s} cv PR-AUC {r['cv_pr_auc_mean']:.4f} "
                f"± {r['cv_pr_auc_std']:.4f}  valid PR-AUC {r['valid_pr_auc']:.4f}  "
                f"params {json.dumps(r['best_params'])}"
            )


if __name__ == "__main__":
    main()
