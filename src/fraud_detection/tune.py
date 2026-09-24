"""Hyperparameter tuning: RandomizedSearchCV on the TRAIN part only, scored by PR-AUC.

Cross-validation happens inside TRAIN; VALIDATION is used once afterwards to score the
refitted best model (same protocol as Phase 6, so tuned vs untuned is comparable).
The test split is never read.

    python -m fraud_detection.tune --models xgb_weighted xgb --splits stratified time
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
)

from fraud_detection.config import Config, load_config
from fraud_detection.features import MODEL_FEATURES
from fraud_detection.models import XGB_NAMES, build_model
from fraud_detection.schema import TARGET
from fraud_detection.train import (
    SPLIT_NAMES,
    load_part,
    score_and_log,
    setup_mlflow,
    train_scale_pos_weight,
)

logger = logging.getLogger(__name__)

DEFAULT_MODELS: tuple[str, ...] = ("xgb_weighted", "xgb")  # top-2 tunable in Phase 6
N_FOLDS = 5
N_ITER = 20

# Unprefixed XGBoost parameter grid. Every model is a Pipeline whose estimator step is
# called "model", so each key becomes "model__<name>" for the search.
SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [200, 400, 700],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}


def pipeline_space(space: dict[str, list[Any]] = SEARCH_SPACE) -> dict[str, list[Any]]:
    """Prefix parameter names with the pipeline step name ("model__")."""
    return {f"model__{name}": values for name, values in space.items()}


def make_cv(split_name: str, seed: int, n_splits: int = N_FOLDS) -> BaseCrossValidator:
    """Stratified split → shuffled StratifiedKFold; time split → TimeSeriesSplit.

    TimeSeriesSplit always validates on rows LATER than the ones it trains on, which
    mirrors how the time split itself was built. It requires time-ordered rows.
    """
    if split_name == "time":
        return TimeSeriesSplit(n_splits=n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def search(
    model_name: str,
    train: pd.DataFrame,
    split_name: str,
    seed: int,
    n_iter: int = N_ITER,
    n_splits: int = N_FOLDS,
    space: dict[str, list[Any]] = SEARCH_SPACE,
) -> RandomizedSearchCV:
    """Run the randomized search on TRAIN rows and refit the best params on all of TRAIN."""
    if model_name not in XGB_NAMES:
        raise ValueError(f"tuning is defined for XGBoost models only, got '{model_name}'")
    if split_name == "time":
        train = train.sort_values("Time", kind="stable")  # TimeSeriesSplit needs time order
    X, y = train[MODEL_FEATURES], train[TARGET]

    # For xgb_weighted the weight comes from the full TRAIN part (see memory.md caveat).
    spw = train_scale_pos_weight(y) if model_name == "xgb_weighted" else None
    pipeline = build_model(model_name, {"random_state": seed}, scale_pos_weight=spw)

    searcher = RandomizedSearchCV(
        pipeline,
        param_distributions=pipeline_space(space),
        n_iter=n_iter,
        scoring="average_precision",
        cv=make_cv(split_name, seed, n_splits),
        random_state=seed,
        refit=True,  # refit best params on the whole TRAIN part
        n_jobs=1,  # XGBoost already uses all cores
        error_score="raise",
    )
    searcher.fit(X, y)
    return searcher


def best_params_unprefixed(searcher: RandomizedSearchCV) -> dict[str, Any]:
    """best_params_ without the "model__" prefix, e.g. {"max_depth": 4, ...}."""
    return {k.removeprefix("model__"): v for k, v in searcher.best_params_.items()}


def untuned_valid_pr_auc(model_name: str, split_name: str, experiments_csv: Path) -> float | None:
    """Latest Phase 6 (untuned) validation PR-AUC for this model/split, if recorded."""
    if not experiments_csv.exists():
        return None
    df = pd.read_csv(experiments_csv)
    rows = df[(df["model"] == model_name) & (df["split"] == split_name)]
    return float(rows.sort_values("timestamp")["pr_auc"].iloc[-1]) if len(rows) else None


def run_tuning(
    model_name: str, split_name: str, config: Config, n_iter: int = N_ITER
) -> dict[str, Any]:
    """Tune one model on one split, score validation, log, save best params JSON."""
    train = load_part(split_name, "train", config.paths.processed_dir)
    valid = load_part(split_name, "valid", config.paths.processed_dir)
    experiments_csv = config.paths.metrics_dir / "experiments.csv"
    untuned = untuned_valid_pr_auc(model_name, split_name, experiments_csv)

    logger.info(
        "Tuning %s on %s: %d candidates x %d folds", model_name, split_name, n_iter, N_FOLDS
    )
    start = time.perf_counter()
    searcher = search(model_name, train, split_name, config.seed, n_iter=n_iter)
    search_seconds = time.perf_counter() - start

    best = searcher.best_index_
    cv_mean = float(searcher.cv_results_["mean_test_score"][best])
    cv_std = float(searcher.cv_results_["std_test_score"][best])
    params = best_params_unprefixed(searcher)

    with tempfile.TemporaryDirectory() as tmp:
        cv_csv = Path(tmp) / f"cv_results_{model_name}_{split_name}.csv"
        pd.DataFrame(searcher.cv_results_).to_csv(cv_csv, index=False)
        row = score_and_log(
            searcher.best_estimator_,
            f"{model_name}_tuned",
            split_name,
            train,
            valid,
            fit_seconds=float(searcher.refit_time_),
            config=config,
            tags={"phase": "7", "stage": "tuned", "base_model": model_name},
            extra_params={
                "cv": type(searcher.cv).__name__,
                "cv_folds": N_FOLDS,
                "n_iter": n_iter,
                "search_space": json.dumps(SEARCH_SPACE),
            },
            extra_metrics={
                "cv_pr_auc_mean": cv_mean,
                "cv_pr_auc_std": cv_std,
                "search_seconds": search_seconds,
            },
            artifacts=[cv_csv],
        )

    result = {
        "model": model_name,
        "split": split_name,
        "best_params": params,
        "cv": type(searcher.cv).__name__,
        "cv_folds": N_FOLDS,
        "n_iter": n_iter,
        "cv_pr_auc_mean": cv_mean,
        "cv_pr_auc_std": cv_std,
        "valid_pr_auc_tuned": row["pr_auc"],
        "valid_pr_auc_untuned": untuned,
        "search_seconds": round(search_seconds, 1),
        "mlflow_run_id": row["run_id"],
    }
    out = config.paths.metrics_dir / f"best_params_{model_name}_{split_name}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %s", out)
    return result


def summary_markdown(results: list[dict[str, Any]]) -> str:
    """Tuned vs untuned validation PR-AUC table."""
    lines = [
        "# Tuning summary — validation PR-AUC, tuned vs untuned",
        "",
        "Generated by `python -m fraud_detection.tune`. CV on TRAIN only; the refitted best",
        "model is scored once on VALIDATION. Test set not used.",
        "",
        "| model | split | cv | cv PR-AUC (mean ± std) | valid untuned | valid tuned | change |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        untuned = r["valid_pr_auc_untuned"]
        change = f"{r['valid_pr_auc_tuned'] - untuned:+.4f}" if untuned is not None else "n/a"
        untuned_txt = f"{untuned:.4f}" if untuned is not None else "n/a"
        lines.append(
            f"| {r['model']} | {r['split']} | {r['cv']} "
            f"| {r['cv_pr_auc_mean']:.4f} ± {r['cv_pr_auc_std']:.4f} "
            f"| {untuned_txt} | {r['valid_pr_auc_tuned']:.4f} | {change} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Tune XGBoost models (train CV, PR-AUC).")
    parser.add_argument("--models", nargs="+", choices=XGB_NAMES, default=list(DEFAULT_MODELS))
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES))
    parser.add_argument("--n-iter", type=int, default=N_ITER)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    setup_mlflow(config)

    results = [run_tuning(m, s, config, args.n_iter) for s in args.splits for m in args.models]
    md = summary_markdown(results)
    (config.paths.metrics_dir / "tuning_summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
