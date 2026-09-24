"""Experiment runner: fit on TRAIN, score on VALIDATION, log to MLflow + experiments.csv.

The test split is never read here (only evaluate.py may, in Phase 9).

    python -m fraud_detection.train --models dummy logreg iforest --splits stratified time
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from fraud_detection.config import Config, load_config
from fraud_detection.features import MODEL_FEATURES
from fraud_detection.metrics import compute_metrics
from fraud_detection.models import MODEL_NAMES, XGB_NAMES, build_model
from fraud_detection.schema import TARGET

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "fraud-detection"
SPLIT_NAMES: tuple[str, ...] = ("stratified", "time")
VALIDATION_THRESHOLD = 0.5  # fixed for baseline comparison; tuned threshold comes in Phase 8
SEEDED_MODELS = {"logreg", "iforest", *XGB_NAMES}
# MLflow saves sklearn models with skops, which only loads allow-listed types.
# Only the exact types our models need are trusted (found with skops.io.get_untrusted_types).
SKOPS_TRUSTED_TYPES = [
    "imblearn.pipeline.Pipeline",
    "fraud_detection.models.IsolationForestScorer",
    "sklearn.tree._tree.Tree",  # IsolationForest internals
    "xgboost.sklearn.XGBClassifier",
    "xgboost.core.Booster",
    "imblearn.over_sampling._smote.base.SMOTE",
    "sklearn.neighbors._kd_tree.KDTree",  # SMOTE's fitted nearest-neighbour index
    "sklearn.metrics._dist_metrics.EuclideanDistance64",
]


def train_scale_pos_weight(y_train: pd.Series) -> float:
    """n_legit / n_fraud in the training labels (XGBoost's recommended imbalance weight)."""
    n_fraud = int(y_train.sum())
    if n_fraud == 0:
        raise ValueError("training labels contain no fraud; cannot compute scale_pos_weight")
    return (len(y_train) - n_fraud) / n_fraud


def load_part(split_name: str, part: str, processed_dir: Path) -> pd.DataFrame:
    """Read one split part from parquet. Refuses to load the test part."""
    if part == "test":
        raise PermissionError("the test split may only be loaded by evaluate.py (Phase 9)")
    return pd.read_parquet(processed_dir / f"{split_name}_{part}.parquet")


def setup_mlflow(config: Config) -> None:
    """Point MLflow at the project's SQLite store and select the experiment."""
    mlflow.set_tracking_uri(f"sqlite:///{config.paths.mlflow_db.as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=config.paths.mlflow_artifacts_dir.as_uri()
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def append_experiment_row(row: dict[str, Any], csv_path: Path) -> None:
    """Append one row to the experiments CSV, writing the header on first use."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def run_experiment(
    model_name: str, split_name: str, config: Config | None = None
) -> dict[str, Any]:
    """Fit model_name on {split_name}_train, score {split_name}_valid, log everything.

    Returns the row written to reports/metrics/experiments.csv.
    """
    config = config or load_config()
    train = load_part(split_name, "train", config.paths.processed_dir)
    valid = load_part(split_name, "valid", config.paths.processed_dir)
    X_train, y_train = train[MODEL_FEATURES], train[TARGET]

    params = {"random_state": config.seed} if model_name in SEEDED_MODELS else {}
    # Weight computed from TRAIN labels only: valid/test class balance must not leak in.
    scale_pos_weight = train_scale_pos_weight(y_train) if model_name == "xgb_weighted" else None
    model = build_model(model_name, params, scale_pos_weight=scale_pos_weight)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    phase = "6" if model_name in XGB_NAMES else "5"
    return score_and_log(
        model, model_name, split_name, train, valid, fit_seconds, config, tags={"phase": phase}
    )


def score_and_log(
    model: Any,
    model_name: str,
    split_name: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    fit_seconds: float,
    config: Config,
    tags: dict[str, str],
    extra_params: dict[str, Any] | None = None,
    extra_metrics: dict[str, float] | None = None,
    artifacts: list[Path] | None = None,
) -> dict[str, Any]:
    """Score a FITTED model on validation, log run to MLflow, append to experiments.csv.

    model_name is the label used in MLflow and the CSV (e.g. "xgb_weighted_tuned").
    Returns the CSV row.
    """
    y_train = train[TARGET]
    X_valid, y_valid = valid[MODEL_FEATURES], valid[TARGET]
    proba = model.predict_proba(X_valid)[:, 1]
    metrics = compute_metrics(
        y_valid,
        proba,
        threshold=VALIDATION_THRESHOLD,
        amounts=valid["Amount"],
        review_cost=config.costs.review_cost_per_alert,
    )

    with mlflow.start_run(run_name=f"{model_name}_{split_name}") as run:
        mlflow.set_tags({**tags, "model": model_name, "split": split_name})
        mlflow.log_params(
            {
                "model": model_name,
                "split": split_name,
                "n_features": len(MODEL_FEATURES),
                "n_train": len(train),
                "n_valid": len(valid),
                "train_fraud": int(y_train.sum()),
                "valid_fraud": int(y_valid.sum()),
                "review_cost": config.costs.review_cost_per_alert,
                **{
                    f"{step}__{k}": v
                    for step, estimator in model.named_steps.items()
                    for k, v in estimator.get_params().items()
                },
                **(extra_params or {}),
            }
        )
        mlflow.log_metrics({**metrics, **(extra_metrics or {}), "fit_seconds": fit_seconds})
        for path in artifacts or []:
            mlflow.log_artifact(str(path))
        sample = train[MODEL_FEATURES].head(100)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(sample, model.predict(sample)),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        run_id = run.info.run_id

    row = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": run_id,
        "model": model_name,
        "split": split_name,
        "fit_seconds": round(fit_seconds, 2),
        **metrics,
    }
    append_experiment_row(row, config.paths.metrics_dir / "experiments.csv")
    logger.info(
        "%s on %s: valid PR-AUC=%.4f recall=%.3f precision=%.3f",
        model_name, split_name, metrics["pr_auc"], metrics["recall"], metrics["precision"],
    )  # fmt: skip
    return row


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run baseline experiments (validation only).")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    setup_mlflow(config)

    rows = [run_experiment(m, s, config) for s in args.splits for m in args.models]
    columns = ["model", "split", "pr_auc", "roc_auc", "precision", "recall", "f1",
               "alerts_per_1000", "expected_cost", "fit_seconds"]  # fmt: skip
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(rows)[columns].to_string(index=False))


if __name__ == "__main__":
    main()
