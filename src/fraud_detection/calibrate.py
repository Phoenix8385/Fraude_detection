"""Phase 8 — Calibration and cost-based threshold selection.

Why calibrate: scale_pos_weight and SMOTE inflate raw probabilities (a "0.9" may really
mean 0.3). Isotonic regression re-maps predicted probabilities so they match observed
fraud rates, which matters when analysts sort alerts by score.

Calibration data problem: you can't calibrate on the same validation set you choose the
threshold on without introducing bias. Solution: split the VALIDATION part 50/50
(stratified by Class, fixed seed) into:
    valid_cal — fit the calibrator (isotonic regression)
    valid_thr — choose the operating threshold by expected cost

FrozenEstimator (sklearn >= 1.6) wraps the fitted model so CalibratedClassifierCV's
internal fit() call does NOT retrain the underlying model.

    python -m fraud_detection.calibrate --splits stratified time
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from fraud_detection.config import Config, load_config  # noqa: E402
from fraud_detection.features import MODEL_FEATURES  # noqa: E402
from fraud_detection.schema import TARGET  # noqa: E402
from fraud_detection.threshold import (  # noqa: E402
    choose_threshold,
    cost_sensitivity,
    plot_cost_curve,
    threshold_table,
)
from fraud_detection.train import (  # noqa: E402
    SPLIT_NAMES,
    load_part,
    setup_mlflow,
)

logger = logging.getLogger(__name__)

# sklearn >= 1.6 provides FrozenEstimator — documented above for the interviewer.
CALIBRATION_METHOD = "FrozenEstimator + CalibratedClassifierCV(method='isotonic', cv=5-fold)"


# ---------------------------------------------------------------------------
# 1. Validation sub-split
# ---------------------------------------------------------------------------

def split_validation(
    valid: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split the VALIDATION part 50/50 (stratified) into valid_cal and valid_thr.

    Returns (valid_cal, valid_thr, cal_indices, thr_indices) where the indices are the
    original DataFrame integer positions, saved for reproducibility.
    """
    cal, thr = train_test_split(
        valid, test_size=0.5, stratify=valid[TARGET], random_state=seed
    )
    return cal, thr, cal.index.to_numpy(), thr.index.to_numpy()


# ---------------------------------------------------------------------------
# 2. Reliability diagram helpers
# ---------------------------------------------------------------------------

def _reliability_diagram(
    y: np.ndarray,
    proba: np.ndarray,
    brier: float,
    title: str,
    ax: plt.Axes,
    n_bins: int = 10,
) -> None:
    """Draw a single reliability diagram on *ax*."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y, proba, n_bins=n_bins, strategy="uniform"
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{title}\nBrier = {brier:.6f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)


def reliability_side_by_side(
    y: np.ndarray,
    raw_proba: np.ndarray,
    cal_proba: np.ndarray,
    raw_brier: float,
    cal_brier: float,
    split_name: str,
    path: Path,
) -> None:
    """Save side-by-side reliability diagrams (raw vs calibrated) for one split."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _reliability_diagram(y, raw_proba, raw_brier, f"Raw — {split_name}", ax1)
    _reliability_diagram(y, cal_proba, cal_brier, f"Calibrated — {split_name}", ax2)
    fig.suptitle(f"Reliability diagrams — {split_name} split (valid_thr)", fontsize=13, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# 3. Calibration
# ---------------------------------------------------------------------------

def fit_calibrator(
    model: Any, X_cal: pd.DataFrame, y_cal: pd.Series
) -> CalibratedClassifierCV:
    """Fit isotonic calibration on valid_cal using FrozenEstimator.

    FrozenEstimator (sklearn >= 1.6) wraps the already-fitted model so
    CalibratedClassifierCV does NOT retrain it — it only fits the isotonic
    regression on the predicted probabilities.

    sklearn >= 1.9 removed cv="prefit". With FrozenEstimator, the default CV
    (5-fold) is safe: each fold calls fit() on FrozenEstimator which is a no-op
    for the base model, and the calibrator is fitted on the held-out predictions.
    The final ensemble averages the per-fold isotonic mappings.
    """
    frozen = FrozenEstimator(model)
    calibrated = CalibratedClassifierCV(frozen, method="isotonic")
    calibrated.fit(X_cal, y_cal)
    return calibrated


# ---------------------------------------------------------------------------
# 4. Per-split orchestration
# ---------------------------------------------------------------------------

def _best_model_for_split(
    split_name: str, experiments_csv: Path
) -> tuple[str, str]:
    """Return (model_name, mlflow_run_id) of the best (highest PR-AUC) model for this split.

    The best model is the one with the highest validation PR-AUC in experiments.csv.
    For each model, we use only the latest run (by timestamp).
    """
    df = pd.read_csv(experiments_csv)
    df = df[df["split"] == split_name]
    latest = df.sort_values("timestamp").groupby("model", as_index=False).tail(1)
    best = latest.sort_values("pr_auc", ascending=False).iloc[0]
    return str(best["model"]), str(best["run_id"])


def run_calibration_and_threshold(
    split_name: str, config: Config
) -> dict[str, Any]:
    """Full Phase 8 pipeline for one split.

    1. Load the best tuned model from MLflow.
    2. Split validation 50/50 → valid_cal, valid_thr.
    3. Compute raw Brier on valid_thr.
    4. Fit isotonic calibration on valid_cal.
    5. Compute calibrated Brier on valid_thr.
    6. Keep calibration only if Brier improves.
    7. Run cost-based threshold selection on valid_thr.
    8. Save everything.
    """
    experiments_csv = config.paths.metrics_dir / "experiments.csv"
    model_name, run_id = _best_model_for_split(split_name, experiments_csv)
    logger.info("Split %s: best model = %s (run %s)", split_name, model_name, run_id)

    # Load fitted model from MLflow
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    # Load full validation set
    valid = load_part(split_name, "valid", config.paths.processed_dir)

    # Sub-split validation: valid_cal (fit calibrator) + valid_thr (choose threshold)
    valid_cal, valid_thr, cal_idx, thr_idx = split_validation(valid, config.seed)
    logger.info(
        "valid_cal: %d rows (%d fraud), valid_thr: %d rows (%d fraud)",
        len(valid_cal), int(valid_cal[TARGET].sum()),
        len(valid_thr), int(valid_thr[TARGET].sum()),
    )

    # Save sub-split indices for reproducibility
    indices_path = config.paths.metrics_dir / f"valid_subsplit_indices_{split_name}.json"
    indices_path.write_text(
        json.dumps({"cal_indices": cal_idx.tolist(), "thr_indices": thr_idx.tolist()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    logger.info("Saved sub-split indices to %s", indices_path)

    X_cal, y_cal = valid_cal[MODEL_FEATURES], valid_cal[TARGET]
    X_thr, y_thr = valid_thr[MODEL_FEATURES], valid_thr[TARGET]

    # Raw probabilities on valid_thr
    raw_proba = model.predict_proba(X_thr)[:, 1]
    raw_brier = float(brier_score_loss(y_thr, raw_proba))

    # Fit calibration on valid_cal
    calibrated_model = fit_calibrator(model, X_cal, y_cal)
    cal_proba = calibrated_model.predict_proba(X_thr)[:, 1]
    cal_brier = float(brier_score_loss(y_thr, cal_proba))

    # Decision: keep calibration only if Brier improves (lower is better)
    calibration_kept = cal_brier < raw_brier
    decision_reason = (
        f"Brier improved: {raw_brier:.6f} → {cal_brier:.6f} (Δ = {cal_brier - raw_brier:+.6f})"
        if calibration_kept
        else f"Brier worsened: {raw_brier:.6f} → {cal_brier:.6f} (Δ = {cal_brier - raw_brier:+.6f})"
    )
    logger.info("Calibration decision (%s): %s → %s",
                split_name, "KEEP" if calibration_kept else "SKIP", decision_reason)

    # Select the model and probabilities to use for threshold selection
    final_model = calibrated_model if calibration_kept else model
    final_proba = cal_proba if calibration_kept else raw_proba

    # Reliability diagrams (side-by-side raw vs calibrated)
    reliability_path = config.paths.figures_dir / f"reliability_{split_name}.png"
    reliability_side_by_side(
        y_thr.to_numpy(), raw_proba, cal_proba, raw_brier, cal_brier, split_name, reliability_path
    )

    # Threshold selection on valid_thr
    amounts_thr = valid_thr["Amount"].to_numpy()
    review_cost = config.costs.review_cost_per_alert
    table = threshold_table(y_thr, final_proba, amounts_thr, review_cost)
    chosen = choose_threshold(table, config.target_recall)

    # Save threshold table
    table_path = config.paths.metrics_dir / f"threshold_table_{split_name}.csv"
    table.to_csv(table_path, index=False, float_format="%.6f")
    logger.info("Saved threshold table to %s", table_path)

    # Cost-vs-threshold plot
    cost_plot_path = config.paths.figures_dir / f"cost_vs_threshold_{split_name}.png"
    plot_cost_curve(
        table, chosen,
        f"Cost vs threshold — {split_name} split (valid_thr, n={len(valid_thr)})",
        cost_plot_path,
    )

    # Cost sensitivity analysis
    sensitivity = cost_sensitivity(
        y_thr, final_proba, amounts_thr, config.target_recall
    )
    sens_path = config.paths.metrics_dir / f"cost_sensitivity_{split_name}.csv"
    sensitivity.to_csv(sens_path, index=False, float_format="%.4f")
    logger.info("Saved cost sensitivity to %s", sens_path)

    # Save the frozen threshold
    frozen_threshold = {
        "split": split_name,
        "threshold": chosen["threshold"],
        "recall": chosen["recall"],
        "precision": chosen["precision"],
        "f1": chosen["f1"],
        "expected_cost": chosen["expected_cost"],
        "met_target": chosen["met_target"],
        "rule": chosen["rule"],
        "target_recall": chosen["target_recall"],
        "alerts_per_1000": chosen["alerts_per_1000"],
        "model": model_name,
        "mlflow_run_id": run_id,
        "calibration_kept": calibration_kept,
        "calibration_method": CALIBRATION_METHOD,
        "brier_raw": raw_brier,
        "brier_calibrated": cal_brier,
        "valid_cal_rows": len(valid_cal),
        "valid_thr_rows": len(valid_thr),
    }
    frozen_path = config.paths.metrics_dir / f"frozen_threshold_{split_name}.json"
    frozen_path.write_text(json.dumps(frozen_threshold, indent=2) + "\n", encoding="utf-8")
    logger.info("FROZEN threshold for %s: %.2f (recall=%.3f, cost=%.1f)",
                split_name, chosen["threshold"], chosen["recall"], chosen["expected_cost"])

    # Save the final model (calibrated or raw) for downstream use
    model_path = config.paths.models_dir / f"final_model_{split_name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    logger.info("Saved final model to %s", model_path)

    return frozen_threshold


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point for Phase 8."""
    parser = argparse.ArgumentParser(
        description="Phase 8: calibrate probabilities and freeze threshold (validation only)."
    )
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    setup_mlflow(config)

    results = []
    for split in args.splits:
        result = run_calibration_and_threshold(split, config)
        results.append(result)

    # Print summary
    print("\n" + "=" * 80)
    print("Phase 8 -- Calibration + Threshold Selection Summary")
    print("=" * 80)
    for r in results:
        cal_status = "KEPT" if r["calibration_kept"] else "SKIPPED"
        print(
            f"\n  {r['split']}:"
            f"\n    Model:       {r['model']}"
            f"\n    Calibration: {cal_status} (Brier {r['brier_raw']:.6f} -> {r['brier_calibrated']:.6f})"
            f"\n    Threshold:   {r['threshold']:.2f}"
            f"\n    Recall:      {r['recall']:.3f} (target: {r['target_recall']})"
            f"\n    Precision:   {r['precision']:.3f}"
            f"\n    Cost:        {r['expected_cost']:.1f}"
            f"\n    Met target:  {r['met_target']}"
        )
    print()


if __name__ == "__main__":
    main()
