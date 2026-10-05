"""Phase 8 — calibration and the two-threshold policy, on VALIDATION only.

For each split, the FROZEN Phase 7 champion (reports/metrics/champion_{split}.json) is
reloaded from MLflow — no retraining, no re-selection. Then (docs/evaluation.md):

1. VALID is halved: time split -> chronological halves (earlier = valid_cal, later =
   valid_thr); stratified split -> stratified halves. Row indices are saved. Warn if a
   half has < 30 frauds.
2. Calibration candidates none / sigmoid / isotonic. Calibrators are fitted on valid_cal
   only; the model itself is never refitted. Every candidate is scored on valid_thr only
   (Brier, log-loss, ECE 10 bins, PR-AUC). Selection: lowest Brier; if the best calibrated
   candidate is within 5% of 'none', keep 'none'.
3. On valid_thr, with the selected probabilities (policy.py):
   t_review = min expected cost s.t. recall >= 0.85 (fallback: highest-recall threshold);
   t_block = smallest threshold with precision >= 0.90 and >= 5 TP, else None.
4. Threshold stability (200 stratified bootstrap resamples, median + IQR) and cost
   sensitivity (review cost 2 / 5 / 20) are reported; they never change the frozen thresholds.

The test split is never read.

    python -m fraud_detection.calibrate --splits stratified time
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # file output only, no window

import matplotlib.pyplot as plt  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from fraud_detection.config import Config, load_config  # noqa: E402
from fraud_detection.features import MODEL_FEATURES  # noqa: E402
from fraud_detection.metrics import calibration_metrics  # noqa: E402
from fraud_detection.policy import choose_t_block, choose_t_review, tier_table  # noqa: E402
from fraud_detection.schema import TARGET  # noqa: E402
from fraud_detection.threshold import (  # noqa: E402
    DEFAULT_GRID,
    cost_sensitivity,
    plot_cost_curve,
    threshold_stability,
    threshold_table,
)
from fraud_detection.train import SPLIT_NAMES, load_part, setup_mlflow  # noqa: E402

try:  # sklearn >= 1.6
    from sklearn.frozen import FrozenEstimator  # noqa: E402

    CALIBRATION_API = (
        f"sklearn {sklearn.__version__}: CalibratedClassifierCV(FrozenEstimator(champion), "
        "method=...) — ensemble='auto' -> one calibrator on all valid_cal predictions"
    )
except ImportError:  # pragma: no cover - older sklearn
    FrozenEstimator = None
    CALIBRATION_API = (
        f"sklearn {sklearn.__version__}: CalibratedClassifierCV(champion, method=..., cv='prefit')"
    )

logger = logging.getLogger(__name__)

CANDIDATES: tuple[str, ...] = ("none", "sigmoid", "isotonic")


def split_validation(
    valid: pd.DataFrame, split_name: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Halve VALID into (valid_cal, valid_thr).

    time: chronological halves — earlier rows calibrate, later rows pick thresholds, so
    thresholds are chosen on data that comes after everything the model has seen.
    stratified: stratified 50/50 halves with the config seed.
    """
    if split_name == "time":
        ordered = valid.sort_values("Time", kind="stable")
        half = len(ordered) // 2
        return ordered.iloc[:half], ordered.iloc[half:]
    cal, thr = train_test_split(valid, test_size=0.5, stratify=valid[TARGET], random_state=seed)
    return cal, thr


def half_summary(
    cal: pd.DataFrame, thr: pd.DataFrame, min_positives: int
) -> tuple[dict[str, Any], list[str]]:
    """Rows / frauds / Time range per half, plus a warning per half below min_positives."""
    summary, warnings = {}, []
    for name, part in (("valid_cal", cal), ("valid_thr", thr)):
        fraud = int(part[TARGET].sum())
        summary[name] = {
            "rows": len(part),
            "fraud": fraud,
            "time_min": float(part["Time"].min()),
            "time_max": float(part["Time"].max()),
        }
        if fraud < min_positives:
            warnings.append(f"{name} has only {fraud} frauds (< {min_positives})")
    return summary, warnings


def fit_calibrator(model: Any, X_cal: pd.DataFrame, y_cal: pd.Series, method: str) -> Any:
    """Fit a sigmoid or isotonic calibrator on valid_cal without refitting the model.

    sklearn >= 1.6: FrozenEstimator (ensemble="auto" -> ONE calibrator fitted on the model's
    predictions for all of valid_cal; the model's fit() is a no-op). Older: cv="prefit".
    """
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    else:  # pragma: no cover - older sklearn
        calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")
    return calibrator.fit(X_cal, y_cal)


def select_calibration(briers: dict[str, float], tolerance: float) -> dict[str, Any]:
    """Lowest Brier wins, unless it improves on 'none' by no more than `tolerance` (relative).

    "Within 5% of 'none' -> 'none'": a calibrator is kept only if its Brier is MORE than
    5% lower than the uncalibrated Brier. Exactly 5% counts as "within".
    """
    best = min(briers, key=lambda m: briers[m])
    improvement = (briers["none"] - briers[best]) / briers["none"] if briers["none"] else 0.0
    # isclose guards float noise such as 0.05000000000000004 at the boundary
    beats = improvement > tolerance and not math.isclose(improvement, tolerance)
    selected = best if best != "none" and beats else "none"
    if best == "none":
        how = "'none' has the lowest Brier"
    elif selected == "none":
        how = f"'{best}' is lowest but only {improvement:.1%} below 'none' (within {tolerance:.0%})"
    else:
        how = f"'{best}' is lowest and {improvement:.1%} below 'none' (more than {tolerance:.0%})"
    return {
        "selected": selected,
        "lowest_brier": best,
        "relative_improvement_vs_none": float(improvement),
        "rule": f"lowest Brier on valid_thr; within {tolerance:.0%} of 'none' -> 'none'",
        "how_applied": how,
    }


def plot_reliability(
    y: np.ndarray, probas: dict[str, np.ndarray], briers: dict[str, float], title: str, path: Path
) -> None:
    """Reliability curves of every calibration candidate on ONE figure (valid_thr, 10 bins)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    for name, proba in probas.items():
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "o-", label=f"{name} (Brier {briers[name]:.6f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of fraud")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def grid_diagnostics(table: pd.DataFrame, y: pd.Series, proba: np.ndarray) -> dict[str, Any]:
    """Facts about what the 0.01-0.95 grid can reach on valid_thr (report only)."""
    fraud = np.asarray(y).astype(int) == 1
    return {
        "grid": f"{DEFAULT_GRID.min():.2f}-{DEFAULT_GRID.max():.2f} step 0.01",
        "max_recall_on_grid": float(table["recall"].max()),
        "max_precision_on_grid": float(table["precision"].max()),
        "fraud_scored_below_grid_min": int(np.sum(fraud & (proba < DEFAULT_GRID.min()))),
        "fraud_total": int(fraud.sum()),
    }


def run_phase8(split: str, config: Config) -> dict[str, Any]:
    """Calibrate the frozen Phase 7 champion and freeze t_review / t_block for one split."""
    metrics_dir, figures_dir = config.paths.metrics_dir, config.paths.figures_dir
    champion = json.loads((metrics_dir / f"champion_{split}.json").read_text(encoding="utf-8"))
    model_key = champion["decision"]["champion"]
    run_id = champion["models"][model_key]["mlflow_run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    logger.info("%s: frozen champion %s (run %s)", split, model_key, run_id)

    valid = load_part(split, "valid", config.paths.processed_dir)
    cal, thr = split_validation(valid, split, config.seed)
    halves, warnings = half_summary(cal, thr, config.calibration.min_half_positives)
    for w in warnings:
        logger.warning("%s: %s", split, w)
    (metrics_dir / f"valid_subsplit_indices_{split}.json").write_text(
        json.dumps({"cal_indices": cal.index.tolist(), "thr_indices": thr.index.tolist()})
        + "\n",
        encoding="utf-8",
    )

    # Calibrators see valid_cal only; every candidate is scored on valid_thr only.
    X_thr, y_thr = thr[MODEL_FEATURES], thr[TARGET]
    fitted: dict[str, Any] = {"none": model}
    for method in ("sigmoid", "isotonic"):
        fitted[method] = fit_calibrator(model, cal[MODEL_FEATURES], cal[TARGET], method)
    probas = {name: m.predict_proba(X_thr)[:, 1] for name, m in fitted.items()}
    scores = {name: calibration_metrics(y_thr, p, n_bins=10) for name, p in probas.items()}
    briers = {name: s["brier"] for name, s in scores.items()}
    selection = select_calibration(briers, config.calibration.none_tolerance)
    method = selection["selected"]
    proba = probas[method]
    logger.info("%s: %s -> calibration %s", split, selection["how_applied"], method)

    review_cost = config.costs.review_cost_per_alert
    amounts = thr["Amount"]
    t_review = choose_t_review(y_thr, proba, amounts, review_cost, config.target_recall)
    t_block = choose_t_block(
        y_thr, proba, config.policy.block_precision, config.policy.block_min_tp
    )
    if t_review["fallback"]:
        warnings.append(
            f"FALLBACK: no grid threshold reaches recall {config.target_recall} on valid_thr; "
            "t_review = highest-recall threshold (cheapest among ties)"
        )
    if not t_block["exists"]:
        warnings.append("t_block does not exist (None): no HIGH/HOLD tier")
    elif t_block["threshold"] < t_review["threshold"]:
        warnings.append("t_block < t_review: the MEDIUM/REVIEW tier is empty")

    table = threshold_table(y_thr, proba, amounts, review_cost)
    tiers = tier_table(y_thr, proba, t_review["threshold"], t_block["threshold"])
    sensitivity = cost_sensitivity(y_thr, proba, amounts, config.target_recall)
    stability = threshold_stability(
        y_thr, proba, amounts, review_cost, config.target_recall,
        config.policy.block_precision, config.policy.block_min_tp,
        config.policy.stability_resamples, config.seed,
    )  # fmt: skip

    table.to_csv(metrics_dir / f"threshold_table_{split}.csv", index=False, float_format="%.6f")
    tiers.to_csv(metrics_dir / f"tier_table_{split}.csv", index=False, float_format="%.6f")
    sensitivity.to_csv(
        metrics_dir / f"cost_sensitivity_{split}.csv", index=False, float_format="%.4f"
    )
    plot_reliability(
        y_thr.to_numpy(), probas, briers,
        f"Reliability — {split} split, {model_key} (valid_thr, fraud = {int(y_thr.sum())})",
        figures_dir / f"reliability_{split}.png",
    )  # fmt: skip
    plot_cost_curve(
        table, {**t_review, "target_recall": config.target_recall},
        f"Cost vs threshold — {split}, {model_key}, calibration={method} (valid_thr)",
        figures_dir / f"cost_vs_threshold_{split}.png",
    )  # fmt: skip

    calibration = {
        "split": split,
        "model": model_key,
        "mlflow_run_id": run_id,
        "api": CALIBRATION_API,
        "fitted_on": "valid_cal",
        "scored_on": "valid_thr",
        "halves": halves,
        "candidates_valid_thr": scores,
        **selection,
    }
    (metrics_dir / f"calibration_{split}.json").write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )
    policy = {
        "split": split,
        "model": model_key,
        "mlflow_run_id": run_id,
        "calibration": method,
        "chosen_on": "valid_thr",
        "valid_thr_rows": halves["valid_thr"]["rows"],
        "valid_thr_fraud": halves["valid_thr"]["fraud"],
        "review_cost": review_cost,
        "t_review": t_review["threshold"],
        "t_block": t_block["threshold"],
        # operating point at t_review
        "recall": t_review["recall"],
        "precision": t_review["precision"],
        "alerts_per_1000": t_review["alerts_per_1000"],
        "expected_cost": t_review["expected_cost"],
        "t_review_detail": t_review,
        "t_block_detail": t_block,
        "tiers_valid_thr": tiers.to_dict(orient="records"),
        "stability": stability,
        "cost_sensitivity": sensitivity.to_dict(orient="records"),
        "grid_diagnostics": grid_diagnostics(table, y_thr, proba),
        "warnings": warnings,
    }
    (metrics_dir / f"policy_{split}.json").write_text(
        json.dumps(policy, indent=2) + "\n", encoding="utf-8"
    )
    model_path = config.paths.models_dir / f"final_model_{split}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[method], model_path)
    logger.info("%s: saved policy_%s.json and %s", split, split, model_path.name)
    return {"calibration": calibration, "policy": policy}


def summary_line(result: dict[str, Any]) -> str:
    """One readable block per split for the CLI (ASCII only, Windows console safe)."""
    c, p = result["calibration"], result["policy"]
    tr, tb = p["t_review_detail"], p["t_block_detail"]
    block = (
        f"t_block {tb['threshold']:.2f} (precision {tb['precision']:.3f}, tp {tb['tp']:.0f})"
        if tb["exists"]
        else "t_block DOES NOT EXIST (None)"
    )
    briers = ", ".join(f"{k} {v['brier']:.6f}" for k, v in c["candidates_valid_thr"].items())
    return (
        f"{p['split']}: model {p['model']}, calibration {c['selected']} (Brier {briers})\n"
        f"  t_review {tr['threshold']:.2f}: recall {tr['recall']:.3f}, "
        f"precision {tr['precision']:.3f}, cost EUR {tr['expected_cost']:.2f}, "
        f"fallback {tr['fallback']}\n"
        f"  {block}\n"
        f"  warnings: {p['warnings'] or 'none'}"
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Phase 8: calibration + two thresholds.")
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    setup_mlflow(config)
    logger.info("Calibration API: %s", CALIBRATION_API)
    for split in args.splits:
        print(summary_line(run_phase8(split, config)))


if __name__ == "__main__":
    main()
