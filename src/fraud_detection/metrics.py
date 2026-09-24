"""The one trusted metrics function. Every reported number goes through compute_metrics.

Pure: no I/O, no global state. See docs/evaluation.md for definitions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_div(numerator: float, denominator: float) -> float:
    """numerator / denominator, or 0.0 when the denominator is 0 (e.g. no alerts)."""
    return float(numerator / denominator) if denominator else 0.0


def compute_metrics(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    threshold: float,
    amounts: ArrayLike,
    review_cost: float,
) -> dict[str, Any]:
    """Score fraud probabilities against true labels.

    Args:
        y_true: 1 = fraud, 0 = legit.
        y_proba: predicted fraud probability (or any score in [0, 1]).
        threshold: an alert is raised when y_proba >= threshold.
        amounts: transaction Amount per row, used for money-based metrics.
        review_cost: cost of an analyst reviewing one alert.

    Returns:
        Dict of metrics. pr_auc / roc_auc are threshold-free (ranking quality);
        everything else depends on the threshold. pr_auc and roc_auc are NaN when
        y_true contains only one class, because they are undefined then.
    """
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    amt = np.asarray(amounts, dtype=float)
    if not (len(y) == len(proba) == len(amt)):
        raise ValueError("y_true, y_proba and amounts must have the same length")

    alert = proba >= threshold
    fraud = y == 1
    tp = int(np.sum(alert & fraud))
    fp = int(np.sum(alert & ~fraud))
    tn = int(np.sum(~alert & ~fraud))
    fn = int(np.sum(~alert & fraud))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    both_classes = 0 < fraud.sum() < len(y)

    return {
        "pr_auc": float(average_precision_score(y, proba)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y, proba)) if both_classes else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "specificity": _safe_div(tn, tn + fp),
        "fpr": _safe_div(fp, fp + tn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "alerts_per_1000": _safe_div(1000 * (tp + fp), len(y)),
        "fraud_amount_caught_pct": 100 * _safe_div(amt[alert & fraud].sum(), amt[fraud].sum()),
        "expected_cost": float(amt[~alert & fraud].sum() + review_cost * (tp + fp)),
        "threshold": float(threshold),
    }
