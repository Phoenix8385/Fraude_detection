"""Bootstrap uncertainty (docs/evaluation.md, "Uncertainty").

Stratified bootstrap: every resample draws fraud rows and legit rows separately, with
replacement, keeping both class counts fixed. With only ~57-95 validation frauds an
unstratified resample could change the fraud count noticeably and inflate the spread.

Paired comparison: every model is scored on the SAME resampled indices, so the per-resample
difference between two models removes the "which rows were drawn" noise they share.

Pure functions: no I/O. Every metric goes through metrics.compute_metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from fraud_detection.metrics import compute_metrics

# Headline metrics that get a CI (docs/evaluation.md, "Uncertainty").
BOOTSTRAP_METRICS: tuple[str, ...] = (
    "pr_auc", "recall", "precision", "alerts_per_1000", "expected_cost",
)  # fmt: skip


def stratified_resample(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row positions of one stratified bootstrap resample (class counts unchanged)."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate(
        [rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)]
    )


def bootstrap_samples(
    y_true: ArrayLike,
    probas: Mapping[str, ArrayLike],
    thresholds: Mapping[str, float],
    amounts: ArrayLike,
    review_cost: float,
    n_resamples: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Metric values on each resample, for every model, on SHARED resample indices.

    Args:
        y_true: labels (both classes must be present).
        probas: model name -> predicted fraud probability per row.
        thresholds: model name -> alert threshold for the threshold-dependent metrics.
        amounts: transaction Amount per row.
        review_cost: cost per alert.
        n_resamples: number of bootstrap resamples.
        seed: seeds the resampling, so results are reproducible.

    Returns:
        model name -> DataFrame with n_resamples rows and BOOTSTRAP_METRICS columns.
    """
    y = np.asarray(y_true).astype(int)
    amt = np.asarray(amounts, dtype=float)
    p = {name: np.asarray(values, dtype=float) for name, values in probas.items()}
    if not 0 < y.sum() < len(y):
        raise ValueError("bootstrap needs both classes in y_true")
    if set(p) != set(thresholds):
        raise ValueError("probas and thresholds must name the same models")

    rng = np.random.default_rng(seed)
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in p}
    for _ in range(n_resamples):
        idx = stratified_resample(y, rng)  # one draw, shared by every model -> paired
        for name, proba in p.items():
            m = compute_metrics(y[idx], proba[idx], thresholds[name], amt[idx], review_cost)
            rows[name].append({k: m[k] for k in BOOTSTRAP_METRICS})
    return {name: pd.DataFrame(r, columns=list(BOOTSTRAP_METRICS)) for name, r in rows.items()}


def percentile_ci(values: ArrayLike, level: float) -> tuple[float, float]:
    """Percentile interval, e.g. level 0.95 -> (2.5th, 97.5th percentile)."""
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(np.asarray(values, dtype=float), [tail, 100.0 - tail])
    return float(low), float(high)


def summarize_ci(
    point: Mapping[str, float], samples: pd.DataFrame, level: float
) -> dict[str, dict[str, float]]:
    """{metric: {"point", "ci_low", "ci_high"}}; point = metric on the full (unresampled) data."""
    out: dict[str, dict[str, float]] = {}
    for metric in samples.columns:
        low, high = percentile_ci(samples[metric], level)
        out[metric] = {"point": float(point[metric]), "ci_low": low, "ci_high": high}
    return out


def paired_difference(
    samples_a: ArrayLike,
    samples_b: ArrayLike,
    point_a: float,
    point_b: float,
    level: float,
) -> dict[str, Any]:
    """Paired bootstrap of (a - b) for one metric.

    prob_a_better is the share of resamples where a beats b (a > b). The CI "includes 0"
    when ci_low <= 0 <= ci_high, i.e. the data cannot tell the two models apart.
    """
    diff = np.asarray(samples_a, dtype=float) - np.asarray(samples_b, dtype=float)
    low, high = percentile_ci(diff, level)
    return {
        "diff": float(point_a - point_b),
        "ci_low": low,
        "ci_high": high,
        "ci_level": level,
        "prob_a_better": float(np.mean(diff > 0)),
        "ci_includes_zero": bool(low <= 0.0 <= high),
        "n_resamples": int(len(diff)),
    }
