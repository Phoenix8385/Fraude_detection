"""Cost-based threshold selection (docs/evaluation.md, "Threshold rule").

Pure functions: build a table of metrics for every candidate threshold, then pick the
cheapest threshold that still meets the recall target. Used on valid_thr only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from fraud_detection.metrics import compute_metrics

DEFAULT_GRID: np.ndarray = np.round(np.arange(0.01, 0.96, 0.01), 2)  # 0.01 ... 0.95
SENSITIVITY_REVIEW_COSTS: tuple[float, ...] = (2.0, 5.0, 20.0)

TABLE_COLUMNS = [
    "threshold", "recall", "precision", "f1", "alerts_per_1000", "tp", "fp", "fn",
    "fraud_amount_caught_pct", "missed_fraud_cost", "review_cost_total", "expected_cost",
]  # fmt: skip


def threshold_table(
    y: ArrayLike,
    proba: ArrayLike,
    amounts: ArrayLike,
    review_cost: float,
    grid: ArrayLike = DEFAULT_GRID,
) -> pd.DataFrame:
    """One row per threshold, every number computed by metrics.compute_metrics.

    expected_cost is split into its two parts:
    missed_fraud_cost (sum of Amount of missed fraud) + review_cost_total (review_cost x alerts).
    """
    rows = []
    for t in np.asarray(grid, dtype=float):
        m = compute_metrics(y, proba, float(t), amounts, review_cost)
        review_total = review_cost * (m["tp"] + m["fp"])
        rows.append(
            {
                **{k: m[k] for k in TABLE_COLUMNS if k in m},
                "review_cost_total": review_total,
                "missed_fraud_cost": m["expected_cost"] - review_total,
            }
        )
    return pd.DataFrame(rows)[TABLE_COLUMNS]


def choose_threshold(table: pd.DataFrame, target_recall: float) -> dict[str, Any]:
    """Pick the minimum-expected-cost threshold with recall >= target_recall.

    If no threshold reaches the target, fall back to the highest-recall threshold
    (cheapest among ties) and flag met_target=False so it gets documented.
    Remaining ties are broken by the lower threshold.

    Returns the chosen table row as a dict plus "met_target" and "rule".
    """
    qualifying = table[table["recall"] >= target_recall]
    if len(qualifying):
        pool, met, rule = qualifying, True, f"min expected_cost s.t. recall >= {target_recall}"
    else:
        pool = table[table["recall"] == table["recall"].max()]
        met, rule = False, f"no threshold reaches recall {target_recall}; max recall, min cost"
    best = pool.sort_values(["expected_cost", "threshold"]).iloc[0]
    return {**best.to_dict(), "met_target": met, "rule": rule, "target_recall": target_recall}


def choose_block_threshold(
    table: pd.DataFrame, min_precision: float, min_tp: int
) -> dict[str, Any]:
    """t_block: the SMALLEST threshold with precision >= min_precision and >= min_tp true
    positives at or above it (docs/evaluation.md, "Decision policy").

    If no threshold qualifies, t_block does not exist: returns exists=False, threshold=None.
    The rule is never relaxed.
    """
    rule = f"smallest threshold with precision >= {min_precision} and tp >= {min_tp}"
    ok = table[(table["precision"] >= min_precision) & (table["tp"] >= min_tp)]
    if not len(ok):
        return {"exists": False, "threshold": None, "rule": rule}
    best = ok.sort_values("threshold").iloc[0]
    return {"exists": True, **best.to_dict(), "rule": rule}


def threshold_stability(
    y: ArrayLike,
    proba: ArrayLike,
    amounts: ArrayLike,
    review_cost: float,
    target_recall: float,
    block_precision: float,
    block_min_tp: int,
    n_resamples: int,
    seed: int,
    grid: ArrayLike = DEFAULT_GRID,
) -> dict[str, Any]:
    """Re-run both threshold rules on stratified bootstrap resamples; median + IQR.

    Report only: the frozen thresholds are NOT changed by this (docs/evaluation.md).
    """
    from fraud_detection.bootstrap import stratified_resample

    y_arr = np.asarray(y).astype(int)
    p_arr = np.asarray(proba, dtype=float)
    amt = np.asarray(amounts, dtype=float)
    rng = np.random.default_rng(seed)
    review, met, block = [], [], []
    for _ in range(n_resamples):
        idx = stratified_resample(y_arr, rng)
        table = threshold_table(y_arr[idx], p_arr[idx], amt[idx], review_cost, grid)
        chosen = choose_threshold(table, target_recall)
        review.append(chosen["threshold"])
        met.append(chosen["met_target"])
        b = choose_block_threshold(table, block_precision, block_min_tp)
        if b["exists"]:
            block.append(b["threshold"])

    def spread(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        return {"median": float(median), "q25": float(q25), "q75": float(q75)}

    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "t_review": {**(spread(review) or {}), "share_met_target": float(np.mean(met))},
        "t_block": {"spread": spread(block), "share_exists": len(block) / n_resamples},
    }


def cost_sensitivity(
    y: ArrayLike,
    proba: ArrayLike,
    amounts: ArrayLike,
    target_recall: float,
    review_costs: tuple[float, ...] = SENSITIVITY_REVIEW_COSTS,
    grid: ArrayLike = DEFAULT_GRID,
) -> pd.DataFrame:
    """Repeat the threshold choice for several review costs (the EUR 5 is an assumption)."""
    rows = []
    for cost in review_costs:
        choice = choose_threshold(threshold_table(y, proba, amounts, cost, grid), target_recall)
        rows.append({"review_cost": cost, **{k: choice[k] for k in ("met_target", *TABLE_COLUMNS)}})
    return pd.DataFrame(rows)


def plot_cost_curve(table: pd.DataFrame, chosen: dict[str, Any], title: str, path: Path) -> None:
    """Expected cost (and its two parts) vs threshold, with the chosen threshold marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_cost, ax_recall) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_cost.plot(table["threshold"], table["expected_cost"], label="expected cost", lw=2)
    ax_cost.plot(table["threshold"], table["missed_fraud_cost"], "--", label="missed fraud €")
    ax_cost.plot(table["threshold"], table["review_cost_total"], ":", label="review €")
    ax_cost.set_ylabel("EUR (validation)")
    ax_cost.set_title(title)
    ax_cost.legend()

    ax_recall.plot(table["threshold"], table["recall"], label="recall")
    ax_recall.plot(table["threshold"], table["precision"], label="precision")
    ax_recall.axhline(chosen["target_recall"], color="grey", ls="--", lw=1, label="target recall")
    ax_recall.set_xlabel("Threshold")
    ax_recall.set_ylim(0, 1.02)
    ax_recall.legend()

    for ax in (ax_cost, ax_recall):
        ax.axvline(chosen["threshold"], color="red", lw=1)
        ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
