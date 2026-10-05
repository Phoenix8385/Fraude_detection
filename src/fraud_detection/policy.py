"""Two-threshold decision policy (docs/architecture.md, docs/evaluation.md "Decision policy").

    t_review = min expected cost s.t. recall >= target (0.85) on the 0.01-0.95 grid;
               if no threshold reaches the target: the highest-recall threshold (cheapest
               among ties), reported as a FALLBACK.
    t_block  = SMALLEST threshold with precision >= 0.90 and >= 5 true positives;
               if none qualifies: None (no HIGH tier). Never relaxed.

    p >= t_block            -> HIGH   / HOLD
    p >= t_review           -> MEDIUM / REVIEW
    otherwise               -> LOW    / APPROVE
Equality meets a threshold. Pure functions, no I/O; threshold metrics come from
threshold.threshold_table (i.e. metrics.compute_metrics).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from fraud_detection.threshold import (
    DEFAULT_GRID,
    choose_block_threshold,
    choose_threshold,
    threshold_table,
)

HIGH: tuple[str, str] = ("HIGH", "HOLD")
MEDIUM: tuple[str, str] = ("MEDIUM", "REVIEW")
LOW: tuple[str, str] = ("LOW", "APPROVE")

OPERATING_KEYS: tuple[str, ...] = (
    "threshold", "recall", "precision", "f1", "alerts_per_1000", "tp", "fp", "fn",
    "fraud_amount_caught_pct", "expected_cost",
)  # fmt: skip


def choose_t_review(
    y: ArrayLike,
    p: ArrayLike,
    amounts: ArrayLike,
    review_cost: float,
    target_recall: float,
    grid: ArrayLike = DEFAULT_GRID,
) -> dict[str, Any]:
    """t_review and its operating point; fallback=True when the recall target is unreachable."""
    chosen = choose_threshold(threshold_table(y, p, amounts, review_cost, grid), target_recall)
    return {
        **{k: chosen[k] for k in OPERATING_KEYS},
        "met_target": bool(chosen["met_target"]),
        "fallback": not chosen["met_target"],
        "target_recall": target_recall,
        "rule": chosen["rule"],
    }


def choose_t_block(
    y: ArrayLike,
    p: ArrayLike,
    min_precision: float = 0.90,
    min_tp: int = 5,
    grid: ArrayLike = DEFAULT_GRID,
) -> dict[str, Any]:
    """t_block (or threshold=None with exists=False). Amounts/costs do not affect it."""
    n = len(np.asarray(y))
    table = threshold_table(y, p, np.zeros(n), review_cost=0.0, grid=grid)
    chosen = choose_block_threshold(table, min_precision, min_tp)
    if not chosen["exists"]:
        return {"exists": False, "threshold": None, "rule": chosen["rule"]}
    keys = ("threshold", "recall", "precision", "f1", "alerts_per_1000", "tp", "fp", "fn")
    return {"exists": True, **{k: chosen[k] for k in keys}, "rule": chosen["rule"]}


def decide(p: float, t_review: float, t_block: float | None) -> tuple[str, str]:
    """(risk_tier, recommended_action) for one fraud probability."""
    if t_block is not None and p >= t_block:
        return HIGH
    if p >= t_review:
        return MEDIUM
    return LOW


def tier_table(
    y: ArrayLike, p: ArrayLike, t_review: float, t_block: float | None
) -> pd.DataFrame:
    """One row per tier: count, frauds, fraud share, share of all fraud, rows per 1,000.

    fraud_share = frauds in the tier / rows in the tier (how pure the tier is);
    share_of_all_fraud = frauds in the tier / all frauds (how much fraud lands there).
    """
    y_arr = np.asarray(y).astype(int)
    tiers = np.asarray([decide(float(v), t_review, t_block)[0] for v in np.asarray(p, float)])
    n, n_fraud = len(y_arr), int(y_arr.sum())
    rows = []
    for tier, action in (HIGH, MEDIUM, LOW):
        mask = tiers == tier
        count, fraud = int(mask.sum()), int(y_arr[mask].sum())
        rows.append(
            {
                "tier": tier,
                "action": action,
                "count": count,
                "fraud": fraud,
                "fraud_share": fraud / count if count else 0.0,
                "share_of_all_fraud": fraud / n_fraud if n_fraud else 0.0,
                "alerts_per_1000": 1000 * count / n if n else 0.0,
            }
        )
    return pd.DataFrame(rows)
