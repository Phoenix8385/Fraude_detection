"""Two-threshold decision policy (docs/architecture.md, docs/evaluation.md "Decision policy").

    p >= t_block            -> HIGH   / HOLD
    p >= t_review           -> MEDIUM / REVIEW
    otherwise               -> LOW    / APPROVE

If t_block does not exist (None), there is no HIGH tier. Pure functions, no I/O.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

HIGH: tuple[str, str] = ("HIGH", "HOLD")
MEDIUM: tuple[str, str] = ("MEDIUM", "REVIEW")
LOW: tuple[str, str] = ("LOW", "APPROVE")


def decide(p: float, t_review: float, t_block: float | None) -> tuple[str, str]:
    """(risk_tier, recommended_action) for one fraud probability."""
    if t_block is not None and p >= t_block:
        return HIGH
    if p >= t_review:
        return MEDIUM
    return LOW


def tier_summary(
    y: ArrayLike, proba: ArrayLike, t_review: float, t_block: float | None
) -> dict[str, dict[str, Any]]:
    """Rows and frauds landing in each tier."""
    y_arr = np.asarray(y).astype(int)
    tiers = [decide(float(p), t_review, t_block)[0] for p in np.asarray(proba, dtype=float)]
    tiers_arr = np.asarray(tiers)
    out = {}
    for tier, action in (HIGH, MEDIUM, LOW):
        mask = tiers_arr == tier
        out[tier] = {
            "action": action,
            "rows": int(mask.sum()),
            "fraud": int(y_arr[mask].sum()),
            "per_1000": 1000 * float(mask.mean()) if len(mask) else 0.0,
        }
    return out
