"""SYNTHETIC request examples for the OpenAPI docs only.

These are NOT real transactions and were NOT copied from any dataset split (CLAUDE.md:
raw V1..V28 values are never stored). Values are hand-written placeholders chosen only to
show the request shape; no outcome or score is claimed for them.
"""

from __future__ import annotations

from typing import Any

_V = [f"V{i}" for i in range(1, 29)]


def _transaction(transaction_id: str, time: float, amount: float, **v: float) -> dict[str, Any]:
    return {"transaction_id": transaction_id, "Time": time, "Amount": amount,
            **{name: v.get(name, 0.0) for name in _V}}  # fmt: skip


SYNTHETIC_TYPICAL = _transaction("synthetic-0001", 3600.0, 25.0)
SYNTHETIC_UNUSUAL = _transaction("synthetic-0002", 86400.0, 1.0, V4=4.0, V10=-5.0,
                                 V12=-6.0, V14=-8.0)  # fmt: skip

TRANSACTION_EXAMPLES: dict[str, dict[str, Any]] = {
    "synthetic_typical": {
        "summary": "SYNTHETIC example (all V = 0)",
        "description": "Synthetic placeholder, not a real transaction.",
        "value": SYNTHETIC_TYPICAL,
    },
    "synthetic_unusual": {
        "summary": "SYNTHETIC example (a few large V values)",
        "description": "Synthetic placeholder, not a real transaction; no score is implied.",
        "value": SYNTHETIC_UNUSUAL,
    },
}
BATCH_EXAMPLES: dict[str, dict[str, Any]] = {
    "synthetic_batch": {
        "summary": "SYNTHETIC batch of 2",
        "description": "Synthetic placeholders, not real transactions.",
        "value": {"transactions": [SYNTHETIC_TYPICAL, SYNTHETIC_UNUSUAL]},
    }
}
