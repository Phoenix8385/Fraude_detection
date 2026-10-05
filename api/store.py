"""Prediction store (docs/architecture.md): PredictionStore protocol + NullStore (v1.0).

Phase 11 has NO database: NullStore accepts records and drops them. A SqlStore (Neon) is
Phase 15. Store failures are caught, logged and counted; they never change or fail the API
response (rule 18). Records hold no raw V1..V28 values (rule 17): amount, score, tier,
version, latency and feature NAMES only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

logger = logging.getLogger("api.store")


@dataclass(frozen=True)
class PredictionRecord:
    """One row of the future predictions table (no feature values)."""

    request_id: str
    transaction_id: str | None
    amount: float
    fraud_probability: float
    risk_tier: str
    recommended_action: str
    model_version: str
    latency_ms: float
    top_features: list[str] = field(default_factory=list)  # names only
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class PredictionStore(Protocol):
    """Anything that can persist prediction records."""

    name: str

    def save(self, records: list[PredictionRecord]) -> None: ...


class NullStore:
    """v1.0 store: keeps nothing. Counts records so tests can see it was called."""

    name = "null"

    def __init__(self) -> None:
        self.received = 0

    def save(self, records: list[PredictionRecord]) -> None:
        self.received += len(records)


@dataclass
class StoreHealth:
    """Failure counter shown by /ready."""

    failures: int = 0


def safe_save(store: PredictionStore, records: list[PredictionRecord], health: StoreHealth) -> None:
    """Save without ever raising: a store failure must not fail a prediction."""
    try:
        store.save(records)
    except Exception as exc:  # noqa: BLE001 - any store error is swallowed by design
        health.failures += 1
        logger.error("store %s failed: %s", getattr(store, "name", "?"), type(exc).__name__)
