"""JSON logs: one line per prediction, privacy-safe.

Only whitelisted fields are ever written (rule 17): request id, model version, fraud
probability, tier, action, latency and Amount. Raw V1..V28 values are never logged.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

PREDICTION_LOGGER = "api.predictions"
ALLOWED_FIELDS: frozenset[str] = frozenset({
    "request_id", "model_version", "fraud_probability", "risk_tier", "recommended_action",
    "latency_ms", "amount", "endpoint", "batch_size",
})  # fmt: skip


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON line; only ALLOWED_FIELDS from `fields` are kept."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None) or {}
        payload.update({k: v for k, v in fields.items() if k in ALLOWED_FIELDS})
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    """Send the api.* loggers to stderr as JSON lines (idempotent)."""
    root = logging.getLogger("api")
    if not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.setLevel(level)


def log_prediction(**fields: Any) -> None:
    """Log one prediction event; unknown keys are dropped (never feature values)."""
    safe = {k: v for k, v in fields.items() if k in ALLOWED_FIELDS}
    logging.getLogger(PREDICTION_LOGGER).info("prediction", extra={"fields": safe})
