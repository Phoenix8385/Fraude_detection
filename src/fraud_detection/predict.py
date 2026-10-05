"""Predictor: the production artifact + frozen decision policy, loaded once, used per request.

Everything comes from the Phase 10 artifact (models/model.joblib + model_meta.json):
model_version, feature order and the frozen thresholds (t_review, t_block). Nothing is
hard-coded, refitted or recalibrated here.

Per transaction: add_features() -> columns in the artifact's feature order -> calibrated
fraud_probability (isotonic) -> policy.decide() tier/action. is_flagged means an alert is
raised (probability >= t_review), i.e. tier MEDIUM or HIGH.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.artifacts import load_artifact
from fraud_detection.contributions import TOP_K, explain_row
from fraud_detection.features import add_features
from fraud_detection.policy import LOW, decide
from fraud_detection.schema import FEATURE_COLUMNS


class Predictor:
    """Scores transactions with the frozen model and policy."""

    def __init__(self, model: Any, meta: Mapping[str, Any]) -> None:
        self.model = model
        self.meta = dict(meta)
        self.model_version: str = meta["model_version"]
        self.features: list[str] = list(meta["features"])
        self.input_columns: list[str] = list(meta.get("input_columns", FEATURE_COLUMNS))
        self.t_review: float = float(meta["thresholds"]["review"])
        block = meta["thresholds"]["block"]
        self.t_block: float | None = None if block is None else float(block)

    @classmethod
    def from_dir(cls, models_dir: Path) -> Predictor:
        """Load model.joblib + model_meta.json (MD5, features and schema are verified)."""
        model, meta = load_artifact(models_dir)
        return cls(model, meta)

    @property
    def thresholds(self) -> dict[str, float | None]:
        return {"review": self.t_review, "block": self.t_block}

    def frame(self, records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """Raw input dicts -> model matrix in the artifact's exact feature order.

        Keys other than the raw input columns (e.g. transaction_id) are ignored here.
        """
        raw = pd.DataFrame([{c: r[c] for c in self.input_columns} for r in records])
        return add_features(raw.astype(float))[self.features]

    def predict_batch(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """One result dict per record, in input order."""
        if not records:
            return []
        proba = self.model.predict_proba(self.frame(records))[:, 1]
        out = []
        for p in proba:
            tier, action = decide(float(p), self.t_review, self.t_block)
            out.append({
                "fraud_probability": float(p),
                "risk_tier": tier,
                "recommended_action": action,
                "is_flagged": tier != LOW[0],
            })  # fmt: skip
        return out

    def predict_one(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self.predict_batch([record])[0]

    def explain_one(self, record: Mapping[str, Any], top_k: int = TOP_K) -> dict[str, Any]:
        """Prediction + top-k model contributions (log-odds of the UNCALIBRATED LR)."""
        X = self.frame([record])
        return {**self.predict_one(record), **explain_row(self.model, X, top_k=top_k)}
