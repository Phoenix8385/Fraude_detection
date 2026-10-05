"""Pydantic v2 request/response contract for /v1 (docs/architecture.md).

Transaction: Time (>= 0), Amount (>= 0), V1..V28, optional transaction_id (<= 64 chars).
Extra fields, missing fields, NaN and +/-inf are rejected (422).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from api.settings import MAX_BATCH

_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False)

Tier = Literal["HIGH", "MEDIUM", "LOW"]
Action = Literal["HOLD", "REVIEW", "APPROVE"]


class _TransactionBase(BaseModel):
    model_config = _STRICT

    transaction_id: str | None = Field(default=None, max_length=64)
    Time: float = Field(ge=0, description="Seconds since the first transaction in the dataset")
    Amount: float = Field(ge=0, description="Transaction amount")


# V1..V28 (anonymised PCA components) generated rather than written out 28 times.
Transaction = create_model(
    "Transaction",
    __base__=_TransactionBase,
    **{f"V{i}": (float, Field(description="Anonymised PCA component")) for i in range(1, 29)},
)


class BatchRequest(BaseModel):
    model_config = _STRICT

    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH)  # type: ignore[valid-type]


class Thresholds(BaseModel):
    review: float
    block: float | None


class PredictionResponse(BaseModel):
    transaction_id: str | None
    fraud_probability: float = Field(description="Calibrated (isotonic) probability")
    risk_tier: Tier
    recommended_action: Action
    is_flagged: bool = Field(description="True when fraud_probability >= thresholds.review")
    thresholds: Thresholds
    model_version: str
    latency_ms: float


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    model_version: str
    latency_ms: float


class Contribution(BaseModel):
    feature: str
    value: float = Field(description="The caller's own input value, echoed back")
    contribution: float = Field(description="Log-odds contribution (uncalibrated model)")


class ExplainResponse(PredictionResponse):
    base_value: float = Field(description="Intercept: log-odds at the training mean")
    log_odds: float = Field(description="Uncalibrated log-odds = base_value + all contributions")
    units: str
    note: str
    contributions: list[Contribution]


class ModelInfo(BaseModel):
    model_version: str
    model_name: str
    headline_split: str
    calibration: str
    thresholds: Thresholds
    t_review_validation_fallback: bool
    features: list[str]
    schema_hash: str
    explanations: str
    final_test_metrics: dict


class Health(BaseModel):
    status: Literal["ok"]


class Ready(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    model_version: str | None
    db_ok: bool
    store: str
    store_failures: int


class ErrorDetail(BaseModel):
    code: int
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
