"""FastAPI risk-scoring service, /v1 core (docs/architecture.md).

    GET  /health              liveness (always 200, no auth)
    GET  /ready               model loaded + store status (503 if no model; no auth)
    GET  /model-info          frozen model metadata (auth)
    POST /v1/predict          one transaction (auth, 60/min per IP)
    POST /v1/predict/batch    up to 500 transactions (auth, 60/min per IP)
    POST /v1/explain          prediction + top-5 model contributions (auth, 20/min per IP)

The model, version and thresholds come from the Phase 10 artifact (models/model.joblib +
model_meta.json); nothing is hard-coded. The artifact is loaded once at startup; if that
fails the service still starts, /ready and the scoring routes return 503.
Errors: {"error": {"code", "message", "request_id"}}. Logs: one JSON line per prediction,
never raw V1..V28 values. Persistence: injected PredictionStore (NullStore in v1.0); a store
failure never fails a prediction.

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException

from api.examples import BATCH_EXAMPLES, TRANSACTION_EXAMPLES
from api.logging_config import configure_logging, log_prediction
from api.schemas import (
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    ExplainResponse,
    Health,
    ModelInfo,
    PredictionResponse,
    Ready,
    Transaction,
)
from api.settings import Settings
from api.store import NullStore, PredictionRecord, PredictionStore, StoreHealth, safe_save
from fraud_detection.predict import Predictor

logger = logging.getLogger("api.main")

EXPLAIN_NOTE = (
    "Contributions are in log-odds of the UNCALIBRATED Logistic Regression (coefficient x "
    "scaled feature value; base_value = intercept). They describe the model's arithmetic, "
    "not causes of fraud. fraud_probability is the calibrated (isotonic) probability."
)
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 422, 429, 503)
}


def _error(request: Request, code: int, message: str) -> JSONResponse:
    rid = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=code,
        content={"error": {"code": code, "message": message, "request_id": rid}},
        headers={"X-Request-ID": rid},
    )


def create_app(
    settings: Settings | None = None,
    predictor_loader: Callable[[Path], Predictor] = Predictor.from_dir,
    store: PredictionStore | None = None,
) -> FastAPI:
    """Build the app. Tests inject settings, a loader and a store; production uses env."""
    settings = settings or Settings.from_env()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.predictor = predictor_loader(settings.models_dir)
            logger.info("model loaded: %s", app.state.predictor.model_version)
        except Exception as exc:  # noqa: BLE001 - start degraded, report via /ready
            app.state.predictor = None
            logger.error("model NOT loaded (%s): %s", type(exc).__name__, exc)
        yield

    app = FastAPI(
        title="Fraud risk-scoring API",
        version="1.0.0",
        description="Risk-scoring API for card transactions (not a real-time fraud platform). "
        "Request examples in these docs are SYNTHETIC.",
        lifespan=lifespan,
    )
    app.state.predictor = None
    app.state.store = store or NullStore()
    app.state.store_health = StoreHealth()
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        )

    @app.middleware("http")
    async def request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        incoming = request.headers.get("X-Request-ID", "")
        rid = incoming if 0 < len(incoming) <= 64 else uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # field locations + messages only: never echo the submitted values back into logs
        parts = [".".join(str(p) for p in e["loc"]) + ": " + e["msg"] for e in exc.errors()]
        return _error(request, 422, "; ".join(parts)[:1000])

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return _error(request, exc.status_code, str(exc.detail))

    @app.exception_handler(RateLimitExceeded)
    async def rate_limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return _error(request, 429, f"rate limit exceeded ({exc.detail})")

    def require_api_key(request: Request) -> None:
        if settings.api_key is None:
            return
        given = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(given.encode(), settings.api_key.encode()):
            raise HTTPException(401, "invalid or missing API key")

    def get_predictor(request: Request) -> Predictor:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(503, "model not loaded")
        return predictor

    def respond(
        predictor: Predictor, tx: Any, result: dict[str, Any], latency_ms: float,
        request: Request, endpoint: str, top: list[str] | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        """Shared response body + privacy-safe log line + store record."""
        rid = request.state.request_id
        log_prediction(request_id=rid, endpoint=endpoint, model_version=predictor.model_version,
                       fraud_probability=result["fraud_probability"],
                       risk_tier=result["risk_tier"],
                       recommended_action=result["recommended_action"],
                       latency_ms=latency_ms, amount=tx.Amount)  # fmt: skip
        record = PredictionRecord(
            request_id=rid, transaction_id=tx.transaction_id, amount=tx.Amount,
            fraud_probability=result["fraud_probability"], risk_tier=result["risk_tier"],
            recommended_action=result["recommended_action"],
            model_version=predictor.model_version, latency_ms=latency_ms,
            top_features=top or [],
        )  # fmt: skip
        safe_save(request.app.state.store, [record], request.app.state.store_health)
        return {
            "transaction_id": tx.transaction_id,
            "fraud_probability": result["fraud_probability"],
            "risk_tier": result["risk_tier"],
            "recommended_action": result["recommended_action"],
            "is_flagged": result["is_flagged"],
            "thresholds": predictor.thresholds,
            "model_version": predictor.model_version,
            "latency_ms": latency_ms,
        }

    @app.get("/health", response_model=Health, tags=["ops"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", response_model=Ready, tags=["ops"], responses={503: {"model": Ready}})
    def ready(request: Request) -> Any:
        predictor = request.app.state.predictor
        st, health_ = request.app.state.store, request.app.state.store_health
        body = {
            "status": "ready" if predictor else "not_ready",
            "model_loaded": predictor is not None,
            "model_version": predictor.model_version if predictor else None,
            "db_ok": health_.failures == 0,
            "store": getattr(st, "name", type(st).__name__),
            "store_failures": health_.failures,
        }
        return body if predictor else JSONResponse(status_code=503, content=body)

    @app.get("/model-info", response_model=ModelInfo, tags=["model"], responses=ERROR_RESPONSES,
             dependencies=[Depends(require_api_key)])  # fmt: skip
    def model_info(predictor: Predictor = Depends(get_predictor)) -> dict[str, Any]:
        m = predictor.meta
        keys = ("model_version", "model_name", "headline_split", "calibration",
                "t_review_validation_fallback", "features", "schema_hash", "explanations",
                "final_test_metrics")  # fmt: skip
        return {**{k: m[k] for k in keys}, "thresholds": predictor.thresholds}

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["v1"],
              responses=ERROR_RESPONSES, dependencies=[Depends(require_api_key)])  # fmt: skip
    @limiter.limit(settings.rate_limit_predict)
    def predict(
        request: Request,
        tx: Transaction = Body(openapi_examples=TRANSACTION_EXAMPLES),  # type: ignore[valid-type]
        predictor: Predictor = Depends(get_predictor),
    ) -> dict[str, Any]:
        start = time.perf_counter()
        result = predictor.predict_one(tx.model_dump())
        latency = round((time.perf_counter() - start) * 1000, 3)
        return respond(predictor, tx, result, latency, request, "predict")

    @app.post("/v1/predict/batch", response_model=BatchResponse, tags=["v1"],
              responses=ERROR_RESPONSES, dependencies=[Depends(require_api_key)])  # fmt: skip
    @limiter.limit(settings.rate_limit_batch)
    def predict_batch(
        request: Request,
        batch: BatchRequest = Body(openapi_examples=BATCH_EXAMPLES),
        predictor: Predictor = Depends(get_predictor),
    ) -> dict[str, Any]:
        start = time.perf_counter()
        results = predictor.predict_batch([t.model_dump() for t in batch.transactions])
        latency = round((time.perf_counter() - start) * 1000, 3)
        per_row = round(latency / len(results), 3)
        rows = [respond(predictor, t, r, per_row, request, "predict_batch")
                for t, r in zip(batch.transactions, results, strict=True)]  # fmt: skip
        return {"predictions": rows, "count": len(rows),
                "model_version": predictor.model_version, "latency_ms": latency}  # fmt: skip

    @app.post("/v1/explain", response_model=ExplainResponse, tags=["v1"],
              responses=ERROR_RESPONSES, dependencies=[Depends(require_api_key)])  # fmt: skip
    @limiter.limit(settings.rate_limit_explain)
    def explain(
        request: Request,
        tx: Transaction = Body(openapi_examples=TRANSACTION_EXAMPLES),  # type: ignore[valid-type]
        predictor: Predictor = Depends(get_predictor),
    ) -> dict[str, Any]:
        start = time.perf_counter()
        result = predictor.explain_one(tx.model_dump(), top_k=5)
        latency = round((time.perf_counter() - start) * 1000, 3)
        top = [c["feature"] for c in result["contributions"]]
        body = respond(predictor, tx, result, latency, request, "explain", top=top)
        return {**body, "base_value": result["base_value"], "log_odds": result["log_odds"],
                "units": result["units"], "note": EXPLAIN_NOTE,
                "contributions": result["contributions"]}  # fmt: skip

    return app


app = create_app()
