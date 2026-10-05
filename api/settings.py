"""Runtime settings, from environment variables only (no secrets in code or config files).

API_KEY          if set, /v1/* and /model-info require header X-API-Key (not /health,
                 /ready, /docs).
ALLOWED_ORIGINS  comma-separated CORS origins; empty = no CORS headers.
Rate limits (Phase 11 contract, per client IP, fixed — not configurable by env):
                 /v1/predict 60/minute, /v1/predict/batch 60/minute, /v1/explain 20/minute.
MODEL_PATH       path to the artifact file model.joblib (model_meta.json must sit next to it).
                 Takes precedence over MODELS_DIR. Tests point it at a synthetic artifact.
MODELS_DIR       folder holding model.joblib + model_meta.json (default: config paths).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from fraud_detection.artifacts import ARTIFACT_NAME
from fraud_detection.config import load_config

RATE_LIMIT_PREDICT = "60/minute"  # Phase 11 contract
RATE_LIMIT_BATCH = "60/minute"
RATE_LIMIT_EXPLAIN = "20/minute"
MAX_BATCH = 500  # docs/architecture.md: POST /v1/predict/batch (max 500)


@dataclass(frozen=True)
class Settings:
    """API settings. Build with from_env() in production; construct directly in tests."""

    models_dir: Path
    api_key: str | None = None
    allowed_origins: list[str] = field(default_factory=list)
    # route-specific limits (contract defaults; overridden only by tests / the benchmark)
    rate_limit_predict: str = RATE_LIMIT_PREDICT
    rate_limit_batch: str = RATE_LIMIT_BATCH
    rate_limit_explain: str = RATE_LIMIT_EXPLAIN

    @classmethod
    def from_env(cls) -> Settings:
        origins = os.environ.get("ALLOWED_ORIGINS", "")
        return cls(
            models_dir=models_dir_from_env(),
            api_key=os.environ.get("API_KEY") or None,
            allowed_origins=[o.strip() for o in origins.split(",") if o.strip()],
        )


def models_dir_from_env() -> Path:
    """MODEL_PATH (artifact file) > MODELS_DIR (folder) > configs/config.yaml models_dir.

    The artifact is always loaded as <folder>/model.joblib with its model_meta.json, so a
    MODEL_PATH naming any other file is rejected rather than silently ignored.
    """
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        path = Path(model_path)
        if path.name != ARTIFACT_NAME:
            raise ValueError(f"MODEL_PATH must point to {ARTIFACT_NAME}, got {path.name}")
        return path.parent
    models_dir = os.environ.get("MODELS_DIR")
    return Path(models_dir) if models_dir else load_config().paths.models_dir
