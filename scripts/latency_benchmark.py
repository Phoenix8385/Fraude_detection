"""Latency benchmark: N sequential POST /v1/predict calls through the in-process TestClient.

Uses the real production artifact (models/model.joblib) and a SYNTHETIC request body (no real
transaction rows). Reports end-to-end client latency (validation + model + serialisation,
no network) as p50 / p95 / p99 in milliseconds. PRD target: p95 < 50 ms locally.

    python scripts/latency_benchmark.py [--n 1000]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root: import api

from api.examples import SYNTHETIC_TYPICAL  # noqa: E402
from api.main import create_app  # noqa: E402
from api.settings import Settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    logging.getLogger("api").setLevel(logging.WARNING)  # keep per-request log lines quiet
    # the benchmark measures latency, not the rate limiter: lift the limit for this app only
    lifted = f"{args.n * 10}/minute"
    settings = dataclasses.replace(Settings.from_env(), rate_limit_predict=lifted)
    app = create_app(settings)
    times = []
    with TestClient(app) as client:
        if client.get("/ready").status_code != 200:
            raise SystemExit("model not loaded; run python -m fraud_detection.artifacts first")
        version = client.get("/ready").json()["model_version"]
        for _ in range(args.warmup):
            client.post("/v1/predict", json=SYNTHETIC_TYPICAL)
        for _ in range(args.n):
            start = time.perf_counter()
            r = client.post("/v1/predict", json=SYNTHETIC_TYPICAL)
            times.append((time.perf_counter() - start) * 1000)
            r.raise_for_status()
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    print(json.dumps({"model_version": version, "n": args.n, "p50_ms": round(p50, 2),
                      "p95_ms": round(p95, 2), "p99_ms": round(p99, 2),
                      "max_ms": round(max(times), 2)}))  # fmt: skip


if __name__ == "__main__":
    main()
