"""API tests (docs/evaluation.md "Required tests" + "Additional tests").

Synthetic artifact from conftest; the real model, data and MLflow are never used.
"""

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.examples import SYNTHETIC_TYPICAL, TRANSACTION_EXAMPLES
from api.logging_config import JsonFormatter
from api.main import EXPLAIN_NOTE, create_app
from api.settings import MAX_BATCH, Settings
from api.store import NullStore, PredictionRecord


def client(models_dir: Path, **kw) -> TestClient:  # type: ignore[no-untyped-def]
    store = kw.pop("store", None)
    return TestClient(create_app(Settings(models_dir=models_dir, **kw), store=store))


@pytest.fixture
def api(synthetic_artifact_dir):  # type: ignore[no-untyped-def]
    with client(synthetic_artifact_dir) as c:
        yield c


# --- health / readiness / model info -------------------------------------------------------


def test_health_and_ready(api) -> None:  # type: ignore[no-untyped-def]
    assert api.get("/health").json() == {"status": "ok"}
    r = api.get("/ready").json()
    assert r["status"] == "ready" and r["model_version"] == "synthetic-test-v0"
    assert r["store"] == "null" and r["db_ok"] is True


def test_model_info_comes_from_metadata(api) -> None:  # type: ignore[no-untyped-def]
    info = api.get("/model-info").json()
    assert info["model_version"] == "synthetic-test-v0"
    assert info["thresholds"] == {"review": 0.3, "block": 0.7}
    assert "artifact_md5" not in info and "library_versions" not in info


# --- predict --------------------------------------------------------------------------------


def test_predict_200_contract(api, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    r = api.post("/v1/predict", json={**synthetic_transaction, "transaction_id": "t-1"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"transaction_id", "fraud_probability", "risk_tier",
                         "recommended_action", "is_flagged", "thresholds", "model_version",
                         "latency_ms"}  # fmt: skip
    assert body["transaction_id"] == "t-1" and 0 <= body["fraud_probability"] <= 1
    assert body["thresholds"] == {"review": 0.3, "block": 0.7}
    assert r.headers["X-Request-ID"]


def test_high_risk_input_maps_to_hold(api, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    body = api.post("/v1/predict", json={**synthetic_transaction, "V14": -12.0}).json()
    assert (body["risk_tier"], body["recommended_action"], body["is_flagged"]) == (
        "HIGH", "HOLD", True)  # fmt: skip


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.pop("V7"),  # missing field
        lambda t: t.update(V29=1.0),  # extra field
        lambda t: t.update(Amount=-1.0),  # negative amount
        lambda t: t.update(transaction_id="x" * 65),  # id too long
    ],
    ids=["missing", "extra", "negative_amount", "long_id"],
)
def test_invalid_payload_422_with_error_body(api, synthetic_transaction, mutate) -> None:  # type: ignore[no-untyped-def]
    tx = dict(synthetic_transaction)
    mutate(tx)
    r = api.post("/v1/predict", json=tx)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == 422 and err["request_id"] == r.headers["X-Request-ID"]


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_inf_rejected(api, synthetic_transaction, bad) -> None:  # type: ignore[no-untyped-def]
    text = json.dumps(synthetic_transaction).replace('"V5": 0.0', f'"V5": {bad}')
    r = api.post("/v1/predict", content=text, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_validation_error_does_not_echo_values(api, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    r = api.post("/v1/predict", json={**synthetic_transaction, "Amount": -987.654321})
    assert "987.654321" not in r.text


# --- batch ----------------------------------------------------------------------------------


def test_batch_returns_one_prediction_per_row(api, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    rows = [{**synthetic_transaction, "transaction_id": f"b{i}"} for i in range(3)]
    body = api.post("/v1/predict/batch", json={"transactions": rows}).json()
    assert body["count"] == 3 and [p["transaction_id"] for p in body["predictions"]] == [
        "b0", "b1", "b2"]  # fmt: skip


@pytest.mark.parametrize("n", [0, MAX_BATCH + 1])
def test_batch_size_limits(api, synthetic_transaction, n) -> None:  # type: ignore[no-untyped-def]
    r = api.post("/v1/predict/batch", json={"transactions": [synthetic_transaction] * n})
    assert r.status_code == 422


def test_batch_max_is_500(api, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    assert MAX_BATCH == 500
    r = api.post("/v1/predict/batch", json={"transactions": [synthetic_transaction] * 500})
    assert r.status_code == 200 and r.json()["count"] == 500


# --- explain --------------------------------------------------------------------------------


def test_explain_top5_labelled_not_causes(api, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    body = api.post("/v1/explain", json={**synthetic_transaction, "V14": -6.0}).json()
    assert len(body["contributions"]) == 5
    assert body["contributions"][0]["feature"] == "V14"
    assert body["units"] == "log-odds, uncalibrated model" and body["note"] == EXPLAIN_NOTE
    assert "not causes" in body["note"] and "UNCALIBRATED" in body["note"]


# --- model missing / auth / rate limit / CORS -----------------------------------------------


def test_503_without_model(tmp_path: Path, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    with client(tmp_path) as c:  # empty dir: artifact missing
        assert c.get("/health").status_code == 200  # liveness unaffected
        assert c.get("/ready").status_code == 503
        for path in ("/v1/predict", "/v1/explain"):
            r = c.post(path, json=synthetic_transaction)
            assert r.status_code == 503 and r.json()["error"]["message"] == "model not loaded"
        assert c.get("/model-info").status_code == 503


def test_api_key(synthetic_artifact_dir, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    with client(synthetic_artifact_dir, api_key="secret") as c:
        assert c.post("/v1/predict", json=synthetic_transaction).status_code == 401
        wrong = c.post("/v1/predict", json=synthetic_transaction, headers={"X-API-Key": "nope"})
        assert wrong.status_code == 401 and wrong.json()["error"]["code"] == 401
        ok = c.post("/v1/predict", json=synthetic_transaction, headers={"X-API-Key": "secret"})
        assert ok.status_code == 200
        assert c.get("/model-info").status_code == 401
        for open_path in ("/health", "/ready", "/docs", "/openapi.json"):
            assert c.get(open_path).status_code == 200


def test_contract_rate_limit_values() -> None:
    s = Settings(models_dir=Path("."))
    assert (s.rate_limit_predict, s.rate_limit_batch, s.rate_limit_explain) == (
        "60/minute", "60/minute", "20/minute")  # fmt: skip


@pytest.mark.parametrize(
    ("path", "limit"), [("/v1/predict", 60), ("/v1/predict/batch", 60), ("/v1/explain", 20)]
)
def test_route_specific_rate_limits_429(
    synthetic_artifact_dir, synthetic_transaction, path, limit
) -> None:  # type: ignore[no-untyped-def]
    """Contract defaults: the (limit+1)-th call within a minute from one IP gets 429."""
    body = {"transactions": [synthetic_transaction]} if path.endswith("batch") else (
        synthetic_transaction)  # fmt: skip
    with client(synthetic_artifact_dir) as c:  # default Settings = contract limits
        codes = [c.post(path, json=body).status_code for _ in range(limit + 1)]
        assert codes[:limit] == [200] * limit and codes[limit] == 429
        assert c.post(path, json=body).json()["error"]["code"] == 429
        assert c.get("/health").status_code == 200  # ops routes are not limited


def test_rate_limits_are_counted_per_route(synthetic_artifact_dir, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    with client(synthetic_artifact_dir) as c:
        for _ in range(21):
            c.post("/v1/explain", json=synthetic_transaction)
        assert c.post("/v1/explain", json=synthetic_transaction).status_code == 429
        assert c.post("/v1/predict", json=synthetic_transaction).status_code == 200


def test_cors_only_for_allowed_origins(synthetic_artifact_dir) -> None:  # type: ignore[no-untyped-def]
    with client(synthetic_artifact_dir, allowed_origins=["https://dash.example"]) as c:
        ok = c.get("/health", headers={"Origin": "https://dash.example"})
        bad = c.get("/health", headers={"Origin": "https://evil.example"})
        assert ok.headers.get("access-control-allow-origin") == "https://dash.example"
        assert "access-control-allow-origin" not in bad.headers


# --- store: NullStore by injection, failures never fail a prediction ------------------------


class FailingStore:
    name = "failing"

    def save(self, records: list[PredictionRecord]) -> None:
        raise ConnectionError("database down")


def test_store_failure_never_fails_prediction(
    synthetic_artifact_dir, synthetic_transaction
) -> None:  # type: ignore[no-untyped-def]
    with client(synthetic_artifact_dir, store=FailingStore()) as c:
        assert c.post("/v1/predict", json=synthetic_transaction).status_code == 200
        ready = c.get("/ready").json()
        assert ready["store_failures"] == 1 and ready["db_ok"] is False


def test_null_store_receives_records_without_feature_values(
    synthetic_artifact_dir, synthetic_transaction
) -> None:  # type: ignore[no-untyped-def]
    store = NullStore()
    with client(synthetic_artifact_dir, store=store) as c:
        c.post("/v1/predict/batch", json={"transactions": [synthetic_transaction] * 2})
    assert store.received == 2
    fields = set(PredictionRecord.__dataclass_fields__)
    assert not any(f.startswith("V") for f in fields)


# --- logging privacy ------------------------------------------------------------------------


def test_logs_are_json_and_never_contain_feature_values(
    api, synthetic_transaction, caplog
) -> None:  # type: ignore[no-untyped-def]
    marker = 4.56789012345  # distinctive V value that must never appear in logs
    with caplog.at_level(logging.INFO, logger="api"):
        api.post("/v1/predict", json={**synthetic_transaction, "V9": marker, "Amount": 12.5})
        api.post("/v1/explain", json={**synthetic_transaction, "V9": marker})
    lines = [JsonFormatter().format(r) for r in caplog.records if r.name == "api.predictions"]
    assert len(lines) == 2
    for line in lines:
        payload = json.loads(line)
        assert str(marker) not in line and not any(k.startswith("V") for k in payload)
        assert {"request_id", "model_version", "fraud_probability", "risk_tier",
                "latency_ms", "amount"} <= set(payload)  # fmt: skip
    assert json.loads(lines[0])["amount"] == 12.5


# --- OpenAPI examples are synthetic ---------------------------------------------------------


def test_openapi_examples_are_labelled_synthetic(api) -> None:  # type: ignore[no-untyped-def]
    spec = api.get("/openapi.json").json()
    examples = spec["paths"]["/v1/predict"]["post"]["requestBody"]["content"][
        "application/json"]["examples"]  # fmt: skip
    assert set(examples) == set(TRANSACTION_EXAMPLES)
    assert all("SYNTHETIC" in e["summary"] for e in examples.values())
    assert SYNTHETIC_TYPICAL["transaction_id"].startswith("synthetic-")


# --- Phase 12: log privacy with distinctive markers (all loggers, success + validation error) -

MARKERS = {"V1": 987654.321, "V2": -123456.789}
MARKER_TEXT = ("987654.321", "123456.789", "987654", "123456")


def _all_log_text(caplog) -> str:  # type: ignore[no-untyped-def]
    """Every captured record from EVERY logger, raw message + args + JSON rendering."""
    fmt = JsonFormatter()
    return "\n".join(f"{r.getMessage()} {r.args} {getattr(r, 'fields', '')} {fmt.format(r)}"
                     for r in caplog.records)  # fmt: skip


@pytest.mark.parametrize("path", ["/v1/predict", "/v1/explain", "/v1/predict/batch"])
def test_marker_feature_values_never_reach_logs(api, synthetic_transaction, caplog, path) -> None:  # type: ignore[no-untyped-def]
    tx = {**synthetic_transaction, **MARKERS}
    body = {"transactions": [tx]} if path.endswith("batch") else tx
    with caplog.at_level(logging.DEBUG):  # root level: capture all loggers
        assert api.post(path, json=body).status_code == 200
    text = _all_log_text(caplog)
    assert "prediction" in text  # the request WAS logged ...
    assert not any(m in text for m in MARKER_TEXT)  # ... without any feature value


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.update(Amount=-1.0),  # invalid sibling field
        lambda t: t.update(V3="not-a-number"),  # type error on a feature
        lambda t: t.update(V29=987654.321),  # extra field carrying a marker
    ],
    ids=["invalid_amount", "type_error", "extra_field"],
)
def test_validation_errors_echo_no_feature_values(
    api, synthetic_transaction, caplog, mutate
) -> None:  # type: ignore[no-untyped-def]
    tx = {**synthetic_transaction, **MARKERS}
    mutate(tx)
    with caplog.at_level(logging.DEBUG):
        r = api.post("/v1/predict", json=tx)
    assert r.status_code == 422
    assert not any(m in r.text for m in MARKER_TEXT)  # response body
    assert not any(m in _all_log_text(caplog) for m in MARKER_TEXT)  # logs


# --- Phase 12: MODEL_PATH selects the artifact; no real model needed ------------------------


def test_model_path_env_is_used(synthetic_model_path) -> None:  # type: ignore[no-untyped-def]
    """The session fixture points MODEL_PATH at the SYNTHETIC artifact; from_env obeys it."""
    s = Settings.from_env()
    assert s.models_dir == Path(synthetic_model_path).parent
    with TestClient(create_app(s)) as c:
        assert c.get("/ready").json()["model_version"] == "synthetic-test-v0"


def test_model_path_must_name_the_artifact(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "fraud_model.joblib"))
    with pytest.raises(ValueError, match="model.joblib"):
        Settings.from_env()


def test_models_dir_fallback_when_no_model_path(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    assert Settings.from_env().models_dir == tmp_path


def test_missing_model_path_target_gives_503(
    monkeypatch, tmp_path: Path, synthetic_transaction
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))  # does not exist
    with TestClient(create_app(Settings.from_env())) as c:
        assert c.get("/ready").status_code == 503
        assert c.post("/v1/predict", json=synthetic_transaction).status_code == 503


# --- regression: every operation shown in Swagger is actually routable (never 404) --------


def test_post_v1_predict_is_routed_not_404(api, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    """Regression for "Swagger shows POST /v1/predict but the server answers 404"."""
    r = api.post("/v1/predict", json={**synthetic_transaction, "transaction_id": "synthetic-0001"})
    assert r.status_code == 200, r.text
    assert r.json()["transaction_id"] == "synthetic-0001"
    assert any(getattr(rt, "path", None) == "/v1/predict" and "POST" in rt.methods
               for rt in api.app.routes)  # fmt: skip


def test_every_openapi_operation_is_routable(api, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    """Each path+method in /openapi.json must reach a handler: 404/405 = schema and routing
    disagree. Bodies are minimal valid payloads; any non-404/405 status proves routing."""
    bodies = {"/v1/predict/batch": {"transactions": [synthetic_transaction]}}
    spec = api.get("/openapi.json").json()
    for path, ops in spec["paths"].items():
        for method in ops:
            r = api.request(method.upper(), path, json=bodies.get(path, synthetic_transaction)
                            if method == "post" else None)  # fmt: skip
            assert r.status_code not in (404, 405), f"{method.upper()} {path} -> {r.status_code}"
