# Architecture v2

## Flow
CSV -> data.py (validate, dedupe) -> features.py -> splits.py (stratified | time) -> train.py (MLflow)
-> tune.py (Optuna, bootstrap CIs) -> calibrate.py (valid_cal) -> policy.py (valid_thr: t_review, t_block)
-> evaluate.py (TEST once) -> explain.py -> artifacts.py -> predict.py -> api/ -> Docker -> CI -> Render
Product layer: api/store.py -> Neon Postgres -> /v1/stats/* -> Streamlit dashboard; drift.py (PSI).

## Modules
| Module | Contract |
|---|---|
| policy.py | decide(p,t_review,t_block) -> (tier, action): HIGH/HOLD, MEDIUM/REVIEW, LOW/APPROVE |
| drift.py | psi(reference, current) using models/reference_profile.json |
| store.py | PredictionStore protocol: NullStore (v1.0), SqlStore (v1.1); failures swallowed + counted |
| artifacts.py | model.joblib + model_meta.json (version, thresholds, features, schema_hash, lib versions, git commit) |

## API (v1)
- GET /health (liveness) · GET /ready (model loaded; reports db_ok) · GET /model-info
- POST /v1/predict {transaction_id?, Time, Amount, V1..V28} -> {transaction_id, fraud_probability, risk_tier,
  recommended_action, is_flagged, thresholds{review,block}, model_version, latency_ms}
- POST /v1/predict/batch (max 500) · POST /v1/explain (top 5 contributions, log-odds, uncalibrated)
- GET /v1/stats/summary?hours= · GET /v1/predictions/recent?limit= · GET /v1/stats/drift?hours=
- Auth: X-API-Key if API_KEY set (not on /health, /ready, /docs). Rate limit via slowapi. CORS from ALLOWED_ORIGINS.
- Errors: {"error": {"code","message","request_id"}}. 401, 422, 429, 503.
- Logs: JSON, one line per prediction, no feature values.

## predictions table
id, request_id, transaction_id, created_at(idx), amount, fraud_probability, risk_tier,
recommended_action, model_version, latency_ms, top_features(jsonb, names only). 30-day retention.

## Deployment
API: Render (Docker, free; sleeps ~15 min idle). DB: Neon free. Dashboard: Streamlit Community Cloud.
CI: ruff, pytest --cov-fail-under=85, docker smoke, deploy hook, post-deploy /ready retries.
