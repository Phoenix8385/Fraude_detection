# Architecture

## Flow
raw CSV → data.py (load, Pandera validate, dedupe) → features.py (log_amount, hour_of_day)
→ splits.py (stratified | time, 60/20/20) → train.py (experiment matrix, MLflow)
→ tune.py → calibrate.py (valid_cal half) → threshold.py (valid_thr half)
→ evaluate.py (TEST, once) → explain.py (SHAP) → artifacts.py (joblib + meta.json)
→ predict.py (Predictor) → api/main.py (FastAPI) → Docker → Render

## Module contracts
| Module | Input | Output | Rule |
|---|---|---|---|
| schema.py | — | FEATURE_COLUMNS (30), TARGET, Pandera schema | single source of column truth |
| data.py | CSV path | clean DataFrame + summary dict | dedupe before any split |
| features.py | DataFrame | new DataFrame + MODEL_FEATURES | pure, no fitting; shared by train & API |
| splits.py | DataFrame | {train, valid, test} parquet | deterministic, seed from config |
| metrics.py | y, proba, threshold, amounts | metrics dict | pure function |
| models.py | name, params | sklearn/imblearn Pipeline | resampling only inside pipeline |
| threshold.py | y, proba, amounts | table + chosen threshold | validation only |
| evaluate.py | frozen model + threshold | final_*.json | only module that loads TEST |
| artifacts.py | pipeline, calibrator, meta | models/fraud_model.joblib, model_meta.json | meta carries feature order + versions |
| predict.py | dict / list[dict] | prediction dicts | orders columns from meta |

## Artifact: models/model_meta.json
model_version, model_name, headline_split, threshold, calibrated, features[],
trained_at, test_metrics{}, library_versions{}, git_commit.

## API contract
- GET /health → {status: "ok"|"degraded", model_loaded, model_version}
- GET /model-info → meta subset (version, name, threshold, features, test_metrics, trained_at)
- POST /predict → body: Time, Amount, V1..V28 (floats, finite), optional transaction_id;
  extra fields rejected → {transaction_id, fraud_probability, is_fraud, threshold,
  risk_level (low|medium|high), model_version, latency_ms}
- POST /predict/batch → {transactions: [...]} max 1000 → {predictions: [...]}
- POST /explain → prediction + base_value + top 5 {feature, value, contribution}
  (log-odds of uncalibrated model)
- Errors: 422 validation, 401 bad/missing X-API-Key (only if API_KEY set), 503 no model.
- Model loaded once in lifespan. Raw feature values never logged.

## Deployment
python:3.11-slim + libgomp1, requirements-api.txt only, non-root user,
HEALTHCHECK /health, PORT from env. CI: ruff + pytest --cov + docker build.
