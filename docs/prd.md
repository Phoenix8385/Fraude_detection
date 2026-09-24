# PRD — Fraud & Anomaly Detection System

## Problem
Card fraud is rare (0.173% of transactions in the dataset) and costly. A naive model
that labels everything legitimate is 99.8% accurate and catches nothing. A fraud team
needs (1) a risk score that ranks fraud well, (2) an alert threshold that balances
missed fraud against analyst workload, and (3) a reason for each alert.

## Users
- Fraud analyst: reviews flagged transactions, needs probability + top contributing features.
- Integrating service/engineer: calls a REST API, needs a stable contract and low latency.
- Reviewer (interviewer): needs a reproducible, honest evaluation.

## Dataset
Kaggle mlg-ulb/creditcardfraud — 284,807 rows, 492 fraud, features Time, V1–V28 (PCA,
anonymised), Amount; target Class. Historical, 2 days, Sept 2013. Not committed to git.

## Scope — v1 (FROZEN)
1. Validated data loading with de-duplication before splitting.
2. Two split designs (stratified 60/20/20, time-based 60/20/20 — primary).
3. Model comparison: Dummy, Logistic Regression, Isolation Forest, XGBoost plain,
   XGBoost weighted, XGBoost + SMOTE (1.0 and 0.1 sampling), tracked in MLflow.
4. Hyperparameter tuning scored by PR-AUC.
5. Calibration check + cost-based threshold selection on validation.
6. One-shot final test evaluation.
7. SHAP global + per-transaction explanations.
8. Versioned model artifact.
9. FastAPI: /health, /model-info, /predict, /predict/batch, /explain.
10. Pytest (>=80% coverage), Docker, GitHub Actions CI, Render deployment.
11. JSON prediction logs + PSI drift-check script.

## Out of scope (v1)
Streaming (Kafka), Spark, Kubernetes, Airflow, LLMs/RAG/agents, feature store,
automatic retraining, user accounts, a frontend UI.

## Success criteria
- Evaluation protocol in docs/evaluation.md followed exactly (test opened once).
- Final model beats Logistic Regression and Isolation Forest on validation PR-AUC.
- Chosen threshold meets recall >= 0.85 on validation (or deviation documented).
- API p95 latency < 50 ms locally for single prediction.
- Clone → docker build → docker run → /docs works with no manual steps.

## Honesty constraints
- Never say "accuracy" as the headline metric.
- Never claim meaning for V1–V28.
- Say "real-time inference API", never "real-time fraud detection platform".
- SHAP output is "model contribution", never "cause".

## README sections (16)
Problem · Why it's hard · Dataset · Architecture · EDA findings · Validation design ·
Models compared · Results · Threshold & cost · Calibration · Explainability · API ·
Run locally / Docker · Testing & CI · Limitations · Future work
