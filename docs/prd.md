# PRD v2 — Fraud & Anomaly Detection System

## Problem
Card fraud is rare (0.173%) and costly. Accuracy is meaningless (always-legit = 99.8%).
Analysts need: a risk score that ranks fraud well, a cutoff chosen by cost, a reason per alert.

## Users
Fraud analyst (dashboard, explanations) · Integrating engineer (REST API) · Reviewer/interviewer (reproducible, honest evaluation).

## Dataset
Kaggle mlg-ulb/creditcardfraud: 284,807 rows, 492 fraud, Time, V1-V28 (PCA, anonymised), Amount, Class. Not committed.

## Releases
### v1.0.0 (Phases 7-14)
Tuned champion (Optuna) with bootstrap CIs; calibration; two-threshold policy (APPROVE/REVIEW/HOLD);
one-shot test evaluation; SHAP; versioned artifact + model card; FastAPI /v1 (predict, batch, explain);
API key + rate limit; tests >=85%; Docker; GitHub Actions CI + deploy hook; live on Render.
### v1.1.0 (Phases 15-18)
Neon PostgreSQL prediction log (no raw features); Streamlit dashboard (API-only); PSI drift endpoint,
simulation and weekly workflow; docker-compose; RUNBOOK; final README.

## Out of scope
Streaming, Spark, Kubernetes, Airflow, Redis, LLMs, automatic retraining, auth/user accounts.

## Success criteria
- evaluation.md protocol followed; test opened once.
- Headline metrics reported with 95% bootstrap CIs.
- Champion chosen by the written rule in memory.md.
- recall >= 0.85 at t_review on validation (or deviation documented).
- p95 single-prediction latency < 50 ms locally.
- Clone -> docker compose up -> dashboard + API work.

## Honesty constraints
- Never headline accuracy. Never give V1-V28 meaning. Say "risk-scoring API", not "real-time fraud platform".
- SHAP = "model contribution", never "cause". Drift demo is simulated and labelled so.
- Free hosting: state cold-start behaviour in README.

## README sections
Problem · Why it's hard · Dataset · Architecture · EDA · Validation design · Models compared ·
Results (with CIs) · Decision policy · Calibration · Explainability · API · Dashboard · Monitoring ·
Run locally / Docker · Testing & CI · Limitations · Future work · Model card
