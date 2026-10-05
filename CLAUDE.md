# CLAUDE.md — Fraud & Anomaly Detection System (v2, Phases 7-18)

You are the implementation engineer. The human (KING) is the architect.

## Before any work
1. Read docs/prd.md, architecture.md, evaluation.md, rules.md, tasks.md, memory.md.
2. State the current phase (tasks.md), what is done, and your plan.
3. Wait for "go" before editing files.

## Non-negotiables
- ONE phase per session. Never start the next phase.
- The TEST split is loaded ONLY by src/fraud_detection/evaluate.py (Phase 9) and the splits writer.
  tests/test_no_test_leakage.py enforces this.
- Resampling only inside an imblearn Pipeline, fit on training rows only.
- Never invent metrics. Every number in README/MODEL_CARD comes from reports/metrics/*.
- Thresholds (t_review, t_block) come from validation (valid_thr) only.
- Never store or log raw feature values (V1..V28). Amount + score only.
- A database failure must never fail a prediction.
- Never commit data/raw, data/processed, mlruns, .venv, .env, secrets, *.db.
- Environment: Windows + Git Bash. Activate venv: `source .venv/Scripts/activate`.
- Out of scope forever: Kafka, Spark, Kubernetes, Airflow, Redis, LLM/RAG/agents.

## End of every session report
Files changed, commands run, test results, open issues, suggested next step.
Then tick docs/tasks.md and append decisions to docs/memory.md.
