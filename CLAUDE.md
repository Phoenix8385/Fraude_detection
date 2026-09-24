# CLAUDE.md — Fraud & Anomaly Detection System

You are the implementation engineer. The human (KING) is the architect.

## Before any work
1. Read docs/prd.md, docs/architecture.md, docs/evaluation.md, docs/rules.md, docs/tasks.md, docs/memory.md.
2. State the current phase (from tasks.md), what is done, and your plan.
3. Wait for "go" before editing files.

## Non-negotiables
- ONE phase per session. Never start the next phase.
- The TEST split is only loaded by src/fraud_detection/evaluate.py (Phase 9). Nowhere else.
- SMOTE/any resampling only inside an imblearn Pipeline, fit on training rows only.
- Never invent metrics. Every number in README/docs must come from reports/metrics/*.
- Never commit data/raw, data/processed, mlruns, .venv, secrets.
- Environment: Windows + Git Bash. Activate venv with `source .venv/Scripts/activate`.

## End of every session, report
Files changed · commands run · test results · open issues · suggested next step.
Then update docs/tasks.md (tick boxes) and append decisions to docs/memory.md.
