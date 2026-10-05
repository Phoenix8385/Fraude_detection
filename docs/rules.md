# Coding Rules

1. One phase per session; never implement future phases.
2. Read existing code before changing it; prefer small edits over rewrites.
3. Python 3.11, type hints on all public functions, docstrings on modules and public functions.
4. No hardcoded paths — use config.py (paths relative to project root). No secrets in code.
5. Seeds from config everywhere randomness exists.
6. Pure functions for features, metrics, threshold logic. No global mutable state.
7. Notebooks are for exploration only; reusable logic lives in src/.
8. Tests never depend on the real CSV or the real model artifact (synthetic fixtures).
9. Never load the test split outside evaluate.py.
10. Never write a metric into docs/README that isn't in reports/metrics/.
11. Ruff must pass; pytest must pass before reporting a phase done.
12. Log with the logging module, not print (CLI summaries excepted).
13. Pin nothing silently: if you change a dependency version, say so and why.
14. If a requirement is ambiguous, ask — do not invent one.
15. Git Bash syntax for all commands; activate venv with `source .venv/Scripts/activate`.
# Rules addendum (append to docs/rules.md)
16. Never load the test split outside evaluate.py (enforced by a test).
17. Never store or log V1..V28 values. Amount, score, tier, version, latency only.
18. DB or store errors are caught and logged; they never change the API response.
19. API responses are versioned under /v1; breaking changes need a new prefix and an openapi snapshot update.
20. The dashboard talks to the API only. No direct DB access from the dashboard.
21. Docker images install from requirements-api.lock (exact pins identical to training).
22. Secrets only via environment variables or CI secrets. .env.example has placeholders only.
23. README and MODEL_CARD numbers must be traceable to reports/metrics/*.
24. Label simulated drift as simulated everywhere it appears.
