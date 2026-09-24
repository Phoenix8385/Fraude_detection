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
