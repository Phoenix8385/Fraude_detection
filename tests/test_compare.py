"""Tests for fraud_detection.compare (no MLflow, no real data)."""

from pathlib import Path

import pandas as pd

from fraud_detection.compare import comparison_markdown, latest_runs, to_markdown_table


def _runs_csv(tmp_path: Path) -> Path:
    rows = [
        # an older logreg/time run that must be ignored
        ("2026-01-01T00:00:00", "old", "logreg", "time", 0.10),
        ("2026-01-02T00:00:00", "r1", "logreg", "time", 0.70),
        ("2026-01-02T00:00:00", "r2", "xgb", "time", 0.80),
        ("2026-01-02T00:00:00", "r3", "dummy", "time", 0.001),
        ("2026-01-02T00:00:00", "r4", "xgb", "stratified", 0.90),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "run_id", "model", "split", "pr_auc"])
    df["roc_auc"], df["recall"], df["precision"], df["fit_seconds"] = 0.9, 0.8, 0.5, 1.0
    path = tmp_path / "experiments.csv"
    df.to_csv(path, index=False)
    return path


def test_latest_runs_keeps_newest_per_model_split(tmp_path: Path) -> None:
    runs = latest_runs(_runs_csv(tmp_path))
    assert len(runs) == 4
    assert "old" not in set(runs["run_id"])


def test_markdown_sorted_by_pr_auc(tmp_path: Path) -> None:
    md = comparison_markdown(latest_runs(_runs_csv(tmp_path)), {"stratified": 94, "time": 57})
    time_section = md.split("## time split")[1]
    assert time_section.index("| xgb |") < time_section.index("| logreg |")
    assert time_section.index("| logreg |") < time_section.index("| dummy |")
    assert "recall@0.5" in md and "validation fraud cases: 57" in md


def test_to_markdown_table_format() -> None:
    table = to_markdown_table(pd.DataFrame({"model": ["a"], "pr_auc": [0.123456]}))
    assert table.splitlines() == ["| model | pr_auc |", "|---|---|", "| a | 0.1235 |"]
