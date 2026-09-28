"""Hand-computed tests for fraud_detection.threshold."""

from pathlib import Path

import numpy as np
import pytest

from fraud_detection.threshold import (
    DEFAULT_GRID,
    choose_threshold,
    cost_sensitivity,
    plot_cost_curve,
    threshold_table,
)

# ---------------------------------------------------------------------------
# Same 10 rows as test_metrics.py. Fraud amounts 100, 50, 200 (total 350); legit 10 each.
#   y     = 1    1    1    0    0    0    0    0    0     0
#   proba = 0.9  0.6  0.3  0.8  0.4  0.2  0.1  0.1  0.05  0.0
#
# review_cost = 100, grid 0.1 .. 0.9. cost = missed fraud EUR + 100 x alerts
#   t    alerts  TP FP  missed EUR        recall  cost
#   0.1    8      3  5   0                 1.00   800
#   0.2    6      3  3   0                 1.00   600
#   0.3    5      3  2   0                 1.00   500   <- cheapest with recall >= 0.85
#   0.4    4      2  2   200               0.67   600
#   0.5    3      2  1   200               0.67   500
#   0.6    3      2  1   200               0.67   500
#   0.7    2      1  1   250               0.33   450
#   0.8    2      1  1   250               0.33   450
#   0.9    1      1  0   250               0.33   350   <- cheapest overall (no constraint)
# ---------------------------------------------------------------------------
Y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
P = [0.9, 0.6, 0.3, 0.8, 0.4, 0.2, 0.1, 0.1, 0.05, 0.0]
AMOUNTS = [100, 50, 200, 10, 10, 10, 10, 10, 10, 10]
GRID = np.round(np.arange(0.1, 1.0, 0.1), 1)


@pytest.fixture
def table():  # type: ignore[no-untyped-def]
    return threshold_table(Y, P, AMOUNTS, review_cost=100, grid=GRID)


def test_table_matches_hand_computation(table) -> None:  # type: ignore[no-untyped-def]
    assert list(table["expected_cost"]) == [800, 600, 500, 600, 500, 500, 450, 450, 350]
    assert list(table["tp"] + table["fp"]) == [8, 6, 5, 4, 3, 3, 2, 2, 1]
    assert list(table["missed_fraud_cost"]) == [0, 0, 0, 200, 200, 200, 250, 250, 250]
    assert (table["missed_fraud_cost"] + table["review_cost_total"] == table["expected_cost"]).all()


def test_chosen_threshold_respects_recall_target(table) -> None:  # type: ignore[no-untyped-def]
    choice = choose_threshold(table, target_recall=0.85)
    assert choice["threshold"] == pytest.approx(0.3)
    assert choice["expected_cost"] == 500
    assert choice["recall"] >= 0.85
    assert choice["met_target"] is True


def test_without_constraint_cheapest_threshold_wins(table) -> None:  # type: ignore[no-untyped-def]
    # proves the recall constraint is what moved the choice from 0.9 to 0.3
    assert choose_threshold(table, target_recall=0.0)["threshold"] == pytest.approx(0.9)


def test_fallback_when_target_unreachable() -> None:
    # grid starts at 0.5: best recall is 2/3 (at 0.5 and 0.6, both cost 500) -> lower threshold
    t = threshold_table(Y, P, AMOUNTS, review_cost=100, grid=[0.5, 0.6, 0.7, 0.8, 0.9])
    choice = choose_threshold(t, target_recall=0.85)
    assert choice["met_target"] is False
    assert choice["threshold"] == pytest.approx(0.5)
    assert choice["recall"] == pytest.approx(2 / 3)
    assert "no threshold reaches" in choice["rule"]


def test_empty_alert_threshold_is_handled() -> None:
    # 0.95 is above every score: no alerts, all fraud money lost, no division error
    t = threshold_table(Y, P, AMOUNTS, review_cost=5, grid=[0.95])
    row = t.iloc[0]
    assert row["tp"] + row["fp"] == 0
    assert row["precision"] == 0.0 and row["recall"] == 0.0
    assert row["expected_cost"] == 350
    assert choose_threshold(t, target_recall=0.85)["met_target"] is False


def test_default_grid() -> None:
    assert DEFAULT_GRID[0] == 0.01 and DEFAULT_GRID[-1] == 0.95 and len(DEFAULT_GRID) == 95


def test_cost_sensitivity_changes_with_review_cost() -> None:
    s = cost_sensitivity(Y, P, AMOUNTS, target_recall=0.0, review_costs=(1.0, 100.0), grid=GRID)
    # review_cost 1: missing a 200 fraud is never worth it -> low threshold (0.3, cost 5)
    # review_cost 100: alerts are expensive -> high threshold (0.9, cost 350)
    assert list(s["threshold"]) == pytest.approx([0.3, 0.9])
    assert list(s["expected_cost"]) == [5, 350]


def test_plot_cost_curve_writes_file(table, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "cost.png"
    plot_cost_curve(table, choose_threshold(table, 0.85), "test", out)
    assert out.exists() and out.stat().st_size > 0
