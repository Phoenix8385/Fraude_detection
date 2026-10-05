"""Hand-computed tests for fraud_detection.threshold."""

from pathlib import Path

import numpy as np
import pytest

from fraud_detection.threshold import (
    DEFAULT_GRID,
    choose_block_threshold,
    choose_threshold,
    cost_sensitivity,
    plot_cost_curve,
    threshold_stability,
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


# t_block on the same table. precision per threshold (TP / alerts):
#   t     0.1   0.2  0.3  0.4  0.5   0.6   0.7  0.8  0.9
#   prec  .375  .50  .60  .50  .667  .667  .50  .50  1.0   (TP 3,3,3,2,2,2,1,1,1)
# Precision is NOT monotonic in the threshold, so "smallest qualifying" matters.


@pytest.mark.parametrize(
    ("min_precision", "min_tp", "expected"),
    [
        (0.90, 1, 0.9),  # only 0.9 reaches precision 0.90
        (0.60, 3, 0.3),  # 0.3 qualifies and is the smallest that does
        (0.65, 2, 0.5),  # 0.3 fails precision; 0.5 is the smallest that passes
    ],
)
def test_block_threshold_hand_computed(table, min_precision, min_tp, expected) -> None:  # type: ignore[no-untyped-def]
    b = choose_block_threshold(table, min_precision, min_tp)
    assert b["exists"] is True
    assert b["threshold"] == pytest.approx(expected)
    assert b["precision"] >= min_precision and b["tp"] >= min_tp


def test_block_threshold_does_not_exist_when_tp_floor_unmet(table) -> None:  # type: ignore[no-untyped-def]
    # precision 0.90 only at 0.9, which has 1 TP: with min_tp 2 nothing qualifies, no relaxing
    b = choose_block_threshold(table, 0.90, 2)
    assert b["exists"] is False and b["threshold"] is None
    assert set(b) == {"exists", "threshold", "rule"}
    assert "tp >= 2" in b["rule"]


def test_threshold_stability_is_reproducible_and_bounded() -> None:
    y, p, amt = Y * 5, P * 5, AMOUNTS * 5
    kw = {"review_cost": 100, "target_recall": 0.85, "block_precision": 0.9,
          "block_min_tp": 1, "n_resamples": 30, "seed": 3, "grid": GRID}  # fmt: skip
    a = threshold_stability(y, p, amt, **kw)
    assert a == threshold_stability(y, p, amt, **kw)
    r = a["t_review"]
    assert 0.1 <= r["q25"] <= r["median"] <= r["q75"] <= 0.9
    assert 0.0 <= r["share_met_target"] <= 1.0
    assert 0.0 <= a["t_block"]["share_exists"] <= 1.0 and a["n_resamples"] == 30
