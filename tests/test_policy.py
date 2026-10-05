"""Tests for fraud_detection.policy on a hand-made 12-row example (default 0.01-0.95 grid).

    fraud  p = 0.90 0.85 0.80 0.70 0.60 0.20   Amount 100 each
    legit  p = 0.55 0.30 0.15 0.10 0.05 0.02   Amount  10 each
    review_cost = 5, target_recall = 0.85

t_review: recall >= 0.85 needs all 6 frauds (5/6 = 0.833), so t <= 0.20.
    t = 0.15: 9 alerts, nothing missed -> cost 45
    t = 0.16 .. 0.20: 8 alerts (6 fraud + legit 0.55, 0.30), nothing missed -> cost 40
    cheapest qualifying, ties -> lower threshold => t_review = 0.16
    (recall 1.0, precision 6/8 = 0.75, alerts per 1,000 = 8/12 * 1000)
t_block: precision >= 0.90 with >= 5 TP
    t = 0.21 .. 0.30: 7 alerts, 5 TP -> 0.714;  t = 0.31 .. 0.55: 6 alerts, 5 TP -> 0.833
    t = 0.56: 5 alerts, 5 TP -> 1.0  => t_block = 0.56
"""

import numpy as np
import pytest

from fraud_detection.policy import (
    HIGH,
    LOW,
    MEDIUM,
    choose_t_block,
    choose_t_review,
    decide,
    tier_table,
)

Y = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
P = [0.90, 0.85, 0.80, 0.70, 0.60, 0.20, 0.55, 0.30, 0.15, 0.10, 0.05, 0.02]
AMOUNTS = [100] * 6 + [10] * 6


def test_t_review_hand_computed() -> None:
    r = choose_t_review(Y, P, AMOUNTS, review_cost=5, target_recall=0.85)
    assert r["threshold"] == pytest.approx(0.16)
    assert r["recall"] == 1.0 and r["precision"] == pytest.approx(0.75)
    assert r["expected_cost"] == 40
    assert r["alerts_per_1000"] == pytest.approx(8 / 12 * 1000)
    assert r["met_target"] is True and r["fallback"] is False


def test_t_review_fallback_is_flagged() -> None:
    # the 0.20 fraud drops to 0.005 (below the grid): max reachable recall is 5/6 < 0.85.
    # Every threshold <= 0.60 keeps recall 5/6; cheapest is 0.56..0.60 (5 alerts):
    # cost = 100 missed + 5 x 5 alerts = 125, tie -> 0.56
    p = [0.90, 0.85, 0.80, 0.70, 0.60, 0.005] + P[6:]
    r = choose_t_review(Y, p, AMOUNTS, review_cost=5, target_recall=0.85)
    assert r["fallback"] is True and r["met_target"] is False
    assert r["threshold"] == pytest.approx(0.56)
    assert r["recall"] == pytest.approx(5 / 6) and r["expected_cost"] == 125
    assert "no threshold reaches" in r["rule"]


def test_t_block_hand_computed() -> None:
    b = choose_t_block(Y, P, min_precision=0.90)
    assert b["exists"] is True
    assert b["threshold"] == pytest.approx(0.56)
    assert b["precision"] == 1.0 and b["tp"] == 5


def test_t_block_none_when_rule_cannot_be_met() -> None:
    # 6 TP only exists at t <= 0.20 where precision is 0.75: no threshold qualifies
    b = choose_t_block(Y, P, min_precision=0.90, min_tp=6)
    assert b == {"exists": False, "threshold": None, "rule": b["rule"]}


def test_decide_boundaries_use_the_frozen_thresholds() -> None:
    t_review = choose_t_review(Y, P, AMOUNTS, 5, 0.85)["threshold"]
    t_block = choose_t_block(Y, P)["threshold"]
    assert decide(t_block, t_review, t_block) == HIGH  # p == t_block -> HIGH
    assert decide(t_review, t_review, t_block) == MEDIUM  # p == t_review -> MEDIUM
    assert decide(np.nextafter(t_block, 0), t_review, t_block) == MEDIUM
    assert decide(np.nextafter(t_review, 0), t_review, t_block) == LOW
    assert decide(1.0, t_review, t_block) == HIGH and decide(0.0, t_review, t_block) == LOW


def test_decide_without_t_block() -> None:
    assert decide(0.999, t_review=0.16, t_block=None) == MEDIUM
    assert decide(0.16, t_review=0.16, t_block=None) == MEDIUM
    assert decide(0.10, t_review=0.16, t_block=None) == LOW


def test_actions_match_architecture() -> None:
    assert (HIGH, MEDIUM, LOW) == (("HIGH", "HOLD"), ("MEDIUM", "REVIEW"), ("LOW", "APPROVE"))


def test_tier_table_hand_computed() -> None:
    t = tier_table(Y, P, t_review=0.16, t_block=0.56).set_index("tier")
    assert list(t.index) == ["HIGH", "MEDIUM", "LOW"]
    assert list(t["action"]) == ["HOLD", "REVIEW", "APPROVE"]
    assert list(t["count"]) == [5, 3, 4] and list(t["fraud"]) == [5, 1, 0]
    assert t.loc["MEDIUM", "fraud_share"] == pytest.approx(1 / 3)
    assert t.loc["HIGH", "share_of_all_fraud"] == pytest.approx(5 / 6)
    assert t["alerts_per_1000"].sum() == pytest.approx(1000)
    assert t.loc["HIGH", "alerts_per_1000"] == pytest.approx(5 / 12 * 1000)


def test_empty_alert_case_has_no_nan_or_crash() -> None:
    # every score is 0: no threshold on the grid raises an alert
    p = [0.0] * 12
    r = choose_t_review(Y, p, AMOUNTS, review_cost=5, target_recall=0.85)
    assert r["fallback"] is True and r["recall"] == 0.0 and r["precision"] == 0.0
    assert r["alerts_per_1000"] == 0.0 and r["expected_cost"] == 600  # all fraud missed
    assert not any(np.isnan(v) for v in r.values() if isinstance(v, float))
    assert choose_t_block(Y, p)["threshold"] is None
    t = tier_table(Y, p, r["threshold"], None).set_index("tier")
    assert t.loc["LOW", "count"] == 12 and t.loc["HIGH", "count"] == 0
    assert t.loc["HIGH", "fraud_share"] == 0.0  # empty tier, no division by zero
