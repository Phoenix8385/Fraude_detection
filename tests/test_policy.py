"""Tests for fraud_detection.policy: tier boundaries are inclusive (p >= threshold)."""

import pytest

from fraud_detection.policy import HIGH, LOW, MEDIUM, decide, tier_summary


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.95, HIGH), (0.80, HIGH), (0.79, MEDIUM), (0.30, MEDIUM), (0.29, LOW), (0.0, LOW)],
)
def test_decide_with_both_thresholds(p: float, expected: tuple[str, str]) -> None:
    assert decide(p, t_review=0.30, t_block=0.80) == expected


def test_no_high_tier_when_t_block_does_not_exist() -> None:
    assert decide(0.999, t_review=0.30, t_block=None) == MEDIUM
    assert decide(0.1, t_review=0.30, t_block=None) == LOW


def test_actions_match_architecture() -> None:
    assert (HIGH, MEDIUM, LOW) == (("HIGH", "HOLD"), ("MEDIUM", "REVIEW"), ("LOW", "APPROVE"))


def test_tier_summary_counts_rows_and_fraud() -> None:
    y = [1, 1, 0, 1, 0, 0]
    p = [0.9, 0.5, 0.85, 0.1, 0.4, 0.0]
    s = tier_summary(y, p, t_review=0.4, t_block=0.8)
    assert (s["HIGH"]["rows"], s["HIGH"]["fraud"]) == (2, 1)
    assert (s["MEDIUM"]["rows"], s["MEDIUM"]["fraud"]) == (2, 1)
    assert (s["LOW"]["rows"], s["LOW"]["fraud"]) == (2, 1)
    assert s["HIGH"]["action"] == "HOLD"
    assert sum(t["per_1000"] for t in s.values()) == pytest.approx(1000)
