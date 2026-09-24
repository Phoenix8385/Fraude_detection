"""Hand-computed tests for fraud_detection.metrics.compute_metrics."""

import math

import numpy as np
import pytest

from fraud_detection.metrics import compute_metrics

# ---------------------------------------------------------------------------
# 10-row example, threshold 0.5, review_cost 5
#
#  row  y  proba  amount  alert(proba>=0.5)  outcome
#   0   1  0.90    100        yes             TP
#   1   1  0.60     50        yes             TP
#   2   1  0.30    200        no              FN   <- missed fraud, 200 lost
#   3   0  0.80     10        yes             FP
#   4   0  0.40     10        no              TN
#   5   0  0.20     10        no              TN
#   6   0  0.10     10        no              TN
#   7   0  0.10     10        no              TN
#   8   0  0.05     10        no              TN
#   9   0  0.00     10        no              TN
#
#  TP=2 FP=1 TN=6 FN=1
#  precision = 2/3, recall = 2/3, f1 = 2/3, specificity = 6/7, fpr = 1/7
#  alerts_per_1000 = 3 alerts / 10 rows * 1000 = 300
#  fraud_amount_caught_pct = (100+50) / (100+50+200) * 100 = 42.857...
#  expected_cost = 200 (missed) + 5 * 3 alerts = 215
#  PR-AUC (average precision): rank by proba -> 0.9(F) 0.8(L) 0.6(F) 0.4(L) 0.3(F)
#    precision at each fraud: 1/1, 2/3, 3/5 -> mean = (1 + 2/3 + 3/5) / 3 = 0.7556
#  ROC-AUC: of 3*7 = 21 (fraud, legit) pairs, fraud scores higher in 7 + 6 + 5 = 18
#    -> 18/21 = 0.8571
# ---------------------------------------------------------------------------
Y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
P = [0.9, 0.6, 0.3, 0.8, 0.4, 0.2, 0.1, 0.1, 0.05, 0.0]
AMOUNTS = [100, 50, 200, 10, 10, 10, 10, 10, 10, 10]


def test_ten_row_hand_example() -> None:
    m = compute_metrics(Y, P, threshold=0.5, amounts=AMOUNTS, review_cost=5.0)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 6, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(6 / 7)
    assert m["fpr"] == pytest.approx(1 / 7)
    assert m["alerts_per_1000"] == pytest.approx(300.0)
    assert m["fraud_amount_caught_pct"] == pytest.approx(150 / 350 * 100)
    assert m["expected_cost"] == pytest.approx(215.0)
    assert m["pr_auc"] == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)
    assert m["roc_auc"] == pytest.approx(18 / 21)
    assert m["threshold"] == 0.5


def test_threshold_is_inclusive() -> None:
    # proba exactly equal to the threshold raises an alert
    m = compute_metrics([1, 0], [0.5, 0.1], threshold=0.5, amounts=[1, 1], review_cost=0.0)
    assert m["tp"] == 1


def test_perfect_predictions() -> None:
    y = [1, 1, 0, 0, 0]
    proba = [0.99, 0.8, 0.1, 0.2, 0.0]
    m = compute_metrics(y, proba, 0.5, amounts=[10, 20, 1, 1, 1], review_cost=5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 0, 3, 0)
    for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "specificity"):
        assert m[key] == pytest.approx(1.0), key
    assert m["fpr"] == 0.0
    assert m["fraud_amount_caught_pct"] == pytest.approx(100.0)
    assert m["expected_cost"] == pytest.approx(5 * 2)  # nothing missed, 2 alerts reviewed


def test_all_negative_predictions_no_division_error() -> None:
    y = [1, 0, 0, 1, 0]
    amounts = [30, 5, 5, 70, 5]
    m = compute_metrics(y, np.zeros(5), 0.5, amounts=amounts, review_cost=5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (0, 0, 3, 2)
    assert m["precision"] == 0.0  # no alerts -> defined as 0, not an error
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["alerts_per_1000"] == 0.0
    assert m["fraud_amount_caught_pct"] == 0.0
    assert m["expected_cost"] == pytest.approx(100.0)  # all fraud money lost, no review cost
    assert m["pr_auc"] == pytest.approx(2 / 5)  # constant score -> AP = fraud rate
    assert m["roc_auc"] == pytest.approx(0.5)


def test_single_class_gives_nan_auc() -> None:
    m = compute_metrics([0, 0, 0], [0.1, 0.9, 0.2], 0.5, amounts=[1, 1, 1], review_cost=5)
    assert math.isnan(m["pr_auc"]) and math.isnan(m["roc_auc"])
    assert m["fp"] == 1 and m["recall"] == 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_metrics([1, 0], [0.5], 0.5, amounts=[1, 1], review_cost=5)


def test_inputs_not_mutated() -> None:
    proba = np.array(P)
    before = proba.copy()
    compute_metrics(Y, proba, 0.5, amounts=AMOUNTS, review_cost=5)
    np.testing.assert_array_equal(proba, before)
