"""Tests for fraud_detection.bootstrap (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.bootstrap import (
    BOOTSTRAP_METRICS,
    bootstrap_samples,
    paired_difference,
    percentile_ci,
    stratified_resample,
    summarize_ci,
)
from fraud_detection.metrics import compute_metrics


@pytest.fixture
def scored() -> dict[str, np.ndarray]:
    """500 rows, 50 fraud; 'good' ranks fraud well, 'weak' barely."""
    rng = np.random.default_rng(0)
    y = np.zeros(500, dtype=int)
    y[:50] = 1
    good = np.clip(0.7 * y + rng.uniform(0, 0.4, 500), 0, 1)
    weak = np.clip(0.1 * y + rng.uniform(0, 0.9, 500), 0, 1)
    return {"y": y, "good": good, "weak": weak, "amounts": rng.uniform(1, 200, 500)}


def test_stratified_resample_keeps_class_counts() -> None:
    y = np.array([0] * 90 + [1] * 10)
    idx = stratified_resample(y, np.random.default_rng(0))
    assert len(idx) == 100
    assert y[idx].sum() == 10


def test_percentile_ci_hand_computed() -> None:
    low, high = percentile_ci(np.arange(101), 0.90)  # 5th and 95th percentile of 0..100
    assert (low, high) == pytest.approx((5.0, 95.0))


def test_samples_are_shared_reproducible_and_bracket_point(scored: dict) -> None:
    y, amt = scored["y"], scored["amounts"]
    probas = {"good": scored["good"], "weak": scored["weak"]}
    thr = {"good": 0.5, "weak": 0.5}
    a = bootstrap_samples(y, probas, thr, amt, 5.0, n_resamples=200, seed=7)
    b = bootstrap_samples(y, probas, thr, amt, 5.0, n_resamples=200, seed=7)
    pd.testing.assert_frame_equal(a["good"], b["good"])
    assert list(a["good"].columns) == list(BOOTSTRAP_METRICS)
    assert len(a["weak"]) == 200

    point = compute_metrics(y, scored["good"], 0.5, amt, 5.0)
    ci = summarize_ci(point, a["good"], 0.95)
    assert ci["pr_auc"]["ci_low"] <= point["pr_auc"] <= ci["pr_auc"]["ci_high"]
    assert ci["pr_auc"]["ci_low"] < ci["pr_auc"]["ci_high"]


def test_paired_difference_detects_clear_winner(scored: dict) -> None:
    y, amt = scored["y"], scored["amounts"]
    probas = {"good": scored["good"], "weak": scored["weak"]}
    s = bootstrap_samples(y, probas, {"good": 0.5, "weak": 0.5}, amt, 5.0, 300, seed=1)
    pa = compute_metrics(y, scored["good"], 0.5, amt, 5.0)["pr_auc"]
    pb = compute_metrics(y, scored["weak"], 0.5, amt, 5.0)["pr_auc"]
    res = paired_difference(s["good"]["pr_auc"], s["weak"]["pr_auc"], pa, pb, 0.95)
    assert res["diff"] == pytest.approx(pa - pb)
    assert res["prob_a_better"] == 1.0
    assert not res["ci_includes_zero"]


def test_paired_difference_of_identical_models_includes_zero() -> None:
    same = np.array([0.1, 0.2, 0.3])
    res = paired_difference(same, same, 0.5, 0.5, 0.95)
    assert res["ci_includes_zero"] and res["prob_a_better"] == 0.0 and res["diff"] == 0.0


def test_bootstrap_rejects_single_class_and_mismatched_names(scored: dict) -> None:
    amt = scored["amounts"]
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_samples(np.zeros(500), {"a": scored["good"]}, {"a": 0.5}, amt, 5.0, 2, 0)
    with pytest.raises(ValueError, match="same models"):
        bootstrap_samples(scored["y"], {"a": scored["good"]}, {"b": 0.5}, amt, 5.0, 2, 0)
