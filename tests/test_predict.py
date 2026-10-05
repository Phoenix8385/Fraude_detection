"""Tests for fraud_detection.predict.Predictor (synthetic artifact from conftest)."""

import random

import pytest

from fraud_detection.policy import decide
from fraud_detection.predict import Predictor


@pytest.fixture(scope="module")
def predictor(synthetic_artifact_dir) -> Predictor:  # type: ignore[no-untyped-def]
    return Predictor.from_dir(synthetic_artifact_dir)


def test_thresholds_and_version_come_from_metadata(predictor) -> None:  # type: ignore[no-untyped-def]
    assert predictor.model_version == "synthetic-test-v0"
    assert predictor.thresholds == {"review": 0.3, "block": 0.7}


def test_probability_in_unit_interval_and_tier_matches_policy(
    predictor, synthetic_transaction
) -> None:  # type: ignore[no-untyped-def]
    for v14 in (0.0, -2.0, -4.0, -8.0):
        r = predictor.predict_one({**synthetic_transaction, "V14": v14})
        assert 0.0 <= r["fraud_probability"] <= 1.0
        tier, action = decide(r["fraud_probability"], 0.3, 0.7)
        assert (r["risk_tier"], r["recommended_action"]) == (tier, action)
        assert r["is_flagged"] == (r["fraud_probability"] >= 0.3)


def test_input_key_order_and_extra_keys_do_not_matter(predictor, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    tx = {**synthetic_transaction, "V14": -4.0, "V3": 1.5}
    items = list(tx.items())
    random.Random(0).shuffle(items)
    shuffled = {**dict(items), "transaction_id": "abc"}
    assert predictor.predict_one(shuffled) == predictor.predict_one(tx)


def test_frame_uses_artifact_feature_order(predictor, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    X = predictor.frame([synthetic_transaction])
    assert list(X.columns) == predictor.features
    assert X["log_amount"].iloc[0] == pytest.approx(3.258096538)  # log1p(25)
    assert X["hour_of_day"].iloc[0] == 1.0  # 3600 s -> hour 1


def test_batch_length_and_order(predictor, synthetic_transaction) -> None:  # type: ignore[no-untyped-def]
    batch = [{**synthetic_transaction, "V14": v} for v in (0.0, -8.0, 0.0)]
    out = predictor.predict_batch(batch)
    assert len(out) == 3 and out[0] == out[2]
    assert out[1]["fraud_probability"] >= out[0]["fraud_probability"]
    assert predictor.predict_batch([]) == []


def test_explain_one_returns_top5_uncalibrated_contributions(
    predictor, synthetic_transaction
) -> None:  # type: ignore[no-untyped-def]
    e = predictor.explain_one({**synthetic_transaction, "V14": -6.0})
    assert len(e["contributions"]) == 5
    assert e["units"] == "log-odds, uncalibrated model"
    assert e["contributions"][0]["feature"] == "V14"
    assert {"fraud_probability", "risk_tier", "base_value", "log_odds"} <= set(e)
