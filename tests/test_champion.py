"""Tests for fraud_detection.champion: the pre-registered rule, hand-built inputs."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.champion import (
    choose_champion,
    operating_point,
    seed_stability,
    summary_markdown,
)
from fraud_detection.features import MODEL_FEATURES
from fraud_detection.schema import TARGET

TIE = {"ci_includes_zero": True}
CLEAR = {"ci_includes_zero": False}


def op(met: bool, cost: float) -> dict:
    return {"met_target": met, "expected_cost": cost}


def test_clear_pr_auc_winner_takes_it_regardless_of_cost() -> None:
    d = choose_champion({"xgb_u": 0.80, "logreg": 0.75},
                        {"xgb_u": op(True, 900), "logreg": op(True, 100)}, CLEAR)  # fmt: skip
    assert d["champion"] == "xgb_u" and d["step"] == "rule"


def test_clear_winner_can_be_logreg() -> None:
    d = choose_champion({"xgb_u": 0.70, "logreg": 0.80},
                        {"xgb_u": op(True, 1), "logreg": op(True, 2)}, CLEAR)  # fmt: skip
    assert d["champion"] == "logreg" and d["step"] == "rule"


def test_tie_break_1_lower_cost() -> None:
    d = choose_champion({"xgb_u": 0.80, "logreg": 0.79},
                        {"xgb_u": op(True, 500), "logreg": op(True, 400)}, TIE)  # fmt: skip
    assert d["champion"] == "logreg" and d["step"] == "tie-break 1"


def test_tie_break_1_only_model_meeting_recall() -> None:
    d = choose_champion({"xgb_u": 0.80, "logreg": 0.79},
                        {"xgb_u": op(True, 900), "logreg": op(False, 100)}, TIE)  # fmt: skip
    assert d["champion"] == "xgb_u" and d["step"] == "tie-break 1"


@pytest.mark.parametrize(
    "ops",
    [
        {"xgb_u": op(False, 100), "logreg": op(False, 900)},  # neither meets the target
        {"xgb_u": op(True, 300), "logreg": op(True, 300)},  # equal cost
    ],
)
def test_tie_break_2_simpler_model(ops: dict) -> None:
    d = choose_champion({"xgb_u": 0.80, "logreg": 0.79}, ops, TIE)
    assert d["champion"] == "logreg" and d["step"] == "tie-break 2"


def test_operating_point_follows_threshold_rule() -> None:
    y = pd.Series([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    amounts = pd.Series([10.0, 10.0, 10.0, 100.0, 100.0])
    p = operating_point(y, proba, amounts, review_cost=5.0, target_recall=1.0)
    # thresholds 0.61..0.70 alert exactly the 2 frauds: recall 1, cost 2 alerts x 5 = 10
    assert p["met_target"] and p["recall"] == 1.0
    assert p["expected_cost"] == 10.0 and p["fp"] == 0
    assert p["threshold"] == pytest.approx(0.61)


def test_seed_stability_reports_each_seed() -> None:
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(rng.normal(size=(n, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    df["Amount"] = rng.uniform(1, 100, n)
    df[TARGET] = (np.arange(n) % 10 == 0).astype(int)
    df.loc[df[TARGET] == 1, "V14"] -= 3
    params = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.7,
              "colsample_bytree": 0.7, "min_child_weight": 1.0, "reg_lambda": 1.0, "u": 1.0}
    res = seed_stability("xgb_u", params, df.iloc[:200], df.iloc[200:], (1, 2, 3), 5.0)
    assert res["seeds"] == [1, 2, 3] and len(res["valid_pr_auc"]) == 3
    assert res["mean"] == pytest.approx(np.mean(res["valid_pr_auc"]))
    assert res["std"] == pytest.approx(np.std(res["valid_pr_auc"], ddof=1))


def test_summary_markdown_prints_on_a_windows_console() -> None:
    """The CLI prints the summary; Windows consoles default to cp1252."""
    ci = {"point": 0.5, "ci_low": 0.4, "ci_high": 0.6}
    model = {
        "best_params": {"C": 1.0}, "cv_pr_auc_mean": 0.5, "cv_pr_auc_std": 0.1,
        "valid": dict.fromkeys(("pr_auc", "recall", "precision", "alerts_per_1000",
                                "expected_cost"), ci),
        "operating_point": {"threshold": 0.1, "met_target": False},
        "seed_stability": {"mean": 0.5, "std": 0.0},
    }  # fmt: skip
    result = {
        "split": "time", "valid_rows": 100, "valid_fraud": 5,
        "bootstrap": {"n_resamples": 10}, "models": {"xgb_u": model, "logreg": model},
        "paired_pr_auc": {"a": "xgb_u", "b": "logreg", "diff": 0.0, "ci_low": -0.1,
                          "ci_high": 0.1, "prob_a_better": 0.5, "ci_includes_zero": True},
        "decision": {"champion": "logreg", "step": "tie-break 2", "reason": "simpler"},
        "untuned_valid_pr_auc_reference": {"logreg": 0.5},
    }  # fmt: skip
    md = summary_markdown([result])
    md.encode("cp1252")  # raises UnicodeEncodeError on characters like U+2212
    assert "Champion: logreg" in md
