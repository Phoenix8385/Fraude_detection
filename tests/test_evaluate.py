"""Tests for fraud_detection.evaluate — synthetic data and temp dirs only; the real test
split, real models and MLflow are never touched."""

import ast
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.evaluate import (
    RERUN_WARNING,
    check_artifact,
    file_md5,
    guard_rerun,
    load_test,
    reference_costs,
    score_frozen,
    summary_markdown,
    verify_frozen,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "fraud_detection" / "evaluate.py"

POLICY = {
    "model": "logreg", "mlflow_run_id": "run-1", "calibration": "isotonic",
    "t_review": 0.02, "t_block": 0.23, "review_cost": 5.0,
}  # fmt: skip
CHAMPION = {"decision": {"champion": "logreg"}, "models": {"logreg": {"mlflow_run_id": "run-1"}}}
PREREG = {
    "splits": {
        "time": {
            "champion": "logreg", "mlflow_run_id": "run-1", "calibration": "isotonic",
            "t_review": 0.02, "t_block": 0.23, "review_cost": 5.0,
        }
    }
}  # fmt: skip


def test_verify_frozen_accepts_consistent_configuration() -> None:
    verify_frozen("time", POLICY, CHAMPION, PREREG)


@pytest.mark.parametrize(
    ("field", "value"),
    [("t_review", 0.03), ("t_block", None), ("calibration", "none"), ("mlflow_run_id", "x")],
)
def test_verify_frozen_rejects_policy_drift(field: str, value: object) -> None:
    policy = {**POLICY, field: value}
    with pytest.raises(ValueError, match=field):
        verify_frozen("time", policy, CHAMPION, PREREG)


def test_verify_frozen_rejects_champion_mismatch() -> None:
    champion = copy.deepcopy(CHAMPION)
    champion["decision"]["champion"] = "xgb_u"
    with pytest.raises(ValueError, match="different models"):
        verify_frozen("time", POLICY, champion, PREREG)


def test_check_artifact_md5(tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"frozen")
    check_artifact(artifact, file_md5(artifact))
    with pytest.raises(ValueError, match="md5"):
        check_artifact(artifact, "0" * 32)


def test_guard_refuses_rerun_without_force(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    guard_rerun(["time"], tmp_path, force=False)  # nothing yet: allowed
    (tmp_path / "final_time.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        guard_rerun(["time", "stratified"], tmp_path, force=False)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "REFUSING" in out and "final_time.json" in out and "--force" in out
    guard_rerun(["time"], tmp_path, force=True)  # explicit override passes
    guard_rerun(["stratified"], tmp_path, force=False)  # other split not done yet


def test_rerun_warning_explains_why() -> None:
    assert "optimistic" in RERUN_WARNING and "second validation set" in RERUN_WARNING


def test_reference_costs_hand_computed() -> None:
    y = pd.Series([1, 0, 0, 1, 0])
    amounts = pd.Series([100.0, 5.0, 5.0, 40.0, 5.0])
    ref = reference_costs(y, amounts, review_cost=5.0)
    assert ref["no_model"]["expected_cost"] == 140.0  # all fraud Amounts lost
    assert ref["flag_everything"]["expected_cost"] == 25.0  # 5 rows x 5, nothing missed


def test_score_frozen_applies_given_thresholds_without_recomputing() -> None:
    y = pd.Series([1, 1, 0, 0, 1, 0])
    proba = np.array([0.90, 0.30, 0.25, 0.10, 0.01, 0.0])
    amounts = pd.Series([10.0] * 6)
    s = score_frozen(y, proba, amounts, t_review=0.25, t_block=0.80, review_cost=5.0)
    r, b = s["at_t_review"], s["at_t_block"]
    assert (r["threshold"], r["tp"], r["fp"], r["fn"]) == (0.25, 2, 1, 1)
    assert (b["threshold"], b["tp"], b["fp"]) == (0.80, 1, 0)
    tiers = {t["tier"]: t["count"] for t in s["tiers"]}
    assert tiers == {"HIGH": 1, "MEDIUM": 2, "LOW": 3}


def test_score_frozen_without_t_block() -> None:
    y = pd.Series([1, 0])
    s = score_frozen(y, np.array([0.9, 0.1]), pd.Series([1.0, 1.0]), 0.5, None, 5.0)
    assert s["at_t_block"] is None
    assert {t["tier"]: t["count"] for t in s["tiers"]}["HIGH"] == 0


def test_load_test_reads_the_named_part(tmp_path: Path) -> None:
    pd.DataFrame({"a": [1, 2]}).to_parquet(tmp_path / "time_test.parquet")
    assert len(load_test("time", tmp_path)) == 2


def test_summary_has_placeholder_for_king_and_marks_headline() -> None:
    ci = {"point": 0.5, "ci_low": 0.4, "ci_high": 0.6}
    cis = dict.fromkeys(("pr_auc", "recall", "precision", "alerts_per_1000", "expected_cost"), ci)
    result = {
        "split": "time", "headline": True,
        "frozen_policy": {"model": "logreg", "calibration": "isotonic", "t_review": 0.02},
        "at_t_review_ci": cis, "at_t_block": None, "at_t_block_ci": None,
        "reference_costs": {"no_model": {"expected_cost": 100.0},
                            "flag_everything": {"expected_cost": 200.0}},
        "context_baselines": {"dummy": {"pr_auc": 0.001, "pr_auc_ci": [0.0, 0.002],
                                        "roc_auc": 0.5}},
    }  # fmt: skip
    md = summary_markdown([result])
    assert "time (headline)" in md and "TODO (KING)" in md and "does not exist" in md


def test_evaluate_never_imports_selection_or_calibration_code() -> None:
    """evaluate.py applies frozen decisions; it must not be able to re-derive them."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    imported_modules, imported_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module)
            imported_names.update(a.name for a in node.names)
    forbidden_modules = {"fraud_detection.calibrate", "fraud_detection.tune",
                         "fraud_detection.champion", "fraud_detection.threshold"}  # fmt: skip
    forbidden_names = {"choose_t_review", "choose_t_block", "choose_threshold",
                       "choose_block_threshold", "threshold_table", "fit_calibrator",
                       "load_part", "choose_champion"}  # fmt: skip
    assert not imported_modules & forbidden_modules
    assert not imported_names & forbidden_names
    assert '"valid"' not in SRC.read_text(encoding="utf-8")  # never loads validation data
