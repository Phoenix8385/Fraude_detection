"""End-to-end pipeline on SYNTHETIC data in a temporary project root (Phase 12).

Runs every stage's real CLI entry point — data -> splits -> train -> compare -> tune ->
champion -> calibrate -> evaluate -> artifacts -> explain — against generated data, a temp
config (tiny budgets) and a temp MLflow store. Nothing here reads the real CSV, the real
processed splits, the real model or real reports, and nothing is written into the repo:
`load_config` is redirected to the temp root in every module (and artifacts.PROJECT_ROOT).
The final test checks that the repo's frozen artifacts were not touched.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection import (
    artifacts,
    calibrate,
    champion,
    compare,
    data,
    evaluate,
    explain,
    splits,
    train,
    tune,
)
from fraud_detection.config import PROJECT_ROOT, load_config
from fraud_detection.schema import FEATURE_COLUMNS, TARGET

MODULES_USING_CONFIG = (data, splits, train, compare, tune, champion, calibrate, evaluate,
                        artifacts, explain)  # fmt: skip
N_ROWS, N_DUPLICATES = 3000, 5
REAL_FROZEN = [PROJECT_ROOT / "models" / "model.joblib",
               PROJECT_ROOT / "reports" / "metrics" / "final_time.json"]  # fmt: skip


def _md5(path: Path) -> str | None:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None


def _synthetic_csv(path: Path) -> None:
    """Creditcard-shaped CSV: linear fraud signal, fraud spread over time, a few duplicates."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(size=(N_ROWS, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df["Time"] = np.sort(rng.integers(0, 172_800, N_ROWS)).astype(float)
    df["Amount"] = rng.uniform(0, 300, N_ROWS).round(2)
    y = (rng.uniform(size=N_ROWS) < 0.05).astype(int)
    for col, shift in (("V14", -2.5), ("V4", 1.5), ("V10", -1.5)):
        df.loc[y == 1, col] += shift
    df[TARGET] = y
    df = pd.concat([df, df.iloc[:N_DUPLICATES]], ignore_index=True)  # exact duplicates
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _tiny_config(root: Path) -> Path:
    text = (PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    for old, new in (("n_trials: 50", "n_trials: 2"), ("cv_folds: 5", "cv_folds: 2"),
                     ("n_resamples: 1000", "n_resamples: 20"),
                     ("[42, 43, 44, 45, 46]", "[42, 43]"),
                     ("stability_resamples: 200", "stability_resamples: 10")):  # fmt: skip
        assert old in text
        text = text.replace(old, new)
    path = root / "configs" / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_preregistration(cfg) -> None:  # type: ignore[no-untyped-def]
    """Same shape as the real pre-registration, built from the synthetic outputs."""
    m = cfg.paths.metrics_dir
    policy = json.loads((m / "policy_time.json").read_text(encoding="utf-8"))
    runs = compare.latest_runs(m / "experiments.csv")
    base = runs[(runs["split"] == "time") & runs["model"].isin(["dummy", "logreg", "iforest"])]
    prereg = {
        "phase8_commit": "synthetic", "headline_split": "time", "primary_metric": "pr_auc",
        "bootstrap": {"n_resamples": cfg.bootstrap.n_resamples, "ci_level": 0.95,
                      "stratified": True, "seed": cfg.seed},
        "splits": {"time": {
            "champion": policy["model"], "mlflow_run_id": policy["mlflow_run_id"],
            "calibration": policy["calibration"], "t_review": policy["t_review"],
            "t_block": policy["t_block"], "review_cost": policy["review_cost"],
            "valid_t_review_recall": policy["recall"],
            "fallback": policy["t_review_detail"]["fallback"],
            "artifact": "models/final_model_time.joblib",
            "artifact_md5": _md5(cfg.paths.models_dir / "final_model_time.joblib"),
        }},
        "context_baselines_phase5": {"time": dict(zip(base["model"], base["run_id"], strict=True))},
    }  # fmt: skip
    (m / "preregistration.json").write_text(json.dumps(prereg), encoding="utf-8")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    """Run the whole pipeline once; tests below inspect its outputs."""
    root = tmp_path_factory.mktemp("synthetic_project")
    before = {p: _md5(p) for p in REAL_FROZEN}
    cfg = load_config(_tiny_config(root), root=root)
    _synthetic_csv(cfg.paths.raw_data)

    with pytest.MonkeyPatch.context() as mp:
        for module in MODULES_USING_CONFIG:
            mp.setattr(module, "load_config", lambda *a, **k: cfg)
        mp.setattr(artifacts, "PROJECT_ROOT", root)

        data.main()
        splits.main()
        train.main(["--models", "dummy", "logreg", "iforest", "xgb_weighted",
                    "--splits", "stratified", "time"])  # fmt: skip
        compare.main()
        tune.main(["--models", "xgb_u", "logreg", "--splits", "time", "--n-trials", "2"])
        champion.main(["--splits", "time"])
        calibrate.main(["--splits", "time"])
        _write_preregistration(cfg)
        evaluate.main(["--splits", "time"])
        with pytest.raises(SystemExit) as rerun:  # one-shot guard on a second run
            evaluate.main(["--splits", "time"])
        artifacts.main()
        explain.main()
    yield {"cfg": cfg, "root": root, "before": before, "rerun_code": rerun.value.code}


def _read(cfg, name: str) -> dict:  # type: ignore[no-untyped-def]
    return json.loads((cfg.paths.metrics_dir / name).read_text(encoding="utf-8"))


def test_data_and_splits(pipeline) -> None:  # type: ignore[no-untyped-def]
    cfg = pipeline["cfg"]
    summary = _read(cfg, "data_summary.json")
    assert summary["duplicates_dropped"] == N_DUPLICATES
    assert summary["rows"] == N_ROWS
    for name in ("stratified", "time"):
        s = _read(cfg, f"splits_{name}.json")
        assert sum(s[p]["rows"] for p in ("train", "valid", "test")) == N_ROWS
    t = _read(cfg, "splits_time.json")
    assert t["train"]["time_max"] <= t["valid"]["time_min"] <= t["valid"]["time_max"]


def test_training_and_comparison(pipeline) -> None:  # type: ignore[no-untyped-def]
    cfg = pipeline["cfg"]
    runs = pd.read_csv(cfg.paths.metrics_dir / "experiments.csv")
    assert {"dummy", "logreg", "iforest", "xgb_weighted", "xgb_u_tuned", "logreg_tuned"} <= set(
        runs["model"])  # fmt: skip
    assert (cfg.paths.metrics_dir / "model_comparison.md").exists()
    assert (cfg.paths.figures_dir / "pr_curves_time.png").exists()


def test_tuning_and_champion(pipeline) -> None:  # type: ignore[no-untyped-def]
    cfg = pipeline["cfg"]
    for model in ("xgb_u", "logreg"):
        bp = _read(cfg, f"best_params_{model}_time.json")
        assert bp["n_trials"] == 2 and 0 <= bp["valid_pr_auc"] <= 1
    ch = _read(cfg, "champion_time.json")
    assert ch["decision"]["champion"] in ("xgb_u", "logreg")
    assert ch["bootstrap"]["n_resamples"] == 20
    assert 0 <= ch["paired_pr_auc"]["prob_a_better"] <= 1
    assert len(ch["models"]["logreg"]["seed_stability"]["valid_pr_auc"]) == 2
    assert (cfg.paths.metrics_dir / "champion_summary.md").exists()


def test_calibration_and_policy(pipeline) -> None:  # type: ignore[no-untyped-def]
    cfg = pipeline["cfg"]
    cal, pol = _read(cfg, "calibration_time.json"), _read(cfg, "policy_time.json")
    assert cal["selected"] in ("none", "sigmoid", "isotonic")
    assert set(cal["candidates_valid_thr"]) == {"none", "sigmoid", "isotonic"}
    assert cal["halves"]["valid_cal"]["time_max"] <= cal["halves"]["valid_thr"]["time_min"]
    assert 0.01 <= pol["t_review"] <= 0.95
    assert pol["t_block"] is None or pol["t_block"] >= 0.01
    assert pol["stability"]["n_resamples"] == 10
    assert (cfg.paths.models_dir / "final_model_time.joblib").exists()


def test_one_shot_evaluation(pipeline) -> None:  # type: ignore[no-untyped-def]
    cfg = pipeline["cfg"]
    final, pol = _read(cfg, "final_time.json"), _read(cfg, "policy_time.json")
    assert final["frozen_policy"]["t_review"] == pol["t_review"]  # applied, not recomputed
    assert final["test_rows"] == _read(cfg, "splits_time.json")["test"]["rows"]
    assert set(final["context_baselines"]) == {"dummy", "logreg", "iforest"}
    assert pipeline["rerun_code"] == 1  # second run refused without --force
    assert (cfg.paths.metrics_dir / "final_summary.md").exists()


def test_artifact_and_explanations(pipeline) -> None:  # type: ignore[no-untyped-def]
    cfg = pipeline["cfg"]
    models = cfg.paths.models_dir
    assert _md5(models / "model.joblib") == _md5(models / "final_model_time.joblib")
    meta = json.loads((models / "model_meta.json").read_text(encoding="utf-8"))
    pol = _read(cfg, "policy_time.json")
    assert meta["thresholds"] == {"review": pol["t_review"], "block": pol["t_block"]}
    profile = json.loads((models / "reference_profile.json").read_text(encoding="utf-8"))
    assert abs(sum(profile["features"]["Amount"]["proportions"]) - 1) < 1e-9
    exp = _read(cfg, "explain_time.json")
    assert len(exp["global_importance"]) == 32
    assert (cfg.paths.figures_dir / "shap_waterfall_fraud_time.png").exists()


def test_real_repo_artifacts_untouched(pipeline) -> None:  # type: ignore[no-untyped-def]
    assert {p: _md5(p) for p in REAL_FROZEN} == pipeline["before"]
    assert str(pipeline["cfg"].paths.models_dir).startswith(str(pipeline["root"]))
