"""Phase 10 — explanations for the headline production model (time split, tuned Logistic
Regression + isotonic calibration). The stratified model is not explained here.

Method: for a scaled linear model the log-odds are exactly
    intercept + sum_j coef_j * z_j,   z_j = (x_j - mean_j) / std_j  (the pipeline's scaler)
so each feature's contribution is coef_j * z_j and the base value is the intercept (the
log-odds of a transaction sitting at the training mean). This equals linear SHAP with the
training mean as background, needs no `shap` library at serving time, and sums exactly to
the model's log-odds.

IMPORTANT: contributions are in log-odds of the UNCALIBRATED Logistic Regression. The served
fraud_probability additionally passes through the isotonic calibrator, which is monotonic, so
the ranking is unchanged but the numbers are not probability deltas. A contribution is the
model's arithmetic ("model contribution"), never a cause of fraud.

Global plots use 2,000 valid_thr rows (docs/evaluation.md: valid_thr is the SHAP sample); the
`shap` library is used for plotting only. No raw V1..V28 values are written to any file.
The test split is never read.

    python -m fraud_detection.explain
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # file output only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraud_detection.config import Config, load_config  # noqa: E402
from fraud_detection.features import MODEL_FEATURES  # noqa: E402
from fraud_detection.schema import TARGET  # noqa: E402
from fraud_detection.train import load_part  # noqa: E402

logger = logging.getLogger(__name__)

HEADLINE_SPLIT = "time"
SAMPLE_ROWS = 2000
TOP_K = 5


def linear_pipeline(model: Any) -> Any:
    """The scaler + LogisticRegression pipeline inside the production model.

    Accepts the pipeline itself or CalibratedClassifierCV(FrozenEstimator(pipeline)).
    """
    est = model
    if hasattr(est, "calibrated_classifiers_"):
        est = est.calibrated_classifiers_[0].estimator
    while not hasattr(est, "named_steps") and hasattr(est, "estimator"):
        est = est.estimator  # FrozenEstimator -> wrapped pipeline
    steps = getattr(est, "named_steps", {})
    if "scaler" not in steps or not hasattr(steps.get("model"), "coef_"):
        raise TypeError("expected a fitted scaler + linear model pipeline")
    return est


def linear_contributions(model: Any, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per-row, per-feature log-odds contributions and the base value (intercept).

    base + contributions.sum(axis=1) == the uncalibrated model's decision_function(X).
    """
    pipe = linear_pipeline(model)
    z = pipe.named_steps["scaler"].transform(X[MODEL_FEATURES])
    lr = pipe.named_steps["model"]
    contrib = pd.DataFrame(z * lr.coef_[0], columns=MODEL_FEATURES, index=X.index)
    return contrib, float(lr.intercept_[0])


def explain_row(model: Any, row: pd.DataFrame, top_k: int = TOP_K) -> dict[str, Any]:
    """Top-k contributions for ONE transaction, sorted by |contribution|.

    Returned values are in log-odds of the UNCALIBRATED model (see module docstring);
    "value" echoes the caller's own input and is not stored anywhere by this function.
    """
    if len(row) != 1:
        raise ValueError("explain_row expects exactly one row")
    contrib, base = linear_contributions(model, row)
    c = contrib.iloc[0]
    order = c.abs().sort_values(ascending=False).index[:top_k]
    return {
        "base_value": base,
        "log_odds": base + float(c.sum()),
        "units": "log-odds, uncalibrated model",
        "contributions": [
            {"feature": f, "value": float(row[f].iloc[0]), "contribution": float(c[f])}
            for f in order
        ],
    }


def global_importance(contrib: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Mean |contribution| per feature on the sample, with the coefficient (scaled units)."""
    coef = linear_pipeline(model).named_steps["model"].coef_[0]
    out = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "mean_abs_contribution": contrib.abs().mean().to_numpy(),
            "mean_contribution": contrib.mean().to_numpy(),
            "coefficient_scaled": coef,
        }
    )
    return out.sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)


def valid_thr_rows(split: str, config: Config) -> pd.DataFrame:
    """The valid_thr half saved in Phase 8 (validation data only)."""
    valid = load_part(split, "valid", config.paths.processed_dir)
    idx_path = config.paths.metrics_dir / f"valid_subsplit_indices_{split}.json"
    thr_idx = json.loads(idx_path.read_text(encoding="utf-8"))["thr_indices"]
    return valid.loc[thr_idx]


def _explanation(contrib: pd.DataFrame, base: float, z: np.ndarray) -> Any:
    """shap.Explanation for PLOTTING only. data = scaled values (colour), never raw V values."""
    import shap

    return shap.Explanation(
        values=contrib.to_numpy(),
        base_values=np.full(len(contrib), base),
        data=z,
        feature_names=MODEL_FEATURES,
    )


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved %s", path)


def run_explain(model: Any, config: Config) -> dict[str, Any]:
    """Global + local explanation outputs for the headline model on valid_thr."""
    import shap

    figures, metrics = config.paths.figures_dir, config.paths.metrics_dir
    thr = valid_thr_rows(HEADLINE_SPLIT, config)
    sample = thr.sample(n=min(SAMPLE_ROWS, len(thr)), random_state=config.seed)
    contrib, base = linear_contributions(model, sample)
    z = linear_pipeline(model).named_steps["scaler"].transform(sample[MODEL_FEATURES])
    expl = _explanation(contrib, base, z)

    shap.plots.beeswarm(expl, max_display=15, show=False)
    plt.title("Model contributions (log-odds, uncalibrated LR) — valid_thr sample")
    _save(figures / "shap_beeswarm_time.png")
    shap.plots.bar(expl, max_display=15, show=False)
    plt.title("Mean |contribution| (log-odds) — valid_thr sample")
    _save(figures / "shap_bar_time.png")

    # one true fraud and one true legit row from valid_thr, seeded
    rng = np.random.default_rng(config.seed)
    examples = {}
    for label, name in ((1, "fraud"), (0, "legit")):
        pick = thr.loc[[rng.choice(thr.index[thr[TARGET] == label])]]
        c_row, _ = linear_contributions(model, pick)
        # data=None: the waterfall shows feature names only, no raw values
        row_expl = shap.Explanation(values=c_row.to_numpy()[0], base_values=base,
                                    data=None, feature_names=MODEL_FEATURES)  # fmt: skip
        shap.plots.waterfall(row_expl, max_display=10, show=False)
        plt.title(f"Example true {name} (valid_thr) — log-odds, uncalibrated LR")
        _save(figures / f"shap_waterfall_{name}_time.png")
        exp = explain_row(model, pick)
        examples[name] = {
            "base_value": exp["base_value"],
            "log_odds": exp["log_odds"],
            # contributions only: raw feature values are never written to disk
            "top_contributions": [
                {"feature": c["feature"], "contribution": c["contribution"]}
                for c in exp["contributions"]
            ],
        }

    importance = global_importance(contrib, model)
    result = {
        "model": "logreg (tuned, time split) + isotonic calibration",
        "method": "coef_j * scaled x_j; base = intercept; log-odds of the UNCALIBRATED LR",
        "interpretation": "model contribution, not cause",
        "sample": f"{len(sample)} valid_thr rows (time split), seed {config.seed}",
        "base_value": base,
        "global_importance": importance.to_dict(orient="records"),
        "examples_valid_thr": examples,
    }
    (metrics / "explain_time.json").write_text(json.dumps(result, indent=2) + "\n",
                                                encoding="utf-8")  # fmt: skip
    return result


def main() -> None:
    """CLI: explain the production artifact (models/model.joblib) on valid_thr."""
    from fraud_detection.artifacts import load_artifact

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    model, meta = load_artifact(config.paths.models_dir)
    logger.info("Explaining %s", meta["model_version"])
    result = run_explain(model, config)
    top = [r["feature"] for r in result["global_importance"][:5]]
    print(f"{meta['model_version']}: top features by mean |contribution|: {top}")


if __name__ == "__main__":
    main()
