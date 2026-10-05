"""Per-transaction model contributions for the production Logistic Regression.

Light module (numpy/pandas only) so the API can explain predictions without importing the
plotting / MLflow code in explain.py. Moved unchanged from explain.py (Phase 10), which
re-exports these names.

For a scaled linear model the log-odds are exactly
    intercept + sum_j coef_j * z_j,   z_j = (x_j - mean_j) / std_j  (the pipeline's scaler)
so each feature's contribution is coef_j * z_j and the base value is the intercept.

IMPORTANT: contributions are in log-odds of the UNCALIBRATED Logistic Regression; the served
fraud_probability additionally passes through the isotonic calibrator. A contribution is the
model's arithmetic ("model contribution"), never a cause of fraud.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fraud_detection.features import MODEL_FEATURES

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
