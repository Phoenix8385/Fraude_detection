"""Model factory: every model is returned as an imblearn Pipeline with the same interface.

imblearn's Pipeline behaves like sklearn's, and additionally allows resampling steps
(e.g. SMOTE in Phase 6) that run on training data only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.pipeline import Pipeline
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

MODEL_NAMES: tuple[str, ...] = ("dummy", "logreg", "iforest")


class IsolationForestScorer(ClassifierMixin, BaseEstimator):
    """Unsupervised Isolation Forest exposed through a classifier-style predict_proba.

    fit() ignores the labels entirely: the forest only learns what "normal" data looks
    like. The anomaly score (higher = more anomalous) is min-max scaled using the range
    seen on the TRAINING data, then clipped to [0, 1] so it can be thresholded like a
    probability. It is a ranking score, not a calibrated fraud probability.
    """

    def __init__(
        self, n_estimators: int = 100, max_samples: Any = "auto", random_state: int | None = None
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> IsolationForestScorer:
        """Fit the forest on X only; y is accepted for Pipeline compatibility and ignored."""
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X)
        train_scores = self._anomaly_score(X)
        self.score_min_ = float(train_scores.min())
        self.score_max_ = float(train_scores.max())
        self.classes_ = np.array([0, 1])
        return self

    def _anomaly_score(self, X: ArrayLike) -> np.ndarray:
        # sklearn's score_samples is higher for NORMAL points; negate so higher = anomalous
        return -self.forest_.score_samples(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Columns: [1 - score, score], score = train-min-max-scaled anomaly score."""
        check_is_fitted(self, "forest_")
        span = self.score_max_ - self.score_min_
        scaled = (self._anomaly_score(X) - self.score_min_) / span if span else np.zeros(len(X))
        score = np.clip(scaled, 0.0, 1.0)
        return np.column_stack([1.0 - score, score])

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Label 1 when the scaled anomaly score is >= 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_model(
    name: str, params: dict[str, Any] | None = None, scale_pos_weight: float | None = None
) -> Pipeline:
    """Return an unfitted Pipeline for the named model.

    Args:
        name: one of MODEL_NAMES.
        params: keyword arguments for the model step (e.g. {"random_state": 42}).
        scale_pos_weight: class-weight ratio for XGBoost (Phase 6); not used by these models.
    """
    params = dict(params or {})
    if scale_pos_weight is not None:
        raise ValueError(f"scale_pos_weight is not supported for model '{name}'")

    if name == "dummy":
        return Pipeline([("model", DummyClassifier(strategy="most_frequent", **params))])
    if name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000, **params)),
            ]
        )
    if name == "iforest":
        return Pipeline([("model", IsolationForestScorer(**params))])
    raise ValueError(f"unknown model '{name}'; expected one of {MODEL_NAMES}")
