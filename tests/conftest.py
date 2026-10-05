"""Shared synthetic fixtures. Tests never read the real CSV or the real model."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.schema import FEATURE_COLUMNS, TARGET


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Ten valid rows shaped like creditcard.csv: 2 fraud, no duplicates."""
    rng = np.random.default_rng(0)
    n = 10
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df["Time"] = np.arange(n, dtype="float64") * 10.0
    df["Amount"] = rng.uniform(0, 500, size=n).round(2)
    df[TARGET] = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    return df


@pytest.fixture(scope="session")
def synthetic_artifact_dir(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    """A tiny model packaged exactly like the production artifact (scaler + LR + isotonic,
    model.joblib + model_meta.json), trained on synthetic data. Thresholds 0.3 / 0.7 differ
    on purpose from production (0.02 / 0.23): code must read them from the metadata."""
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from fraud_detection.artifacts import file_md5, library_versions, save_artifact, schema_hash
    from fraud_detection.features import MODEL_FEATURES, add_features

    rng = np.random.default_rng(0)
    n = 600
    raw = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    raw["Time"] = rng.uniform(0, 172_000, n)
    raw["Amount"] = rng.uniform(0, 500, n)
    y = pd.Series((np.arange(n) % 10 == 0).astype(int))
    raw.loc[y == 1, "V14"] -= 4
    X = add_features(raw)[MODEL_FEATURES]
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
    pipe.fit(X.iloc[:400], y.iloc[:400])
    model = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    model.fit(X.iloc[400:], y.iloc[400:])

    root = tmp_path_factory.mktemp("artifact")
    source = root / "final_model_time.joblib"
    joblib.dump(model, source)
    models_dir = root / "models"
    models_dir.mkdir()
    meta = {
        "model_version": "synthetic-test-v0", "model_name": "logreg_tuned",
        "headline_split": "time", "calibrated": True, "calibration": "isotonic",
        "thresholds": {"review": 0.3, "block": 0.7}, "t_review_validation_fallback": False,
        "features": MODEL_FEATURES, "input_columns": FEATURE_COLUMNS,
        "schema_hash": schema_hash(), "artifact_md5": file_md5(source),
        "library_versions": library_versions(), "explanations": "synthetic",
        "final_test_metrics": {"source": "synthetic"},
    }  # fmt: skip
    save_artifact(source, models_dir, meta)
    return models_dir


@pytest.fixture
def synthetic_transaction() -> dict:
    """One synthetic request body (all V = 0)."""
    return {"Time": 3600.0, "Amount": 25.0, **{f"V{i}": 0.0 for i in range(1, 29)}}


@pytest.fixture(scope="session", autouse=True)
def synthetic_model_path(synthetic_artifact_dir):  # type: ignore[no-untyped-def]
    """Point MODEL_PATH at the synthetic artifact for the WHOLE session.

    Any Settings.from_env() during tests therefore loads the synthetic model, never the real
    models/model.joblib (rule 8). MODELS_DIR is cleared so it cannot override this.
    """
    import os

    saved = {k: os.environ.get(k) for k in ("MODEL_PATH", "MODELS_DIR")}
    os.environ["MODEL_PATH"] = str(synthetic_artifact_dir / "model.joblib")
    os.environ.pop("MODELS_DIR", None)
    yield os.environ["MODEL_PATH"]
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
