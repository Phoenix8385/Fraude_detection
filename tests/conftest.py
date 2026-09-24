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
