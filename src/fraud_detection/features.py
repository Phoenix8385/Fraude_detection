"""Feature engineering shared by training and the API.

Pure functions only: no fitting, no global state. Whatever is computed here at training
time is computed identically at serving time, so there is no train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.schema import FEATURE_COLUMNS

ENGINEERED_FEATURES: list[str] = ["log_amount", "hour_of_day"]
MODEL_FEATURES: list[str] = FEATURE_COLUMNS + ENGINEERED_FEATURES

_SECONDS_PER_HOUR = 3600


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a NEW DataFrame with engineered columns added; the input is not modified.

    - log_amount = log1p(Amount): compresses the long right tail of amounts; log1p(0) = 0.
    - hour_of_day = (Time // 3600) % 24.

    Note: Time is seconds elapsed since the first transaction in the dataset, not a clock
    time. hour_of_day therefore equals the real hour of day only if the data starts at
    midnight, which the dataset does not document. Treat it as "hour within a 24h cycle".
    """
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"])
    out["hour_of_day"] = (out["Time"] // _SECONDS_PER_HOUR) % 24
    return out
