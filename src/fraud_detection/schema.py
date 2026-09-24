"""Single source of truth for dataset columns and the raw-data validation schema."""

from __future__ import annotations

import pandera.pandas as pa

TARGET: str = "Class"

# Written out explicitly (not generated) so the column contract is readable at a glance.
FEATURE_COLUMNS: list[str] = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]  # fmt: skip

_NON_NEGATIVE: set[str] = {"Time", "Amount"}


def _feature_column(name: str) -> pa.Column:
    """Float, never null; Time and Amount must also be >= 0."""
    checks = [pa.Check.ge(0)] if name in _NON_NEGATIVE else []
    return pa.Column(float, checks=checks, nullable=False)


RAW_SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
    columns={
        **{name: _feature_column(name) for name in FEATURE_COLUMNS},
        TARGET: pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False),
    },
    strict=True,  # any extra column is an error
    name="creditcard_raw",
)
