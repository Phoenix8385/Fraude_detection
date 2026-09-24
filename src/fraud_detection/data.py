"""Load the raw credit-card CSV, validate it against the schema, and drop exact duplicates.

Run as a script to print and save a data summary:
    python -m fraud_detection.data
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.config import load_config
from fraud_detection.schema import FEATURE_COLUMNS, RAW_SCHEMA, TARGET

logger = logging.getLogger(__name__)


def load_raw(path: Path | str) -> pd.DataFrame:
    """Read the CSV and cast feature columns to float64.

    The CSV stores Time as whole seconds, so pandas reads it as int64; casting makes
    every feature float as the schema requires. Values are unchanged.
    """
    df = pd.read_csv(path)
    present = [col for col in FEATURE_COLUMNS if col in df.columns]
    df[present] = df[present].astype("float64")
    logger.info("Loaded %d rows x %d columns from %s", len(df), df.shape[1], path)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate against RAW_SCHEMA; raises pandera SchemaErrors listing every failure."""
    return RAW_SCHEMA.validate(df, lazy=True)


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows identical in every column (label included), keeping the first occurrence.

    Must run before any train/valid/test split, otherwise a row and its copy can land
    on both sides and leak information into evaluation.
    """
    is_dup = df.duplicated(keep="first")
    n_dropped = int(is_dup.sum())
    n_fraud_dropped = int(df.loc[is_dup, TARGET].sum())
    logger.info("Dropped %d exact duplicate rows (%d of them fraud)", n_dropped, n_fraud_dropped)
    return df.loc[~is_dup].reset_index(drop=True), n_dropped


def load_clean(path: Path | str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, validate and de-duplicate the raw data.

    Args:
        path: CSV to read. Defaults to paths.raw_data from the config.

    Returns:
        The clean DataFrame and a summary dict.
    """
    csv_path = Path(path) if path is not None else load_config().paths.raw_data
    raw = validate(load_raw(csv_path))
    raw_fraud = int(raw[TARGET].sum())

    clean, n_dropped = drop_exact_duplicates(raw)
    fraud_count = int(clean[TARGET].sum())
    summary: dict[str, Any] = {
        "rows_raw": len(raw),
        "rows": len(clean),
        "fraud_count": fraud_count,
        "fraud_rate": fraud_count / len(clean) if len(clean) else 0.0,
        "duplicates_dropped": n_dropped,
        "duplicate_fraud_dropped": raw_fraud - fraud_count,
    }
    return clean, summary


def save_summary(summary: dict[str, Any], path: Path) -> None:
    """Write the summary dict as pretty JSON, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved data summary to %s", path)


def main() -> None:
    """CLI: load clean data, save reports/metrics/data_summary.json, print the summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    _, summary = load_clean(config.paths.raw_data)
    save_summary(summary, config.paths.metrics_dir / "data_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
