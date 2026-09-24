"""Train / valid / test splits: stratified (random) and time-based (primary).

Both split functions keep the original row index, so every row can be traced back to the
clean dataset and overlap between parts can be checked.

Run as a script to build and save both split sets:
    python -m fraud_detection.splits
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import SplitConfig, load_config
from fraud_detection.schema import TARGET

logger = logging.getLogger(__name__)

PARTS: tuple[str, ...] = ("train", "valid", "test")


def stratified_split(df: pd.DataFrame, ratios: SplitConfig, seed: int) -> dict[str, pd.DataFrame]:
    """Random split that keeps the fraud rate equal in every part.

    Two successive train_test_split calls: first carve off test, then split the rest
    into train and valid. valid's share of the remainder is valid / (train + valid),
    e.g. 0.2 / 0.8 = 0.25, which gives 60/20/20 of the whole.
    """
    rest, test = train_test_split(
        df, test_size=ratios.test, stratify=df[TARGET], random_state=seed
    )
    valid_share = ratios.valid / (ratios.train + ratios.valid)
    train, valid = train_test_split(
        rest, test_size=valid_share, stratify=rest[TARGET], random_state=seed
    )
    return {"train": train, "valid": valid, "test": test}


def time_split(df: pd.DataFrame, ratios: SplitConfig) -> dict[str, pd.DataFrame]:
    """Chronological split: oldest rows train, newest rows test.

    Mimics production, where a model trained on the past scores the future. Stable sort
    keeps rows with equal Time in their original order, so the result is deterministic.
    """
    ordered = df.sort_values("Time", kind="stable")
    n = len(ordered)
    train_end = round(n * ratios.train)
    valid_end = round(n * (ratios.train + ratios.valid))
    return {
        "train": ordered.iloc[:train_end],
        "valid": ordered.iloc[train_end:valid_end],
        "test": ordered.iloc[valid_end:],
    }


def summarize_splits(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Rows, fraud count, fraud rate and Time range for each part."""
    summary: dict[str, dict[str, Any]] = {}
    for part in PARTS:
        frame = splits[part]
        fraud = int(frame[TARGET].sum())
        summary[part] = {
            "rows": len(frame),
            "fraud_count": fraud,
            "fraud_rate": fraud / len(frame) if len(frame) else 0.0,
            "time_min": float(frame["Time"].min()),
            "time_max": float(frame["Time"].max()),
        }
    return summary


def save_splits(
    splits: dict[str, pd.DataFrame],
    name: str,
    processed_dir: Path | None = None,
    metrics_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Write {name}_{part}.parquet for each part and splits_{name}.json; return the summary.

    Directories default to paths.processed_dir and paths.metrics_dir from the config.
    """
    if processed_dir is None or metrics_dir is None:
        paths = load_config().paths
        processed_dir = processed_dir or paths.processed_dir
        metrics_dir = metrics_dir or paths.metrics_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for part in PARTS:
        out = processed_dir / f"{name}_{part}.parquet"
        splits[part].to_parquet(out)  # index kept: it is the row id in the clean dataset
        logger.info("Wrote %s (%d rows)", out.name, len(splits[part]))

    summary = summarize_splits(splits)
    summary_path = metrics_dir / f"splits_{name}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved split summary to %s", summary_path)
    return summary


def main() -> None:
    """CLI: build stratified and time splits from the clean, feature-engineered data."""
    from fraud_detection.data import load_clean
    from fraud_detection.features import add_features

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    df, _ = load_clean(config.paths.raw_data)
    df = add_features(df)

    results = {
        "stratified": save_splits(stratified_split(df, config.split, config.seed), "stratified"),
        "time": save_splits(time_split(df, config.split), "time"),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
