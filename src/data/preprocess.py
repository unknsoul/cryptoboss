"""Data preprocessing helpers."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.data.schema import standardize_ohlcv


TIMEFRAME_MAP = {
    "1m": "1T",
    "3m": "3T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "1d": "1D",
}


def clean_ohlcv(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Sort, drop duplicates, and normalize timestamp column."""
    cleaned = standardize_ohlcv(df, keep_extra_columns=True, timestamp_col=timestamp_col)
    return cleaned.reset_index(drop=True)


def resample_ohlcv(df: pd.DataFrame, timeframe: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Resample OHLCV data to a new timeframe."""
    rule = TIMEFRAME_MAP.get(timeframe, timeframe)
    working = standardize_ohlcv(df, keep_extra_columns=False, timestamp_col=timestamp_col)

    if "timestamp" in working.columns:
        working = working.set_index("timestamp")

    aggregated = working.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"])
    aggregated = aggregated.reset_index().rename(columns={timestamp_col: "timestamp"})
    return standardize_ohlcv(aggregated)


def align_timeframes(frames: Dict[str, pd.DataFrame], timestamp_col: str = "timestamp") -> Dict[str, pd.DataFrame]:
    """Align multiple timeframes on shared timestamps."""
    indexed = {}
    for name, frame in frames.items():
        working = frame.copy()
        if timestamp_col in working.columns:
            working = standardize_ohlcv(working, keep_extra_columns=True, timestamp_col=timestamp_col)
            working = working.set_index("timestamp")
        indexed[name] = working.sort_index()

    common_index = None
    for frame in indexed.values():
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)

    if common_index is None:
        return frames

    aligned = {}
    for name, frame in indexed.items():
        aligned_frame = frame.loc[common_index].reset_index().rename(columns={"index": "timestamp"})
        aligned[name] = aligned_frame

    return aligned
