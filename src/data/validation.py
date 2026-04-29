"""Data validation helpers for OHLCV and alternative feeds."""

from __future__ import annotations

import pandas as pd

from src.data.schema import standardize_ohlcv


def validate_ohlcv(
    df: pd.DataFrame,
    max_duplicate_candles: int = 10,
    timestamp_col: str = "timestamp",
) -> None:
    """Validate OHLCV integrity and raise ValueError on failure."""
    df = standardize_ohlcv(
        df,
        keep_extra_columns=True,
        timestamp_col=timestamp_col,
        drop_invalid_rows=False,
    )

    if df is None or df.empty:
        raise ValueError("OHLCV dataframe is empty")

    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Missing OHLCV columns: %s" % missing)

    if df[required].isnull().any().any():
        raise ValueError("NaN detected in OHLCV columns")

    if (df["volume"] < 0).any():
        raise ValueError("Negative volume detected")

    if (df["high"] < df["low"]).any():
        raise ValueError("High < low detected")

    if ((df["open"] < df["low"]) | (df["open"] > df["high"])).any():
        raise ValueError("Open outside high/low range")

    if ((df["close"] < df["low"]) | (df["close"] > df["high"])).any():
        raise ValueError("Close outside high/low range")

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        if timestamps.isnull().any():
            raise ValueError("Invalid timestamps detected")
        if not timestamps.is_monotonic_increasing:
            raise ValueError("Timestamps not sorted")
    elif not df.index.is_monotonic_increasing:
        raise ValueError("Index not sorted")

    duplicates = int(df.duplicated(subset=["open", "high", "low", "close"]).sum())
    if duplicates > max_duplicate_candles:
        raise ValueError("Too many duplicate candles: %s" % duplicates)
