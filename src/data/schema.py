"""Canonical market data schema utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
MODEL_EXCLUDED_COLUMNS = set(OHLCV_COLUMNS)


def standardize_ohlcv(
    raw_data: Any,
    *,
    keep_extra_columns: bool = False,
    timestamp_col: str = "timestamp",
    drop_invalid_rows: bool = True,
) -> pd.DataFrame:
    """Return OHLCV data in canonical order and dtype."""
    if isinstance(raw_data, pd.DataFrame):
        df = raw_data.copy()
    else:
        df = pd.DataFrame(raw_data, columns=OHLCV_COLUMNS)

    if timestamp_col != "timestamp" and timestamp_col in df.columns:
        df = df.rename(columns={timestamp_col: "timestamp"})

    if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df["timestamp"] = df.index

    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    selected_cols = list(df.columns) if keep_extra_columns else OHLCV_COLUMNS
    normalized = df[selected_cols].copy()
    normalized["timestamp"] = _normalize_timestamp(normalized["timestamp"])

    for col in ["open", "high", "low", "close", "volume"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    if drop_invalid_rows:
        normalized = normalized.dropna(subset=OHLCV_COLUMNS)
    normalized = normalized.drop_duplicates(subset=["timestamp"], keep="last")
    normalized = normalized.sort_values("timestamp")
    return normalized.reset_index(drop=True)


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return consistent model feature columns for train and live."""
    return [col for col in frame.columns if col not in MODEL_EXCLUDED_COLUMNS]


def _normalize_timestamp(ts: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(ts):
        return pd.to_datetime(ts, utc=True, errors="coerce")

    numeric = pd.to_numeric(ts, errors="coerce")
    if numeric.notna().all():
        return pd.to_datetime(numeric.astype("int64"), unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(ts, utc=True, errors="coerce")
