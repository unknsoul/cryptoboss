"""Alternative data feature transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.statistical import rolling_zscore


def funding_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute funding rate regime features."""
    data = df.copy()
    data["funding_rate"] = pd.to_numeric(data.get("funding_rate"), errors="coerce")
    data["funding_regime"] = np.sign(data["funding_rate"]) * np.log1p(data["funding_rate"].abs())
    return data[["timestamp", "funding_rate", "funding_regime"]].dropna()


def open_interest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute open interest change metrics."""
    data = df.copy()
    data["open_interest"] = pd.to_numeric(data.get("open_interest"), errors="coerce")
    data["oi_change"] = data["open_interest"].pct_change()
    data["oi_zscore"] = rolling_zscore(data["open_interest"], window=50)
    return data[["timestamp", "open_interest", "oi_change", "oi_zscore"]].dropna()


def liquidation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate liquidation intensity by timestamp."""
    data = df.copy()
    data["quantity"] = pd.to_numeric(data.get("quantity"), errors="coerce")
    data = data.dropna(subset=["quantity", "timestamp"])

    grouped = data.groupby("timestamp")["quantity"].sum().reset_index()
    grouped["liq_intensity"] = rolling_zscore(grouped["quantity"], window=30)
    return grouped.rename(columns={"quantity": "liq_volume"})


def options_skew_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute options skew from call/put implied vols."""
    data = df.copy()
    if "option_type" in data.columns and "mark_iv" in data.columns:
        calls = data[data["option_type"] == "call"]["mark_iv"].mean()
        puts = data[data["option_type"] == "put"]["mark_iv"].mean()
        if pd.notna(calls) and pd.notna(puts):
            skew_value = float(puts - calls)
        else:
            skew_value = 0.0
        timestamp = data["timestamp"].max() if "timestamp" in data.columns else pd.Timestamp.utcnow()
        return pd.DataFrame([{"timestamp": timestamp, "options_skew": skew_value}])

    if "put_iv_25d" in data.columns and "call_iv_25d" in data.columns:
        data["options_skew"] = pd.to_numeric(data["put_iv_25d"], errors="coerce") - pd.to_numeric(data["call_iv_25d"], errors="coerce")
        return data[["timestamp", "options_skew"]].dropna()

    return pd.DataFrame(columns=["timestamp", "options_skew"])


def onchain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute on-chain netflow z-score."""
    data = df.copy()
    data["netflow"] = pd.to_numeric(data.get("netflow"), errors="coerce")
    data["netflow_zscore"] = rolling_zscore(data["netflow"], window=60)
    return data[["timestamp", "netflow", "netflow_zscore"]].dropna()
