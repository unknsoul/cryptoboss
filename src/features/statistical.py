"""Statistical feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def volatility_cluster(returns: pd.Series, short_window: int = 20, long_window: int = 100) -> pd.Series:
    """Return short/long volatility ratio."""
    short_vol = returns.rolling(short_window).std(ddof=0)
    long_vol = returns.rolling(long_window).std(ddof=0)
    return short_vol / long_vol.replace(0, np.nan)


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Estimate Hurst exponent for a price series."""
    values = series.dropna().astype(float).values
    if len(values) < max_lag + 2:
        return 0.5

    lags = range(2, max_lag + 1)
    tau = [np.std(values[lag:] - values[:-lag]) for lag in lags]
    if any(val <= 0 for val in tau):
        return 0.5

    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(poly[0] * 2.0)


def rolling_hurst(series: pd.Series, window: int = 200, max_lag: int = 20) -> pd.Series:
    """Compute rolling Hurst exponent."""
    hurst_values = []
    for i in range(len(series)):
        if i < window:
            hurst_values.append(np.nan)
            continue
        window_slice = series.iloc[i - window : i]
        hurst_values.append(hurst_exponent(window_slice, max_lag=max_lag))
    return pd.Series(hurst_values, index=series.index)
