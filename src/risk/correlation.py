"""Correlation guard utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def correlation_matrix(price_df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Compute correlation matrix on trailing window."""
    if price_df.empty:
        return pd.DataFrame()
    returns = price_df.pct_change().tail(window)
    return returns.corr()


def correlation_guard(
    new_symbol: str,
    open_symbols: Iterable[str],
    price_df: pd.DataFrame,
    threshold: float = 0.7,
    scale_factor: float = 0.5,
) -> float:
    """Return size multiplier based on portfolio correlation."""
    symbols = list(open_symbols) + [new_symbol]
    available = [s for s in symbols if s in price_df.columns]
    if len(available) < 2:
        return 1.0

    corr = correlation_matrix(price_df[available])
    if corr.empty:
        return 1.0

    new_corrs = corr.loc[new_symbol].drop(labels=[new_symbol], errors="ignore")
    avg_corr = float(new_corrs.mean()) if not new_corrs.empty else 0.0

    if avg_corr > threshold:
        return scale_factor
    return 1.0
