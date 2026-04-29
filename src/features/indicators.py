"""Technical indicator feature wrappers."""

from __future__ import annotations

import pandas as pd

from src.analysis.indicators import IndicatorEngine


def build_indicator_features(df: pd.DataFrame, include_ohlcv: bool = True) -> pd.DataFrame:
    """Return indicator-enriched DataFrame."""
    engine = IndicatorEngine()
    enriched = engine.compute_all(df)
    if include_ohlcv:
        return enriched

    drop_cols = [col for col in ["open", "high", "low", "close", "volume", "timestamp"] if col in enriched.columns]
    return enriched.drop(columns=drop_cols, errors="ignore")
