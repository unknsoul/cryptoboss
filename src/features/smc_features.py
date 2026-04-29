"""SMC feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.smc.bos_choch import MarketTrend, StructureAnalyzer
from src.smc.order_blocks import OrderBlockDetector
from src.smc.fvg import FairValueGapDetector


def build_smc_features(df: pd.DataFrame, timeframe: str = "5m") -> pd.DataFrame:
    """Return SMC event features aligned to the input index."""
    if df.empty:
        return pd.DataFrame(index=df.index)

    analyzer = StructureAnalyzer(timeframe=timeframe)
    analyzer.analyze(df)

    features = pd.DataFrame(index=df.index)
    features["smc_bos"] = 0
    features["smc_choch"] = 0

    for event in analyzer.structure_breaks:
        if event.index >= len(features):
            continue
        value = 1 if event.direction == "bullish" else -1
        if event.is_bos:
            features.iloc[event.index, features.columns.get_loc("smc_bos")] = value
        if event.is_choch:
            features.iloc[event.index, features.columns.get_loc("smc_choch")] = value

    trend_map = {
        MarketTrend.BULLISH: 1,
        MarketTrend.BEARISH: -1,
        MarketTrend.RANGING: 0,
    }
    features["smc_trend"] = trend_map.get(analyzer.current_trend, 0)

    ob_detector = OrderBlockDetector(timeframe=timeframe)
    obs = ob_detector.detect(df)
    latest_ob_strength = obs[-1].strength if obs else 0.0
    features["smc_ob_strength"] = float(latest_ob_strength)

    fvg_detector = FairValueGapDetector(timeframe=timeframe)
    fvg_detector.detect(df)
    open_fvgs = fvg_detector.get_open_fvgs()
    features["smc_fvg_open_count"] = len(open_fvgs)

    return features
