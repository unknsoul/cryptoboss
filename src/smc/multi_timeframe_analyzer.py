"""Multi-timeframe trend alignment helper for SMC workflows."""

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .bos_choch import MarketTrend, StructureAnalyzer


@dataclass
class TimeframeAlignment:
    timeframe: str
    trend: str


class MultiTimeframeAnalyzer:
    """Computes trend alignment across multiple timeframes."""

    def __init__(self, swing_lookback: int = 5, trend_lookback: int = 3):
        self.swing_lookback = swing_lookback
        self.trend_lookback = trend_lookback

    def analyze(self, data: Dict[str, pd.DataFrame]) -> Dict:
        alignment = {}
        bullish = 0
        bearish = 0
        ranging = 0

        for timeframe, df in data.items():
            if df is None or len(df) < self.swing_lookback * 2 + 5:
                alignment[timeframe] = "unknown"
                continue

            analyzer = StructureAnalyzer(
                timeframe=timeframe,
                swing_lookback=self.swing_lookback,
                trend_lookback=self.trend_lookback,
            )
            analyzer.analyze(df)
            trend = analyzer.current_trend
            alignment[timeframe] = trend.value

            if trend == MarketTrend.BULLISH:
                bullish += 1
            elif trend == MarketTrend.BEARISH:
                bearish += 1
            else:
                ranging += 1

        dominant = "ranging"
        if bullish > bearish and bullish >= ranging:
            dominant = "bullish"
        elif bearish > bullish and bearish >= ranging:
            dominant = "bearish"

        score = 0.0
        total = max(len([tf for tf, trend in alignment.items() if trend != "unknown"]), 1)
        if dominant == "bullish":
            score = bullish / total
        elif dominant == "bearish":
            score = bearish / total
        else:
            score = ranging / total

        return {
            "alignment": alignment,
            "dominant": dominant,
            "alignment_score": round(float(score), 3),
            "counts": {
                "bullish": bullish,
                "bearish": bearish,
                "ranging": ranging,
            },
        }
