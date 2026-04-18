"""Market structure engine for BOS/CHoCH and trend alignment."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .config import MarketStructureConfig


class MarketStructureEngine:
    """Detects BOS, CHoCH, and multi-timeframe trend alignment."""

    def __init__(self, config: Optional[MarketStructureConfig] = None):
        self.config = config or MarketStructureConfig()
        self.last_analysis: Dict[str, object] = {}

    def analyze(self, htf_frame: pd.DataFrame, ltf_frame: pd.DataFrame) -> Dict[str, object]:
        htf = self._prepare(htf_frame)
        ltf = self._prepare(ltf_frame)

        htf_bos = self.detect_bos(htf)
        ltf_bos = self.detect_bos(ltf)

        htf_choch = self.detect_choch(htf, htf_bos)
        ltf_choch = self.detect_choch(ltf, ltf_bos)

        htf_trend = self.detect_trend(htf)
        ltf_trend = self.detect_trend(ltf)
        trend_alignment = htf_trend != "sideways" and htf_trend == ltf_trend

        analysis = {
            "bos": {
                "htf": htf_bos,
                "ltf": ltf_bos,
                "confirmed": bool(htf_bos or ltf_bos),
            },
            "choch": {
                "htf": htf_choch,
                "ltf": ltf_choch,
                "confirmed": bool(htf_choch or ltf_choch),
            },
            "trend_detection": {
                "method": self.config.trend_method,
                "htf_trend": htf_trend,
                "ltf_trend": ltf_trend,
                "trend_alignment": trend_alignment,
                "logic": "HTF trend + LTF confirmation",
            },
            "latest_structure_direction": self._latest_direction(htf_bos, htf_choch, ltf_bos, ltf_choch),
        }

        self.last_analysis = analysis
        return analysis

    def detect_bos(self, frame: pd.DataFrame, lookback: int = 5) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        if len(frame) < lookback + 2:
            return events

        volume_avg = frame["tick_volume"].rolling(20, min_periods=3).mean()

        for idx in range(lookback, len(frame)):
            previous_high = float(frame["high"].iloc[idx - lookback : idx].max())
            previous_low = float(frame["low"].iloc[idx - lookback : idx].min())

            close = float(frame["close"].iloc[idx])
            volume = float(frame["tick_volume"].iloc[idx])
            avg_volume = float(volume_avg.iloc[idx]) if not pd.isna(volume_avg.iloc[idx]) else volume
            volume_spike = volume >= avg_volume * self.config.volume_spike_mult

            bullish_break = close > previous_high + self.config.bos_threshold
            bearish_break = close < previous_low - self.config.bos_threshold

            if bullish_break and volume_spike:
                events.append(
                    {
                        "timestamp": frame.index[idx],
                        "direction": "bullish",
                        "close": close,
                        "broken_level": previous_high,
                        "volume_spike": True,
                    }
                )

            if bearish_break and volume_spike:
                events.append(
                    {
                        "timestamp": frame.index[idx],
                        "direction": "bearish",
                        "close": close,
                        "broken_level": previous_low,
                        "volume_spike": True,
                    }
                )

        return events[-20:]

    def detect_choch(self, frame: pd.DataFrame, bos_events: List[Dict[str, object]]) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        if len(bos_events) < 2 or len(frame) < 10:
            return events

        for idx in range(1, len(bos_events)):
            previous_direction = str(bos_events[idx - 1]["direction"])
            current_direction = str(bos_events[idx]["direction"])
            if previous_direction == current_direction:
                continue

            ts = bos_events[idx]["timestamp"]
            position = frame.index.get_indexer([ts], method="nearest")
            candle_index = int(position[0]) if len(position) else -1
            if candle_index < 3:
                continue

            internal_high = float(frame["high"].iloc[candle_index - 3 : candle_index].max())
            internal_low = float(frame["low"].iloc[candle_index - 3 : candle_index].min())
            close = float(frame["close"].iloc[candle_index])

            internal_structure_broken = (
                close > internal_high if current_direction == "bullish" else close < internal_low
            )
            if not internal_structure_broken:
                continue

            events.append(
                {
                    "timestamp": ts,
                    "from": previous_direction,
                    "to": current_direction,
                    "internal_structure_break": True,
                }
            )

        return events[-20:]

    def detect_trend(self, frame: pd.DataFrame) -> str:
        if len(frame) < 50:
            return "sideways"

        fast_ema = frame["close"].ewm(span=20, adjust=False).mean()
        slow_ema = frame["close"].ewm(span=50, adjust=False).mean()

        latest_fast = float(fast_ema.iloc[-1])
        latest_slow = float(slow_ema.iloc[-1])

        if latest_fast > latest_slow * 1.001:
            return "bullish"
        if latest_fast < latest_slow * 0.999:
            return "bearish"
        return "sideways"

    @staticmethod
    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            raise ValueError("Frame is empty for market structure analysis")

        df = frame.copy()
        if "tick_volume" not in df.columns:
            if "volume" in df.columns:
                df["tick_volume"] = df["volume"]
            else:
                df["tick_volume"] = 1.0

        return df.sort_index()

    @staticmethod
    def _latest_direction(*event_groups: List[Dict[str, object]]) -> str:
        flattened = [event for group in event_groups for event in group]
        if not flattened:
            return "neutral"

        flattened.sort(key=lambda item: item["timestamp"])
        latest = flattened[-1]
        if "direction" in latest:
            return str(latest["direction"])
        return str(latest.get("to", "neutral"))
