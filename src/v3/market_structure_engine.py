"""Market structure engine for BOS/CHoCH and trend alignment."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .config import MarketStructureConfig
from .quantified_smc import detect_bos as quantify_bos
from .quantified_smc import detect_swing_points_fractal


class MarketStructureEngine:
    """Detects BOS, CHoCH, and multi-timeframe trend alignment."""

    def __init__(self, config: Optional[MarketStructureConfig] = None):
        self.config = config or MarketStructureConfig()
        self.last_analysis: Dict[str, object] = {}

    def analyze(self, htf_frame: pd.DataFrame, ltf_frame: pd.DataFrame) -> Dict[str, object]:
        htf = self._prepare(htf_frame)
        ltf = self._prepare(ltf_frame)

        htf_swings = self.detect_swings(htf)
        ltf_swings = self.detect_swings(ltf)

        htf_bos = self.detect_bos(htf)
        ltf_bos = self.detect_bos(ltf)

        htf_choch = self.detect_choch(htf, htf_bos)
        ltf_choch = self.detect_choch(ltf, ltf_bos)

        htf_trend = self.detect_trend(htf)
        ltf_trend = self.detect_trend(ltf)
        trend_alignment = htf_trend != "sideways" and htf_trend == ltf_trend

        analysis = {
            "bos": {
                "definition": "break of previous swing high/low with strong momentum candle",
                "conditions": [
                    "close > previous_high + threshold",
                    "volume_spike == true",
                ],
                "htf": htf_bos,
                "ltf": ltf_bos,
                "confirmed": bool(htf_bos or ltf_bos),
            },
            "choch": {
                "definition": "change in trend direction",
                "conditions": [
                    "bullish_to_bearish OR bearish_to_bullish",
                    "break of internal structure",
                ],
                "htf": htf_choch,
                "ltf": ltf_choch,
                "confirmed": bool(htf_choch or ltf_choch),
            },
            "swing_points": {
                "method": self.config.swing_method,
                "left": self.config.swing_left,
                "right": self.config.swing_right,
                "htf": htf_swings,
                "ltf": ltf_swings,
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

    def detect_swings(self, frame: pd.DataFrame) -> List[Dict[str, object]]:
        swings = detect_swing_points_fractal(
            frame,
            left=self.config.swing_left,
            right=self.config.swing_right,
        )

        events: List[Dict[str, object]] = []
        for idx in swings.index:
            swing_high = int(swings.at[idx, "swing_high"])
            swing_low = int(swings.at[idx, "swing_low"])
            if swing_high == 0 and swing_low == 0:
                continue

            if swing_high:
                events.append(
                    {
                        "timestamp": idx,
                        "type": "swing_high",
                        "price": float(swings.at[idx, "high"]),
                    }
                )
            if swing_low:
                events.append(
                    {
                        "timestamp": idx,
                        "type": "swing_low",
                        "price": float(swings.at[idx, "low"]),
                    }
                )

        return events[-40:]

    def detect_bos(self, frame: pd.DataFrame) -> List[Dict[str, object]]:
        scored = quantify_bos(
            frame,
            lookback=self.config.bos_lookback,
            break_threshold=self.config.bos_threshold,
            volume_spike_mult=self.config.volume_spike_mult,
        )

        events: List[Dict[str, object]] = []
        for idx in scored.index:
            bos_value = int(scored.at[idx, "bos"])
            if bos_value == 0:
                continue

            direction = "bullish" if bos_value > 0 else "bearish"
            events.append(
                {
                    "timestamp": idx,
                    "direction": direction,
                    "close": float(scored.at[idx, "close"]),
                    "broken_level": float(scored.at[idx, "bos_level"]),
                    "volume_spike": bool(scored.at[idx, "bos_volume_spike"]),
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
