"""BOS/CHoCH structure analyzer for Smart Money Concepts (SMC)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd


class StructureType(Enum):
    BOS_BULLISH = "bos_bullish"
    BOS_BEARISH = "bos_bearish"
    CHOCH_BULLISH = "choch_bullish"
    CHOCH_BEARISH = "choch_bearish"


class MarketTrend(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"


@dataclass
class StructureBreak:
    break_id: str
    break_type: StructureType
    timeframe: str
    index: int
    timestamp: pd.Timestamp
    break_price: float
    broken_level: float
    impulse_size: float
    confirmed: bool = False
    metadata: Dict = field(default_factory=dict)

    @property
    def is_choch(self) -> bool:
        return "choch" in self.break_type.value

    @property
    def is_bos(self) -> bool:
        return "bos" in self.break_type.value

    @property
    def direction(self) -> str:
        return "bullish" if "bullish" in self.break_type.value else "bearish"


class StructureAnalyzer:
    """Finds swing highs/lows and classifies BOS/CHoCH events."""

    def __init__(
        self,
        timeframe: str = "15m",
        swing_lookback: int = 5,
        trend_lookback: int = 3,
    ) -> None:
        self.timeframe = timeframe
        self.swing_lookback = swing_lookback
        self.trend_lookback = trend_lookback
        self.structure_breaks: List[StructureBreak] = []
        self.swing_highs: List[Tuple[int, float]] = []
        self.swing_lows: List[Tuple[int, float]] = []
        self.current_trend: MarketTrend = MarketTrend.RANGING
        self._counter = 0

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy().reset_index(drop=False)
        if "timestamp" not in normalized.columns:
            if "index" in normalized.columns:
                normalized["timestamp"] = normalized["index"]
            else:
                normalized["timestamp"] = normalized.index
        return normalized

    def _find_swings(self, df: pd.DataFrame) -> None:
        n = self.swing_lookback
        highs: List[Tuple[int, float]] = []
        lows: List[Tuple[int, float]] = []

        for i in range(n, len(df) - n):
            window_high = df["high"].iloc[i - n : i + n + 1]
            window_low = df["low"].iloc[i - n : i + n + 1]

            if float(df["high"].iloc[i]) == float(window_high.max()):
                highs.append((i, float(df["high"].iloc[i])))

            if float(df["low"].iloc[i]) == float(window_low.min()):
                lows.append((i, float(df["low"].iloc[i])))

        self.swing_highs = highs
        self.swing_lows = lows

    def _determine_trend(self) -> MarketTrend:
        n = self.trend_lookback
        if len(self.swing_highs) < n or len(self.swing_lows) < n:
            return MarketTrend.RANGING

        recent_highs = [price for _, price in self.swing_highs[-n:]]
        recent_lows = [price for _, price in self.swing_lows[-n:]]

        hh = all(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs)))
        hl = all(recent_lows[i] > recent_lows[i - 1] for i in range(1, len(recent_lows)))
        ll = all(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows)))
        lh = all(recent_highs[i] < recent_highs[i - 1] for i in range(1, len(recent_highs)))

        if hh and hl:
            return MarketTrend.BULLISH
        if ll and lh:
            return MarketTrend.BEARISH
        return MarketTrend.RANGING

    def analyze(self, df: pd.DataFrame) -> List[StructureBreak]:
        if len(df) < self.swing_lookback * 2 + 5:
            return []

        working_df = self._normalize_df(df)

        self._find_swings(working_df)
        self.current_trend = self._determine_trend()
        self.structure_breaks = []

        checked_high_indices = set()
        checked_low_indices = set()

        # Focus on recent bars for responsive intraday behavior.
        for i in range(len(working_df) - 1, max(len(working_df) - 50, 0), -1):
            current_close = float(working_df["close"].iloc[i])
            current_ts = pd.Timestamp(working_df["timestamp"].iloc[i])

            for swing_idx, swing_price in self.swing_highs:
                if swing_idx >= i or swing_idx in checked_high_indices:
                    continue
                if current_close > swing_price:
                    checked_high_indices.add(swing_idx)
                    self._counter += 1
                    is_choch = self.current_trend == MarketTrend.BEARISH
                    break_type = StructureType.CHOCH_BULLISH if is_choch else StructureType.BOS_BULLISH
                    self.structure_breaks.append(
                        StructureBreak(
                            break_id=f"SB_{self._counter}",
                            break_type=break_type,
                            timeframe=self.timeframe,
                            index=i,
                            timestamp=current_ts,
                            break_price=current_close,
                            broken_level=swing_price,
                            impulse_size=current_close - swing_price,
                            confirmed=True,
                        )
                    )

            for swing_idx, swing_price in self.swing_lows:
                if swing_idx >= i or swing_idx in checked_low_indices:
                    continue
                if current_close < swing_price:
                    checked_low_indices.add(swing_idx)
                    self._counter += 1
                    is_choch = self.current_trend == MarketTrend.BULLISH
                    break_type = StructureType.CHOCH_BEARISH if is_choch else StructureType.BOS_BEARISH
                    self.structure_breaks.append(
                        StructureBreak(
                            break_id=f"SB_{self._counter}",
                            break_type=break_type,
                            timeframe=self.timeframe,
                            index=i,
                            timestamp=current_ts,
                            break_price=current_close,
                            broken_level=swing_price,
                            impulse_size=swing_price - current_close,
                            confirmed=True,
                        )
                    )

        self.structure_breaks.sort(key=lambda event: event.index, reverse=True)
        return self.structure_breaks

    @property
    def latest_break(self) -> Optional[StructureBreak]:
        return self.structure_breaks[0] if self.structure_breaks else None

    @property
    def latest_choch(self) -> Optional[StructureBreak]:
        for item in self.structure_breaks:
            if item.is_choch:
                return item
        return None

    @property
    def latest_bos(self) -> Optional[StructureBreak]:
        for item in self.structure_breaks:
            if item.is_bos:
                return item
        return None

    def to_dict_list(self) -> List[Dict]:
        return [
            {
                "id": item.break_id,
                "type": item.break_type.value,
                "timeframe": item.timeframe,
                "timestamp": str(item.timestamp),
                "break_price": item.break_price,
                "broken_level": item.broken_level,
                "impulse_size": round(item.impulse_size, 4),
                "is_choch": item.is_choch,
                "is_bos": item.is_bos,
                "direction": item.direction,
                "confirmed": item.confirmed,
            }
            for item in self.structure_breaks
        ]
