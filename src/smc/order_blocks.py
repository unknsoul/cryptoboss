"""Order Block detector for Smart Money Concepts (SMC)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class OBType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class OBStatus(Enum):
    ACTIVE = "active"
    TESTED = "tested"
    MITIGATED = "mitigated"
    BROKEN = "broken"


@dataclass
class OrderBlock:
    ob_id: str
    ob_type: OBType
    status: OBStatus
    timeframe: str
    index: int
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    top: float
    bottom: float
    midpoint: float
    strength: float
    volume: float
    touches: int = 0
    last_touch_ts: Optional[pd.Timestamp] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status in (OBStatus.ACTIVE, OBStatus.TESTED)

    @property
    def zone_size(self) -> float:
        return self.top - self.bottom


class OrderBlockDetector:
    """Detects and tracks order blocks on OHLCV data."""

    def __init__(
        self,
        timeframe: str = "5m",
        impulse_multiplier: float = 1.5,
        lookback: int = 100,
        min_strength: float = 0.3,
        use_body_only: bool = False,
    ) -> None:
        self.timeframe = timeframe
        self.impulse_multiplier = impulse_multiplier
        self.lookback = lookback
        self.min_strength = min_strength
        self.use_body_only = use_body_only
        self.order_blocks: List[OrderBlock] = []
        self._ob_counter = 0

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy().reset_index(drop=False)
        if "timestamp" not in normalized.columns:
            if "index" in normalized.columns:
                normalized["timestamp"] = normalized["index"]
            else:
                normalized["timestamp"] = normalized.index
        return normalized

    @staticmethod
    def _candle_body(row: pd.Series) -> float:
        return abs(float(row["close"]) - float(row["open"]))

    @staticmethod
    def _candle_range(row: pd.Series) -> float:
        return float(row["high"]) - float(row["low"])

    @staticmethod
    def _is_bullish(row: pd.Series) -> bool:
        return float(row["close"]) > float(row["open"])

    @staticmethod
    def _is_bearish(row: pd.Series) -> bool:
        return float(row["close"]) < float(row["open"])

    def detect(self, df: pd.DataFrame) -> List[OrderBlock]:
        if len(df) < 5:
            return []

        working_df = self._normalize_df(df)
        self.order_blocks = []

        end = len(working_df)
        start = max(0, end - self.lookback)

        for i in range(start + 2, end - 2):
            candle = working_df.iloc[i]

            if self._is_bearish(candle):
                impulse = 0.0
                for j in range(i + 1, min(i + 4, end)):
                    next_candle = working_df.iloc[j]
                    if self._is_bullish(next_candle):
                        impulse += self._candle_body(next_candle)
                    else:
                        break

                ob_size = self._candle_body(candle) if self.use_body_only else self._candle_range(candle)
                if ob_size > 0 and impulse >= ob_size * self.impulse_multiplier:
                    strength = min(impulse / (ob_size * self.impulse_multiplier * 2.0), 1.0)
                    if strength >= self.min_strength:
                        self._ob_counter += 1
                        top = float(candle["high"]) if not self.use_body_only else max(float(candle["open"]), float(candle["close"]))
                        bottom = float(candle["low"]) if not self.use_body_only else min(float(candle["open"]), float(candle["close"]))
                        self.order_blocks.append(
                            OrderBlock(
                                ob_id=f"OB_{self._ob_counter}",
                                ob_type=OBType.BULLISH,
                                status=OBStatus.ACTIVE,
                                timeframe=self.timeframe,
                                index=i,
                                timestamp=pd.Timestamp(candle["timestamp"]),
                                open=float(candle["open"]),
                                high=float(candle["high"]),
                                low=float(candle["low"]),
                                close=float(candle["close"]),
                                top=top,
                                bottom=bottom,
                                midpoint=(top + bottom) / 2.0,
                                strength=float(strength),
                                volume=float(candle.get("volume", 0.0)),
                            )
                        )

            if self._is_bullish(candle):
                impulse = 0.0
                for j in range(i + 1, min(i + 4, end)):
                    next_candle = working_df.iloc[j]
                    if self._is_bearish(next_candle):
                        impulse += self._candle_body(next_candle)
                    else:
                        break

                ob_size = self._candle_body(candle) if self.use_body_only else self._candle_range(candle)
                if ob_size > 0 and impulse >= ob_size * self.impulse_multiplier:
                    strength = min(impulse / (ob_size * self.impulse_multiplier * 2.0), 1.0)
                    if strength >= self.min_strength:
                        self._ob_counter += 1
                        top = float(candle["high"]) if not self.use_body_only else max(float(candle["open"]), float(candle["close"]))
                        bottom = float(candle["low"]) if not self.use_body_only else min(float(candle["open"]), float(candle["close"]))
                        self.order_blocks.append(
                            OrderBlock(
                                ob_id=f"OB_{self._ob_counter}",
                                ob_type=OBType.BEARISH,
                                status=OBStatus.ACTIVE,
                                timeframe=self.timeframe,
                                index=i,
                                timestamp=pd.Timestamp(candle["timestamp"]),
                                open=float(candle["open"]),
                                high=float(candle["high"]),
                                low=float(candle["low"]),
                                close=float(candle["close"]),
                                top=top,
                                bottom=bottom,
                                midpoint=(top + bottom) / 2.0,
                                strength=float(strength),
                                volume=float(candle.get("volume", 0.0)),
                            )
                        )

        return self.order_blocks

    def update_status(self, df: pd.DataFrame) -> None:
        if df.empty or not self.order_blocks:
            return

        current_high = float(df["high"].iloc[-1])
        current_low = float(df["low"].iloc[-1])
        current_close = float(df["close"].iloc[-1])

        index_value = df.index[-1]
        touch_ts = pd.Timestamp(index_value) if not isinstance(index_value, int) else pd.Timestamp.utcnow()

        for ob in self.order_blocks:
            if ob.status == OBStatus.MITIGATED:
                continue

            if ob.ob_type == OBType.BULLISH:
                if current_low <= ob.top and current_high >= ob.bottom:
                    ob.touches += 1
                    ob.last_touch_ts = touch_ts
                    ob.status = OBStatus.TESTED
                if current_close < ob.bottom:
                    ob.status = OBStatus.MITIGATED

            if ob.ob_type == OBType.BEARISH:
                if current_high >= ob.bottom and current_low <= ob.top:
                    ob.touches += 1
                    ob.last_touch_ts = touch_ts
                    ob.status = OBStatus.TESTED
                if current_close > ob.top:
                    ob.status = OBStatus.MITIGATED

    def get_active_obs(self, ob_type: Optional[OBType] = None) -> List[OrderBlock]:
        return [
            ob
            for ob in self.order_blocks
            if ob.is_valid and (ob_type is None or ob.ob_type == ob_type)
        ]

    def nearest_ob(self, price: float, ob_type: Optional[OBType] = None) -> Optional[OrderBlock]:
        candidates = self.get_active_obs(ob_type)
        if not candidates:
            return None
        return min(candidates, key=lambda ob: abs(ob.midpoint - price))

    def price_in_ob(self, price: float) -> Optional[OrderBlock]:
        for ob in self.get_active_obs():
            if ob.bottom <= price <= ob.top:
                return ob
        return None

    def to_dict_list(self) -> List[Dict]:
        return [
            {
                "id": ob.ob_id,
                "type": ob.ob_type.value,
                "status": ob.status.value,
                "timeframe": ob.timeframe,
                "timestamp": str(ob.timestamp),
                "top": ob.top,
                "bottom": ob.bottom,
                "midpoint": ob.midpoint,
                "strength": round(ob.strength, 3),
                "touches": ob.touches,
                "volume": ob.volume,
            }
            for ob in self.order_blocks
        ]
