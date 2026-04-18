"""Liquidity pool and stop-hunt detector for Smart Money Concepts (SMC)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class LiquidityType(Enum):
    BUYSIDE = "buyside"
    SELLSIDE = "sellside"


class LiquidityStatus(Enum):
    UNTAPPED = "untapped"
    SWEPT = "swept"
    HUNTED = "hunted"


@dataclass
class LiquidityLevel:
    liq_id: str
    liq_type: LiquidityType
    status: LiquidityStatus
    timeframe: str
    price: float
    strength: float
    touches: int
    timestamp_formed: pd.Timestamp
    is_equal_high_low: bool = False
    swept_at: Optional[pd.Timestamp] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status == LiquidityStatus.UNTAPPED


@dataclass
class StopHunt:
    hunt_id: str
    direction: str
    timeframe: str
    timestamp: pd.Timestamp
    swept_level: float
    reversal_size: float
    is_confirmed: bool


class LiquidityHunter:
    """Detects equal highs/lows, liquidity pools, and stop hunts."""

    def __init__(
        self,
        timeframe: str = "15m",
        equality_threshold: float = 0.001,
        min_touches: int = 2,
        stop_hunt_reversal_ratio: float = 1.5,
        lookback: int = 100,
    ) -> None:
        self.timeframe = timeframe
        self.equality_threshold = equality_threshold
        self.min_touches = min_touches
        self.stop_hunt_reversal_ratio = stop_hunt_reversal_ratio
        self.lookback = lookback
        self.liquidity_levels: List[LiquidityLevel] = []
        self.stop_hunts: List[StopHunt] = []
        self._counter = 0

    def detect_equal_highs_lows(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        working_df = df.iloc[-self.lookback :].copy()
        highs = working_df["high"].values
        lows = working_df["low"].values

        for i in range(len(highs)):
            base_high = float(highs[i])
            if base_high == 0:
                continue

            matches = 1
            for j in range(i + 1, len(highs)):
                diff = abs(base_high - float(highs[j])) / base_high
                if diff <= self.equality_threshold:
                    matches += 1

            if matches >= self.min_touches:
                self._counter += 1
                level = LiquidityLevel(
                    liq_id=f"LIQ_{self._counter}",
                    liq_type=LiquidityType.BUYSIDE,
                    status=LiquidityStatus.UNTAPPED,
                    timeframe=self.timeframe,
                    price=base_high,
                    strength=min(matches / 5.0, 1.0),
                    touches=matches,
                    timestamp_formed=pd.Timestamp.utcnow(),
                    is_equal_high_low=True,
                )
                if not any(
                    abs(existing.price - level.price) / max(level.price, 1e-9) <= self.equality_threshold
                    for existing in self.liquidity_levels
                ):
                    self.liquidity_levels.append(level)

        for i in range(len(lows)):
            base_low = float(lows[i])
            if base_low == 0:
                continue

            matches = 1
            for j in range(i + 1, len(lows)):
                diff = abs(base_low - float(lows[j])) / base_low
                if diff <= self.equality_threshold:
                    matches += 1

            if matches >= self.min_touches:
                self._counter += 1
                level = LiquidityLevel(
                    liq_id=f"LIQ_{self._counter}",
                    liq_type=LiquidityType.SELLSIDE,
                    status=LiquidityStatus.UNTAPPED,
                    timeframe=self.timeframe,
                    price=base_low,
                    strength=min(matches / 5.0, 1.0),
                    touches=matches,
                    timestamp_formed=pd.Timestamp.utcnow(),
                    is_equal_high_low=True,
                )
                if not any(
                    abs(existing.price - level.price) / max(level.price, 1e-9) <= self.equality_threshold
                    for existing in self.liquidity_levels
                ):
                    self.liquidity_levels.append(level)

    def detect_stop_hunts(self, df: pd.DataFrame) -> List[StopHunt]:
        self.stop_hunts = []
        if len(df) < 5:
            return self.stop_hunts

        working_df = df.iloc[-self.lookback :].copy().reset_index(drop=False)
        if "timestamp" not in working_df.columns:
            if "index" in working_df.columns:
                working_df["timestamp"] = working_df["index"]
            else:
                working_df["timestamp"] = working_df.index

        for i in range(3, len(working_df) - 1):
            candle = working_df.iloc[i]
            previous = working_df.iloc[i - 1]
            next_candle = working_df.iloc[i + 1]

            candle_range = float(candle["high"]) - float(candle["low"])
            if candle_range == 0:
                continue

            upper_wick = float(candle["high"]) - max(float(candle["open"]), float(candle["close"]))
            lower_wick = min(float(candle["open"]), float(candle["close"])) - float(candle["low"])

            if (
                float(candle["high"]) > float(previous["high"])
                and upper_wick / candle_range > 0.6
                and float(next_candle["close"]) < float(candle["open"])
            ):
                reversal_size = float(candle["high"]) - float(next_candle["close"])
                spike_size = upper_wick
                if reversal_size >= spike_size * self.stop_hunt_reversal_ratio:
                    self._counter += 1
                    self.stop_hunts.append(
                        StopHunt(
                            hunt_id=f"SH_{self._counter}",
                            direction="bearish",
                            timeframe=self.timeframe,
                            timestamp=pd.Timestamp(candle["timestamp"]),
                            swept_level=float(candle["high"]),
                            reversal_size=reversal_size,
                            is_confirmed=True,
                        )
                    )

            if (
                float(candle["low"]) < float(previous["low"])
                and lower_wick / candle_range > 0.6
                and float(next_candle["close"]) > float(candle["open"])
            ):
                reversal_size = float(next_candle["close"]) - float(candle["low"])
                spike_size = lower_wick
                if reversal_size >= spike_size * self.stop_hunt_reversal_ratio:
                    self._counter += 1
                    self.stop_hunts.append(
                        StopHunt(
                            hunt_id=f"SH_{self._counter}",
                            direction="bullish",
                            timeframe=self.timeframe,
                            timestamp=pd.Timestamp(candle["timestamp"]),
                            swept_level=float(candle["low"]),
                            reversal_size=reversal_size,
                            is_confirmed=True,
                        )
                    )

        return self.stop_hunts

    def update_sweep_status(self, current_price: float) -> None:
        for level in self.liquidity_levels:
            if level.status != LiquidityStatus.UNTAPPED:
                continue
            threshold = abs(level.price) * self.equality_threshold
            if abs(current_price - level.price) <= threshold:
                level.status = LiquidityStatus.SWEPT
                level.swept_at = pd.Timestamp.utcnow()

    def get_nearest_liquidity(
        self,
        price: float,
        liq_type: Optional[LiquidityType] = None,
    ) -> Optional[LiquidityLevel]:
        candidates = [
            level
            for level in self.liquidity_levels
            if level.is_active and (liq_type is None or level.liq_type == liq_type)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda level: abs(level.price - price))

    def to_dict_list(self) -> List[Dict]:
        return [
            {
                "id": level.liq_id,
                "type": level.liq_type.value,
                "status": level.status.value,
                "price": level.price,
                "strength": level.strength,
                "touches": level.touches,
                "is_equal_high_low": level.is_equal_high_low,
                "swept_at": str(level.swept_at) if level.swept_at else None,
            }
            for level in self.liquidity_levels
        ]
