"""Fair Value Gap (FVG) detector for Smart Money Concepts (SMC)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class FVGType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class FVGStatus(Enum):
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"


@dataclass
class FairValueGap:
    fvg_id: str
    fvg_type: FVGType
    status: FVGStatus
    timeframe: str
    index: int
    timestamp: pd.Timestamp
    top: float
    bottom: float
    size: float
    fill_pct: float = 0.0
    strength: float = 0.0
    formed_by_volume: float = 0.0
    is_institutional: bool = False
    metadata: Dict = field(default_factory=dict)

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def is_open(self) -> bool:
        return self.status != FVGStatus.FILLED


class FairValueGapDetector:
    """Detects bullish and bearish fair value gaps in OHLCV data."""

    def __init__(
        self,
        timeframe: str = "5m",
        min_gap_atr_ratio: float = 0.3,
        atr_period: int = 14,
        lookback: int = 100,
    ) -> None:
        self.timeframe = timeframe
        self.min_gap_atr_ratio = min_gap_atr_ratio
        self.atr_period = atr_period
        self.lookback = lookback
        self.fvgs: List[FairValueGap] = []
        self._counter = 0

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy().reset_index(drop=False)
        if "timestamp" not in normalized.columns:
            if "index" in normalized.columns:
                normalized["timestamp"] = normalized["index"]
            else:
                normalized["timestamp"] = normalized.index
        return normalized

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def detect(self, df: pd.DataFrame) -> List[FairValueGap]:
        if len(df) < self.atr_period + 3:
            return []

        working_df = self._normalize_df(df)
        atr = self._compute_atr(working_df)

        self.fvgs = []
        start = max(1, len(working_df) - self.lookback)

        for i in range(start, len(working_df) - 1):
            prev_row = working_df.iloc[i - 1]
            curr_row = working_df.iloc[i]
            next_row = working_df.iloc[i + 1]
            current_atr = atr.iloc[i]

            if pd.isna(current_atr) or current_atr == 0:
                continue

            bull_gap_bottom = float(prev_row["high"])
            bull_gap_top = float(next_row["low"])
            if bull_gap_top > bull_gap_bottom:
                bull_gap_size = bull_gap_top - bull_gap_bottom
                if bull_gap_size >= current_atr * self.min_gap_atr_ratio:
                    strength = bull_gap_size / current_atr
                    self._counter += 1
                    self.fvgs.append(
                        FairValueGap(
                            fvg_id=f"FVG_{self._counter}",
                            fvg_type=FVGType.BULLISH,
                            status=FVGStatus.OPEN,
                            timeframe=self.timeframe,
                            index=i,
                            timestamp=pd.Timestamp(curr_row["timestamp"]),
                            top=bull_gap_top,
                            bottom=bull_gap_bottom,
                            size=bull_gap_size,
                            strength=round(float(strength), 2),
                            formed_by_volume=float(curr_row.get("volume", 0.0)),
                            is_institutional=bool(strength >= 2.0),
                        )
                    )

            bear_gap_top = float(prev_row["low"])
            bear_gap_bottom = float(next_row["high"])
            if bear_gap_top > bear_gap_bottom:
                bear_gap_size = bear_gap_top - bear_gap_bottom
                if bear_gap_size >= current_atr * self.min_gap_atr_ratio:
                    strength = bear_gap_size / current_atr
                    self._counter += 1
                    self.fvgs.append(
                        FairValueGap(
                            fvg_id=f"FVG_{self._counter}",
                            fvg_type=FVGType.BEARISH,
                            status=FVGStatus.OPEN,
                            timeframe=self.timeframe,
                            index=i,
                            timestamp=pd.Timestamp(curr_row["timestamp"]),
                            top=bear_gap_top,
                            bottom=bear_gap_bottom,
                            size=bear_gap_size,
                            strength=round(float(strength), 2),
                            formed_by_volume=float(curr_row.get("volume", 0.0)),
                            is_institutional=bool(strength >= 2.0),
                        )
                    )

        return self.fvgs

    def update_fill_status(self, df: pd.DataFrame) -> None:
        if df.empty or not self.fvgs:
            return

        current_high = float(df["high"].iloc[-1])
        current_low = float(df["low"].iloc[-1])
        current_close = float(df["close"].iloc[-1])

        for fvg in self.fvgs:
            if fvg.status == FVGStatus.FILLED:
                continue

            if fvg.fvg_type == FVGType.BULLISH:
                if current_low <= fvg.top:
                    penetration = min(fvg.top - current_low, fvg.size)
                    fvg.fill_pct = round(max(penetration, 0.0) / fvg.size * 100.0, 1)
                    fvg.status = FVGStatus.PARTIALLY_FILLED
                if current_close <= fvg.bottom:
                    fvg.fill_pct = 100.0
                    fvg.status = FVGStatus.FILLED

            if fvg.fvg_type == FVGType.BEARISH:
                if current_high >= fvg.bottom:
                    penetration = min(current_high - fvg.bottom, fvg.size)
                    fvg.fill_pct = round(max(penetration, 0.0) / fvg.size * 100.0, 1)
                    fvg.status = FVGStatus.PARTIALLY_FILLED
                if current_close >= fvg.top:
                    fvg.fill_pct = 100.0
                    fvg.status = FVGStatus.FILLED

    def get_open_fvgs(self, fvg_type: Optional[FVGType] = None) -> List[FairValueGap]:
        return [
            fvg
            for fvg in self.fvgs
            if fvg.is_open and (fvg_type is None or fvg.fvg_type == fvg_type)
        ]

    def nearest_fvg(self, price: float, fvg_type: Optional[FVGType] = None) -> Optional[FairValueGap]:
        candidates = self.get_open_fvgs(fvg_type)
        if not candidates:
            return None
        return min(candidates, key=lambda fvg: abs(fvg.midpoint - price))

    def price_in_fvg(self, price: float) -> Optional[FairValueGap]:
        for fvg in self.get_open_fvgs():
            if fvg.bottom <= price <= fvg.top:
                return fvg
        return None

    def to_dict_list(self) -> List[Dict]:
        return [
            {
                "id": fvg.fvg_id,
                "type": fvg.fvg_type.value,
                "status": fvg.status.value,
                "timeframe": fvg.timeframe,
                "timestamp": str(fvg.timestamp),
                "top": fvg.top,
                "bottom": fvg.bottom,
                "size": fvg.size,
                "fill_pct": fvg.fill_pct,
                "strength": fvg.strength,
                "is_institutional": fvg.is_institutional,
            }
            for fvg in self.fvgs
        ]
