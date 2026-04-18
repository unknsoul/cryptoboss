"""Smart money concepts engine for v3 architecture."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.smc.fvg import FairValueGapDetector
from src.smc.liquidity import LiquidityHunter
from src.smc.order_blocks import OBStatus, OrderBlockDetector

from .config import SmartMoneyConfig


class SmartMoneyEngine:
    """Detects order blocks, FVGs, and liquidity behavior."""

    def __init__(self, config: Optional[SmartMoneyConfig] = None):
        self.config = config or SmartMoneyConfig()
        self._ob_detectors: Dict[str, OrderBlockDetector] = {}
        self._fvg_detectors: Dict[str, FairValueGapDetector] = {}
        self._liq_hunters: Dict[str, LiquidityHunter] = {}

    def analyze(self, frame: pd.DataFrame, timeframe: str = "5m") -> Dict[str, object]:
        if frame is None or frame.empty:
            raise ValueError("Frame is empty for smart money analysis")

        detector_ob = self._ob_detectors.setdefault(timeframe, OrderBlockDetector(timeframe=timeframe))
        detector_fvg = self._fvg_detectors.setdefault(timeframe, FairValueGapDetector(timeframe=timeframe))
        hunter_liq = self._liq_hunters.setdefault(timeframe, LiquidityHunter(timeframe=timeframe))

        df = frame.copy().sort_index()
        current_price = float(df["close"].iloc[-1])

        obs = detector_ob.detect(df)
        detector_ob.update_status(df)
        fvgs = detector_fvg.detect(df)
        detector_fvg.update_fill_status(df)

        hunter_liq.detect_equal_highs_lows(df)
        stop_hunts = hunter_liq.detect_stop_hunts(df)
        hunter_liq.update_sweep_status(current_price)

        validated_obs = [
            ob for ob in obs
            if ob.strength >= self.config.impulse_strength_threshold
            and (
                not self.config.mitigation_pending_required
                or ob.status in (OBStatus.ACTIVE, OBStatus.TESTED)
            )
        ]

        price_at_ob = detector_ob.price_in_ob(current_price)
        price_in_fvg = detector_fvg.price_in_fvg(current_price)

        previous_high = float(df["high"].iloc[-50:].max())
        previous_low = float(df["low"].iloc[-50:].min())
        liquidity_zones = {
            "equal_highs_lows": hunter_liq.to_dict_list(),
            "previous_high": previous_high,
            "previous_low": previous_low,
        }

        return {
            "order_blocks": [
                {
                    "id": ob.ob_id,
                    "type": ob.ob_type.value,
                    "status": ob.status.value,
                    "top": ob.top,
                    "bottom": ob.bottom,
                    "strength": ob.strength,
                    "mitigation_pending": ob.status in (OBStatus.ACTIVE, OBStatus.TESTED),
                    "definition": (
                        "last bearish candle before bullish impulse"
                        if ob.ob_type.value == "bullish"
                        else "last bullish candle before bearish impulse"
                    ),
                }
                for ob in validated_obs
            ],
            "fvg": [
                {
                    "id": fvg.fvg_id,
                    "type": fvg.fvg_type.value,
                    "status": fvg.status.value,
                    "top": fvg.top,
                    "bottom": fvg.bottom,
                    "size": fvg.size,
                    "definition": "3 candle imbalance",
                }
                for fvg in fvgs
            ],
            "liquidity": {
                "zones": self.config.liquidity_zones,
                "levels": liquidity_zones,
                "stop_hunts": [
                    {
                        "id": hunt.hunt_id,
                        "direction": hunt.direction,
                        "swept_level": hunt.swept_level,
                        "confirmed": hunt.is_confirmed,
                    }
                    for hunt in stop_hunts
                ],
            },
            "price_at_ob": price_at_ob is not None,
            "price_in_fvg": price_in_fvg is not None,
            "liquidity_sweep": any(hunt.is_confirmed for hunt in stop_hunts),
            "current_price": current_price,
        }

    def analyze_multi_timeframe(self, frames: Dict[str, pd.DataFrame]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for timeframe, frame in frames.items():
            if frame is None or frame.empty:
                continue
            result[timeframe] = self.analyze(frame, timeframe=timeframe)

        return result
