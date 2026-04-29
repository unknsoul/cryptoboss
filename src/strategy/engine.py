"""Structure-first strategy engine (phase 2 foundation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.analysis.market_structure import MarketStructureEngine
from src.data.schema import standardize_ohlcv
from src.regime.detector import RegimeDetector
from src.smc.fvg import FairValueGapDetector
from src.smc.order_blocks import OrderBlockDetector


@dataclass(slots=True)
class StrategySnapshot:
    """Compact state snapshot for decision engines."""

    structure: dict[str, Any]
    order_blocks: list[dict[str, Any]]
    fair_value_gaps: list[dict[str, Any]]
    liquidity_zones: list[dict[str, float]]
    regime: str
    trend: str


class StructureStrategyEngine:
    """Build structural context before any indicator-based decision."""

    def __init__(self, timeframe: str = "5m") -> None:
        self.timeframe = timeframe
        self.structure_engine = MarketStructureEngine()
        self.order_block_detector = OrderBlockDetector(timeframe=timeframe)
        self.fvg_detector = FairValueGapDetector(timeframe=timeframe)
        self.regime_detector = RegimeDetector()

    def snapshot(self, ohlcv: pd.DataFrame) -> StrategySnapshot:
        """Create strategy context from OHLCV."""
        frame = standardize_ohlcv(ohlcv, keep_extra_columns=True).set_index("timestamp")

        structure = self.structure_engine.get_structure_snapshot(frame, timeframe=self.timeframe)
        order_blocks = self.order_block_detector.detect(frame)
        fair_value_gaps = self.fvg_detector.detect(frame)
        regime = self.regime_detector.detect(ohlcv).regime
        trend = structure.trend_state.value

        return StrategySnapshot(
            structure=structure.to_dict(),
            order_blocks=self.order_block_detector.to_dict_list(),
            fair_value_gaps=self.fvg_detector.to_dict_list(),
            liquidity_zones=self._detect_liquidity_zones(frame),
            regime=regime,
            trend=trend,
        )

    @staticmethod
    def _detect_liquidity_zones(frame: pd.DataFrame, window: int = 20) -> list[dict[str, float]]:
        """Simple liquidity proxy from rolling highs/lows."""
        if len(frame) < window:
            return []

        recent = frame.tail(window)
        return [
            {
                "type": "sell_side_liquidity",
                "price": float(recent["low"].min()),
            },
            {
                "type": "buy_side_liquidity",
                "price": float(recent["high"].max()),
            },
        ]
