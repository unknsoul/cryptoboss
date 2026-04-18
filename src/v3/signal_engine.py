"""Signal engine for v3 intraday scalper architecture."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import SignalEngineConfig
from .models import SignalOutput


class SignalEngine:
    """Combines structure and SMC evidence into BUY/SELL/HOLD outputs."""

    def __init__(self, config: Optional[SignalEngineConfig] = None):
        self.config = config or SignalEngineConfig()

    def evaluate(
        self,
        market_structure: Dict[str, object],
        smart_money: Dict[str, object],
        ltf_frame: pd.DataFrame,
    ) -> SignalOutput:
        if ltf_frame is None or ltf_frame.empty:
            return SignalOutput(action="HOLD", confidence=0.0, reason="No LTF market data")

        trend_alignment = bool(
            market_structure.get("trend_detection", {}).get("trend_alignment", False)
        )
        bos_confirmed = bool(market_structure.get("bos", {}).get("confirmed", False))
        choch_confirmed = bool(market_structure.get("choch", {}).get("confirmed", False))

        price_at_ob = bool(smart_money.get("price_at_ob", False))
        price_in_fvg = bool(smart_money.get("price_in_fvg", False))
        liquidity_sweep = bool(smart_money.get("liquidity_sweep", False))

        entry_gate = (
            (not self.config.require_trend_alignment or trend_alignment)
            and (not self.config.require_structure_confirmation or (bos_confirmed or choch_confirmed))
            and (not self.config.require_price_in_ob_or_fvg or (price_at_ob or price_in_fvg))
            and (not self.config.require_liquidity_sweep or liquidity_sweep)
        )

        if not entry_gate:
            return SignalOutput(
                action="HOLD",
                confidence=0.2,
                reason="Entry logic not satisfied",
                metadata={
                    "trend_alignment": trend_alignment,
                    "bos_confirmed": bos_confirmed,
                    "choch_confirmed": choch_confirmed,
                    "price_at_ob": price_at_ob,
                    "price_in_fvg": price_in_fvg,
                    "liquidity_sweep": liquidity_sweep,
                },
            )

        bullish_engulfing = self._is_bullish_engulfing(ltf_frame)
        bearish_engulfing = self._is_bearish_engulfing(ltf_frame)
        momentum_spike = self._has_momentum_spike(ltf_frame)

        if "engulfing_candle" in self.config.confirmations and not (bullish_engulfing or bearish_engulfing):
            return SignalOutput(action="HOLD", confidence=0.3, reason="Engulfing confirmation missing")

        if "momentum_spike" in self.config.confirmations and not momentum_spike:
            return SignalOutput(action="HOLD", confidence=0.3, reason="Momentum spike confirmation missing")

        structure_direction = str(market_structure.get("latest_structure_direction", "neutral"))
        if structure_direction == "neutral":
            structure_direction = "bullish" if bullish_engulfing else "bearish" if bearish_engulfing else "neutral"

        latest_price = float(ltf_frame["close"].iloc[-1])

        if structure_direction == "bullish" and bullish_engulfing:
            return SignalOutput(
                action="BUY",
                confidence=0.82,
                reason="Trend alignment + BOS/CHoCH + OB/FVG + liquidity sweep + bullish confirmation",
                direction="long",
                entry_price=latest_price,
                metadata={"confirmation": ["engulfing_candle", "momentum_spike"]},
            )

        if structure_direction == "bearish" and bearish_engulfing:
            return SignalOutput(
                action="SELL",
                confidence=0.82,
                reason="Trend alignment + BOS/CHoCH + OB/FVG + liquidity sweep + bearish confirmation",
                direction="short",
                entry_price=latest_price,
                metadata={"confirmation": ["engulfing_candle", "momentum_spike"]},
            )

        return SignalOutput(action="HOLD", confidence=0.4, reason="Direction conflict in confirmations")

    @staticmethod
    def _is_bullish_engulfing(frame: pd.DataFrame) -> bool:
        if len(frame) < 2:
            return False

        prev_candle = frame.iloc[-2]
        curr_candle = frame.iloc[-1]

        return (
            float(prev_candle["close"]) < float(prev_candle["open"])
            and float(curr_candle["close"]) > float(curr_candle["open"])
            and float(curr_candle["open"]) <= float(prev_candle["close"])
            and float(curr_candle["close"]) >= float(prev_candle["open"])
        )

    @staticmethod
    def _is_bearish_engulfing(frame: pd.DataFrame) -> bool:
        if len(frame) < 2:
            return False

        prev_candle = frame.iloc[-2]
        curr_candle = frame.iloc[-1]

        return (
            float(prev_candle["close"]) > float(prev_candle["open"])
            and float(curr_candle["close"]) < float(curr_candle["open"])
            and float(curr_candle["open"]) >= float(prev_candle["close"])
            and float(curr_candle["close"]) <= float(prev_candle["open"])
        )

    @staticmethod
    def _has_momentum_spike(frame: pd.DataFrame) -> bool:
        if len(frame) < 20:
            return False

        returns = frame["close"].pct_change().fillna(0.0)
        latest_abs = abs(float(returns.iloc[-1]))
        baseline = float(returns.abs().rolling(20, min_periods=5).mean().iloc[-1])
        threshold = max(baseline * 1.8, 0.0008)
        return latest_abs >= threshold
