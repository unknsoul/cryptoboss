"""Signal engine for v3 intraday scalper architecture."""

from __future__ import annotations

from typing import Dict, Optional

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
        win_probability: Optional[float] = None,
    ) -> SignalOutput:
        if ltf_frame is None or ltf_frame.empty:
            return SignalOutput(action="HOLD", confidence=0.0, reason="No LTF market data")

        weights = self.config.score_weights
        trend_alignment = bool(market_structure.get("trend_detection", {}).get("trend_alignment", False))
        trend_direction = str(market_structure.get("latest_structure_direction", "neutral"))

        bos_confirmed = bool(market_structure.get("bos", {}).get("confirmed", False))
        bos_direction = self._latest_bos_direction(market_structure)

        ob_signal = int(smart_money.get("ob_signal", 0))
        fvg_signal = int(smart_money.get("fvg_signal", 0))
        liquidity_sweep = bool(smart_money.get("liquidity_sweep", False))

        momentum_direction = self._momentum_direction(ltf_frame)

        buy_score = 0
        sell_score = 0
        score_breakdown: Dict[str, object] = {
            "trend_alignment": trend_alignment,
            "bos_confirmed": bos_confirmed,
            "ob_signal": ob_signal,
            "fvg_signal": fvg_signal,
            "momentum_direction": momentum_direction,
            "liquidity_sweep": liquidity_sweep,
        }

        if trend_alignment:
            if trend_direction == "bullish":
                buy_score += int(weights.get("trend", 0))
            elif trend_direction == "bearish":
                sell_score += int(weights.get("trend", 0))

        if bos_confirmed:
            if bos_direction > 0:
                buy_score += int(weights.get("bos", 0))
            elif bos_direction < 0:
                sell_score += int(weights.get("bos", 0))

        if ob_signal > 0:
            buy_score += int(weights.get("ob_touch", 0))
        elif ob_signal < 0:
            sell_score += int(weights.get("ob_touch", 0))

        if fvg_signal > 0:
            buy_score += int(weights.get("fvg_touch", 0))
        elif fvg_signal < 0:
            sell_score += int(weights.get("fvg_touch", 0))

        if momentum_direction > 0:
            buy_score += int(weights.get("momentum", 0))
        elif momentum_direction < 0:
            sell_score += int(weights.get("momentum", 0))

        if liquidity_sweep:
            if momentum_direction > 0:
                buy_score += 1
            elif momentum_direction < 0:
                sell_score += 1

        if win_probability is None:
            win_probability = 0.5

        probability_pass = win_probability >= self.config.probability_threshold
        score_breakdown["buy_score"] = buy_score
        score_breakdown["sell_score"] = sell_score
        score_breakdown["win_probability"] = win_probability
        score_breakdown["probability_threshold"] = self.config.probability_threshold

        max_score = max(buy_score, sell_score)
        signal_action = "HOLD"
        reason = "No score threshold met"

        buy_threshold = self.config.buy_threshold
        sell_threshold = self.config.sell_threshold

        if not probability_pass:
            buy_threshold += self.config.threshold_relaxation
            sell_threshold += self.config.threshold_relaxation
            score_breakdown["probability_penalty"] = True

        if buy_score >= buy_threshold and buy_score > sell_score:
            signal_action = "BUY"
            reason = "Weighted scoring BUY threshold met"
        elif sell_score >= sell_threshold and sell_score > buy_score:
            signal_action = "SELL"
            reason = "Weighted scoring SELL threshold met"
        elif self.config.force_trade_if_no_signal:
            if buy_score > sell_score and buy_score > 0:
                signal_action = "BUY"
                reason = "Forced BUY fallback based on positive relative score"
            elif sell_score > buy_score and sell_score > 0:
                signal_action = "SELL"
                reason = "Forced SELL fallback based on positive relative score"
            else:
                reason = "Forced mode active but no directional edge"

        if max_score < self.config.min_trade_score and signal_action in ("BUY", "SELL"):
            if self.config.force_trade_if_no_signal:
                reason = f"{reason} | trade score below ideal threshold ({self.config.min_trade_score})"
            else:
                signal_action = "HOLD"
                reason = f"Trade score below minimum threshold ({self.config.min_trade_score})"

        latest_price = float(ltf_frame["close"].iloc[-1])
        confidence = min(max_score / max(self.config.buy_threshold, self.config.sell_threshold, 1), 1.0)

        direction = "neutral"
        if signal_action == "BUY":
            direction = "long"
        elif signal_action == "SELL":
            direction = "short"

        return SignalOutput(
            action=signal_action,
            confidence=float(confidence),
            reason=reason,
            direction=direction,
            entry_price=latest_price,
            metadata=score_breakdown,
        )

    @staticmethod
    def _momentum_direction(frame: pd.DataFrame) -> int:
        if len(frame) < 20:
            return 0

        returns = frame["close"].pct_change().fillna(0.0)
        latest = float(returns.iloc[-1])
        baseline = float(returns.abs().rolling(20, min_periods=5).mean().iloc[-1])
        spike_threshold = max(baseline * 1.8, 0.0008)

        if latest > spike_threshold:
            return 1
        if latest < -spike_threshold:
            return -1
        return 0

    @staticmethod
    def _latest_bos_direction(market_structure: Dict[str, object]) -> int:
        bos = market_structure.get("bos", {})
        events = []
        events.extend(bos.get("htf", []))
        events.extend(bos.get("ltf", []))
        if not events:
            return 0

        events = sorted(events, key=lambda event: event.get("timestamp"))
        latest = str(events[-1].get("direction", ""))
        if latest == "bullish":
            return 1
        if latest == "bearish":
            return -1
        return 0
