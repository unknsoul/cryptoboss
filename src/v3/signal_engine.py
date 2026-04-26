"""Signal engine for v3 intraday scalper architecture.

Enhanced with technical indicators (RSI, EMA, ATR) on top of
existing Smart Money Concepts (BOS, OB, FVG) scoring.

Designed for 1m–5m scalping timeframes.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import SignalEngineConfig
from .models import SignalOutput


class SignalEngine:
    """Combines structure, SMC evidence, and technical indicators into BUY/SELL/HOLD outputs."""

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

        # === Technical Indicators ===
        rsi_signal = self._rsi_signal(ltf_frame)
        ema_signal = self._ema_crossover_signal(ltf_frame)
        atr_filter = self._atr_volatility_filter(ltf_frame)

        buy_score = 0
        sell_score = 0
        reasons = []
        score_breakdown: Dict[str, object] = {
            "trend_alignment": trend_alignment,
            "bos_confirmed": bos_confirmed,
            "ob_signal": ob_signal,
            "fvg_signal": fvg_signal,
            "momentum_direction": momentum_direction,
            "liquidity_sweep": liquidity_sweep,
            "rsi_signal": rsi_signal,
            "ema_signal": ema_signal,
            "atr_pass": atr_filter,
        }

        # --- SMC scoring (existing) ---
        if trend_alignment:
            if trend_direction == "bullish":
                buy_score += int(weights.get("trend", 0))
                reasons.append("trend=bullish")
            elif trend_direction == "bearish":
                sell_score += int(weights.get("trend", 0))
                reasons.append("trend=bearish")

        if bos_confirmed:
            if bos_direction > 0:
                buy_score += int(weights.get("bos", 0))
                reasons.append("bos=bullish")
            elif bos_direction < 0:
                sell_score += int(weights.get("bos", 0))
                reasons.append("bos=bearish")

        if ob_signal > 0:
            buy_score += int(weights.get("ob_touch", 0))
            reasons.append("ob=bullish")
        elif ob_signal < 0:
            sell_score += int(weights.get("ob_touch", 0))
            reasons.append("ob=bearish")

        if fvg_signal > 0:
            buy_score += int(weights.get("fvg_touch", 0))
            reasons.append("fvg=bullish")
        elif fvg_signal < 0:
            sell_score += int(weights.get("fvg_touch", 0))
            reasons.append("fvg=bearish")

        if momentum_direction > 0:
            buy_score += int(weights.get("momentum", 0))
            reasons.append("momentum=up")
        elif momentum_direction < 0:
            sell_score += int(weights.get("momentum", 0))
            reasons.append("momentum=down")

        if liquidity_sweep:
            if momentum_direction > 0:
                buy_score += 1
                reasons.append("liq_sweep=bullish")
            elif momentum_direction < 0:
                sell_score += 1
                reasons.append("liq_sweep=bearish")

        # --- Technical indicator scoring (new) ---
        # RSI: +2 for oversold buy / overbought sell
        if rsi_signal > 0:
            buy_score += 2
            reasons.append("rsi=oversold_buy")
        elif rsi_signal < 0:
            sell_score += 2
            reasons.append("rsi=overbought_sell")

        # EMA crossover: +2 for bullish/bearish cross
        if ema_signal > 0:
            buy_score += 2
            reasons.append("ema=bullish_cross")
        elif ema_signal < 0:
            sell_score += 2
            reasons.append("ema=bearish_cross")

        # ATR filter: if volatility too low, penalize both scores
        if not atr_filter:
            buy_score = max(0, buy_score - 2)
            sell_score = max(0, sell_score - 2)
            reasons.append("atr=low_vol_penalty")

        # === Win probability adjustment ===
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
            reason = f"BUY threshold met ({buy_score}/{buy_threshold}): {', '.join(reasons)}"
        elif sell_score >= sell_threshold and sell_score > buy_score:
            signal_action = "SELL"
            reason = f"SELL threshold met ({sell_score}/{sell_threshold}): {', '.join(reasons)}"
        elif self.config.force_trade_if_no_signal:
            if buy_score > sell_score and buy_score > 0:
                signal_action = "BUY"
                reason = f"Forced BUY ({buy_score}): {', '.join(reasons)}"
            elif sell_score > buy_score and sell_score > 0:
                signal_action = "SELL"
                reason = f"Forced SELL ({sell_score}): {', '.join(reasons)}"
            else:
                reason = f"No directional edge: {', '.join(reasons) if reasons else 'flat'}"

        if max_score < self.config.min_trade_score and signal_action in ("BUY", "SELL"):
            if self.config.force_trade_if_no_signal:
                reason = f"{reason} | below ideal threshold ({self.config.min_trade_score})"
            else:
                signal_action = "HOLD"
                reason = f"Score below minimum threshold ({self.config.min_trade_score}): {', '.join(reasons)}"

        latest_price = float(ltf_frame["close"].iloc[-1])
        # Max possible score: trend(2) + bos(2) + ob(3) + fvg(2) + momentum(1) + liq(1) + rsi(2) + ema(2) = 15
        max_possible = max(
            self.config.buy_threshold, self.config.sell_threshold, 1
        ) + 4  # +4 for RSI and EMA additions
        confidence = min(max_score / max_possible, 1.0)

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

    # === Technical Indicator Methods ===

    @staticmethod
    def _rsi_signal(frame: pd.DataFrame, period: int = 14) -> int:
        """
        RSI-based signal.
        
        Returns:
            1 if oversold (<30), -1 if overbought (>70), 0 otherwise
        """
        if len(frame) < period + 1:
            return 0

        close = frame["close"].astype(float)
        delta = close.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
        avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        if rsi < 30:
            return 1  # Oversold — buy signal
        elif rsi > 70:
            return -1  # Overbought — sell signal
        return 0

    @staticmethod
    def _ema_crossover_signal(frame: pd.DataFrame, fast: int = 9, slow: int = 21) -> int:
        """
        EMA crossover signal.
        
        Returns:
            1 if fast EMA crossed above slow (bullish), -1 if below (bearish), 0 if no cross
        """
        if len(frame) < slow + 2:
            return 0

        close = frame["close"].astype(float)
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        # Check for crossover in last 2 bars
        curr_diff = float(ema_fast.iloc[-1] - ema_slow.iloc[-1])
        prev_diff = float(ema_fast.iloc[-2] - ema_slow.iloc[-2])

        if curr_diff > 0 and prev_diff <= 0:
            return 1  # Bullish crossover
        elif curr_diff < 0 and prev_diff >= 0:
            return -1  # Bearish crossover

        # Also consider current position (not just crossover)
        if curr_diff > 0:
            return 1  # Fast above slow = bullish bias
        elif curr_diff < 0:
            return -1  # Fast below slow = bearish bias
        return 0

    @staticmethod
    def _atr_volatility_filter(frame: pd.DataFrame, period: int = 14, threshold_mult: float = 0.5) -> bool:
        """
        ATR-based volatility filter.
        
        Returns True if current ATR >= threshold_mult * average ATR (enough volatility).
        Returns False if volatility is too low for scalping.
        """
        if len(frame) < max(period * 2, 20):
            return True  # Not enough data — allow trading

        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period, min_periods=period).mean()
        if atr.empty or pd.isna(atr.iloc[-1]):
            return False
        current_atr = float(atr.iloc[-1])

        # Average ATR over longer period
        avg_atr = float(atr.iloc[-period * 2: -1].mean()) if len(atr) > period * 2 else current_atr
        if pd.isna(avg_atr) or avg_atr <= 0:
            return False

        return current_atr >= threshold_mult * avg_atr

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
