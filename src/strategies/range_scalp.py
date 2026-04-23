"""Range scalp strategy for mean-reversion setups in ranging regimes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base_strategy import BaseStrategy, SignalResult, StrategyConfig


@dataclass(slots=True)
class RangeScalpSettings:
    """Config settings for RangeScalpStrategy."""

    timeframe: str = "15m"
    lookback_bars: int = 96
    support_resistance_tolerance_pct: float = 0.30
    min_confluence: float = 0.60
    max_sl_pct: float = 1.5


class RangeScalpStrategy(BaseStrategy):
    """Scalp strategy that fades range extremes with confirmation filters."""

    def __init__(
        self,
        strategy_id: str = "range_scalp",
        symbol: str = "BTC/USDT",
        settings: RangeScalpSettings | None = None,
    ) -> None:
        self.settings = settings or RangeScalpSettings()

        super().__init__(
            StrategyConfig(
                strategy_id=strategy_id,
                version="12.0",
                symbol=symbol,
                min_confidence=self.settings.min_confluence,
                cooldown_seconds=30,
            )
        )

    def generate_signal(self, df: pd.DataFrame, i: int, current_price: float) -> SignalResult:
        if i < self.settings.lookback_bars:
            return SignalResult(action="HOLD", reason="warmup: insufficient candles for range")

        window = df.iloc[i - self.settings.lookback_bars + 1 : i + 1].copy()
        high = float(window["high"].max())
        low = float(window["low"].min())
        range_size = high - low

        if range_size <= 0:
            return SignalResult(action="HOLD", reason="invalid range")

        # Skip when breakout risk is high (range not stable).
        if self._is_breakout_window(window):
            return SignalResult(action="HOLD", reason="range instability detected")

        tolerance = range_size * (self.settings.support_resistance_tolerance_pct / 100.0)
        near_low = current_price <= (low + tolerance)
        near_high = current_price >= (high - tolerance)

        momentum = self._momentum(window)
        rsi = self._rsi(window["close"].astype(float), period=14)

        if near_low:
            confluence = 0.0
            confluence += 0.35
            if momentum > 0:
                confluence += 0.30
            if rsi < 40:
                confluence += 0.20
            if self._support_touches(window, threshold=low + tolerance) >= 2:
                confluence += 0.15

            if confluence >= self.settings.min_confluence:
                stop_loss = min(low - (range_size * 0.1), current_price * (1.0 - self.settings.max_sl_pct / 100.0))
                tp_mid = low + range_size * 0.5
                return SignalResult(
                    action="BUY",
                    reason="range support long setup",
                    confidence=min(1.0, confluence),
                    size=1.0,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=tp_mid,
                    signal_strength=confluence,
                    metadata={
                        "range_low": low,
                        "range_high": high,
                        "rsi": rsi,
                    },
                )

        if near_high:
            confluence = 0.0
            confluence += 0.35
            if momentum < 0:
                confluence += 0.30
            if rsi > 60:
                confluence += 0.20
            if self._resistance_touches(window, threshold=high - tolerance) >= 2:
                confluence += 0.15

            if confluence >= self.settings.min_confluence:
                stop_loss = max(high + (range_size * 0.1), current_price * (1.0 + self.settings.max_sl_pct / 100.0))
                tp_mid = high - range_size * 0.5
                return SignalResult(
                    action="SELL",
                    reason="range resistance short setup",
                    confidence=min(1.0, confluence),
                    size=1.0,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=tp_mid,
                    signal_strength=confluence,
                    metadata={
                        "range_low": low,
                        "range_high": high,
                        "rsi": rsi,
                    },
                )

        return SignalResult(action="HOLD", reason="no valid range setup")

    @staticmethod
    def _momentum(window: pd.DataFrame) -> float:
        close = window["close"].astype(float)
        return float(close.ewm(span=8, adjust=False).mean().iloc[-1] - close.ewm(span=21, adjust=False).mean().iloc[-1])

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> float:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period, min_periods=period).mean()

        rs = gain / loss.replace(0.0, 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty and pd.notna(rsi.iloc[-1]) else 50.0

    @staticmethod
    def _is_breakout_window(window: pd.DataFrame) -> bool:
        close = window["close"].astype(float)
        recent = close.tail(8)
        if len(recent) < 8:
            return False
        move = abs(float(recent.iloc[-1] - recent.iloc[0]))
        total_range = float(window["high"].max() - window["low"].min())
        if total_range <= 0:
            return False
        return move / total_range > 0.65

    @staticmethod
    def _support_touches(window: pd.DataFrame, threshold: float) -> int:
        lows = window["low"].astype(float)
        return int((lows <= threshold).sum())

    @staticmethod
    def _resistance_touches(window: pd.DataFrame, threshold: float) -> int:
        highs = window["high"].astype(float)
        return int((highs >= threshold).sum())
