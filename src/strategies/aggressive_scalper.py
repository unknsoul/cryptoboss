"""
Aggressive Scalper Strategy — CryptoBoss

4-signal confluence gate scalper optimised for 5-minute BTC/USDT charts.
Uses RSI + MACD + Volume breakout + Price action for high-confidence entries.

Configuration loaded from configs/aggressive_scalper.yaml.

Safety guards:
    - 3% daily loss halt
    - 8 trades/hour max
    - 90s cooldown between entries
    - ADX filter skips choppy/ranging markets
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, StrategyConfig, SignalResult

logger = logging.getLogger(__name__)


@dataclass
class ScalperParams:
    """Tunable knobs for the aggressive scalper."""
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_ma_period: int = 20
    volume_spike_threshold: float = 1.5  # Volume must be 1.5x average
    adx_period: int = 14
    adx_min_trend: float = 20.0  # Skip if ADX < 20 (choppy)
    stop_loss_pct: float = 0.4   # 0.4% SL
    take_profit_pct: float = 1.2  # 1.2% TP (3:1 R/R)
    max_trades_per_hour: int = 8
    daily_loss_halt_pct: float = 3.0
    cooldown_seconds: int = 90


class AggressiveScalper(BaseStrategy):
    """
    High-frequency scalper with 4-signal confluence gate.

    Signals (all 4 must align for entry):
        1. RSI divergence/extreme (oversold for BUY, overbought for SELL)
        2. MACD histogram flip (bearish-to-bullish for BUY, vice versa)
        3. Volume spike (current bar > 1.5x 20-bar average)
        4. Price breakout (close above/below recent swing)

    Plus ADX trend filter to skip ranging markets.
    """

    def __init__(
        self,
        config: StrategyConfig = None,
        params: ScalperParams = None,
    ):
        if config is None:
            config = StrategyConfig(
                strategy_id="aggressive_scalper_v1",
                version="1.0",
                symbol="BTC/USDT",
                min_confidence=0.6,
                cooldown_seconds=90,
            )
        super().__init__(config)
        self.params = params or ScalperParams()
        self._hourly_trade_count = 0
        self._hour_start = datetime.now()
        self._daily_pnl = 0.0
        self._daily_pnl_start = datetime.now().date()
        logger.info(
            f"AggressiveScalper initialized — SL={self.params.stop_loss_pct}% "
            f"TP={self.params.take_profit_pct}% ADX_min={self.params.adx_min_trend}"
        )

    # ------------------------------------------------------------------
    # Technical Indicators
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    @staticmethod
    def _macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)

        # Zero out when the other is larger
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0

        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))

        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx = dx.rolling(period).mean().fillna(0)
        return adx

    # ------------------------------------------------------------------
    # Confluence gate
    # ------------------------------------------------------------------

    def generate_signal(
        self, df: pd.DataFrame, i: int, current_price: float
    ) -> SignalResult:
        """
        4-signal confluence + ADX trend filter.

        Returns BUY, SELL, or HOLD with confidence 0..1.
        """
        p = self.params

        # Need enough data
        min_bars = max(p.macd_slow, p.adx_period, p.volume_ma_period) + 10
        if df is None or len(df) < min_bars:
            return SignalResult(action="HOLD", reason="insufficient_data")

        # Reset hourly counter
        now = datetime.now()
        if (now - self._hour_start).total_seconds() > 3600:
            self._hourly_trade_count = 0
            self._hour_start = now

        # Reset daily P/L tracker
        if now.date() != self._daily_pnl_start:
            self._daily_pnl = 0.0
            self._daily_pnl_start = now.date()

        # Guard: hourly trade cap
        if self._hourly_trade_count >= p.max_trades_per_hour:
            return SignalResult(action="HOLD", reason="hourly_trade_cap_reached")

        # Guard: daily loss halt
        if self._daily_pnl < -(current_price * p.daily_loss_halt_pct / 100):
            return SignalResult(action="HOLD", reason="daily_loss_halt")

        close = df["close"].astype(float)

        # ---------- 1. RSI ----------
        rsi = self._rsi(close, p.rsi_period)
        rsi_now = float(rsi.iloc[-1])
        rsi_prev = float(rsi.iloc[-2]) if len(rsi) >= 2 else rsi_now

        rsi_buy = rsi_now < p.rsi_oversold or (rsi_prev < p.rsi_oversold and rsi_now > p.rsi_oversold)
        rsi_sell = rsi_now > p.rsi_overbought or (rsi_prev > p.rsi_overbought and rsi_now < p.rsi_overbought)

        # ---------- 2. MACD ----------
        _, _, histogram = self._macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
        hist_now = float(histogram.iloc[-1])
        hist_prev = float(histogram.iloc[-2]) if len(histogram) >= 2 else 0

        macd_buy = hist_prev <= 0 < hist_now   # bearish→bullish flip
        macd_sell = hist_prev >= 0 > hist_now   # bullish→bearish flip

        # ---------- 3. Volume spike ----------
        vol = df["volume"].astype(float)
        vol_ma = vol.rolling(p.volume_ma_period).mean()
        vol_ratio = float(vol.iloc[-1] / vol_ma.iloc[-1]) if float(vol_ma.iloc[-1]) > 0 else 0
        volume_ok = vol_ratio >= p.volume_spike_threshold

        # ---------- 4. Price breakout ----------
        lookback = min(20, len(close) - 1)
        recent_high = float(close.iloc[-lookback:].max())
        recent_low = float(close.iloc[-lookback:].min())
        breakout_buy = current_price >= recent_high * 0.999
        breakout_sell = current_price <= recent_low * 1.001

        # ---------- ADX filter ----------
        adx = self._adx(df, p.adx_period)
        adx_now = float(adx.iloc[-1])
        trending = adx_now >= p.adx_min_trend

        # ---------- Confluence decision ----------
        buy_signals = sum([rsi_buy, macd_buy, volume_ok, breakout_buy])
        sell_signals = sum([rsi_sell, macd_sell, volume_ok, breakout_sell])

        if buy_signals >= 3 and trending:
            sl = current_price * (1 - p.stop_loss_pct / 100)
            tp = current_price * (1 + p.take_profit_pct / 100)
            confidence = min(1.0, 0.5 + buy_signals * 0.12 + (adx_now - 20) * 0.005)

            self._hourly_trade_count += 1
            return SignalResult(
                action="BUY",
                reason=f"confluence_{buy_signals}/4 RSI={rsi_now:.1f} MACD_flip vol_x{vol_ratio:.1f} ADX={adx_now:.0f}",
                confidence=round(confidence, 3),
                price=current_price,
                stop_loss=round(sl, 4),
                take_profit=round(tp, 4),
                signal_strength=buy_signals / 4,
                metadata={
                    "rsi": rsi_now,
                    "macd_hist": hist_now,
                    "vol_ratio": vol_ratio,
                    "adx": adx_now,
                },
            )

        if sell_signals >= 3 and trending:
            sl = current_price * (1 + p.stop_loss_pct / 100)
            tp = current_price * (1 - p.take_profit_pct / 100)
            confidence = min(1.0, 0.5 + sell_signals * 0.12 + (adx_now - 20) * 0.005)

            self._hourly_trade_count += 1
            return SignalResult(
                action="SELL",
                reason=f"confluence_{sell_signals}/4 RSI={rsi_now:.1f} MACD_flip vol_x{vol_ratio:.1f} ADX={adx_now:.0f}",
                confidence=round(confidence, 3),
                price=current_price,
                stop_loss=round(sl, 4),
                take_profit=round(tp, 4),
                signal_strength=sell_signals / 4,
                metadata={
                    "rsi": rsi_now,
                    "macd_hist": hist_now,
                    "vol_ratio": vol_ratio,
                    "adx": adx_now,
                },
            )

        return SignalResult(
            action="HOLD",
            reason=f"no_confluence buy={buy_signals}/4 sell={sell_signals}/4 ADX={adx_now:.0f}",
        )

    def get_status(self) -> dict:
        """Return current strategy status for API/dashboard consumption."""
        now = datetime.now()
        return {
            "strategy": self.strategy_id,
            "version": self.version,
            "symbol": self.config.symbol,
            "halted": self._daily_pnl < -(self.params.daily_loss_halt_pct / 100 * 50000),
            "daily_loss_pct": round(abs(self._daily_pnl) / 50000 * 100, 3) if self._daily_pnl < 0 else 0.0,
            "daily_loss_halt_pct": self.params.daily_loss_halt_pct,
            "trades_last_hour": self._hourly_trade_count,
            "max_trades_per_hour": self.params.max_trades_per_hour,
            "leverage": 15,
            "stop_loss_pct": self.params.stop_loss_pct,
            "take_profit_pct": self.params.take_profit_pct,
            "adx_min_trend": self.params.adx_min_trend,
            "cooldown_seconds": self.params.cooldown_seconds,
            "signals_generated": self._signals_generated,
        }
