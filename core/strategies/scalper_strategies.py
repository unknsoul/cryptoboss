"""
Professional Scalper Strategies - REFACTORED
Now uses:
- BaseStrategyV2 for consistent interface
- TechnicalIndicators for shared calculations (no more code duplication)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime
import logging

from .protocol import BaseStrategyV2
from ..indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class TrendScalper(BaseStrategyV2):
    """
    Professional Trend Scalping Strategy
    
    Trades pullbacks and breakouts in the direction of the trend.
    Uses EMA cloud + RSI for entries, ATR for stops.
    """
    
    def __init__(
        self,
        ema_fast: int = 8,
        ema_slow: int = 21,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_multiplier_sl: float = 1.2,
        atr_multiplier_tp: float = 1.8
    ):
        super().__init__("TrendScalper", "momentum")
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        
        self.parameters = {
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'rsi_period': rsi_period,
            'atr_period': atr_period
        }
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate scalping signal using shared TechnicalIndicators"""
        if not self.validate_data(df, min_rows=50):
            return None
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # Use shared TechnicalIndicators (NO MORE DUPLICATION)
        ema_fast = TechnicalIndicators.ema(closes, self.ema_fast)
        ema_slow = TechnicalIndicators.ema(closes, self.ema_slow)
        rsi = TechnicalIndicators.rsi(closes, self.rsi_period)
        atr_array = TechnicalIndicators.atr(highs, lows, closes, self.atr_period)
        atr = atr_array[-1] if len(atr_array) > 0 else closes[-1] * 0.01
        
        current_price = closes[-1]
        prev_price = closes[-2]
        ema_fast_now = ema_fast[-1]
        ema_slow_now = ema_slow[-1]
        rsi_now = rsi[-1]
        
        # Trend detection using EMA cloud
        is_uptrend = ema_fast_now > ema_slow_now and closes[-1] > ema_fast_now
        is_downtrend = ema_fast_now < ema_slow_now and closes[-1] < ema_fast_now
        
        # Pullback detection
        price_at_fast_ema = abs(current_price - ema_fast_now) < atr * 0.5
        bouncing_up = current_price > prev_price
        bouncing_down = current_price < prev_price
        
        # LONG SCALP
        if is_uptrend and price_at_fast_ema and rsi_now > 35 and rsi_now < 70 and bouncing_up:
            return self.create_signal(
                action='LONG',
                confidence=0.75,
                stop=atr * self.atr_multiplier_sl,
                target=atr * self.atr_multiplier_tp,
                reasons=['Uptrend pullback entry', f'RSI: {rsi_now:.0f}', 'Price at EMA support'],
                metadata={'strategy': self.name, 'entry_type': 'pullback_long', 'atr': atr}
            )
        
        # SHORT SCALP
        if is_downtrend and price_at_fast_ema and rsi_now < 65 and rsi_now > 30 and bouncing_down:
            return self.create_signal(
                action='SHORT',
                confidence=0.75,
                stop=atr * self.atr_multiplier_sl,
                target=atr * self.atr_multiplier_tp,
                reasons=['Downtrend pullback entry', f'RSI: {rsi_now:.0f}', 'Price at EMA resistance'],
                metadata={'strategy': self.name, 'entry_type': 'pullback_short', 'atr': atr}
            )
        
        return None


class BreakoutScalper(BaseStrategyV2):
    """
    Breakout Scalping Strategy
    Captures momentum on breakouts with volume confirmation.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        volume_threshold: float = 1.3,
        atr_period: int = 14
    ):
        super().__init__("BreakoutScalper", "momentum")
        self.lookback = lookback
        self.volume_threshold = volume_threshold
        self.atr_period = atr_period
        
        self.parameters = {
            'lookback': lookback,
            'volume_threshold': volume_threshold,
            'atr_period': atr_period
        }
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate breakout signal using shared TechnicalIndicators"""
        if not self.validate_data(df, min_rows=self.lookback + 5):
            return None
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
        
        # Calculate range
        recent_high = np.max(highs[-self.lookback:-1])
        recent_low = np.min(lows[-self.lookback:-1])
        
        current_close = closes[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-self.lookback:-1])
        
        # Use shared TechnicalIndicators (NO MORE DUPLICATION)
        atr_array = TechnicalIndicators.atr(highs, lows, closes, self.atr_period)
        atr = atr_array[-1] if len(atr_array) > 0 else closes[-1] * 0.01
        
        volume_spike = current_volume > avg_volume * self.volume_threshold
        
        # Breakout LONG
        if current_close > recent_high and volume_spike:
            return self.create_signal(
                action='LONG',
                confidence=0.80,
                stop=atr * 1.5,
                target=atr * 2.5,
                reasons=['Breakout above range', 'Volume spike confirmed', f'Range high: ${recent_high:,.0f}'],
                metadata={'strategy': self.name, 'entry_type': 'breakout_long', 'range_high': recent_high}
            )
        
        # Breakout SHORT
        if current_close < recent_low and volume_spike:
            return self.create_signal(
                action='SHORT',
                confidence=0.80,
                stop=atr * 1.5,
                target=atr * 2.5,
                reasons=['Breakdown below range', 'Volume spike confirmed', f'Range low: ${recent_low:,.0f}'],
                metadata={'strategy': self.name, 'entry_type': 'breakout_short', 'range_low': recent_low}
            )
        
        return None


class RegimeAwareScalper(BaseStrategyV2):
    """
    Regime-Aware Scalping Strategy
    Automatically selects trade direction based on detected market regime.
    """
    
    def __init__(self):
        super().__init__("RegimeAwareScalper", "adaptive")
        self.trend_scalper = TrendScalper()
        self.breakout_scalper = BreakoutScalper()
    
    def _detect_regime(self, df: pd.DataFrame) -> str:
        """Quick regime detection using shared indicators"""
        if len(df) < 50:
            return 'unknown'
        
        closes = df['close'].values
        
        # Use shared TechnicalIndicators
        ema20 = TechnicalIndicators.ema(closes, 20)
        ema50 = TechnicalIndicators.ema(closes, 50)
        
        ema20_now = ema20[-1]
        ema50_now = ema50[-1]
        ema20_prev = ema20[-10]
        
        ema_slope = (ema20_now - ema20_prev) / ema20_prev * 100
        
        if ema20_now > ema50_now and ema_slope > 0.1:
            return 'trending_up'
        elif ema20_now < ema50_now and ema_slope < -0.1:
            return 'trending_down'
        else:
            return 'ranging'
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate regime-aware signal"""
        regime = self._detect_regime(df)
        
        trend_signal = self.trend_scalper.generate_signal(df)
        breakout_signal = self.breakout_scalper.generate_signal(df)
        
        if regime == 'trending_up':
            if trend_signal and trend_signal['action'] == 'LONG':
                trend_signal['reasons'].insert(0, f'Regime: {regime.upper()}')
                return trend_signal
            if breakout_signal and breakout_signal['action'] == 'LONG':
                breakout_signal['reasons'].insert(0, f'Regime: {regime.upper()}')
                return breakout_signal
        
        elif regime == 'trending_down':
            if trend_signal and trend_signal['action'] == 'SHORT':
                trend_signal['reasons'].insert(0, f'Regime: {regime.upper()}')
                return trend_signal
            if breakout_signal and breakout_signal['action'] == 'SHORT':
                breakout_signal['reasons'].insert(0, f'Regime: {regime.upper()}')
                return breakout_signal
        
        else:  # Ranging
            if breakout_signal:
                breakout_signal['reasons'].insert(0, 'Regime: RANGING')
                return breakout_signal
        
        return None


def register_scalper_strategies(strategy_manager):
    """Register all scalper strategies with the strategy manager"""
    try:
        strategy_manager.register_strategy('trend_scalper', TrendScalper())
        strategy_manager.register_strategy('breakout_scalper', BreakoutScalper())
        strategy_manager.register_strategy('regime_scalper', RegimeAwareScalper())
        logger.info("Scalper strategies registered: trend_scalper, breakout_scalper, regime_scalper")
    except Exception as e:
        logger.error(f"Failed to register scalper strategies: {e}")
