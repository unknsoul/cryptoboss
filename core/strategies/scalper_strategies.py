"""
Professional Scalper Strategies
Fast, trend-following scalping strategies for active markets

Key Features:
- Quick entries on pullbacks to trend
- Tight stops and targets (1:1.5 to 1:2 R:R)
- Multiple entries allowed (pyramiding)
- Respects higher timeframe trend direction
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TrendScalper:
    """
    Professional Trend Scalping Strategy
    
    Trades pullbacks and breakouts in the direction of the trend.
    Uses EMA cloud + RSI for entries, ATR for stops.
    
    Entry Rules:
    - LONG: In uptrend, price pulls back to EMA support, RSI > 40
    - SHORT: In downtrend, price pulls back to EMA resistance, RSI < 60
    """
    
    def __init__(
        self,
        ema_fast: int = 8,
        ema_slow: int = 21,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_multiplier_sl: float = 1.2,  # Tight stop for scalps
        atr_multiplier_tp: float = 1.8   # 1.5x R:R target
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.name = "TrendScalper"
        self.strategy_type = "momentum"  # Works in trending markets
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _calculate_rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(data))
        avg_loss = np.zeros(len(data))
        
        # Initial SMA
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        # EMA for remaining
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate ATR"""
        tr = []
        for i in range(1, len(closes)):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate scalping signal"""
        if len(df) < 50:
            return None
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # Calculate indicators
        ema_fast = self._calculate_ema(closes, self.ema_fast)
        ema_slow = self._calculate_ema(closes, self.ema_slow)
        rsi = self._calculate_rsi(closes, self.rsi_period)
        atr = self._calculate_atr(highs, lows, closes, self.atr_period)
        
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
        
        # LONG SCALP: Uptrend + price near EMA support + RSI not overbought + bouncing
        if is_uptrend and price_at_fast_ema and rsi_now > 35 and rsi_now < 70 and bouncing_up:
            return {
                'action': 'LONG',
                'confidence': 0.75,
                'stop': atr * self.atr_multiplier_sl,
                'target': atr * self.atr_multiplier_tp,
                'reasons': [f'Uptrend pullback entry', f'RSI: {rsi_now:.0f}', 'Price at EMA support'],
                'metadata': {
                    'strategy': self.name,
                    'entry_type': 'pullback_long',
                    'atr': atr,
                    'ema_fast': ema_fast_now,
                    'ema_slow': ema_slow_now
                }
            }
        
        # SHORT SCALP: Downtrend + price near EMA resistance + RSI not oversold + bouncing down
        if is_downtrend and price_at_fast_ema and rsi_now < 65 and rsi_now > 30 and bouncing_down:
            return {
                'action': 'SHORT',
                'confidence': 0.75,
                'stop': atr * self.atr_multiplier_sl,
                'target': atr * self.atr_multiplier_tp,
                'reasons': [f'Downtrend pullback entry', f'RSI: {rsi_now:.0f}', 'Price at EMA resistance'],
                'metadata': {
                    'strategy': self.name,
                    'entry_type': 'pullback_short',
                    'atr': atr,
                    'ema_fast': ema_fast_now,
                    'ema_slow': ema_slow_now
                }
            }
        
        return None


class BreakoutScalper:
    """
    Breakout Scalping Strategy
    
    Captures momentum on breakouts with volume confirmation.
    Quick entries and exits on range breakouts.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        volume_threshold: float = 1.3,
        atr_period: int = 14
    ):
        self.lookback = lookback
        self.volume_threshold = volume_threshold
        self.atr_period = atr_period
        self.name = "BreakoutScalper"
        self.strategy_type = "momentum"
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate ATR"""
        tr = []
        for i in range(1, len(closes)):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate breakout signal"""
        if len(df) < self.lookback + 5:
            return None
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
        
        # Calculate range
        recent_high = np.max(highs[-self.lookback:-1])  # Exclude current bar
        recent_low = np.min(lows[-self.lookback:-1])
        
        current_close = closes[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-self.lookback:-1])
        
        atr = self._calculate_atr(highs, lows, closes, self.atr_period)
        volume_spike = current_volume > avg_volume * self.volume_threshold
        
        # Breakout LONG
        if current_close > recent_high and volume_spike:
            return {
                'action': 'LONG',
                'confidence': 0.80,
                'stop': atr * 1.5,
                'target': atr * 2.5,
                'reasons': ['Breakout above range', 'Volume spike confirmed', f'Range high: ${recent_high:,.0f}'],
                'metadata': {
                    'strategy': self.name,
                    'entry_type': 'breakout_long',
                    'range_high': recent_high,
                    'volume_ratio': current_volume / avg_volume
                }
            }
        
        # Breakout SHORT
        if current_close < recent_low and volume_spike:
            return {
                'action': 'SHORT',
                'confidence': 0.80,
                'stop': atr * 1.5,
                'target': atr * 2.5,
                'reasons': ['Breakdown below range', 'Volume spike confirmed', f'Range low: ${recent_low:,.0f}'],
                'metadata': {
                    'strategy': self.name,
                    'entry_type': 'breakout_short',
                    'range_low': recent_low,
                    'volume_ratio': current_volume / avg_volume
                }
            }
        
        return None


class RegimeAwareScalper:
    """
    Regime-Aware Scalping Strategy
    
    Automatically selects trade direction based on detected market regime.
    - Trending Up: Only LONG signals
    - Trending Down: Only SHORT signals
    - Ranging: Both directions at extremes
    """
    
    def __init__(self):
        self.trend_scalper = TrendScalper()
        self.breakout_scalper = BreakoutScalper()
        self.name = "RegimeAwareScalper"
        self.strategy_type = "adaptive"
    
    def _detect_regime(self, df: pd.DataFrame) -> str:
        """Quick regime detection"""
        if len(df) < 50:
            return 'unknown'
        
        closes = df['close'].values
        
        # Simple trend detection with slope
        ema20 = self.trend_scalper._calculate_ema(closes, 20)
        ema50 = self.trend_scalper._calculate_ema(closes, 50)
        
        current_price = closes[-1]
        ema20_now = ema20[-1]
        ema50_now = ema50[-1]
        ema20_prev = ema20[-10]  # 10 bars ago
        
        ema_slope = (ema20_now - ema20_prev) / ema20_prev * 100  # % change
        
        if ema20_now > ema50_now and ema_slope > 0.1:
            return 'trending_up'
        elif ema20_now < ema50_now and ema_slope < -0.1:
            return 'trending_down'
        else:
            return 'ranging'
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate regime-aware signal"""
        regime = self._detect_regime(df)
        
        # Get signals from both strategies
        trend_signal = self.trend_scalper.generate_signal(df)
        breakout_signal = self.breakout_scalper.generate_signal(df)
        
        # Filter based on regime
        if regime == 'trending_up':
            # Only accept LONG signals
            if trend_signal and trend_signal['action'] == 'LONG':
                trend_signal['reasons'].insert(0, f'Regime: {regime.upper()}')
                return trend_signal
            if breakout_signal and breakout_signal['action'] == 'LONG':
                breakout_signal['reasons'].insert(0, f'Regime: {regime.upper()}')
                return breakout_signal
        
        elif regime == 'trending_down':
            # Only accept SHORT signals
            if trend_signal and trend_signal['action'] == 'SHORT':
                trend_signal['reasons'].insert(0, f'Regime: {regime.upper()}')
                return trend_signal
            if breakout_signal and breakout_signal['action'] == 'SHORT':
                breakout_signal['reasons'].insert(0, f'Regime: {regime.upper()}')
                return breakout_signal
        
        else:  # Ranging - accept signals at extremes only
            # Prefer breakouts in ranging markets
            if breakout_signal:
                breakout_signal['reasons'].insert(0, 'Regime: RANGING')
                return breakout_signal
        
        return None


# Register strategies with manager
def register_scalper_strategies(strategy_manager):
    """Register all scalper strategies with the strategy manager"""
    try:
        strategy_manager.register_strategy('trend_scalper', TrendScalper())
        strategy_manager.register_strategy('breakout_scalper', BreakoutScalper())
        strategy_manager.register_strategy('regime_scalper', RegimeAwareScalper())
        logger.info("Scalper strategies registered: trend_scalper, breakout_scalper, regime_scalper")
    except Exception as e:
        logger.error(f"Failed to register scalper strategies: {e}")


if __name__ == '__main__':
    import pandas as pd
    
    print("=" * 60)
    print("PROFESSIONAL SCALPER STRATEGIES")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 100
    price = 90000 + np.cumsum(np.random.randn(n) * 100)
    
    df = pd.DataFrame({
        'close': price,
        'high': price + np.random.rand(n) * 50,
        'low': price - np.random.rand(n) * 50,
        'volume': np.random.rand(n) * 1000 + 500
    })
    
    # Test strategies
    strategies = [
        TrendScalper(),
        BreakoutScalper(),
        RegimeAwareScalper()
    ]
    
    for strat in strategies:
        signal = strat.generate_signal(df)
        print(f"\n{strat.name}:")
        if signal:
            print(f"  Action: {signal['action']}")
            print(f"  Confidence: {signal['confidence']:.0%}")
            print(f"  Reasons: {signal['reasons']}")
        else:
            print("  No signal")
    
    print("\n" + "=" * 60)
    print("✓ All scalper strategies initialized successfully")
    print("=" * 60)
