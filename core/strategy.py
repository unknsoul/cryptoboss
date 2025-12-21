"""
Professional Trading Indicators and Strategy
Implements ADX, RSI, and advanced trend-following strategy
"""

import numpy as np


class ProfessionalIndicators:
    """Advanced technical indicators for professional trading"""
    
    @staticmethod
    def ema(prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None
        
        alpha = 2.0 / (period + 1)
        ema_value = prices[0]
        
        for price in prices[1:]:
            ema_value = alpha * price + (1 - alpha) * ema_value
        
        return ema_value
    
    @staticmethod
    def atr(highs, lows, closes, period=14):
        """Calculate Average True Range"""
        if len(closes) < period + 1:
            return None
        
        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            return None
        
        return np.mean(true_ranges[-period:])
    
    @staticmethod
    def adx(highs, lows, closes, period=14):
        """
        Calculate ADX (Average Directional Index) and directional indicators
        Returns: (adx_value, plus_di, minus_di)
        """
        if len(closes) < period + 1:
            return None, None, None
        
        # Calculate +DM and -DM
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(highs)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]
            
            # +DM
            if high_diff > low_diff and high_diff > 0:
                plus_dm.append(high_diff)
            else:
                plus_dm.append(0)
            
            # -DM
            if low_diff > high_diff and low_diff > 0:
                minus_dm.append(low_diff)
            else:
                minus_dm.append(0)
        
        # Calculate True Range
        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            true_ranges.append(tr)
        
        if len(true_ranges) < period or len(plus_dm) < period:
            return None, None, None
        
        # Smooth using EMA-like approach
        atr_smooth = np.mean(true_ranges[:period])
        plus_dm_smooth = np.mean(plus_dm[:period])
        minus_dm_smooth = np.mean(minus_dm[:period])
        
        for i in range(period, len(true_ranges)):
            atr_smooth = (atr_smooth * (period - 1) + true_ranges[i]) / period
            plus_dm_smooth = (plus_dm_smooth * (period - 1) + plus_dm[i]) / period
            minus_dm_smooth = (minus_dm_smooth * (period - 1) + minus_dm[i]) / period
        
        # Calculate +DI and -DI
        if atr_smooth == 0:
            return None, None, None
        
        plus_di = 100 * (plus_dm_smooth / atr_smooth)
        minus_di = 100 * (minus_dm_smooth / atr_smooth)
        
        # Calculate DX and ADX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return None, plus_di, minus_di
        
        dx = 100 * abs(plus_di - minus_di) / di_sum
        
        # For simplicity, return DX as ADX (proper ADX would need smoothing over multiple DX values)
        # In production, you'd maintain a rolling ADX calculation
        adx_value = dx
        
        return adx_value, plus_di, minus_di
    
    @staticmethod
    def rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes
        changes = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        
        # Calculate average gain and loss
        if len(gains) < period:
            return None
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def volatility_percentile(current_atr, atr_history, lookback=100):
        """
        Calculate what percentile current ATR is in historical distribution
        Returns: 0-100, where higher = more volatile than usual
        """
        if len(atr_history) < lookback:
            return 50  # Default to median if not enough data
        
        recent_atrs = atr_history[-lookback:]
        percentile = np.percentile(recent_atrs, np.searchsorted(np.sort(recent_atrs), current_atr) / len(recent_atrs) * 100)
        
        # Simpler approach
        sorted_atrs = sorted(recent_atrs)
        rank = sum(1 for atr in sorted_atrs if atr < current_atr)
        percentile = (rank / len(sorted_atrs)) * 100
        
        return percentile


class ProfessionalTrendStrategy:
    """
    Advanced trend-following strategy with:
    - ADX trend strength filter
    - RSI momentum confirmation
    - Donchian breakout entries
    - EMA trend filter
    - Volatility-based position sizing
    - Market regime detection
    """
    
    def __init__(self, 
                 donchian_period=20,
                 ema_fast=12,
                 ema_slow=50,
                 atr_period=14,
                 atr_multiplier=3.0,
                 adx_period=14,
                 adx_threshold=25,
                 rsi_period=14):
        
        # Donchian parameters
        self.donchian_period = donchian_period
        
        # EMA parameters
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        
        # ATR parameters
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        # ADX parameters
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        
        # RSI parameters
        self.rsi_period = rsi_period
        
        # Track ATR history for volatility percentile
        self.atr_history = []
    
    def signal(self, highs, lows, closes):
        """
        Generate trading signal with professional filters
        
        Returns: dict with signal info or None
        {
            'action': 'LONG' or 'SHORT',
            'stop': stop distance,
            'volatility_pct': volatility percentile,
            'adx': ADX value,
            'regime': 'STRONG_TREND', 'WEAK_TREND', or 'RANGING'
        }
        """
        # Need enough data
        required_bars = max(self.ema_slow, self.donchian_period + 1, 
                           self.atr_period + 1, self.adx_period + 1, self.rsi_period + 1)
        
        if len(closes) < required_bars:
            return None
        
        current_price = closes[-1]
        
        # Calculate all indicators
        ema_fast = ProfessionalIndicators.ema(closes[-self.ema_fast:], self.ema_fast)
        ema_slow = ProfessionalIndicators.ema(closes[-self.ema_slow:], self.ema_slow)
        atr = ProfessionalIndicators.atr(highs, lows, closes, self.atr_period)
        adx, plus_di, minus_di = ProfessionalIndicators.adx(highs, lows, closes, self.adx_period)
        rsi = ProfessionalIndicators.rsi(closes, self.rsi_period)
        
        # Check if all indicators calculated successfully
        if any(x is None for x in [ema_fast, ema_slow, atr, adx, plus_di, minus_di, rsi]):
            return None
        
        # Track ATR history
        self.atr_history.append(atr)
        if len(self.atr_history) > 200:  # Keep last 200 values
            self.atr_history = self.atr_history[-200:]
        
        # Calculate volatility percentile
        volatility_pct = ProfessionalIndicators.volatility_percentile(
            atr, self.atr_history, lookback=100
        )
        
        # MARKET REGIME DETECTION
        if adx < 20:
            regime = 'RANGING'
            return None  # Don't trade in ranging markets
        elif adx < self.adx_threshold:
            regime = 'WEAK_TREND'
            return None  # Don't trade weak trends
        else:
            regime = 'STRONG_TREND'
        
        # Calculate Donchian channels (exclude current candle)
        donchian_high = max(highs[-self.donchian_period - 1:-1])
        donchian_low = min(lows[-self.donchian_period - 1:-1])
        
        # Calculate stop distance (wider in high volatility)
        base_stop = atr * self.atr_multiplier
        
        # Adjust stop based on volatility percentile
        if volatility_pct > 75:  # High volatility
            stop_distance = base_stop * 1.2  # 20% wider stop
        elif volatility_pct < 25:  # Low volatility
            stop_distance = base_stop * 0.9  # 10% tighter stop
        else:
            stop_distance = base_stop
        
        # ===== LONG ENTRY CONDITIONS =====
        # Must satisfy ALL of:
        # 1. Donchian breakout
        # 2. EMA trend (fast > slow)
        # 3. ADX > threshold (strong trend)
        # 4. +DI > -DI (bullish directional movement)
        # 5. RSI > 50 (bullish momentum)
        
        if (current_price > donchian_high and
            ema_fast > ema_slow and
            adx >= self.adx_threshold and
            plus_di > minus_di and
            rsi > 50 and rsi < 70):  # Not overbought
            
            return {
                'action': 'LONG',
                'stop': stop_distance,
                'volatility_pct': volatility_pct,
                'adx': adx,
                'rsi': rsi,
                'regime': regime
            }
        
        # ===== SHORT ENTRY CONDITIONS =====
        # Must satisfy ALL of:
        # 1. Donchian breakout (downward)
        # 2. EMA trend (fast < slow)
        # 3. ADX > threshold (strong trend)
        # 4. -DI > +DI (bearish directional movement)
        # 5. RSI < 50 (bearish momentum)
        
        if (current_price < donchian_low and
            ema_fast < ema_slow and
            adx >= self.adx_threshold and
            minus_di > plus_di and
            rsi < 50 and rsi > 30):  # Not oversold
            
            return {
                'action': 'SHORT',
                'stop': stop_distance,
                'volatility_pct': volatility_pct,
                'adx': adx,
                'rsi': rsi,
                'regime': regime
            }
        
        return None
    
    def check_exit(self, highs, lows, closes, position_side):
        """
        Check if position should be exited based on ADX
        Returns: True if should exit, False otherwise
        """
        if len(closes) < self.adx_period + 1:
            return False
        
        adx, plus_di, minus_di = ProfessionalIndicators.adx(highs, lows, closes, self.adx_period)
        
        if adx is None:
            return False
        
        # Exit if trend weakens (ADX falls below 20)
        if adx < 20:
            return True
        
        # Exit long if bearish directional change
        if position_side == 'LONG' and minus_di > plus_di and adx > 25:
            return True
        
        # Exit short if bullish directional change
        if position_side == 'SHORT' and plus_di > minus_di and adx > 25:
            return True
        
        return False
