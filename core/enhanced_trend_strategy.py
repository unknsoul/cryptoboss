"""
Enhanced Trend Strategy - Tier 1 Improvements
Builds on SimpleTrendStrategy with advanced entry filters and exit logic
"""

import numpy as np
from .strategy import SimpleTrendStrategy, Indicators
from .indicators import TechnicalIndicators


class EnhancedTrendStrategy(SimpleTrendStrategy):
    """
    Enhanced trend-following strategy with:
    - Volume confirmation (prevents low-volume false breakouts)
    - ATR volatility filter (avoids low-volatility whipsaws)
    - ADX trend strength filter (ensures strong trends)
    - Time-of-day filter (optional, avoids low liquidity hours)
    
    Expected improvements:
    - Win rate: 39% → 48-52%
    - Walk-forward efficiency: -0.25 → 0.5-0.7
    - Expectancy: $31.70 → $50-70
    """
    
    def __init__(self,
                 ema_fast=50,
                 ema_slow=200,
                 donchian_period=20,
                 atr_period=14,
                 atr_multiplier=2.0,
                 # New filter parameters
                 use_volume_filter=True,
                 volume_period=20,
                 volume_threshold=0.8,
                 use_volatility_filter=True,
                 volatility_percentile_threshold=30,
                 volatility_lookback=100,
                 use_adx_filter=True,
                 adx_period=14,
                 adx_threshold=25,
                 use_time_filter=False,
                 avoid_hours=None):  # List of hours to avoid, e.g., [0, 1, 2, 23]
        
        super().__init__(ema_fast, ema_slow, donchian_period, atr_period, atr_multiplier)
        
        # Filter settings
        self.use_volume_filter = use_volume_filter
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold
        
        self.use_volatility_filter = use_volatility_filter
        self.volatility_percentile_threshold = volatility_percentile_threshold
        self.volatility_lookback = volatility_lookback
        
        self.use_adx_filter = use_adx_filter
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        
        self.use_time_filter = use_time_filter
        self.avoid_hours = avoid_hours or [0, 1, 2, 22, 23]  # Default: avoid late night/early morning
    
    def _check_volume_filter(self, volumes):
        """Check if current volume is above average"""
        if not self.use_volume_filter or volumes is None:
            return True
        
        if len(volumes) < self.volume_period:
            return False
        
        avg_volume = np.mean(volumes[-self.volume_period:])
        current_volume = volumes[-1]
        
        return current_volume > avg_volume * self.volume_threshold
    
    def _check_volatility_filter(self, highs, lows, closes):
        """Check if volatility is high enough (prevents trading in dead markets)"""
        if not self.use_volatility_filter:
            return True
        
        lookback = min(self.volatility_lookback, len(closes))
        if lookback < self.atr_period + 10:
            return True  # Not enough data
        
        # Calculate current ATR
        current_atr = Indicators.atr(highs, lows, closes, self.atr_period)
        if current_atr is None:
            return False
        
        # Calculate ATR history for percentile
        atr_history = []
        for i in range(self.atr_period + 1, len(closes)):
            hist_atr = Indicators.atr(
                highs[:i], 
                lows[:i], 
                closes[:i], 
                self.atr_period
            )
            if hist_atr is not None:
                atr_history.append(hist_atr)
        
        if len(atr_history) < 10:
            return True  # Not enough history
        
        # Use recent history for percentile calculation
        recent_history = atr_history[-lookback:] if len(atr_history) > lookback else atr_history
        
        # Calculate percentile: what % of historical ATRs are less than current
        count_below = sum(1.0 for x in recent_history if x < current_atr)
        percentile = (count_below / len(recent_history)) * 100.0
        
        return percentile >= self.volatility_percentile_threshold
    
    def _check_adx_filter(self, highs, lows, closes):
        """Check if trend strength is sufficient"""
        if not self.use_adx_filter:
            return True
        
        if len(closes) < self.adx_period + 1:
            return False
        
        # Calculate ADX using TechnicalIndicators (returns arrays)
        adx_array, plus_di, minus_di = TechnicalIndicators.adx(
            highs, lows, closes, self.adx_period
        )
        
        # Extract the last value (current ADX)
        adx_value = adx_array[-1] if isinstance(adx_array, np.ndarray) else adx_array
        
        if adx_value is None or np.isnan(adx_value):
            return False
        
        return float(adx_value) >= self.adx_threshold
    
    def _check_time_filter(self, timestamp=None):
        """Check if current time is acceptable for trading"""
        if not self.use_time_filter or timestamp is None:
            return True
        
        # Extract hour from timestamp
        if hasattr(timestamp, 'hour'):
            hour = timestamp.hour
        else:
            # Assume it's a string or can be parsed
            try:
                from datetime import datetime
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour = dt.hour
                else:
                    return True  # Can't parse, allow trade
            except:
                return True  # Can't parse, allow trade
        
        return hour not in self.avoid_hours
    
    def signal(self, highs, lows, closes, volumes=None, timestamp=None):
        """
        Generate trading signal with enhanced filters
        
        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            volumes: Volume data (optional, required for volume filter)
            timestamp: Current timestamp (optional, for time filter)
        
        Returns:
            Signal dict or None
        """
        # Get base signal from SimpleTrendStrategy
        base_signal = super().signal(highs, lows, closes, volumes)
        
        if base_signal is None:
            return None
        
        # Apply enhanced filters
        if not self._check_volume_filter(volumes):
            return None  # Skip low volume breakouts
        
        if not self._check_volatility_filter(highs, lows, closes):
            return None  # Skip low volatility periods
        
        if not self._check_adx_filter(highs, lows, closes):
            return None  # Skip weak trends
        
        if not self._check_time_filter(timestamp):
            return None  # Skip unfavorable hours
        
        # All filters passed - enhance metadata
        base_signal['metadata']['filters_applied'] = {
            'volume': self.use_volume_filter,
            'volatility': self.use_volatility_filter,
            'adx': self.use_adx_filter,
            'time': self.use_time_filter
        }
        
        # Add filter values for debugging
        if volumes is not None and len(volumes) >= self.volume_period:
            base_signal['metadata']['volume_ratio'] = volumes[-1] / np.mean(volumes[-self.volume_period:])
        
        # Calculate and add ADX value
        if len(closes) >= self.adx_period + 1:
            adx_array, _, _ = TechnicalIndicators.adx(highs, lows, closes, self.adx_period)
            if adx_array is not None and len(adx_array) > 0:
                adx_value = adx_array[-1] if isinstance(adx_array, np.ndarray) else adx_array
                if not np.isnan(adx_value):
                    base_signal['metadata']['adx_value'] = float(adx_value)
        
        return base_signal
    
    def check_exit(self, highs, lows, closes, position_side, entry_price, entry_index, current_index):
        """
        Check if position should be exited
        
        For now, uses parent class logic (rely on stops and partial profits in backtest)
        Future enhancement: Add trend exhaustion exits (RSI divergence)
        """
        return super().check_exit(highs, lows, closes, position_side, entry_price, entry_index, current_index)
