"""
Professional Trading Strategy - Clean Implementation
Simple, robust trend-following with proper lookahead bias prevention
"""

import numpy as np


class Indicators:
    """Technical indicators with strict no-lookahead policy"""
    
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
    def donchian_channel(highs, lows, period=20, exclude_current=True):
        """
        Calculate Donchian Channel
        
        Args:
            highs: High prices
            lows: Low prices
            period: Lookback period
            exclude_current: If True, excludes current candle to prevent lookahead bias
        
        Returns:
            (upper_band, lower_band) or (None, None)
        """
        if len(highs) < period + 1:
            return None, None
        
        if exclude_current:
            # CRITICAL: Exclude current candle to prevent lookahead bias
            upper = max(highs[-period-1:-1])
            lower = min(lows[-period-1:-1])
        else:
            upper = max(highs[-period:])
            lower = min(lows[-period:])
        
        return upper, lower


class SimpleTrendStrategy:
    """
    Clean trend-following strategy:
    - EMA 50/200 trend filter
    - Donchian 20 breakouts (NO lookahead)
    - ATR-based stops and position sizing
    """
    
    def __init__(self, 
                 ema_fast=50,
                 ema_slow=200,
                 donchian_period=20,
                 atr_period=14,
                 atr_multiplier=2.0):
        
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.donchian_period = donchian_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    def signal(self, highs, lows, closes, volumes=None):
        """
        Generate trading signal
        
        Returns:
            {'action': 'LONG'/'SHORT', 'stop': stop_distance, 'metadata': dict} or None
        """
        # Need enough bars
        required = max(self.ema_slow, self.donchian_period + 1, self.atr_period + 1)
        if len(closes) < required:
            return None
        
        current_price = closes[-1]
        
        # Calculate indicators
        ema_fast = Indicators.ema(closes[-self.ema_fast:], self.ema_fast)
        ema_slow = Indicators.ema(closes[-self.ema_slow:], self.ema_slow)
        atr = Indicators.atr(highs, lows, closes, self.atr_period)
        donchian_high, donchian_low = Indicators.donchian_channel(
            highs, lows, self.donchian_period, exclude_current=True
        )
        
        if any(x is None for x in [ema_fast, ema_slow, atr, donchian_high, donchian_low]):
            return None
        
        # Calculate stop distance
        stop_distance = atr * self.atr_multiplier
        
        # LONG: Breakout above Donchian high + trending up
        if current_price > donchian_high and ema_fast > ema_slow:
            return {
                'action': 'LONG',
                'stop': stop_distance,
                'metadata': {
                    'entry_price': current_price,
                    'donchian_high': donchian_high,
                    'ema_fast': ema_fast,
                    'ema_slow': ema_slow,
                    'atr': atr
                }
            }
        
        # SHORT: Breakout below Donchian low + trending down
        if current_price < donchian_low and ema_fast < ema_slow:
            return {
                'action': 'SHORT',
                'stop': stop_distance,
                'metadata': {
                    'entry_price': current_price,
                    'donchian_low': donchian_low,
                    'ema_fast': ema_fast,
                    'ema_slow': ema_slow,
                    'atr': atr
                }
            }
        
        return None
    
    def check_exit(self, highs, lows, closes, position_side, entry_price, entry_index, current_index):
        """Check if position should be exited (return True to exit)"""
        # For now, rely on stops - can add trend reversal exit later
        return False


# Keep the old class names for backward compatibility
class ProfessionalIndicators(Indicators):
    pass


class ProfessionalTrendStrategy(SimpleTrendStrategy):
    """Alias for backward compatibility"""
    pass
