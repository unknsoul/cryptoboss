"""
Bollinger Breakout Strategy
Volatility breakout strategy for range expansion
"""

import numpy as np
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy
from ..indicators import TechnicalIndicators


class BollingerBreakoutStrategy(BaseStrategy):
    """
    Bollinger Breakout Strategy - Best for breakout opportunities
    
    Entry Rules:
    - Detect BB squeeze (low volatility)
    - LONG: Price breaks above upper BB + volume spike
    - SHORT: Price breaks below lower BB + volume spike
    
    Exit Rules:
    - Return to middle BB
    - Opposite breakout
    - Target hit (2x ATR)
    """
    
    def __init__(self, bb_period=20, bb_std=2, kc_period=20, kc_mult=1.5,
                 volume_period=20, volume_threshold=1.5, min_squeeze_bars=5):
        super().__init__("Bollinger Breakout", "breakout")
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold
        self.min_squeeze_bars = min_squeeze_bars
        
        self.parameters = {
            'bb_period': bb_period,
            'bb_std': bb_std,
            'kc_period': kc_period,
            'kc_mult': kc_mult,
            'volume_period': volume_period,
            'volume_threshold': volume_threshold,
            'min_squeeze_bars': min_squeeze_bars
        }
    
    def signal(self, highs, lows, closes, volumes=None):
        """Generate bollinger breakout signal"""
        if not self.validate_data(highs, lows, closes):
            return None
        
        min_bars = max(self.bb_period, self.kc_period, self.volume_period) + 20
        if len(closes) < min_bars:
            return None
        
        # Calculate indicators
        bb_upper, bb_middle, bb_lower, bandwidth, percent_b = \
            TechnicalIndicators.bollinger_bands(closes, self.bb_period, self.bb_std)
        
        squeeze = TechnicalIndicators.squeeze_detector(
            highs, lows, closes, self.bb_period, self.kc_period, self.kc_mult
        )
        
        atr = TechnicalIndicators.atr(highs, lows, closes, 14)
        
        # Check for recent squeeze
        recent_squeeze = np.sum(squeeze[-self.min_squeeze_bars:]) >= self.min_squeeze_bars - 1
        
        # Volume confirmation
        volume_spike = False
        if volumes is not None and len(volumes) == len(closes):
            vol_ma = TechnicalIndicators.volume_ma(volumes, self.volume_period)
            volume_spike = volumes[-1] > vol_ma[-1] * self.volume_threshold
        else:
            volume_spike = True  # Skip volume check if not available
        
        current_price = closes[-1]
        prev_price = closes[-2]
        current_bb_upper = bb_upper[-1]
        current_bb_lower = bb_lower[-1]
        current_bb_middle = bb_middle[-1]
        prev_bb_upper = bb_upper[-2]
        prev_bb_lower = bb_lower[-2]
        current_atr = atr[-1]
        current_bandwidth = bandwidth[-1]
        
        # LONG signal - Breakout above upper BB after squeeze
        bullish_breakout = prev_price <= prev_bb_upper and current_price > current_bb_upper
        
        if bullish_breakout and volume_spike and not squeeze[-1]:
            # Check if there was a recent squeeze
            if recent_squeeze:
                return {
                    'action': 'LONG',
                    'stop': current_atr * 1.5,
                    'target': current_atr * 3.0,
                    'confidence': min(volume_spike * 0.8 + 0.2, 1.0),
                    'metadata': {
                        'bandwidth': current_bandwidth,
                        'percent_b': percent_b[-1],
                        'had_squeeze': recent_squeeze,
                        'volume_spike': volume_spike,
                        'entry_type': 'bullish_breakout'
                    }
                }
        
        # SHORT signal - Breakout below lower BB after squeeze
        bearish_breakout = prev_price >= prev_bb_lower and current_price < current_bb_lower
        
        if bearish_breakout and volume_spike and not squeeze[-1]:
            if recent_squeeze:
                return {
                    'action': 'SHORT',
                    'stop': current_atr * 1.5,
                    'target': current_atr * 3.0,
                    'confidence': min(volume_spike * 0.8 + 0.2, 1.0),
                    'metadata': {
                        'bandwidth': current_bandwidth,
                        'percent_b': percent_b[-1],
                        'had_squeeze': recent_squeeze,
                        'volume_spike': volume_spike,
                        'entry_type': 'bearish_breakout'
                    }
                }
        
        return None
    
    def check_exit(self, highs, lows, closes, position_side,
                   entry_price, entry_index, current_index):
        """Check if should exit breakout position"""
        if current_index - entry_index < 2:
            return False
        
        # Calculate BB for exit
        bb_upper, bb_middle, bb_lower, _, _ = TechnicalIndicators.bollinger_bands(
            closes[:current_index + 1], self.bb_period, self.bb_std
        )
        
        current_price = closes[current_index]
        current_bb_middle = bb_middle[-1]
        
        if position_side == 'LONG':
            # Exit LONG: price returns to middle BB or below
            if current_price <= current_bb_middle:
                return True
        
        elif position_side == 'SHORT':
            # Exit SHORT: price returns to middle BB or above
            if current_price >= current_bb_middle:
                return True
        
        return False
