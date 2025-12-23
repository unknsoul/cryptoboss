"""
Mean Reversion Strategy
Uses RSI + Bollinger Bands to trade range-bound markets
"""

import numpy as np
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy
from ..indicators import TechnicalIndicators


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy - Best for range-bound markets
    
    Entry Rules:
    - LONG: Price touches lower BB + RSI < 30 (oversold)
    - SHORT: Price touches upper BB + RSI > 70 (overbought)
    
    Exit Rules:
    - LONG: Price touches upper BB or RSI > 70
    - SHORT: Price touches lower BB or RSI < 30
    """
    
    def __init__(self, bb_period=20, bb_std=2, rsi_period=14, 
                 rsi_oversold=30, rsi_overbought=70):
        super().__init__("Mean Reversion", "mean_reversion")
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        self.parameters = {
            'bb_period': bb_period,
            'bb_std': bb_std,
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought
        }
    
    def signal(self, highs, lows, closes, volumes=None):
        """Generate mean reversion signal"""
        if not self.validate_data(highs, lows, closes):
            return None
        
        if len(closes) < max(self.bb_period, self.rsi_period) + 10:
            return None
        
        # Calculate indicators
        bb_upper, bb_middle, bb_lower, bandwidth, percent_b = \
            TechnicalIndicators.bollinger_bands(closes, self.bb_period, self.bb_std)
        
        rsi = TechnicalIndicators.rsi(closes, self.rsi_period)
        
        current_price = closes[-1]
        current_rsi = rsi[-1]
        current_bb_lower = bb_lower[-1]
        current_bb_upper = bb_upper[-1]
        current_bb_middle = bb_middle[-1]
        current_percent_b = percent_b[-1]
        
        # LONG signal - Oversold conditions (relaxed thresholds)
        if current_percent_b < 0.25 and current_rsi < self.rsi_oversold + 5:
            stop_distance = current_price - current_bb_lower
            
            return {
                'action': 'LONG',
                'stop': max(stop_distance, current_price * 0.02),  # Min 2% stop
                'target': current_bb_upper - current_price,
                'confidence': (self.rsi_oversold - current_rsi) / self.rsi_oversold,
                'metadata': {
                    'rsi': current_rsi,
                    'percent_b': current_percent_b,
                    'bandwidth': bandwidth[-1],
                    'entry_type': 'oversold_bounce'
                }
            }
        
        # SHORT signal - Overbought conditions (relaxed thresholds)
        if current_percent_b > 0.75 and current_rsi > self.rsi_overbought - 5:
            stop_distance = current_bb_upper - current_price
            
            return {
                'action': 'SHORT',
                'stop': max(stop_distance, current_price * 0.02),
                'target': current_price - current_bb_lower,
                'confidence': (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought),
                'metadata': {
                    'rsi': current_rsi,
                    'percent_b': current_percent_b,
                    'bandwidth': bandwidth[-1],
                    'entry_type': 'overbought_reversal'
                }
            }
        
        return None
    
    def check_exit(self, highs, lows, closes, position_side, 
                   entry_price, entry_index, current_index):
        """Check if should exit mean reversion position"""
        if current_index - entry_index < 2:
            return False
        
        # Calculate indicators for exit
        bb_upper, bb_middle, bb_lower, _, percent_b = \
            TechnicalIndicators.bollinger_bands(closes[:current_index + 1], self.bb_period, self.bb_std)
        
        rsi = TechnicalIndicators.rsi(closes[:current_index + 1], self.rsi_period)
        
        current_rsi = rsi[-1]
        current_percent_b = percent_b[-1]
        
        if position_side == 'LONG':
            # Exit LONG: reached overbought or upper BB
            if current_rsi > self.rsi_overbought or current_percent_b > 0.8:
                return True
        
        elif position_side == 'SHORT':
            # Exit SHORT: reached oversold or lower BB
            if current_rsi < self.rsi_oversold or current_percent_b < 0.2:
                return True
        
        return False
