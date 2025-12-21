"""
Professional Trend Strategy (Wrapper)
Adapts the existing ProfessionalTrendStrategy to the new base strategy interface
"""

import numpy as np
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy
from ..strategy import ProfessionalTrendStrategy as OriginalProfessionalTrendStrategy


class ProfessionalTrendStrategyWrapper(BaseStrategy):
    """
    Wrapper for the original ProfessionalTrendStrategy
    Adapts it to the new BaseStrategy interface
    """
    
    def __init__(self, donchian_period=20, ema_fast=12, ema_slow=50,
                 atr_period=14, atr_multiplier=3.0, adx_period=14,
                 adx_threshold=25, rsi_period=14):
        super().__init__("Professional Trend", "trend_following")
        
        # Create the original strategy instance
        self.strategy = OriginalProfessionalTrendStrategy(
            donchian_period=donchian_period,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            rsi_period=rsi_period
        )
        
        self.parameters = {
            'donchian_period': donchian_period,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'rsi_period': rsi_period
        }
    
    def signal(self, highs, lows, closes, volumes=None):
        """Generate signal using original strategy"""
        if not self.validate_data(highs, lows, closes):
            return None
        
        # Call original strategy
        signal = self.strategy.signal(highs, lows, closes)
        
        if signal is None:
            return None
        
        # Adapt to new format
        return {
            'action': signal['action'],
            'stop': signal['stop'],
            'target': signal.get('stop', 0) * 2,  # 2:1 risk/reward
            'confidence': 0.8,  # Default confidence
            'metadata': {
                'volatility_pct': signal.get('volatility_pct', 50),
                'adx': signal.get('adx', 0),
                'rsi': signal.get('rsi', 50),
                'regime': signal.get('regime', 'UNKNOWN'),
                'entry_type': 'donchian_breakout'
            }
        }
    
    def check_exit(self, highs, lows, closes, position_side,
                   entry_price, entry_index, current_index):
        """Check exit using original strategy"""
        # Slice data up to current index
        return self.strategy.check_exit(
            highs[:current_index + 1],
            lows[:current_index + 1],
            closes[:current_index + 1],
            position_side
        )
