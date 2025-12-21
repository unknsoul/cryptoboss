"""
Scalping Strategy
High-frequency strategy for quick small profits
"""

import numpy as np
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy
from ..indicators import TechnicalIndicators


class ScalpingStrategy(BaseStrategy):
    """
    Scalping Strategy - High frequency quick profits
    
    Entry Rules:
    - LONG: Fast EMA > Slow EMA + RSI > 50 + price momentum + volume
    - SHORT: Fast EMA < Slow EMA + RSI < 50 + price momentum + volume
    
    Exit Rules:
    - Quick profit target (0.5-1%)
    - Tight stop loss (0.3-0.5%)
    - Time-based exit (max hold time)
    """
    
    def __init__(self, ema_fast=5, ema_slow=15, rsi_period=7,
                 profit_target_pct=0.008, stop_loss_pct=0.004, max_hold_bars=10):
        super().__init__("Scalping", "scalping")
        
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_bars = max_hold_bars
        
        self.parameters = {
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'rsi_period': rsi_period,
            'profit_target_pct': profit_target_pct,
            'stop_loss_pct': stop_loss_pct,
            'max_hold_bars': max_hold_bars
        }
    
    def signal(self, highs, lows, closes, volumes=None):
        """Generate scalping signal"""
        if not self.validate_data(highs, lows, closes):
            return None
        
        min_bars = max(self.ema_slow, self.rsi_period) + 5
        if len(closes) < min_bars:
            return None
        
        # Calculate indicators
        ema_fast = TechnicalIndicators.ema(closes, self.ema_fast)
        ema_slow = TechnicalIndicators.ema(closes, self.ema_slow)
        rsi = TechnicalIndicators.rsi(closes, self.rsi_period)
        
        current_price = closes[-1]
        current_ema_fast = ema_fast[-1]
        current_ema_slow = ema_slow[-1]
        prev_ema_fast = ema_fast[-2]
        prev_ema_slow = ema_slow[-2]
        current_rsi = rsi[-1]
        
        # Price momentum
        price_change = (current_price - closes[-3]) / closes[-3] if len(closes) > 3 else 0
        
        # Volume confirmation (optional)
        volume_ok = True
        if volumes is not None and len(volumes) == len(closes) and len(volumes) > 10:
            recent_vol_avg = np.mean(volumes[-10:-1])
            volume_ok = volumes[-1] > recent_vol_avg * 0.9
        
        # LONG signal - Bullish scalp setup
        bullish_ema = current_ema_fast > current_ema_slow
        bullish_cross = prev_ema_fast <= prev_ema_slow and current_ema_fast > current_ema_slow
        bullish_rsi = current_rsi > 50 and current_rsi < 75  # Not overbought
        bullish_momentum = price_change > 0
        
        if (bullish_cross or (bullish_ema and bullish_momentum)) and bullish_rsi and volume_ok:
            stop = current_price * self.stop_loss_pct
            target = current_price * self.profit_target_pct
            
            return {
                'action': 'LONG',
                'stop': stop,
                'target': target,
                'confidence': min(abs(current_ema_fast - current_ema_slow) / current_price + 0.5, 1.0),
                'metadata': {
                    'ema_fast': current_ema_fast,
                    'ema_slow': current_ema_slow,
                    'rsi': current_rsi,
                    'momentum': price_change,
                    'entry_type': 'bullish_scalp',
                    'max_hold_bars': self.max_hold_bars
                }
            }
        
        # SHORT signal - Bearish scalp setup
        bearish_ema = current_ema_fast < current_ema_slow
        bearish_cross = prev_ema_fast >= prev_ema_slow and current_ema_fast < current_ema_slow
        bearish_rsi = current_rsi < 50 and current_rsi > 25  # Not oversold
        bearish_momentum = price_change < 0
        
        if (bearish_cross or (bearish_ema and bearish_momentum)) and bearish_rsi and volume_ok:
            stop = current_price * self.stop_loss_pct
            target = current_price * self.profit_target_pct
            
            return {
                'action': 'SHORT',
                'stop': stop,
                'target': target,
                'confidence': min(abs(current_ema_fast - current_ema_slow) / current_price + 0.5, 1.0),
                'metadata': {
                    'ema_fast': current_ema_fast,
                    'ema_slow': current_ema_slow,
                    'rsi': current_rsi,
                    'momentum': price_change,
                    'entry_type': 'bearish_scalp',
                    'max_hold_bars': self.max_hold_bars
                }
            }
        
        return None
    
    def check_exit(self, highs, lows, closes, position_side,
                   entry_price, entry_index, current_index):
        """Check if should exit scalp position"""
        # Time-based exit
        if current_index - entry_index >= self.max_hold_bars:
            return True
        
        current_price = closes[current_index]
        
        if position_side == 'LONG':
            # Quick profit or stop loss
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= self.profit_target_pct or profit_pct <= -self.stop_loss_pct:
                return True
        
        elif position_side == 'SHORT':
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= self.profit_target_pct or profit_pct <= -self.stop_loss_pct:
                return True
        
        return False
