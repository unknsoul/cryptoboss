"""
MACD Crossover Strategy
Momentum-based strategy for trending markets
"""

import numpy as np
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy
from ..indicators import TechnicalIndicators


class MACDCrossoverStrategy(BaseStrategy):
    """
    MACD Crossover Strategy - Best for trending markets
    
    Entry Rules:
    - LONG: MACD crosses above signal + price > EMA50 + volume confirmation
    - SHORT: MACD crosses below signal + price < EMA50 + volume confirmation
    
    Exit Rules:
    - Opposite MACD crossover
    - MACD histogram weakening
    """
    
    def __init__(self, macd_fast=12, macd_slow=26, macd_signal=9,
                 ema_trend=50, volume_period=20, volume_threshold=1.2):
        super().__init__("MACD Crossover", "momentum")
        
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_trend = ema_trend
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold
        
        self.parameters = {
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'ema_trend': ema_trend,
            'volume_period': volume_period,
            'volume_threshold': volume_threshold
        }
    
    def signal(self, highs, lows, closes, volumes=None):
        """Generate MACD crossover signal"""
        if not self.validate_data(highs, lows, closes):
            return None
        
        min_bars = max(self.macd_slow, self.ema_trend, self.volume_period) + 10
        if len(closes) < min_bars:
            return None
        
        # Calculate MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )
        
        # Calculate trend EMA
        ema = TechnicalIndicators.ema(closes, self.ema_trend)
        
        # Calculate ATR for stop loss
        atr = TechnicalIndicators.atr(highs, lows, closes, 14)
        
        # Volume confirmation if available
        volume_confirmed = True
        if volumes is not None and len(volumes) == len(closes):
            vol_ma = TechnicalIndicators.volume_ma(volumes, self.volume_period)
            volume_confirmed = float(volumes[-1]) > float(vol_ma[-1]) * self.volume_threshold
        
        # LONG signal - Bullish crossover
        bullish_cross = prev_macd <= prev_signal and current_macd > current_signal
        uptrend = current_price > current_ema
        
        if bullish_cross and uptrend and volume_confirmed and current_histogram > 0:
            return {
                'action': 'LONG',
                'stop': current_atr * 2.0,
                'target': current_atr * 3.0,
                'confidence': 0.8,
                'metadata': {
                    'macd': current_macd,
                    'signal': current_signal,
                    'entry_type': 'bullish_crossover'
                }
            }
        
        # SHORT signal - Bearish crossover
        bearish_cross = prev_macd >= prev_signal and current_macd < current_signal
        downtrend = current_price < current_ema
        
        if bearish_cross and downtrend and volume_confirmed and current_histogram < 0:
            return {
                'action': 'SHORT',
                'stop': current_atr * 2.0,
                'target': current_atr * 3.0,
                'confidence': 0.8,
                'metadata': {
                    'macd': current_macd,
                    'signal': current_signal,
                    'entry_type': 'bearish_crossover'
                }
            }
        
        return None
    
    def check_exit(self, highs, lows, closes, position_side,
                   entry_price, entry_index, current_index):
        """Check if should exit MACD position"""
        try:
            if current_index - entry_index < 2:
                return False
            
            # Calculate MACD for exit
            macd_line, signal_line, histogram = TechnicalIndicators.macd(
                closes[:current_index + 1],
                self.macd_fast, self.macd_slow, self.macd_signal
            )
            
            current_macd = macd_line[-1]
            current_signal = signal_line[-1]
            prev_macd = macd_line[-2] if len(macd_line) > 1 else current_macd
            prev_signal = signal_line[-2] if len(signal_line) > 1 else current_signal
            current_histogram = histogram[-1]
            prev_histogram = histogram[-2] if len(histogram) > 1 else current_histogram
            
            if position_side == 'LONG':
                # Exit LONG: bearish crossover or weakening momentum
                bearish_cross = prev_macd >= prev_signal and current_macd < current_signal
                weakening = current_histogram < prev_histogram and current_histogram < 0
                
                if bearish_cross or weakening:
                    return True
            
            elif position_side == 'SHORT':
                # Exit SHORT: bullish crossover or weakening momentum
                bullish_cross = prev_macd <= prev_signal and current_macd > current_signal
                weakening = current_histogram > prev_histogram and current_histogram > 0
                
                if bullish_cross or weakening:
                    return True
            
            return False
        except Exception as e:
            print(f"DEBUG: Error in check_exit: {e}")
            raise e
