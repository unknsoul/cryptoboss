"""
Enhanced Momentum Strategy
Multi-indicator momentum strategy for strong trending markets
"""

import numpy as np
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy
from ..indicators import TechnicalIndicators


class EnhancedMomentumStrategy(BaseStrategy):
    """
    Enhanced Momentum Strategy - Best for strong trending markets
    
    Entry Rules:
    - LONG: RSI > 60 + MACD bullish + price > EMA50 + ADX > 25 + volume
    - SHORT: RSI < 40 + MACD bearish + price < EMA50 + ADX > 25 + volume
    
    Exit Rules:
    - Momentum weakening (ADX declining + MACD histogram shrinking)
    - RSI divergence
    - Trailing stop
    """
    
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9,
                 ema_period=50, adx_period=14, adx_threshold=25,
                 volume_period=20, volume_threshold=1.1):
        super().__init__("Enhanced Momentum", "momentum")
        
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_period = ema_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold
        
        self.parameters = {
            'rsi_period': rsi_period,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'ema_period': ema_period,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'volume_period': volume_period,
            'volume_threshold': volume_threshold
        }
    
    def signal(self, highs, lows, closes, volumes=None):
        """Generate enhanced momentum signal"""
        if not self.validate_data(highs, lows, closes):
            return None
        
        min_bars = max(self.macd_slow, self.ema_period, self.adx_period, self.volume_period) + 10
        if len(closes) < min_bars:
            return None
        
        # Calculate all indicators
        rsi = TechnicalIndicators.rsi(closes, self.rsi_period)
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )
        ema = TechnicalIndicators.ema(closes, self.ema_period)
        adx, plus_di, minus_di = TechnicalIndicators.adx(highs, lows, closes, self.adx_period)
        atr = TechnicalIndicators.atr(highs, lows, closes, 14)
        
        # Volume confirmation
        volume_confirmed = True
        if volumes is not None and len(volumes) == len(closes):
            vol_ma = TechnicalIndicators.volume_ma(volumes, self.volume_period)
            volume_confirmed = volumes[-1] > vol_ma[-1] * self.volume_threshold
        
        current_price = closes[-1]
        current_rsi = rsi[-1]
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = histogram[-1]
        current_ema = ema[-1]
        current_adx = adx[-1]
        current_plus_di = plus_di[-1]
        current_minus_di = minus_di[-1]
        current_atr = atr[-1]
        
        # Check for increasing ADX (strengthening trend)
        adx_rising = adx[-1] > adx[-3] if len(adx) > 3 else True
        
        # Momentum strength score
        momentum_score = 0
        
        # LONG signal - Strong bullish momentum
        if current_rsi > 60 and current_rsi < 85:  # Strong but not extreme
            momentum_score += 0.2
        
        if current_macd > current_signal and current_histogram > 0:
            momentum_score += 0.25
        
        if current_price > current_ema:
            momentum_score += 0.2
        
        if current_adx > self.adx_threshold and adx_rising:
            momentum_score += 0.2
        
        if current_plus_di > current_minus_di:
            momentum_score += 0.15
        
        if volume_confirmed:
            momentum_score += 0.0  # Bonus for volume
        
        # Strong bullish momentum
        if momentum_score >= 0.8 and current_histogram > 0:
            return {
                'action': 'LONG',
                'stop': current_atr * 2.5,
                'target': current_atr * 5.0,
                'confidence': min(momentum_score, 1.0),
                'metadata': {
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'adx': current_adx,
                    'plus_di': current_plus_di,
                    'minus_di': current_minus_di,
                    'momentum_score': momentum_score,
                    'volume_confirmed': volume_confirmed,
                    'entry_type': 'strong_bullish_momentum'
                }
            }
        
        # SHORT signal - Strong bearish momentum
        momentum_score = 0
        
        if current_rsi < 40 and current_rsi > 15:  # Strong but not extreme
            momentum_score += 0.2
        
        if current_macd < current_signal and current_histogram < 0:
            momentum_score += 0.25
        
        if current_price < current_ema:
            momentum_score += 0.2
        
        if current_adx > self.adx_threshold and adx_rising:
            momentum_score += 0.2
        
        if current_minus_di > current_plus_di:
            momentum_score += 0.15
        
        if volume_confirmed:
            momentum_score += 0.0
        
        # Strong bearish momentum
        if momentum_score >= 0.8 and current_histogram < 0:
            return {
                'action': 'SHORT',
                'stop': current_atr * 2.5,
                'target': current_atr * 5.0,
                'confidence': min(momentum_score, 1.0),
                'metadata': {
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'adx': current_adx,
                    'plus_di': current_plus_di,
                    'minus_di': current_minus_di,
                    'momentum_score': momentum_score,
                    'volume_confirmed': volume_confirmed,
                    'entry_type': 'strong_bearish_momentum'
                }
            }
        
        return None
    
    def check_exit(self, highs, lows, closes, position_side,
                   entry_price, entry_index, current_index):
        """Check if should exit momentum position"""
        if current_index - entry_index < 3:
            return False
        
        # Calculate indicators for exit
        adx, plus_di, minus_di = TechnicalIndicators.adx(
            highs[:current_index + 1],
            lows[:current_index + 1],
            closes[:current_index + 1],
            self.adx_period
        )
        
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            closes[:current_index + 1],
            self.macd_fast, self.macd_slow, self.macd_signal
        )
        
        current_adx = adx[-1]
        prev_adx = adx[-2] if len(adx) > 1 else current_adx
        current_histogram = histogram[-1]
        prev_histogram = histogram[-2] if len(histogram) > 1 else current_histogram
        
        # ADX declining = weakening trend
        adx_weakening = current_adx < prev_adx and current_adx < self.adx_threshold
        
        if position_side == 'LONG':
            # Histogram shrinking
            momentum_weakening = current_histogram < prev_histogram and current_histogram < 0
            # DI crossover
            di_bearish = minus_di[-1] > plus_di[-1]
            
            if adx_weakening or momentum_weakening or di_bearish:
                return True
        
        elif position_side == 'SHORT':
            momentum_weakening = current_histogram > prev_histogram and current_histogram > 0
            di_bullish = plus_di[-1] > minus_di[-1]
            
            if adx_weakening or momentum_weakening or di_bullish:
                return True
        
        return False
