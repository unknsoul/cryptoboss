"""
Support & Resistance Strategy
Dynamic S/R level detection with breakout confirmation
"""

import numpy as np
from .base_strategy import BaseStrategy
from ..strategy import Indicators


class SupportResistanceStrategy(BaseStrategy):
    """
    Support/Resistance breakout strategy
    
    Features:
    - Dynamic S/R level detection
    - Volume-confirmed breakouts
    - False breakout filtering
    """
    
    def __init__(self,
                 lookback=100,
                 cluster_threshold=0.01,
                 volume_multiplier=1.5,
                 atr_period=14,
                 atr_multiplier=2.0):
        super().__init__("Support/Resistance", "breakout")
        
        self.lookback = lookback
        self.cluster_threshold = cluster_threshold
        self.volume_multiplier = volume_multiplier
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    def signal(self, highs, lows, closes, volumes=None):
        """Generate S/R breakout signal"""
        if len(closes) < self.lookback:
            return None
        
        current_price = closes[-1]
        
        # Find S/R levels
        resistance_levels = self._find_resistance(highs[-self.lookback:])
        support_levels = self._find_support(lows[-self.lookback:])
        
        # Calculate ATR for stops
        atr = Indicators.atr(highs, lows, closes, self.atr_period)
        if atr is None:
            return None
        
        stop_distance = atr * self.atr_multiplier
        
        # Check volume confirmation
        volume_ok = True
        if volumes is not None and len(volumes) >= 20:
            avg_volume = np.mean(volumes[-20:])
            volume_ok = volumes[-1] > avg_volume * self.volume_multiplier
        
        # Resistance breakout (LONG)
        if resistance_levels and volume_ok:
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
            if current_price > nearest_resistance * 1.002:  # 0.2% above resistance
                return {
                    'action': 'LONG',
                    'stop': stop_distance,
                    'metadata': {
                        'resistance_broken': nearest_resistance,
                        'volume_confirmed': volume_ok
                    }
                }
        
        # Support breakout (SHORT)
        if support_levels and volume_ok:
            nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
            if current_price < nearest_support * 0.998:  # 0.2% below support
                return {
                    'action': 'SHORT',
                    'stop': stop_distance,
                    'metadata': {
                        'support_broken': nearest_support,
                        'volume_confirmed': volume_ok
                    }
                }
        
        return None
    
    def _find_resistance(self, highs):
        """Find resistance levels"""
        peaks = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                peaks.append(highs[i])
        
        return self._cluster_levels(peaks)
    
    def _find_support(self, lows):
        """Find support levels"""
        troughs = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                troughs.append(lows[i])
        
        return self._cluster_levels(troughs)
    
    def _cluster_levels(self, levels):
        """Cluster nearby levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < self.cluster_threshold:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clustered.append(np.mean(current_cluster))
        
        return clustered
    
    def check_exit(self, highs, lows, closes, position_side, entry_price, entry_index, current_index):
        return False


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Bollinger Band squeeze + ATR expansion
    
    Features:
    - Squeeze detection
    - Direction confirmation
    - Volatility expansion trading
    """
    
    def __init__(self,
                 bb_period=20,
                 bb_std=2,
                 atr_period=14,
                 atr_multiplier=2.0,
                 squeeze_percentile=20):
        super().__init__("Volatility Breakout", "volatility")
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.squeeze_percentile = squeeze_percentile
    
    def signal(self, highs, lows, closes, volumes=None):
        """Generate volatility breakout signal"""
        if len(closes) < max(self.bb_period, self.atr_period) + 50:
            return None
        
        current_price = closes[-1]
        
        # Calculate indicators
        from ..indicators import TechnicalIndicators
        bb_upper, bb_middle, bb_lower, bandwidth, percent_b = TechnicalIndicators.bollinger_bands(
            closes, self.bb_period, self.bb_std
        )
        
        atr = Indicators.atr(highs, lows, closes, self.atr_period)
        if atr is None:
            return None
        
        # Check for squeeze (low bandwidth)
        bandwidth_values = bandwidth[-100:]  # Last 100 values
        current_bandwidth = bandwidth[-1]
        bandwidth_percentile = (np.sum(bandwidth_values < current_bandwidth) / len(bandwidth_values)) * 100
        
        # Not in squeeze anymore, skip
        if bandwidth_percentile > self.squeeze_percentile:
            return None
        
        # Squeeze detected - wait for expansion
        if current_price > bb_upper[-1]:
            # Breakout above upper band
            return {
                'action': 'LONG',
                'stop': atr * self.atr_multiplier,
                'metadata': {
                    'bandwidth_percentile': bandwidth_percentile,
                    'bb_upper': bb_upper[-1],
                    'squeeze_breakout': True
                }
            }
        
        if current_price < bb_lower[-1]:
            # Breakout below lower band
            return {
                'action': 'SHORT',
                'stop': atr * self.atr_multiplier,
                'metadata': {
                    'bandwidth_percentile': bandwidth_percentile,
                    'bb_lower': bb_lower[-1],
                    'squeeze_breakout': True
                }
            }
        
        return None
    
    def check_exit(self, highs, lows, closes, position_side, entry_price, entry_index, current_index):
        return False
