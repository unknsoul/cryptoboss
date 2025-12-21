"""
Statistical Arbitrage Strategy
Professional market-neutral strategy used by quant funds
Works best in ranging markets with mean reversion
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from.base_strategy import BaseStrategy


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical Arbitrage (Pairs Trading / Mean Reversion)
    
    Concept:
    - Identifies when price deviates from statistical mean
    - Trades the reversion back to the mean
    - Uses Z-score to measure deviation
    - Multiple timeframe confirmation
    
    Best for: Ranging markets, low-to-medium volatility
    Used by: Renaissance Technologies, Two Sigma, DE Shaw
    """
    
    def __init__(self, lookback_period: int = 100,
                entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 use_bollinger: bool = True):
        super().__init__("Statistical Arbitrage", "mean_reversion")
        
        self.lookback = lookback_period
        self.entry_z = entry_threshold  # Enter when Z-score > 2.0
        self.exit_z = exit_threshold    # Exit when Z-score < 0.5
        self.use_bollinger = use_bollinger
        
        self.parameters = {
            'lookback_period': lookback_period,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold
        }
    
    def signal(self, highs, lows, closes, volumes=None):
        """Generate mean reversion signal"""
        if not self.validate_data(highs, lows, closes):
            return None
        
        if len(closes) < self.lookback + 20:
            return None
        
        # Calculate statistical measures
        mean = np.mean(closes[-self.lookback:])
        std = np.std(closes[-self.lookback:])
        
        if std == 0:
            return None
        
        # Z-score (how many standard deviations from mean)
        z_score = (closes[-1] - mean) / std
        
        # Bollinger Bands (2 std dev)
        upper_band = mean + (2 * std)
        lower_band = mean - (2 * std)
        
        # Mean Reversion Signal
        action = 'HOLD'
        confidence = 0.0
        
        # Oversold (price too low, expect reversion up)
        if z_score < -self.entry_z:
            action = 'LONG'
            confidence = min(abs(z_score) / 3.0, 0.95)  # Higher Z = higher confidence
        
        # Overbought (price too high, expect reversion down)
        elif z_score > self.entry_z:
            action = 'SHORT'
            confidence = min(abs(z_score) / 3.0, 0.95)
        
        # Additional filters
        if action != 'HOLD':
            # Volume confirmation (require above-average volume)
            if volumes is not None:
                avg_volume = np.mean(volumes[-20:])
                if volumes[-1] < avg_volume * 0.8:
                    confidence *= 0.7  # Reduce confidence
            
            # Volatility filter (avoid trading in extreme volatility)
            recent_vol = np.std(closes[-20:] / closes[-21:-1] - 1)
            hist_vol = np.std(closes[-50:-20] / closes[-51:-21] - 1)
            
            if recent_vol > hist_vol * 2.0:
                confidence *= 0.6  # Too volatile
        
        # Stop loss and target
        stop = std * 1.5  # 1.5 std dev stop
        target = std * 2.5  # 2.5 std dev target (better R:R)
        
        return {
            'action': action,
            'stop': stop,
            'target': target,
            'confidence': confidence,
            'metadata': {
                'z_score': z_score,
                'mean': mean,
                'std': std,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'entry_type': 'statistical_arbitrage'
            }
        }
    
    def check_exit(self, highs, lows, closes, position_side, 
                   entry_price, entry_index, current_index):
        """Exit when price reverts to mean"""
        if current_index >= len(closes):
            return None
        
        # Recalculate mean and z-score
        mean = np.mean(closes[max(0, current_index-self.lookback):current_index+1])
        std = np.std(closes[max(0, current_index-self.lookback):current_index+1])
        
        if std == 0:
            return None
        
        current_price = closes[current_index]
        z_score = (current_price - mean) / std
        
        # Exit when reverted to mean (z-score near 0)
        if abs(z_score) < self.exit_z:
            return {
                'action': 'CLOSE',
                'price': current_price,
                'reason': 'mean_reversion_complete',
                'z_score': z_score
            }
        
        # Standard stop/target exits
        return None


class VolumeProfileStrategy(BaseStrategy):
    """
    Volume Profile Trading
    
    Concept:
    - Trades based on volume-at-price levels
    - Identifies high-volume nodes (support/resistance)
    - Trades bounces from these levels
    
    Best for: All market conditions, especially ranging
    Used by: Professional floor traders, institutions
    """
    
    def __init__(self, vp_period: int = 200, num_bins: int = 20):
        super().__init__("Volume Profile", "support_resistance")
        
        self.vp_period = vp_period
        self.num_bins = num_bins
        
        self.parameters = {
            'vp_period': vp_period,
            'num_bins': num_bins
        }
    
    def _calculate_volume_profile(self, closes, volumes):
        """Calculate volume profile (POC, VAH, VAL)"""
        if len(closes) < self.vp_period:
            return None
        
        recent_closes = closes[-self.vp_period:]
        recent_volumes = volumes[-self.vp_period:]
        
        # Create price bins
        price_min = np.min(recent_closes)
        price_max = np.max(recent_closes)
        bins = np.linspace(price_min, price_max, self.num_bins + 1)
        
        # Aggregate volume at each price level
        volume_at_price = np.zeros(self.num_bins)
        
        for i in range(len(recent_closes)):
            bin_idx = np.digitize(recent_closes[i], bins) - 1
            bin_idx = min(max(bin_idx, 0), self.num_bins - 1)
            volume_at_price[bin_idx] += recent_volumes[i]
        
        # Point of Control (POC) - price level with highest volume
        poc_idx = np.argmax(volume_at_price)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Value Area (70% of volume)
        total_volume = np.sum(volume_at_price)
        value_area_volume = total_volume * 0.70
        
        # Find value area high and low
        cumsum = 0
        val_idx = poc_idx
        
        # Expand from POC until we reach 70% volume
        for offset in range(self.num_bins):
            if cumsum >= value_area_volume:
                break
            
            if val_idx + offset < self.num_bins:
                cumsum += volume_at_price[val_idx + offset]
            if val_idx - offset >= 0:
                cumsum += volume_at_price[val_idx - offset]
        
        vah_price = bins[min(poc_idx + offset, self.num_bins)]  # Value Area High
        val_price = bins[max(poc_idx - offset, 0)]  # Value Area Low
        
        return {
            'poc': poc_price,
            'vah': vah_price,
            'val': val_price,
            'volume_distribution': volume_at_price,
            'bins': bins
        }
    
    def signal(self, highs, lows, closes, volumes=None):
        """Generate volume profile signal"""
        if not self.validate_data(highs, lows, closes):
            return None
        
        if volumes is None or len(volumes) < self.vp_period:
            return None
        
        vp = self._calculate_volume_profile(closes, volumes)
        if vp is None:
            return None
        
        current_price = closes[-1]
        poc = vp['poc']
        vah = vp['vah']
        val = vp['val']
        
        action = 'HOLD'
        confidence = 0.0
        
        # Long at Value Area Low (support)
        if current_price <= val * 1.002:  # Within 0.2% of VAL
            action = 'LONG'
            confidence = 0.75
        
        # Short at Value Area High (resistance)
        elif current_price >= vah * 0.998:  # Within 0.2% of VAH
            action = 'SHORT'
            confidence = 0.75
        
        # Mean reversion to POC
        elif current_price < poc * 0.98:
            action = 'LONG'
            confidence = 0.65
        elif current_price > poc * 1.02:
            action = 'SHORT'
            confidence = 0.65
        
        # Calculate stops and targets based on VP levels
        if action == 'LONG':
            stop = val * 0.996  # Stop below VAL
            target = poc  # Target POC
        elif action == 'SHORT':
            stop = vah * 1.004  # Stop above VAH
            target = poc  # Target POC
        else:
            stop = 0
            target = 0
        
        return {
            'action': action,
            'stop': abs(current_price - stop) if stop else 0,
            'target': abs(target - current_price) if target else 0,
            'confidence': confidence,
            'metadata': {
                'poc': poc,
                'vah': vah,
                'val': val,
                'entry_type': 'volume_profile'
            }
        }
    
    def check_exit(self, highs, lows, closes, position_side,
                   entry_price, entry_index, current_index):
        """Exit at POC or if price breaks key levels"""
        return None  # Use default exits


class BreakoutMomentumStrategy(BaseStrategy):
    """
    Breakout Momentum (Institutional)
    
    Concept:
    - Trades breakouts from consolidation ranges
    - Confirms with volume and momentum
    - Uses multiple timeframe confirmation
    
    Best for: Breakout regimes, trending markets
    Used by: CTA funds, momentum traders
    """
    
    def __init__(self, donchian_period: int = 20, 
                 atr_multiplier: float = 2.0,
                 min_consolidation_bars: int = 10):
        super().__init__("Breakout Momentum", "breakout")
        
        self.donchian_period = donchian_period
        self.atr_mult = atr_multiplier
        self.min_consolidation = min_consolidation_bars
        
        self.parameters = {
            'donchian_period': donchian_period,
            'atr_multiplier': atr_multiplier
        }
    
    def _detect_consolidation(self, highs, lows, closes):
        """Detect if market is consolidating (prerequisite for breakout)"""
        if len(closes) < self.min_consolidation + 5:
            return False
        
        # Check if recent price range is narrow (consolidation)
        recent_range = np.max(highs[-self.min_consolidation:]) - np.min(lows[-self.min_consolidation:])
        historical_range = np.max(highs[-50:-self.min_consolidation]) - np.min(lows[-50:-self.min_consolidation])
        
        # Consolidation if recent range is < 60% of historical
        is_consolidating = recent_range < historical_range * 0.6
        
        return is_consolidating
    
    def signal(self, highs, lows, closes, volumes=None):
        """Generate breakout signal"""
        if not self.validate_data(highs, lows, closes):
            return None
        
        if len(closes) < self.donchian_period + 50:
            return None
        
        # Donchian channels (highest high, lowest low)
        highest = np.max(highs[-self.donchian_period:-1])
        lowest = np.min(lows[-self.donchian_period:-1])
        
        current_price = closes[-1]
        
        # ATR for stops
        atr = self._calculate_atr(highs, lows, closes)
        
        action = 'HOLD'
        confidence = 0.0
        
        # Upside breakout
        if current_price > highest:
            # Check if we were consolidating before
            was_consolidating = self._detect_consolidation(highs, lows, closes)
            
            action = 'LONG'
            confidence = 0.80 if was_consolidating else 0.65
            
            # Volume confirmation
            if volumes is not None:
                avg_volume = np.mean(volumes[-20:])
                if volumes[-1] > avg_volume * 1.3:
                    confidence = min(confidence * 1.2, 0.95)
        
        # Downside breakdown
        elif current_price < lowest:
            was_consolidating = self._detect_consolidation(highs, lows, closes)
            
            action = 'SHORT'
            confidence = 0.80 if was_consolidating else 0.65
            
            if volumes is not None:
                avg_volume = np.mean(volumes[-20:])
                if volumes[-1] > avg_volume * 1.3:
                    confidence = min(confidence * 1.2, 0.95)
        
        stop = atr * self.atr_mult
        target = atr * 4.0  # 2:1 reward:risk
        
        return {
            'action': action,
            'stop': stop,
            'target': target,
            'confidence': confidence,
            'metadata': {
                'highest': highest,
                'lowest': lowest,
                'atr': atr,
                'entry_type': 'breakout_momentum'
            }
        }
    
    def _calculate_atr(self, highs, lows, closes, period: int = 14):
        """Calculate Average True Range"""
        if len(closes) < period + 1:
            return np.mean(highs[-period:] - lows[-period:])
        
        tr_list = []
        for i in range(-period, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        
        return np.mean(tr_list)
    
    def check_exit(self, highs, lows, closes, position_side,
                   entry_price, entry_index, current_index):
        """Exit on momentum loss"""
        if current_index < entry_index + 5:
            return None
        
        # Exit if price comes back inside the channel
        donchian_high = np.max(highs[max(0, current_index-self.donchian_period):current_index])
        donchian_low = np.min(lows[max(0, current_index-self.donchian_period):current_index])
        
        current_price = closes[current_index]
        
        if position_side == 'LONG' and current_price < donchian_high * 0.98:
            return {'action': 'CLOSE', 'reason': 'momentum_lost'}
        elif position_side == 'SHORT' and current_price > donchian_low * 1.02:
            return {'action': 'CLOSE', 'reason': 'momentum_lost'}
        
        return None


if __name__ == "__main__":
    # Test strategies
    print("Testing advanced institutional strategies...")
    
    # Generate test data
    np.random.seed(42)
    closes = 45000 + np.random.randn(300).cumsum() * 50
    highs = closes * 1.002
    lows = closes * 0.998
    volumes = np.random.uniform(800, 1200, 300)
    
    # Test Statistical Arbitrage
    stat_arb = StatisticalArbitrageStrategy()
    signal = stat_arb.signal(highs, lows, closes, volumes)
    print("\n" + "=" * 70)
    print("STATISTICAL ARBITRAGE")
    print("=" * 70)
    print(f"Signal: {signal}")
    
    # Test Volume Profile
    vp_strategy = VolumeProfileStrategy()
    signal = vp_strategy.signal(highs, lows, closes, volumes)
    print("\n" + "=" * 70)
    print("VOLUME PROFILE")
    print("=" * 70)
    print(f"Signal: {signal}")
    
    # Test Breakout Momentum
    breakout = BreakoutMomentumStrategy()
    signal = breakout.signal(highs, lows, closes, volumes)
    print("\n" + "=" * 70)
    print("BREAKOUT MOMENTUM")
    print("=" * 70)
    print(f"Signal: {signal}")
    
    print("\n✅ Advanced strategies test complete")
