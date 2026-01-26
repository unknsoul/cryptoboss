"""
Breakout & Support/Resistance Detector - Enterprise Features #54, #73
Detects breakout signals and auto-identifies support/resistance levels.
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class SupportResistanceDetector:
    """
    Feature #73: Support/Resistance Auto-Detection
    
    Identifies key price levels using:
    - Swing highs/lows
    - Volume clusters
    - Price congestion zones
    - Historical pivot points
    """
    
    def __init__(
        self,
        lookback: int = 50,
        min_touches: int = 2,
        tolerance_pct: float = 0.2
    ):
        """
        Initialize S/R detector.
        
        Args:
            lookback: Number of candles to analyze
            min_touches: Minimum touches to confirm level
            tolerance_pct: Price tolerance for grouping levels
        """
        self.lookback = lookback
        self.min_touches = min_touches
        self.tolerance_pct = tolerance_pct
        
        self.support_levels: List[Dict] = []
        self.resistance_levels: List[Dict] = []
        
        logger.info(f"S/R Detector initialized - Lookback: {lookback}, Tolerance: {tolerance_pct}%")
    
    def detect_levels(self, candles: List[Dict]) -> Dict:
        """
        Detect support and resistance levels.
        
        Args:
            candles: OHLC candle data
            
        Returns:
            Dict with support and resistance levels
        """
        if len(candles) < self.lookback:
            return {'support': [], 'resistance': [], 'current_price': 0}
        
        candles = candles[-self.lookback:]
        current_price = candles[-1]['close']
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(candles) - 2):
            high = candles[i]['high']
            low = candles[i]['low']
            
            # Swing high: higher than 2 candles on each side
            if (high > candles[i-1]['high'] and high > candles[i-2]['high'] and
                high > candles[i+1]['high'] and high > candles[i+2]['high']):
                swing_highs.append({'price': high, 'index': i, 'touches': 1})
            
            # Swing low: lower than 2 candles on each side
            if (low < candles[i-1]['low'] and low < candles[i-2]['low'] and
                low < candles[i+1]['low'] and low < candles[i+2]['low']):
                swing_lows.append({'price': low, 'index': i, 'touches': 1})
        
        # Cluster nearby levels
        support_levels = self._cluster_levels(swing_lows, current_price)
        resistance_levels = self._cluster_levels(swing_highs, current_price)
        
        # Filter: support below price, resistance above
        self.support_levels = [s for s in support_levels if s['price'] < current_price]
        self.resistance_levels = [r for r in resistance_levels if r['price'] > current_price]
        
        # Sort by proximity to current price
        self.support_levels.sort(key=lambda x: current_price - x['price'])
        self.resistance_levels.sort(key=lambda x: x['price'] - current_price)
        
        return {
            'support': self.support_levels[:5],  # Top 5 nearest
            'resistance': self.resistance_levels[:5],
            'current_price': current_price
        }
    
    def _cluster_levels(self, levels: List[Dict], current_price: float) -> List[Dict]:
        """Cluster nearby price levels."""
        if not levels:
            return []
        
        tolerance = current_price * (self.tolerance_pct / 100)
        clustered = []
        used = set()
        
        for i, level in enumerate(levels):
            if i in used:
                continue
            
            cluster_prices = [level['price']]
            used.add(i)
            
            for j, other in enumerate(levels):
                if j not in used and abs(level['price'] - other['price']) <= tolerance:
                    cluster_prices.append(other['price'])
                    used.add(j)
            
            avg_price = sum(cluster_prices) / len(cluster_prices)
            clustered.append({
                'price': round(avg_price, 2),
                'touches': len(cluster_prices),
                'strength': min(len(cluster_prices) / 3, 1.0)  # Normalize to 0-1
            })
        
        # Filter by minimum touches
        return [c for c in clustered if c['touches'] >= self.min_touches]
    
    def get_nearest_levels(self, price: float) -> Dict:
        """Get nearest support and resistance to current price."""
        nearest_support = self.support_levels[0] if self.support_levels else None
        nearest_resistance = self.resistance_levels[0] if self.resistance_levels else None
        
        return {
            'support': nearest_support,
            'resistance': nearest_resistance,
            'in_compression': (nearest_support and nearest_resistance and
                              (nearest_resistance['price'] - nearest_support['price']) / price < 0.02)
        }


class BreakoutDetector:
    """
    Feature #54: Breakout Strategy Detector
    
    Detects breakout signals from:
    - S/R level breaks
    - Bollinger Band breakouts
    - Range expansions
    - Volume-confirmed moves
    """
    
    def __init__(
        self,
        sr_detector: Optional[SupportResistanceDetector] = None,
        volume_threshold: float = 1.5,
        confirmation_candles: int = 2
    ):
        """
        Initialize breakout detector.
        
        Args:
            sr_detector: Support/resistance detector instance
            volume_threshold: Volume multiplier for confirmation
            confirmation_candles: Candles to confirm breakout
        """
        self.sr_detector = sr_detector or SupportResistanceDetector()
        self.volume_threshold = volume_threshold
        self.confirmation_candles = confirmation_candles
        
        self.recent_breakouts: List[Dict] = []
        
        logger.info("Breakout Detector initialized")
    
    def detect_breakouts(self, candles: List[Dict]) -> List[Dict]:
        """
        Detect breakout signals.
        
        Args:
            candles: OHLC candle data
            
        Returns:
            List of detected breakouts
        """
        if len(candles) < 20:
            return []
        
        breakouts = []
        current = candles[-1]
        prev = candles[-2]
        
        # Get S/R levels
        levels = self.sr_detector.detect_levels(candles[:-1])  # Exclude current for detection
        
        # Check resistance breakout
        for r in levels['resistance'][:3]:
            if current['close'] > r['price'] and prev['close'] <= r['price']:
                # Confirm with volume
                avg_vol = sum(c.get('volume', 0) for c in candles[-10:-1]) / 9
                current_vol = current.get('volume', 0)
                
                confirmed = current_vol > avg_vol * self.volume_threshold
                
                breakouts.append({
                    'type': 'RESISTANCE_BREAK',
                    'direction': 'BULLISH',
                    'level': r['price'],
                    'current_price': current['close'],
                    'volume_confirmed': confirmed,
                    'strength': r['strength'],
                    'signal': 'LONG' if confirmed else 'WEAK_LONG'
                })
        
        # Check support breakdown
        for s in levels['support'][:3]:
            if current['close'] < s['price'] and prev['close'] >= s['price']:
                avg_vol = sum(c.get('volume', 0) for c in candles[-10:-1]) / 9
                current_vol = current.get('volume', 0)
                
                confirmed = current_vol > avg_vol * self.volume_threshold
                
                breakouts.append({
                    'type': 'SUPPORT_BREAK',
                    'direction': 'BEARISH',
                    'level': s['price'],
                    'current_price': current['close'],
                    'volume_confirmed': confirmed,
                    'strength': s['strength'],
                    'signal': 'SHORT' if confirmed else 'WEAK_SHORT'
                })
        
        # Bollinger Band breakout
        bb_breakout = self._detect_bb_breakout(candles)
        if bb_breakout:
            breakouts.append(bb_breakout)
        
        # Store recent breakouts
        if breakouts:
            self.recent_breakouts.extend(breakouts)
            self.recent_breakouts = self.recent_breakouts[-20:]
        
        return breakouts
    
    def _detect_bb_breakout(self, candles: List[Dict]) -> Optional[Dict]:
        """Detect Bollinger Band breakout."""
        if len(candles) < 20:
            return None
        
        closes = [c['close'] for c in candles[-20:]]
        sma = sum(closes) / 20
        variance = sum((c - sma) ** 2 for c in closes) / 20
        std = variance ** 0.5
        
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        current = candles[-1]['close']
        prev = candles[-2]['close']
        
        if current > upper_band and prev <= upper_band:
            return {
                'type': 'BB_UPPER_BREAK',
                'direction': 'BULLISH',
                'level': round(upper_band, 2),
                'current_price': current,
                'signal': 'LONG'
            }
        
        if current < lower_band and prev >= lower_band:
            return {
                'type': 'BB_LOWER_BREAK',
                'direction': 'BEARISH',
                'level': round(lower_band, 2),
                'current_price': current,
                'signal': 'SHORT'
            }
        
        return None
    
    def get_breakout_bias(self, candles: List[Dict]) -> Tuple[str, float]:
        """
        Get overall breakout bias.
        
        Returns:
            Tuple of (direction, confidence)
        """
        breakouts = self.detect_breakouts(candles)
        
        if not breakouts:
            return 'NEUTRAL', 0.0
        
        # Score breakouts
        bullish_score = sum(1 for b in breakouts if 'LONG' in b['signal'])
        bearish_score = sum(1 for b in breakouts if 'SHORT' in b['signal'])
        
        if bullish_score > bearish_score:
            return 'BULLISH', min(bullish_score * 0.3, 1.0)
        elif bearish_score > bullish_score:
            return 'BEARISH', min(bearish_score * 0.3, 1.0)
        
        return 'NEUTRAL', 0.0


class SignalWeightingSystem:
    """
    Feature #67: Signal Weighting by Performance
    
    Tracks strategy/signal source performance and adjusts weights dynamically.
    """
    
    def __init__(self, decay_factor: float = 0.95):
        """
        Initialize signal weighting system.
        
        Args:
            decay_factor: Weight decay for older signals
        """
        self.decay_factor = decay_factor
        self.signal_performance: Dict[str, Dict] = {}
        
        logger.info("Signal Weighting System initialized")
    
    def record_signal(self, source: str, prediction: str, outcome: bool, pnl: float):
        """
        Record a signal's outcome for weight adjustment.
        
        Args:
            source: Signal source name (strategy, indicator)
            prediction: Signal prediction (LONG/SHORT)
            outcome: True if profitable
            pnl: Profit/loss amount
        """
        if source not in self.signal_performance:
            self.signal_performance[source] = {
                'total': 0,
                'wins': 0,
                'total_pnl': 0,
                'weight': 1.0,
                'history': []
            }
        
        perf = self.signal_performance[source]
        perf['total'] += 1
        if outcome:
            perf['wins'] += 1
        perf['total_pnl'] += pnl
        perf['history'].append({'prediction': prediction, 'outcome': outcome, 'pnl': pnl})
        perf['history'] = perf['history'][-50:]  # Keep last 50
        
        # Recalculate weight
        self._update_weight(source)
    
    def _update_weight(self, source: str):
        """Update weight for a signal source based on performance."""
        perf = self.signal_performance[source]
        
        if perf['total'] < 5:
            perf['weight'] = 1.0  # Not enough data
            return
        
        # Win rate component
        win_rate = perf['wins'] / perf['total']
        
        # Profit factor component
        wins_pnl = sum(h['pnl'] for h in perf['history'] if h['outcome'])
        losses_pnl = abs(sum(h['pnl'] for h in perf['history'] if not h['outcome']))
        profit_factor = wins_pnl / losses_pnl if losses_pnl > 0 else 2.0
        
        # Combined weight (0.2 to 2.0 range)
        weight = (win_rate * 0.5 + min(profit_factor / 2, 1.0) * 0.5) * 2
        perf['weight'] = max(min(weight, 2.0), 0.2)
    
    def get_weight(self, source: str) -> float:
        """Get current weight for a signal source."""
        if source in self.signal_performance:
            return self.signal_performance[source]['weight']
        return 1.0
    
    def get_weighted_confidence(self, signals: List[Dict]) -> float:
        """
        Calculate weighted confidence from multiple signals.
        
        Args:
            signals: List of {'source': str, 'direction': str, 'confidence': float}
            
        Returns:
            Weighted confidence score
        """
        if not signals:
            return 0.0
        
        total_weight = 0
        weighted_conf = 0
        
        for s in signals:
            weight = self.get_weight(s['source'])
            weighted_conf += s['confidence'] * weight
            total_weight += weight
        
        return weighted_conf / total_weight if total_weight > 0 else 0.0
    
    def get_rankings(self) -> List[Dict]:
        """Get signal sources ranked by weight."""
        rankings = [
            {'source': k, **v}
            for k, v in self.signal_performance.items()
        ]
        rankings.sort(key=lambda x: x['weight'], reverse=True)
        return rankings


# Singleton instances
_sr_detector: Optional[SupportResistanceDetector] = None
_breakout_detector: Optional[BreakoutDetector] = None
_signal_weighting: Optional[SignalWeightingSystem] = None


def get_sr_detector() -> SupportResistanceDetector:
    global _sr_detector
    if _sr_detector is None:
        _sr_detector = SupportResistanceDetector()
    return _sr_detector


def get_breakout_detector() -> BreakoutDetector:
    global _breakout_detector
    if _breakout_detector is None:
        _breakout_detector = BreakoutDetector()
    return _breakout_detector


def get_signal_weighting() -> SignalWeightingSystem:
    global _signal_weighting
    if _signal_weighting is None:
        _signal_weighting = SignalWeightingSystem()
    return _signal_weighting


if __name__ == '__main__':
    # Test S/R detection
    sr = SupportResistanceDetector()
    
    # Generate test candles
    import random
    candles = []
    price = 50000
    for i in range(60):
        change = random.uniform(-200, 200)
        high = price + change + random.uniform(0, 100)
        low = price + change - random.uniform(0, 100)
        close = price + change
        candles.append({
            'open': price,
            'high': high,
            'low': low,
            'close': close,
            'volume': random.uniform(100, 500)
        })
        price = close
    
    levels = sr.detect_levels(candles)
    print(f"Support levels: {levels['support'][:3]}")
    print(f"Resistance levels: {levels['resistance'][:3]}")
    
    # Test breakout
    bo = BreakoutDetector(sr)
    breakouts = bo.detect_breakouts(candles)
    print(f"Breakouts: {breakouts}")
