"""
Candle Pattern Detector - Enterprise Feature #45
Detects common candlestick patterns for trading signals.
"""

import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Candlestick pattern types"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PatternStrength(Enum):
    """Pattern signal strength"""
    STRONG = 3
    MODERATE = 2
    WEAK = 1


class CandlePatternDetector:
    """
    Feature #45: Candle Pattern Detector
    
    Detects classic candlestick patterns:
    - Single candle: Doji, Hammer, Shooting Star, Marubozu
    - Double candle: Engulfing, Harami, Piercing, Dark Cloud
    - Triple candle: Morning Star, Evening Star, Three Soldiers/Crows
    """
    
    def __init__(self, body_threshold: float = 0.1, wick_ratio: float = 2.0):
        """
        Initialize pattern detector.
        
        Args:
            body_threshold: Min body size as % of candle range for significance
            wick_ratio: Min wick/body ratio for hammer/shooting star
        """
        self.body_threshold = body_threshold
        self.wick_ratio = wick_ratio
        
        logger.info("Candle Pattern Detector initialized")
    
    def detect_patterns(self, candles: List[Dict]) -> List[Dict]:
        """
        Detect all patterns in candle data.
        
        Args:
            candles: List of candles with OHLC data
            
        Returns:
            List of detected patterns
        """
        if len(candles) < 3:
            return []
        
        patterns = []
        
        # Single candle patterns (check last 3 candles)
        for i in range(-3, 0):
            single_patterns = self._detect_single_patterns(candles[i])
            for p in single_patterns:
                p['candle_index'] = i
                patterns.append(p)
        
        # Double candle patterns
        double_patterns = self._detect_double_patterns(candles[-2:])
        patterns.extend(double_patterns)
        
        # Triple candle patterns
        triple_patterns = self._detect_triple_patterns(candles[-3:])
        patterns.extend(triple_patterns)
        
        return patterns
    
    def _get_candle_metrics(self, candle: Dict) -> Dict:
        """Extract useful metrics from a candle."""
        o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
        
        body = abs(c - o)
        range_ = h - l if h > l else 0.0001
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        is_bullish = c > o
        
        return {
            'open': o, 'high': h, 'low': l, 'close': c,
            'body': body,
            'range': range_,
            'body_pct': body / range_,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'is_bullish': is_bullish,
            'is_doji': body / range_ < self.body_threshold
        }
    
    def _detect_single_patterns(self, candle: Dict) -> List[Dict]:
        """Detect single candle patterns."""
        patterns = []
        m = self._get_candle_metrics(candle)
        
        # Doji - very small body
        if m['is_doji']:
            patterns.append({
                'name': 'Doji',
                'type': PatternType.NEUTRAL.value,
                'strength': PatternStrength.MODERATE.value,
                'description': 'Indecision - possible reversal'
            })
        
        # Hammer (bullish) - small body at top, long lower wick
        if (m['body_pct'] < 0.3 and 
            m['lower_wick'] > m['body'] * self.wick_ratio and
            m['upper_wick'] < m['body'] * 0.5):
            patterns.append({
                'name': 'Hammer',
                'type': PatternType.BULLISH.value,
                'strength': PatternStrength.MODERATE.value,
                'description': 'Bullish reversal signal'
            })
        
        # Shooting Star (bearish) - small body at bottom, long upper wick
        if (m['body_pct'] < 0.3 and 
            m['upper_wick'] > m['body'] * self.wick_ratio and
            m['lower_wick'] < m['body'] * 0.5):
            patterns.append({
                'name': 'Shooting Star',
                'type': PatternType.BEARISH.value,
                'strength': PatternStrength.MODERATE.value,
                'description': 'Bearish reversal signal'
            })
        
        # Marubozu - full body, no wicks
        if m['body_pct'] > 0.9:
            ptype = PatternType.BULLISH if m['is_bullish'] else PatternType.BEARISH
            patterns.append({
                'name': f"{'Bullish' if m['is_bullish'] else 'Bearish'} Marubozu",
                'type': ptype.value,
                'strength': PatternStrength.STRONG.value,
                'description': 'Strong momentum continuation'
            })
        
        return patterns
    
    def _detect_double_patterns(self, candles: List[Dict]) -> List[Dict]:
        """Detect two-candle patterns."""
        if len(candles) < 2:
            return []
        
        patterns = []
        m1 = self._get_candle_metrics(candles[0])
        m2 = self._get_candle_metrics(candles[1])
        
        # Bullish Engulfing
        if (not m1['is_bullish'] and m2['is_bullish'] and
            m2['body'] > m1['body'] * 1.5 and
            m2['close'] > m1['open'] and m2['open'] < m1['close']):
            patterns.append({
                'name': 'Bullish Engulfing',
                'type': PatternType.BULLISH.value,
                'strength': PatternStrength.STRONG.value,
                'description': 'Strong bullish reversal'
            })
        
        # Bearish Engulfing
        if (m1['is_bullish'] and not m2['is_bullish'] and
            m2['body'] > m1['body'] * 1.5 and
            m2['open'] > m1['close'] and m2['close'] < m1['open']):
            patterns.append({
                'name': 'Bearish Engulfing',
                'type': PatternType.BEARISH.value,
                'strength': PatternStrength.STRONG.value,
                'description': 'Strong bearish reversal'
            })
        
        # Bullish Harami
        if (not m1['is_bullish'] and m2['is_bullish'] and
            m2['body'] < m1['body'] * 0.5 and
            m2['high'] < m1['open'] and m2['low'] > m1['close']):
            patterns.append({
                'name': 'Bullish Harami',
                'type': PatternType.BULLISH.value,
                'strength': PatternStrength.WEAK.value,
                'description': 'Possible bullish reversal'
            })
        
        # Bearish Harami
        if (m1['is_bullish'] and not m2['is_bullish'] and
            m2['body'] < m1['body'] * 0.5 and
            m2['high'] < m1['close'] and m2['low'] > m1['open']):
            patterns.append({
                'name': 'Bearish Harami',
                'type': PatternType.BEARISH.value,
                'strength': PatternStrength.WEAK.value,
                'description': 'Possible bearish reversal'
            })
        
        return patterns
    
    def _detect_triple_patterns(self, candles: List[Dict]) -> List[Dict]:
        """Detect three-candle patterns."""
        if len(candles) < 3:
            return []
        
        patterns = []
        m1 = self._get_candle_metrics(candles[0])
        m2 = self._get_candle_metrics(candles[1])
        m3 = self._get_candle_metrics(candles[2])
        
        # Morning Star (bullish)
        if (not m1['is_bullish'] and m1['body_pct'] > 0.5 and
            m2['is_doji'] and
            m3['is_bullish'] and m3['body_pct'] > 0.5 and
            m3['close'] > (m1['open'] + m1['close']) / 2):
            patterns.append({
                'name': 'Morning Star',
                'type': PatternType.BULLISH.value,
                'strength': PatternStrength.STRONG.value,
                'description': 'Strong bullish reversal after downtrend'
            })
        
        # Evening Star (bearish)
        if (m1['is_bullish'] and m1['body_pct'] > 0.5 and
            m2['is_doji'] and
            not m3['is_bullish'] and m3['body_pct'] > 0.5 and
            m3['close'] < (m1['open'] + m1['close']) / 2):
            patterns.append({
                'name': 'Evening Star',
                'type': PatternType.BEARISH.value,
                'strength': PatternStrength.STRONG.value,
                'description': 'Strong bearish reversal after uptrend'
            })
        
        # Three White Soldiers (bullish)
        if (m1['is_bullish'] and m2['is_bullish'] and m3['is_bullish'] and
            m2['close'] > m1['close'] and m3['close'] > m2['close'] and
            m1['body_pct'] > 0.6 and m2['body_pct'] > 0.6 and m3['body_pct'] > 0.6):
            patterns.append({
                'name': 'Three White Soldiers',
                'type': PatternType.BULLISH.value,
                'strength': PatternStrength.STRONG.value,
                'description': 'Strong bullish continuation'
            })
        
        # Three Black Crows (bearish)
        if (not m1['is_bullish'] and not m2['is_bullish'] and not m3['is_bullish'] and
            m2['close'] < m1['close'] and m3['close'] < m2['close'] and
            m1['body_pct'] > 0.6 and m2['body_pct'] > 0.6 and m3['body_pct'] > 0.6):
            patterns.append({
                'name': 'Three Black Crows',
                'type': PatternType.BEARISH.value,
                'strength': PatternStrength.STRONG.value,
                'description': 'Strong bearish continuation'
            })
        
        return patterns
    
    def get_signal_boost(self, patterns: List[Dict], signal_direction: str) -> float:
        """
        Calculate confidence boost from detected patterns.
        
        Args:
            patterns: Detected patterns
            signal_direction: 'LONG' or 'SHORT'
            
        Returns:
            Confidence boost (-0.2 to +0.2)
        """
        if not patterns:
            return 0.0
        
        boost = 0.0
        expected_type = 'bullish' if signal_direction == 'LONG' else 'bearish'
        opposite_type = 'bearish' if signal_direction == 'LONG' else 'bullish'
        
        for p in patterns:
            strength = p['strength'] / 10  # Convert to 0.1-0.3  
            if p['type'] == expected_type:
                boost += strength
            elif p['type'] == opposite_type:
                boost -= strength
        
        return max(min(boost, 0.2), -0.2)


# Singleton
_pattern_detector: Optional[CandlePatternDetector] = None


def get_pattern_detector() -> CandlePatternDetector:
    global _pattern_detector
    if _pattern_detector is None:
        _pattern_detector = CandlePatternDetector()
    return _pattern_detector


if __name__ == '__main__':
    detector = CandlePatternDetector()
    
    # Test candles - bullish engulfing
    candles = [
        {'open': 100, 'high': 101, 'low': 98, 'close': 99},   # Bearish
        {'open': 98, 'high': 103, 'low': 97, 'close': 102},   # Bullish engulf
        {'open': 102, 'high': 105, 'low': 101, 'close': 104}  # Continuation
    ]
    
    patterns = detector.detect_patterns(candles)
    print(f"Detected {len(patterns)} patterns:")
    for p in patterns:
        print(f"  - {p['name']} ({p['type']}): {p['description']}")
