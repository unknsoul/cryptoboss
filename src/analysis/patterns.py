"""
Candlestick Pattern Detector
Simple but effective pattern recognition for entry confirmation

Patterns detected:
- Bullish/Bearish Engulfing
- Pin Bar (Hammer/Shooting Star)
- Doji (Indecision)
- Inside Bar (Consolidation breakout)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def detect_pattern(df: pd.DataFrame, action: str = None) -> Tuple[bool, str, int]:
    """
    Detect candlestick patterns on the latest candle.
    
    Args:
        df: DataFrame with OHLC data
        action: 'LONG' or 'SHORT' to check if pattern confirms direction
    
    Returns:
        (pattern_found, pattern_name, bonus_points)
        - pattern_found: True if a pattern matching the action was found
        - pattern_name: Description of the pattern
        - bonus_points: Extra points to add to quality score (0-10)
    """
    if len(df) < 3:
        return False, "Insufficient data", 0
    
    # Get last 3 candles
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    o, h, l, c = curr['open'], curr['high'], curr['low'], curr['close']
    po, ph, pl, pc = prev['open'], prev['high'], prev['low'], prev['close']
    
    body = abs(c - o)
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l
    candle_range = h - l
    
    prev_body = abs(pc - po)
    
    # Prevent division by zero
    if candle_range == 0:
        return False, "No range", 0
    
    body_pct = body / candle_range
    
    patterns_found = []
    
    # ========== BULLISH PATTERNS ==========
    
    # Bullish Engulfing: Current candle body engulfs previous bearish candle
    if c > o and pc < po:  # Green candle after red
        if o <= pc and c >= po:  # Body engulfs previous
            patterns_found.append(("Bullish Engulfing", "LONG", 8))
    
    # Hammer (Bullish): Small body at top, long lower wick
    if lower_wick > body * 2 and upper_wick < body * 0.5:
        if c > o:  # Green hammer is stronger
            patterns_found.append(("Hammer", "LONG", 6))
        else:
            patterns_found.append(("Inverted Hammer", "LONG", 4))
    
    # Morning Star (3-candle bullish reversal)
    if len(df) >= 3:
        p2o, p2c = prev2['open'], prev2['close']
        if p2c < p2o:  # First candle bearish
            if abs(pc - po) < abs(p2c - p2o) * 0.3:  # Second candle small
                if c > o and c > (p2o + p2c) / 2:  # Third candle bullish, closes above midpoint
                    patterns_found.append(("Morning Star", "LONG", 10))
    
    # ========== BEARISH PATTERNS ==========
    
    # Bearish Engulfing: Current candle body engulfs previous bullish candle
    if c < o and pc > po:  # Red candle after green
        if o >= pc and c <= po:  # Body engulfs previous
            patterns_found.append(("Bearish Engulfing", "SHORT", 8))
    
    # Shooting Star (Bearish): Small body at bottom, long upper wick
    if upper_wick > body * 2 and lower_wick < body * 0.5:
        if c < o:  # Red shooting star is stronger
            patterns_found.append(("Shooting Star", "SHORT", 6))
        else:
            patterns_found.append(("Hanging Man", "SHORT", 4))
    
    # Evening Star (3-candle bearish reversal)
    if len(df) >= 3:
        p2o, p2c = prev2['open'], prev2['close']
        if p2c > p2o:  # First candle bullish
            if abs(pc - po) < abs(p2c - p2o) * 0.3:  # Second candle small
                if c < o and c < (p2o + p2c) / 2:  # Third candle bearish, closes below midpoint
                    patterns_found.append(("Evening Star", "SHORT", 10))
    
    # ========== NEUTRAL PATTERNS ==========
    
    # Doji: Very small body relative to range
    if body_pct < 0.1:
        patterns_found.append(("Doji", "NEUTRAL", 2))
    
    # Inside Bar: Current candle fully inside previous
    if h < ph and l > pl:
        patterns_found.append(("Inside Bar", "NEUTRAL", 3))
    
    # ========== MATCH TO ACTION ==========
    
    if not patterns_found:
        return False, "No pattern", 0
    
    # If action specified, find matching pattern
    if action:
        for pattern_name, direction, points in patterns_found:
            if direction == action:
                return True, pattern_name, points
            elif direction == "NEUTRAL":
                return True, pattern_name, points // 2  # Half points for neutral
        
        # No matching pattern for the action
        return False, f"Pattern against {action}", 0
    
    # No action specified, return strongest pattern
    best_pattern = max(patterns_found, key=lambda x: x[2])
    return True, best_pattern[0], best_pattern[2]


def get_pattern_confirmation(df: pd.DataFrame, action: str) -> Tuple[bool, str]:
    """
    Check if candlestick patterns confirm the trade direction.
    
    Returns:
        (confirmed, reason)
    """
    pattern_found, pattern_name, points = detect_pattern(df, action)
    
    if pattern_found and points >= 4:
        return True, f"Pattern: {pattern_name} (+{points}pts)"
    elif pattern_found and points > 0:
        return True, f"Pattern: {pattern_name} (weak)"
    else:
        return False, "Pattern: No confirmation"


# Export
__all__ = ['detect_pattern', 'get_pattern_confirmation']


if __name__ == '__main__':
    # Test with sample data
    print("=== PATTERN DETECTOR TEST ===")
    
    # Create sample bullish engulfing
    df = pd.DataFrame({
        'open':  [100, 102, 98],
        'high':  [103, 103, 104],
        'low':   [99, 97, 97],
        'close': [102, 98, 103]
    })
    
    found, name, pts = detect_pattern(df, 'LONG')
    print(f"Bullish Engulfing Test: {found}, {name}, {pts} pts")
    
    # Create sample shooting star
    df = pd.DataFrame({
        'open':  [100, 102, 103],
        'high':  [103, 105, 110],
        'low':   [99, 101, 102],
        'close': [102, 104, 103]
    })
    
    found, name, pts = detect_pattern(df, 'SHORT')
    print(f"Shooting Star Test: {found}, {name}, {pts} pts")
