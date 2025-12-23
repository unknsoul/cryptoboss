"""
Order Flow Analysis for Institutional Trading
Tracks order book imbalance, large orders, and momentum shifts

Key Signals:
- Strong bid/ask imbalance
- Large order absorption
- Momentum divergence
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OrderFlowAnalyzer:
    """
    Analyzes order flow to detect institutional activity and momentum.
    """
    
    def __init__(self, imbalance_threshold: float = 0.3):
        self.imbalance_threshold = imbalance_threshold
        self.flow_history: List[Dict] = []
        self.max_history = 100
    
    def analyze(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        recent_price: float = 0
    ) -> Dict:
        """
        Analyze current order flow.
        
        Args:
            bids: [[price, volume], ...]
            asks: [[price, volume], ...]
            recent_price: Current market price
            
        Returns:
            Analysis dict with imbalance, pressure, and signals
        """
        if not bids or not asks:
            return {
                'imbalance': 0,
                'pressure': 'neutral',
                'strength': 0,
                'signal': None,
                'large_orders': []
            }
        
        # Calculate total volume at each side
        bid_volume = sum(b[1] for b in bids[:10])  # Top 10 levels
        ask_volume = sum(a[1] for a in asks[:10])
        
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            imbalance = 0
        else:
            imbalance = (bid_volume - ask_volume) / total_volume
        
        # Determine pressure
        if imbalance > self.imbalance_threshold:
            pressure = 'bullish'
            strength = min(abs(imbalance), 1.0)
        elif imbalance < -self.imbalance_threshold:
            pressure = 'bearish'
            strength = min(abs(imbalance), 1.0)
        else:
            pressure = 'neutral'
            strength = abs(imbalance)
        
        # Detect large orders (>2x average)
        all_volumes = [b[1] for b in bids] + [a[1] for a in asks]
        avg_volume = np.mean(all_volumes) if all_volumes else 0
        
        large_orders = []
        for i, bid in enumerate(bids[:5]):
            if bid[1] > avg_volume * 2:
                large_orders.append({
                    'side': 'bid',
                    'price': bid[0],
                    'volume': bid[1],
                    'level': i + 1
                })
        for i, ask in enumerate(asks[:5]):
            if ask[1] > avg_volume * 2:
                large_orders.append({
                    'side': 'ask',
                    'price': ask[0],
                    'volume': ask[1],
                    'level': i + 1
                })
        
        # Generate signal based on order flow
        signal = None
        if imbalance > 0.5 and len(large_orders) > 0:
            # Strong bid imbalance with large bid orders
            bid_large = any(o['side'] == 'bid' for o in large_orders)
            if bid_large:
                signal = 'STRONG_BUY'
        elif imbalance < -0.5 and len(large_orders) > 0:
            ask_large = any(o['side'] == 'ask' for o in large_orders)
            if ask_large:
                signal = 'STRONG_SELL'
        elif imbalance > 0.3:
            signal = 'BUY_PRESSURE'
        elif imbalance < -0.3:
            signal = 'SELL_PRESSURE'
        
        result = {
            'imbalance': round(imbalance, 3),
            'bid_volume': round(bid_volume, 2),
            'ask_volume': round(ask_volume, 2),
            'pressure': pressure,
            'strength': round(strength, 2),
            'signal': signal,
            'large_orders': large_orders,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.flow_history.append(result)
        if len(self.flow_history) > self.max_history:
            self.flow_history.pop(0)
        
        return result
    
    def get_momentum_shift(self) -> Optional[str]:
        """
        Detect if there's a momentum shift in order flow.
        Compares recent flow to previous flow.
        
        Returns:
            'bullish_shift', 'bearish_shift', or None
        """
        if len(self.flow_history) < 10:
            return None
        
        recent = self.flow_history[-5:]
        previous = self.flow_history[-10:-5]
        
        recent_imbalance = np.mean([f['imbalance'] for f in recent])
        prev_imbalance = np.mean([f['imbalance'] for f in previous])
        
        shift = recent_imbalance - prev_imbalance
        
        if shift > 0.2:
            return 'bullish_shift'
        elif shift < -0.2:
            return 'bearish_shift'
        
        return None
    
    def get_support_for_signal(self, action: str) -> Tuple[bool, float, str]:
        """
        Check if order flow supports a given signal direction.
        
        Returns:
            (supports, confidence_boost, reason)
        """
        if not self.flow_history:
            return True, 0, "No order flow data"
        
        latest = self.flow_history[-1]
        
        if action == 'LONG':
            if latest['pressure'] == 'bullish':
                return True, 0.1, f"Order flow bullish (imb: {latest['imbalance']:.2f})"
            elif latest['pressure'] == 'bearish':
                return False, -0.1, f"Order flow against LONG (imb: {latest['imbalance']:.2f})"
            else:
                return True, 0, "Order flow neutral"
        
        elif action == 'SHORT':
            if latest['pressure'] == 'bearish':
                return True, 0.1, f"Order flow bearish (imb: {latest['imbalance']:.2f})"
            elif latest['pressure'] == 'bullish':
                return False, -0.1, f"Order flow against SHORT (imb: {latest['imbalance']:.2f})"
            else:
                return True, 0, "Order flow neutral"
        
        return True, 0, "Unknown action"


class SmartMoneyDetector:
    """
    Detects smart money patterns in price action.
    Based on institutional trading concepts.
    """
    
    def find_order_blocks(self, df, lookback: int = 20) -> List[Dict]:
        """
        Find order blocks (last opposite candle before strong move).
        These are areas where institutions placed orders.
        """
        if len(df) < lookback + 5:
            return []
        
        order_blocks = []
        closes = df['close'].values
        opens = df['open'].values if 'open' in df.columns else closes
        highs = df['high'].values
        lows = df['low'].values
        
        for i in range(lookback, len(df) - 3):
            # Bullish Order Block: last bearish candle before strong bullish move
            if closes[i] < opens[i]:  # Bearish candle
                # Check for strong bullish move after
                if closes[i+2] > highs[i] and closes[i+3] > closes[i+2]:
                    order_blocks.append({
                        'type': 'bullish_ob',
                        'high': highs[i],
                        'low': lows[i],
                        'index': i,
                        'strength': (closes[i+3] - lows[i]) / lows[i] * 100
                    })
            
            # Bearish Order Block
            if closes[i] > opens[i]:  # Bullish candle
                if closes[i+2] < lows[i] and closes[i+3] < closes[i+2]:
                    order_blocks.append({
                        'type': 'bearish_ob',
                        'high': highs[i],
                        'low': lows[i],
                        'index': i,
                        'strength': (highs[i] - closes[i+3]) / highs[i] * 100
                    })
        
        return order_blocks[-5:]  # Return most recent 5
    
    def find_fair_value_gaps(self, df, lookback: int = 30) -> List[Dict]:
        """
        Find Fair Value Gaps (FVG) - price imbalances that tend to get filled.
        """
        if len(df) < lookback:
            return []
        
        fvgs = []
        highs = df['high'].values
        lows = df['low'].values
        
        for i in range(2, len(df)):
            # Bullish FVG: Gap up (low of current > high of 2 bars ago)
            if lows[i] > highs[i-2]:
                fvgs.append({
                    'type': 'bullish_fvg',
                    'top': lows[i],
                    'bottom': highs[i-2],
                    'index': i,
                    'size': lows[i] - highs[i-2]
                })
            
            # Bearish FVG: Gap down
            if highs[i] < lows[i-2]:
                fvgs.append({
                    'type': 'bearish_fvg',
                    'top': lows[i-2],
                    'bottom': highs[i],
                    'index': i,
                    'size': lows[i-2] - highs[i]
                })
        
        return fvgs[-5:]
    
    def find_liquidity_zones(self, df, lookback: int = 50) -> Dict:
        """
        Find liquidity zones (where stop losses cluster).
        Above swing highs and below swing lows.
        """
        if len(df) < lookback:
            return {'buy_side': [], 'sell_side': []}
        
        highs = df['high'].values[-lookback:]
        lows = df['low'].values[-lookback:]
        
        # Find swing highs (local maxima)
        swing_highs = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
        
        # Find swing lows
        swing_lows = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])
        
        return {
            'buy_side': sorted(swing_highs)[-3:],  # Above highs = buy side liquidity
            'sell_side': sorted(swing_lows)[:3]    # Below lows = sell side liquidity
        }


# Export
__all__ = ['OrderFlowAnalyzer', 'SmartMoneyDetector']


if __name__ == '__main__':
    print("=" * 60)
    print("ORDER FLOW ANALYZER")
    print("=" * 60)
    
    analyzer = OrderFlowAnalyzer()
    
    # Sample order book
    bids = [[87000, 10], [86990, 15], [86980, 8], [86970, 25], [86960, 12]]
    asks = [[87010, 5], [87020, 8], [87030, 3], [87040, 10], [87050, 6]]
    
    result = analyzer.analyze(bids, asks, 87005)
    
    print(f"\nImbalance: {result['imbalance']}")
    print(f"Pressure: {result['pressure']} (strength: {result['strength']})")
    print(f"Signal: {result['signal']}")
    print(f"Large orders: {len(result['large_orders'])}")
    
    # Test support check
    supports, boost, reason = analyzer.get_support_for_signal('LONG')
    print(f"\nSupports LONG: {supports}")
    print(f"Confidence boost: {boost}")
    print(f"Reason: {reason}")
