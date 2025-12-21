"""
News-Based Event Trading Strategy
Trades based on real-time news sentiment and market reaction
"""

import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy


class NewsEventTradingStrategy(BaseStrategy):
    """
    News Event Trading
    
    Concept:
    - Monitors news sentiment in real-time
    - Trades immediate market reaction
    - Uses volatility expansion as confirmation
    
    Best for: News-driven regimes, high volatility
    Used by: High-frequency traders, event-driven funds
    """
    
    def __init__(self, sentiment_threshold: float = 0.6,
                 volatility_expansion_min: float = 1.5):
        super().__init__("News Event Trading", "event_driven")
        
        self.sentiment_threshold = sentiment_threshold
        self.vol_expansion_min = volatility_expansion_min
        
        self.parameters = {
            'sentiment_threshold': sentiment_threshold,
            'volatility_expansion': volatility_expansion_min
        }
    
    def signal(self, highs, lows, closes, volumes=None, 
              sentiment_score: float = 0.0, volatility_ratio: float = 1.0):
        """
        Generate news-based signal
        
        Additional parameters:
            sentiment_score: -1 to 1 (from news analysis)
            volatility_ratio: Current vol / avg vol
        """
        if not self.validate_data(highs, lows, closes):
            return None
        
        # Require strong sentiment + volatility expansion
        action = 'HOLD'
        confidence = 0.0
        
        # Bullish news
        if sentiment_score > self.sentiment_threshold and volatility_ratio > self.vol_expansion_min:
            action = 'LONG'
            confidence = min(sentiment_score * volatility_ratio / 2.0, 0.95)
        
        # Bearish news
        elif sentiment_score < -self.sentiment_threshold and volatility_ratio > self.vol_expansion_min:
            action = 'SHORT'
            confidence = min(abs(sentiment_score) * volatility_ratio / 2.0, 0.95)
        
        # Calculate dynamic stops based on volatility
        recent_vol = np.std(closes[-20:] / closes[-21:-1] - 1) if len(closes) >= 21 else 0.01
        stop = recent_vol * closes[-1] * 3.0  # Wide stops for news volatility
        target = stop * 2.0  # 2:1 R:R
        
        return {
            'action': action,
            'stop': stop,
            'target': target,
            'confidence': confidence,
            'metadata': {
                'sentiment_score': sentiment_score,
                'volatility_ratio': volatility_ratio,
                'entry_type': 'news_event'
            }
        }
    
    def check_exit(self, highs, lows, closes, position_side,
                   entry_price, entry_index, current_index):
        """Exit quickly - news trades are short-term"""
        # Exit after 10 bars (news impact fades)
        if current_index - entry_index > 10:
            return {'action': 'CLOSE', 'reason': 'news_impact_faded'}
        
        return None


class LiquidityGrabStrategy(BaseStrategy):
    """
    Liquidity Grab Strategy (Market Maker Reversal)
    
    Concept:
    - Identifies liquidity hunts (stop loss raids)
    - Fades the move when liquidity is grabbed
    - Trades the reversal
    
    Best for: Ranging markets, trapping retail traders
    Used by: Market makers, institutional traders
    """
    
    def __init__(self, swing_period: int = 20):
        super().__init__("Liquidity Grab", "reversal")
        
        self.swing_period = swing_period
        
        self.parameters = {'swing_period': swing_period}
    
    def signal(self, highs, lows, closes, volumes=None):
        """Detect liquidity grab and fade"""
        if not self.validate_data(highs, lows, closes):
            return None
        
        if len(closes) < self.swing_period + 10:
            return None
        
        # Identify swing highs/lows (liquidity zones)
        swing_high = np.max(highs[-self.swing_period:-5])
        swing_low = np.min(lows[-self.swing_period:-5])
        
        current_high = highs[-1]
        current_low = lows[-1]
        current_close = closes[-1]
        
        action = 'HOLD'
        confidence = 0.0
        
        # Liquidity grab above swing high (fake breakout)
        if current_high > swing_high * 1.001:  # Broke above by 0.1%
            # Check if it's rejecting (closing back below)
            if current_close < swing_high:
                action = 'SHORT'
                confidence = 0.75
                
                # Higher confidence if volume spike (stop loss hunt)
                if volumes is not None:
                    avg_vol = np.mean(volumes[-20:])
                    if volumes[-1] > avg_vol * 1.5:
                        confidence = 0.85
        
        # Liquidity grab below swing low (fake breakdown)
        elif current_low < swing_low * 0.999:  # Broke below by 0.1%
            # Check if it's rejecting (closing back above)
            if current_close > swing_low:
                action = 'LONG'
                confidence = 0.75
                
                if volumes is not None:
                    avg_vol = np.mean(volumes[-20:])
                    if volumes[-1] > avg_vol * 1.5:
                        confidence = 0.85
        
        # Stops and targets
        range_size = swing_high - swing_low
        stop = range_size * 0.3
        target = range_size * 0.7
        
        return {
            'action': action,
            'stop': stop,
            'target': target,
            'confidence': confidence,
            'metadata': {
                'swing_high': swing_high,
                'swing_low': swing_low,
                'entry_type': 'liquidity_grab'
            }
        }
    
    def check_exit(self, highs, lows, closes, position_side,
                   entry_price, entry_index, current_index):
        """Exit if liquidity grab fails"""
        return None  # Use standard exits


class OrderFlowImbalanceStrategy(BaseStrategy):
    """
    Order Flow Imbalance Strategy
    
    Concept:
    - Trades based on order book imbalance
    - Strong bid = price likely to rise
    - Strong ask = price likely to fall
    
    Best for: Liquid markets with deep order books
    Used by: HFT firms, market makers
    """
    
    def __init__(self, imbalance_threshold: float = 0.4):
        super().__init__("Order Flow Imbalance", "order_flow")
        
        self.imbalance_threshold = imbalance_threshold
        
        self.parameters = {'imbalance_threshold': imbalance_threshold}
    
    def signal(self, highs, lows, closes, volumes=None,
              orderbook_imbalance: float = 0.0):
        """
        Generate signal based on order book
        
        Args:
            orderbook_imbalance: -1 (all asks) to 1 (all bids)
        """
        if not self.validate_data(highs, lows, closes):
            return None
        
        action = 'HOLD'
        confidence = 0.0
        
        # Strong bid support
        if orderbook_imbalance > self.imbalance_threshold:
            action = 'LONG'
            confidence = min(orderbook_imbalance, 0.90)
        
        # Strong ask pressure
        elif orderbook_imbalance < -self.imbalance_threshold:
            action = 'SHORT'
            confidence = min(abs(orderbook_imbalance), 0.90)
        
        # Calculate stops based on order book depth
        recent_range = np.max(highs[-20:]) - np.min(lows[-20:])
        stop = recent_range * 0.02  # Tight stops for order flow
        target = stop * 3.0  # 3:1 R:R
        
        return {
            'action': action,
            'stop': stop,
            'target': target,
            'confidence': confidence,
            'metadata': {
                'orderbook_imbalance': orderbook_imbalance,
                'entry_type': 'order_flow'
            }
        }
    
    def check_exit(self, highs, lows, closes, position_side,
                   entry_price, entry_index, current_index):
        """Exit quickly if order flow changes"""
        # Order flow trades are very short-term
        if current_index - entry_index > 3:
            return {'action': 'CLOSE', 'reason': 'order_flow_timeout'}
        
        return None


if __name__ == "__main__":
    print("Testing event-driven strategies...")
    
    # Generate test data
    np.random.seed(42)
    closes = 45000 + np.random.randn(100).cumsum() * 30
    highs = closes * 1.001
    lows = closes * 0.999
    volumes = np.random.uniform(900, 1100, 100)
    
    # Test News Event Trading
    news_strategy = NewsEventTradingStrategy()
    signal = news_strategy.signal(
        highs, lows, closes, volumes,
        sentiment_score=0.75,  # Bullish news
        volatility_ratio=1.8   # Vol expansion
    )
    print("\nNEWS EVENT TRADING:")
    print(signal)
    
    # Test Liquidity Grab
    liquidity_grab = LiquidityGrabStrategy()
    signal = liquidity_grab.signal(highs, lows, closes, volumes)
    print("\nLIQUIDITY GRAB:")
    print(signal)
    
    # Test Order Flow
    order_flow = OrderFlowImbalanceStrategy()
    signal = order_flow.signal(
        highs, lows, closes, volumes,
        orderbook_imbalance=0.65  # Strong bids
    )
    print("\nORDER FLOW IMBALANCE:")
    print(signal)
    
    print("\n✅ Event-driven strategies test complete")
