"""
Slippage Model & Execution Optimizer - Enterprise Features #62, #152, #153, #331
Models slippage and optimizes order execution.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class SlippageModel:
    """
    Feature #62: Realistic Slippage Modeling
    
    Estimates expected slippage based on:
    - Order size relative to liquidity
    - Current volatility
    - Spread conditions
    - Historical slippage data
    """
    
    def __init__(
        self,
        base_slippage_bps: float = 2.0,      # Base slippage in basis points
        size_impact_factor: float = 0.1,      # Impact per $1000 order
        volatility_factor: float = 0.5,       # Volatility multiplier
        max_slippage_bps: float = 50.0        # Maximum allowed slippage
    ):
        """
        Initialize slippage model.
        
        Args:
            base_slippage_bps: Baseline slippage in basis points
            size_impact_factor: How order size affects slippage
            volatility_factor: Volatility multiplier for slippage
            max_slippage_bps: Maximum slippage threshold for trade rejection
        """
        self.base_slippage_bps = base_slippage_bps
        self.size_impact_factor = size_impact_factor
        self.volatility_factor = volatility_factor
        self.max_slippage_bps = max_slippage_bps
        
        # Historical tracking
        self.slippage_history: List[Dict] = []
        
        logger.info(f"Slippage Model initialized - Base: {base_slippage_bps}bps, Max: {max_slippage_bps}bps")
    
    def estimate_slippage(
        self,
        order_size_usd: float,
        current_price: float,
        atr: float,
        spread: float = 0,
        orderbook_depth: float = 0
    ) -> Dict:
        """
        Estimate expected slippage for an order.
        
        Args:
            order_size_usd: Order size in USD
            current_price: Current market price
            atr: Average True Range (volatility)
            spread: Current bid-ask spread
            orderbook_depth: Sum of top-5 orderbook levels
            
        Returns:
            Slippage estimate with details
        """
        # Base slippage
        slippage_bps = self.base_slippage_bps
        
        # Size impact (larger orders = more slippage)
        size_impact = (order_size_usd / 1000) * self.size_impact_factor
        slippage_bps += size_impact
        
        # Volatility impact
        if atr > 0 and current_price > 0:
            volatility_pct = (atr / current_price) * 100
            vol_impact = volatility_pct * self.volatility_factor * 10  # Scale
            slippage_bps += vol_impact
        
        # Spread impact
        if spread > 0 and current_price > 0:
            spread_bps = (spread / current_price) * 10000
            slippage_bps += spread_bps * 0.5  # Half spread as slippage
        
        # Liquidity impact (less liquidity = more slippage)
        if orderbook_depth > 0 and order_size_usd > 0:
            fill_ratio = order_size_usd / orderbook_depth
            if fill_ratio > 0.1:  # Order is >10% of visible liquidity
                slippage_bps += fill_ratio * 20
        
        # Cap at maximum
        slippage_bps = min(slippage_bps, self.max_slippage_bps)
        
        # Calculate dollar value
        slippage_pct = slippage_bps / 10000
        slippage_usd = order_size_usd * slippage_pct
        
        result = {
            'slippage_bps': round(slippage_bps, 2),
            'slippage_pct': round(slippage_pct * 100, 4),
            'slippage_usd': round(slippage_usd, 2),
            'exceeds_max': slippage_bps >= self.max_slippage_bps,
            'breakdown': {
                'base': self.base_slippage_bps,
                'size_impact': round(size_impact, 2),
                'volatility_impact': round(vol_impact if 'vol_impact' in dir() else 0, 2),
                'spread_impact': round(spread_bps * 0.5 if 'spread_bps' in dir() else 0, 2)
            }
        }
        
        return result
    
    def should_execute(self, slippage_estimate: Dict) -> Tuple[bool, str]:
        """Check if trade should execute based on slippage."""
        if slippage_estimate['exceeds_max']:
            return False, f"Slippage {slippage_estimate['slippage_bps']}bps exceeds max {self.max_slippage_bps}bps"
        return True, "OK"
    
    def record_actual_slippage(
        self,
        expected_price: float,
        actual_price: float,
        order_size: float,
        side: str
    ):
        """Record actual slippage for model calibration."""
        if side == 'BUY':
            slippage = actual_price - expected_price
        else:
            slippage = expected_price - actual_price
        
        slippage_bps = (slippage / expected_price) * 10000
        
        self.slippage_history.append({
            'timestamp': datetime.now().isoformat(),
            'expected': expected_price,
            'actual': actual_price,
            'size': order_size,
            'side': side,
            'slippage_bps': round(slippage_bps, 2)
        })
        
        # Keep last 100
        self.slippage_history = self.slippage_history[-100:]
    
    def get_average_slippage(self) -> float:
        """Get average historical slippage in basis points."""
        if not self.slippage_history:
            return self.base_slippage_bps
        return sum(s['slippage_bps'] for s in self.slippage_history) / len(self.slippage_history)


class SpreadFilter:
    """
    Feature #153: Max Spread Filter
    Blocks trades when spread is too wide.
    """
    
    def __init__(self, max_spread_pct: float = 0.15):
        """
        Initialize spread filter.
        
        Args:
            max_spread_pct: Maximum spread as percentage (0.15 = 0.15%)
        """
        self.max_spread_pct = max_spread_pct
    
    def check_spread(self, orderbook: Dict, price: float) -> Tuple[bool, float]:
        """
        Check if spread is acceptable.
        
        Returns:
            Tuple of (is_acceptable, spread_pct)
        """
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return True, 0  # Can't check, so allow
            
            best_bid = float(bids[0][0]) if isinstance(bids[0], list) else bids[0]
            best_ask = float(asks[0][0]) if isinstance(asks[0], list) else asks[0]
            
            spread = best_ask - best_bid
            spread_pct = (spread / price) * 100
            
            is_ok = spread_pct <= self.max_spread_pct
            
            if not is_ok:
                logger.warning(f"Spread filter triggered: {spread_pct:.3f}% > {self.max_spread_pct}%")
            
            return is_ok, round(spread_pct, 4)
        except Exception:
            return True, 0


class AdaptiveOrderAggressiveness:
    """
    Feature #331: Adaptive Order Aggressiveness
    Adjusts limit/market order mix based on urgency.
    """
    
    def __init__(self):
        """Initialize adaptive order aggressiveness."""
        self.urgency_thresholds = {
            'low': 0.3,       # Use limit orders, wait for fill
            'medium': 0.6,    # Mix of limit/market
            'high': 0.8       # Use market orders
        }
    
    def recommend_order_type(
        self,
        signal_confidence: float,
        volatility_ratio: float,
        position_urgency: str = 'normal'
    ) -> Dict:
        """
        Recommend order type and aggressiveness.
        
        Args:
            signal_confidence: Signal confidence (0-1)
            volatility_ratio: Current vol / avg vol
            position_urgency: 'low', 'normal', 'high', 'critical'
            
        Returns:
            Order recommendation
        """
        # Calculate urgency score
        urgency = signal_confidence * 0.5
        
        if volatility_ratio > 1.5:
            urgency += 0.2  # High vol = more urgent
        
        if position_urgency == 'critical':
            urgency = 1.0
        elif position_urgency == 'high':
            urgency += 0.3
        elif position_urgency == 'low':
            urgency -= 0.2
        
        urgency = min(max(urgency, 0), 1)
        
        # Determine order type
        if urgency >= self.urgency_thresholds['high']:
            order_type = 'MARKET'
            limit_offset = 0
        elif urgency >= self.urgency_thresholds['medium']:
            order_type = 'LIMIT'
            limit_offset = 0.02  # 0.02% from mid
        else:
            order_type = 'LIMIT'
            limit_offset = 0.05  # 0.05% from mid (more passive)
        
        return {
            'urgency_score': round(urgency, 2),
            'order_type': order_type,
            'limit_offset_pct': limit_offset,
            'time_in_force': 'IOC' if urgency >= 0.8 else 'GTC'
        }


# Singleton instances
_slippage_model: Optional[SlippageModel] = None
_spread_filter: Optional[SpreadFilter] = None


def get_slippage_model() -> SlippageModel:
    global _slippage_model
    if _slippage_model is None:
        _slippage_model = SlippageModel()
    return _slippage_model


def get_spread_filter() -> SpreadFilter:
    global _spread_filter
    if _spread_filter is None:
        _spread_filter = SpreadFilter()
    return _spread_filter


if __name__ == '__main__':
    # Test slippage model
    model = SlippageModel()
    
    est = model.estimate_slippage(
        order_size_usd=500,
        current_price=50000,
        atr=1000,
        spread=20
    )
    
    print(f"Slippage estimate: {est}")
    
    # Test spread filter
    sf = SpreadFilter(max_spread_pct=0.1)
    ok, spread = sf.check_spread(
        {'bids': [[49990, 1]], 'asks': [[50010, 1]]},
        50000
    )
    print(f"Spread check: OK={ok}, spread={spread}%")
    
    # Test order aggressiveness
    aggr = AdaptiveOrderAggressiveness()
    rec = aggr.recommend_order_type(0.75, 1.2, 'normal')
    print(f"Order recommendation: {rec}")
