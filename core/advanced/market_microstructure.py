"""
Market Microstructure Analysis
Advanced order book and liquidity analysis for better execution
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class OrderBookSnapshot:
    """Order book snapshot at a point in time"""
    timestamp: datetime
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]  # [(price, size), ...]
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def best_bid(self) -> float:
        """Best bid price"""
        return self.bids[0][0] if self.bids else 0
    
    @property
    def best_ask(self) -> float:
        """Best ask price"""
        return self.asks[0][0] if self.asks else float('inf')
    
    @property
    def spread(self) -> float:
        """Bid-ask spread"""
        return self.best_ask - self.best_bid
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points"""
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return 0


class MarketMicrostructure:
    """
    Advanced market microstructure analysis
    
    Features:
    - Bid-ask spread analysis
    - Order book imbalance
    - Liquidity depth
    - Volume profile
    - Order flow toxicity detection
    """
    
    def __init__(self, depth_levels: int = 10):
        """
        Initialize microstructure analyzer
        
        Args:
            depth_levels: Number of price levels to analyze
        """
        self.depth_levels = depth_levels
        self.orderbook_history: List[OrderBookSnapshot] = []
    
    def add_snapshot(self, snapshot: OrderBookSnapshot):
        """Add order book snapshot to history"""
        self.orderbook_history.append(snapshot)
        
        # Keep last 1000 snapshots
        if len(self.orderbook_history) > 1000:
            self.orderbook_history.pop(0)
    
    def calculate_spread_metrics(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """
        Calculate spread-related metrics
        
        Returns:
            Dictionary with spread metrics
        """
        return {
            'absolute_spread': snapshot.spread,
            'relative_spread_bps': snapshot.spread_bps,
            'mid_price': snapshot.mid_price
        }
    
    def calculate_order_imbalance(self, snapshot: OrderBookSnapshot) -> float:
        """
        Calculate order book imbalance
        
        Values:
        - +1.0: All bids (strong buying pressure)
        - 0.0: Balanced
        - -1.0: All asks (strong selling pressure)
        
        Returns:
            Imbalance ratio (-1 to +1)
        """
        bid_volume = sum(size for _, size in snapshot.bids[:self.depth_levels])
        ask_volume = sum(size for _, size in snapshot.asks[:self.depth_levels])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total_volume
    
    def calculate_liquidity_depth(self, snapshot: OrderBookSnapshot, distance_bps: float = 10) -> Dict[str, float]:
        """
        Calculate liquidity within distance from mid price
        
        Args:
            snapshot: Order book snapshot
            distance_bps: Distance in basis points from mid
            
        Returns:
            Dictionary with bid/ask liquidity
        """
        mid = snapshot.mid_price
        distance = mid * (distance_bps / 10000)
        
        # Calculate bid liquidity
        bid_liquidity = sum(
            size for price, size in snapshot.bids
            if price >= mid - distance
        )
        
        # Calculate ask liquidity
        ask_liquidity = sum(
            size for price, size in snapshot.asks
            if price <= mid + distance
        )
        
        return {
            'bid_liquidity': bid_liquidity,
            'ask_liquidity': ask_liquidity,
            'total_liquidity': bid_liquidity + ask_liquidity,
            'liquidity_ratio': bid_liquidity / ask_liquidity if ask_liquidity > 0 else 0
        }
    
    def detect_spoofing(self, window_seconds: int = 60) -> bool:
        """
        Detect order book spoofing/manipulation
        
        Looks for:
        - Large orders placed and quickly cancelled
        - Unusual order book asymmetry
        
        Args:
            window_seconds: Time window to analyze
            
        Returns:
            True if potential spoofing detected
        """
        if len(self.orderbook_history) < 10:
            return False
        
        # Get recent snapshots
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_snapshots = [
            s for s in self.orderbook_history 
            if s.timestamp >= cutoff_time
        ]
        
        if len(recent_snapshots) < 5:
            return False
        
        # Calculate imbalance volatility (high volatility = potential spoofing)
        imbalances = [self.calculate_order_imbalance(s) for s in recent_snapshots]
        imbalance_std = np.std(imbalances)
        
        # High volatility in order book imbalance
        if imbalance_std > 0.5:  # Threshold
            return True
        
        # Check for large sudden changes
        for i in range(1, len(recent_snapshots)):
            prev_imbalance = self.calculate_order_imbalance(recent_snapshots[i-1])
            curr_imbalance = self.calculate_order_imbalance(recent_snapshots[i])
            
            # Sudden flip
            if abs(curr_imbalance - prev_imbalance) > 0.7:
                return True
        
        return False
    
    def calculate_vwap_levels(self, snapshot: OrderBookSnapshot, target_volume: float) -> Dict[str, float]:
        """
        Calculate VWAP for target volume
        
        Args:
            snapshot: Order book snapshot
            target_volume: Target volume to fill
            
        Returns:
            Dictionary with buy/sell VWAP
        """
        # Buy VWAP (hitting asks)
        remaining_volume = target_volume
        total_cost = 0
        total_volume_filled = 0
        
        for price, size in snapshot.asks:
            if remaining_volume <= 0:
                break
            
            fill_size = min(remaining_volume, size)
            total_cost += price * fill_size
            total_volume_filled += fill_size
            remaining_volume -= fill_size
        
        buy_vwap = total_cost / total_volume_filled if total_volume_filled > 0 else 0
        
        # Sell VWAP (hitting bids)
        remaining_volume = target_volume
        total_revenue = 0
        total_volume_filled = 0
        
        for price, size in snapshot.bids:
            if remaining_volume <= 0:
                break
            
            fill_size = min(remaining_volume, size)
            total_revenue += price * fill_size
            total_volume_filled += fill_size
            remaining_volume -= fill_size
        
        sell_vwap = total_revenue / total_volume_filled if total_volume_filled > 0 else 0
        
        return {
            'buy_vwap': buy_vwap,
            'sell_vwap': sell_vwap,
            'slippage_bps': ((buy_vwap - snapshot.mid_price) / snapshot.mid_price * 10000) if snapshot.mid_price > 0 else 0
        }
    
    def estimate_market_impact(self, snapshot: OrderBookSnapshot, order_size: float, side: str = 'BUY') -> Dict[str, float]:
        """
        Estimate market impact of order
        
        Args:
            snapshot: Order book snapshot
            order_size: Size of intended order
            side: 'BUY' or 'SELL'
            
        Returns:
            Dictionary with impact estimates
        """
        if side == 'BUY':
            # Calculate how much we need to walk up the ask side
            remaining = order_size
            levels_consumed = 0
            worst_price = 0
            
            for price, size in snapshot.asks:
                if remaining <= 0:
                    break
                levels_consumed += 1
                worst_price = price
                remaining -= size
            
            vwap_metrics = self.calculate_vwap_levels(snapshot, order_size)
            avg_price = vwap_metrics['buy_vwap']
            
        else:  # SELL
            remaining = order_size
            levels_consumed = 0
            worst_price = 0
            
            for price, size in snapshot.bids:
                if remaining <= 0:
                    break
                levels_consumed += 1
                worst_price = price
                remaining -= size
            
            vwap_metrics = self.calculate_vwap_levels(snapshot, order_size)
            avg_price = vwap_metrics['sell_vwap']
        
        # Calculate impact
        mid = snapshot.mid_price
        impact_bps = abs((avg_price - mid) / mid * 10000) if mid > 0 else 0
        
        return {
            'levels_consumed': levels_consumed,
            'worst_price': worst_price,
            'average_price': avg_price,
            'market_impact_bps': impact_bps,
            'can_fill': remaining <= 0
        }


if __name__ == '__main__':
    # Example usage
    
    # Create mock order book
    snapshot = OrderBookSnapshot(
        timestamp=datetime.now(),
        bids=[
            (100000, 0.5),  # price, size
            (99950, 1.0),
            (99900, 1.5),
            (99850, 2.0),
        ],
        asks=[
            (100050, 0.5),
            (100100, 1.0),
            (100150, 1.5),
            (100200, 2.0),
        ]
    )
    
    analyzer = MarketMicrostructure(depth_levels=5)
    analyzer.add_snapshot(snapshot)
    
    # Spread analysis
    spread = analyzer.calculate_spread_metrics(snapshot)
    print(f"Spread: ${spread['absolute_spread']:.2f} ({spread['relative_spread_bps']:.2f} bps)")
    
    # Order imbalance
    imbalance = analyzer.calculate_order_imbalance(snapshot)
    print(f"Order Imbalance: {imbalance:.3f} ({'BUY' if imbalance > 0 else 'SELL'} pressure)")
    
    # Liquidity depth
    liquidity = analyzer.calculate_liquidity_depth(snapshot, distance_bps=100)
    print(f"Liquidity (100bps): {liquidity['total_liquidity']:.2f} BTC")
    
    # Market impact
    impact = analyzer.estimate_market_impact(snapshot, order_size=2.0, side='BUY')
    print(f"Market Impact (2 BTC buy): {impact['market_impact_bps']:.2f} bps")
    print(f"Levels consumed: {impact['levels_consumed']}")
    
    print("\n✓ Market microstructure analysis working!")
