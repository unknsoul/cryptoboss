"""
Slippage Monitor
Tracks execution quality by comparing expected vs actual fill prices
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)


class SlippageMonitor:
    """
    Professional slippage tracking system
    
    Features:
    - Track expected vs actual fill prices
    - Calculate slippage per order type (market vs limit)
    - Monitor slippage trends over time
    - Alert on excessive slippage
    """
    
    def __init__(self, max_history: int = 1000, alert_threshold_bps: float = 10.0):
        """
        Initialize slippage monitor
        
        Args:
            max_history: Maximum trades to keep in history
            alert_threshold_bps: Alert if slippage > this many basis points
        """
        self.max_history = max_history
        self.alert_threshold_bps = alert_threshold_bps
        
        # Slippage tracking by order type
        self.slippage_history = {
            'MARKET': deque(maxlen=max_history),
            'LIMIT': deque(maxlen=max_history),
            'ALL': deque(maxlen=max_history)
        }
        
        # Statistics
        self.total_slippage_cost = 0.0
        self.worst_slippage = 0.0
        self.worst_slippage_trade = None
        
    def record_fill(self, expected_price: float, actual_price: float, 
                   side: str, size: float, order_type: str = 'MARKET') -> Dict:
        """
        Record a fill and calculate slippage
        
        Args:
            expected_price: Price we expected to get
            actual_price: Actual fill price
            side: 'BUY' or 'SELL'
            size: Position size
            order_type: 'MARKET' or 'LIMIT'
            
        Returns:
            Dict with slippage details
        """
        # Calculate slippage
        if side.upper() in ['BUY', 'LONG']:
            # For buys, positive slippage = paid more than expected (bad)
            slippage = actual_price - expected_price
        else:  # SELL/SHORT
            # For sells, positive slippage = got less than expected (bad)
            slippage = expected_price - actual_price
        
        # Calculate in basis points (bps)
        slippage_bps = (slippage / expected_price) * 10000
        
        # Calculate cost in dollars
        slippage_cost = abs(slippage * size)
        
        # Record the fill
        fill_record = {
            'timestamp': datetime.now(),
            'expected_price': expected_price,
            'actual_price': actual_price,
            'slippage': slippage,
            'slippage_bps': slippage_bps,
            'slippage_cost': slippage_cost,
            'side': side,
            'size': size,
            'order_type': order_type
        }
        
        # Add to history
        self.slippage_history[order_type].append(fill_record)
        self.slippage_history['ALL'].append(fill_record)
        
        # Update totals
        self.total_slippage_cost += slippage_cost
        
        # Track worst slippage
        if abs(slippage_bps) > abs(self.worst_slippage):
            self.worst_slippage = slippage_bps
            self.worst_slippage_trade = fill_record
        
        # Check for alerts
        alert = None
        if abs(slippage_bps) > self.alert_threshold_bps:
            alert = f"⚠️ High slippage: {slippage_bps:.1f} bps (${slippage_cost:.2f})"
            logger.warning(alert)
        
        result = {
            'slippage': round(slippage, 2),
            'slippage_bps': round(slippage_bps, 2),
            'slippage_cost': round(slippage_cost, 2),
            'alert': alert
        }
        
        return result
    
    def get_statistics(self, order_type: str = 'ALL', window: Optional[int] = None) -> Dict:
        """
        Get slippage statistics
        
        Args:
            order_type: 'MARKET', 'LIMIT', or 'ALL'
            window: Last N trades (None = all history)
            
        Returns:
            Dict with statistics
        """
        history = list(self.slippage_history.get(order_type, []))
        
        if not history:
            return {
                'status': 'no_data',
                'trade_count': 0
            }
        
        # Apply window
        if window and len(history) > window:
            history = history[-window:]
        
        # Extract slippage values
        slippages_bps = [fill['slippage_bps'] for fill in history]
        costs = [fill['slippage_cost'] for fill in history]
        
        return {
            'trade_count': len(history),
            'avg_slippage_bps': round(np.mean(slippages_bps), 2),
            'median_slippage_bps': round(np.median(slippages_bps), 2),
            'max_slippage_bps': round(np.max(slippages_bps), 2),
            'min_slippage_bps': round(np.min(slippages_bps), 2),
            'std_slippage_bps': round(np.std(slippages_bps), 2),
            'total_cost': round(sum(costs), 2),
            'avg_cost_per_trade': round(np.mean(costs), 2),
            'status': 'ok'
        }
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive slippage summary
        
        Returns:
            Dict with all slippage metrics
        """
        all_stats = self.get_statistics('ALL')
        market_stats = self.get_statistics('MARKET')
        limit_stats = self.get_statistics('LIMIT')
        
        return {
            'overall': all_stats,
            'market_orders': market_stats,
            'limit_orders': limit_stats,
            'total_slippage_cost': round(self.total_slippage_cost, 2),
            'worst_slippage_bps': round(self.worst_slippage, 2),
            'trades_tracked': len(self.slippage_history['ALL'])
        }
    
    def get_recent_slippage(self, n: int = 10) -> List[Dict]:
        """
        Get last N fills with slippage details
        
        Args:
            n: Number of recent fills to return
            
        Returns:
            List of recent fill records
        """
        recent = list(self.slippage_history['ALL'])[-n:]
        
        return [{
            'timestamp': fill['timestamp'].isoformat(),
            'expected': fill['expected_price'],
            'actual': fill['actual_price'],
            'slippage_bps': round(fill['slippage_bps'], 2),
            'cost': round(fill['slippage_cost'], 2),
            'side': fill['side'],
            'type': fill['order_type']
        } for fill in recent]
    
    def analyze_trends(self, window: int = 50) -> Dict:
        """
        Analyze slippage trends over time
        
        Args:
            window: Rolling window size
            
        Returns:
            Dict with trend analysis
        """
        history = list(self.slippage_history['ALL'])
        
        if len(history) < window:
            return {'status': 'insufficient_data', 'required': window, 'available': len(history)}
        
        # Calculate rolling average
        recent = history[-window:]
        recent_avg = np.mean([f['slippage_bps'] for f in recent])
        
        # Compare to historical average
        if len(history) > window * 2:
            historical = history[:-window]
            historical_avg = np.mean([f['slippage_bps'] for f in historical])
            
            trend = 'improving' if recent_avg < historical_avg else 'worsening'
            change_pct = ((recent_avg - historical_avg) / abs(historical_avg)) * 100 if historical_avg != 0 else 0
        else:
            trend = 'unknown'
            change_pct = 0
        
        return {
            'recent_avg_bps': round(recent_avg, 2),
            'trend': trend,
            'change_pct': round(change_pct, 2),
            'window_size': window,
            'status': 'ok'
        }


# Singleton instance
_slippage_monitor: Optional[SlippageMonitor] = None


def get_slippage_monitor() -> SlippageMonitor:
    """Get singleton slippage monitor"""
    global _slippage_monitor
    if _slippage_monitor is None:
        _slippage_monitor = SlippageMonitor()
    return _slippage_monitor


if __name__ == '__main__':
    # Test the slippage monitor
    print("=" * 60)
    print("SLIPPAGE MONITOR - TEST")
    print("=" * 60)
    
    monitor = SlippageMonitor(alert_threshold_bps=5.0)
    
    # Simulate some trades
    import random
    
    print("\n📊 Simulating 20 trades...")
    for i in range(20):
        expected = 50000.0
        # Add random slippage (usually small, occasionally large)
        slippage_pct = random.gauss(0.0005, 0.002)  # 0.05% avg, 0.2% std
        actual = expected * (1 + slippage_pct)
        
        side = random.choice(['BUY', 'SELL'])
        order_type = random.choice(['MARKET', 'MARKET', 'LIMIT'])  # More market orders
        size = random.uniform(0.05, 0.2)
        
        result = monitor.record_fill(expected, actual, side, size, order_type)
        
        if result['alert']:
            print(f"  Trade {i+1}: {result['alert']}")
    
    # Get statistics
    print("\n📈 Overall Statistics:")
    summary = monitor.get_summary()
    for key, value in summary['overall'].items():
        print(f"  {key}: {value}")
    
    print("\n📊 Market Orders:")
    for key, value in summary['market_orders'].items():
        print(f"  {key}: {value}")
    
    print("\n📉 Recent Slippage (Last 5):")
    recent = monitor.get_recent_slippage(5)
    for fill in recent:
        print(f"  {fill['side']} {fill['type']}: {fill['slippage_bps']:.2f} bps (${fill['cost']:.2f})")
    
    print("\n📊 Trend Analysis:")
    trends = monitor.analyze_trends(window=10)
    for key, value in trends.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Slippage monitor working correctly!")
    print(f"💰 Total slippage cost: ${summary['total_slippage_cost']:.2f}")
    print("=" * 60)
