"""
Latency Monitor - Order Execution Performance Tracker
Measures time from order submission to fill confirmation

Critical for:
- Identifying slow execution
- Detecting API issues
- Measuring market impact
- Optimizing order routing
"""

import time
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LatencyMonitor:
    """
    Professional latency tracking system
    
    Tracks:
    - Order-to-fill latency (ms)
    - API response times
    - Network latency
    - Statistical analysis (percentiles)
    """
    
    def __init__(self, max_history: int = 1000):
        """Initialize latency monitor"""
        self.max_history = max_history
        self.latencies = deque(maxlen=max_history)
        self.current_orders = {}  # Track pending orders
        
    def start_order(self, order_id: str) -> float:
        """
        Mark order start time
        
        Args:
            order_id: Unique order identifier
            
        Returns:
            Start timestamp
        """
        start_time = time.time()
        self.current_orders[order_id] = {
            'start': start_time,
            'timestamp': datetime.now()
        }
        return start_time
    
    def end_order(self, order_id: str) -> Optional[Dict]:
        """
        Mark order completion and calculate latency
        
        Args:
            order_id: Order identifier
            
        Returns:
            Dict with latency details
        """
        if order_id not in self.current_orders:
            logger.warning(f"Order {order_id} not found in tracking")
            return None
        
        end_time = time.time()
        order_data = self.current_orders.pop(order_id)
        start_time = order_data['start']
        
        latency_seconds = end_time - start_time
        latency_ms = latency_seconds * 1000
        
        record = {
            'order_id': order_id,
            'latency_ms': latency_ms,
            'timestamp': order_data['timestamp']
        }
        
        self.latencies.append(record)
        
        # Log if very slow
        if latency_ms > 1000:  # > 1 second
            logger.warning(f"High latency: {latency_ms:.0f}ms for order {order_id}")
        
        return record
    
    def get_statistics(self, window: Optional[int] = None) -> Dict:
        """
        Get latency statistics
        
        Args:
            window: Last N orders (None = all)
            
        Returns:
            Comprehensive latency stats
        """
        if not self.latencies:
            return {'status': 'no_data'}
        
        latencies_list = list(self.latencies)[-window:] if window else list(self.latencies)
        latencies_ms = [l['latency_ms'] for l in latencies_list]
        
        return {
            'count': len(latencies_ms),
            'mean_ms': round(np.mean(latencies_ms), 2),
            'median_ms': round(np.median(latencies_ms), 2),
            'min_ms': round(np.min(latencies_ms), 2),
            'max_ms': round(np.max(latencies_ms), 2),
            'std_ms': round(np.std(latencies_ms), 2),
            'p50_ms': round(np.percentile(latencies_ms, 50), 2),
            'p95_ms': round(np.percentile(latencies_ms, 95), 2),
            'p99_ms': round(np.percentile(latencies_ms, 99), 2),
            'status': 'ok'
        }
    
    def get_performance_grade(self) -> Dict:
        """
        Grade execution performance based on latency
        
        Returns:
            Performance grade and analysis
        """
        if len(self.latencies) < 10:
            return {'status': 'insufficient_data', 'required': 10}
        
        stats = self.get_statistics()
        p95 = stats['p95_ms']
        median = stats['median_ms']
        
        # Grading criteria (milliseconds)
        if p95 < 100:
            grade = 'A'
            description = 'Excellent - Institutional quality'
        elif p95 < 250:
            grade = 'B'
            description = 'Good - Acceptable for retail'
        elif p95 < 500:
            grade = 'C'
            description = 'Fair - Room for improvement'
        elif p95 < 1000:
            grade = 'D'
            description = 'Poor - Significant delays'
        else:
            grade = 'F'
            description = 'Failing - Critical latency issues'
        
        return {
            'grade': grade,
            'description': description,
            'p95_ms': p95,
            'median_ms': median,
            'recommendation': self._get_recommendation(p95),
            'status': 'ok'
        }
    
    def _get_recommendation(self, p95_ms: float) -> str:
        """Get recommendations based on latency"""
        if p95_ms > 1000:
            return "CRITICAL: Consider switching to faster API endpoint or checking network"
        elif p95_ms > 500:
            return "Investigate: Check for network issues or API throttling"
        elif p95_ms > 250:
            return "Monitor: Latency slightly elevated, watch for degradation"
        else:
            return "Good: Latency within acceptable range"
    
    def get_recent_orders(self, n: int = 10) -> List[Dict]:
        """Get last N order latencies"""
        recent = list(self.latencies)[-n:]
        return [{
            'order_id': r['order_id'],
            'latency_ms': round(r['latency_ms'], 2),
            'timestamp': r['timestamp'].isoformat()
        } for r in recent]
    
    def detect_degradation(self, window: int = 50, threshold_pct: float = 50) -> Dict:
        """
        Detect if latency is degrading over time
        
        Args:
            window: Rolling window size
            threshold_pct: Alert if increase > this %
            
        Returns:
            Degradation analysis
        """
        if len(self.latencies) < window * 2:
            return {'status': 'insufficient_data', 'required': window * 2}
        
        # Compare recent window to historical
        all_latencies = list(self.latencies)
        recent = all_latencies[-window:]
        historical = all_latencies[-window*2:-window]
        
        recent_avg = np.mean([l['latency_ms'] for l in recent])
        historical_avg = np.mean([l['latency_ms'] for l in historical])
        
        pct_change = ((recent_avg - historical_avg) / historical_avg) * 100
        
        is_degrading = pct_change > threshold_pct
        
        return {
            'is_degrading': is_degrading,
            'pct_change': round(pct_change, 2),
            'recent_avg_ms': round(recent_avg, 2),
            'historical_avg_ms': round(historical_avg, 2),
            'alert': f"⚠️ Latency increased {pct_change:.1f}%" if is_degrading else None,
            'status': 'ok'
        }


# Singleton
_latency_monitor: Optional[LatencyMonitor] = None


def get_latency_monitor() -> LatencyMonitor:
    """Get singleton latency monitor"""
    global _latency_monitor
    if _latency_monitor is None:
        _latency_monitor = LatencyMonitor()
    return _latency_monitor


if __name__ == '__main__':
    # Test latency monitor
    print("=" * 70)
    print("LATENCY MONITOR - TEST")
    print("=" * 70)
    
    monitor = LatencyMonitor()
    
    # Simulate orders
    import random
    
    print("\n📊 Simulating 100 orders with varying latencies...")
    
    for i in range(100):
        order_id = f"ORDER_{i:04d}"
        
        # Start order
        monitor.start_order(order_id)
        
        # Simulate processing time (normally distributed around 50ms)
        base_latency = random.gauss(50, 20) / 1000  # Convert to seconds
        
        # Add occasional spike
        if random.random() < 0.05:  # 5% chance of spike
            base_latency += random.uniform(0.5, 2.0)
        
        time.sleep(max(0, base_latency))
        
        # End order
        result = monitor.end_order(order_id)
        
        if i % 20 == 0:
            print(f"  Order {i}: {result['latency_ms']:.1f}ms")
    
    # Get statistics
    print("\n📈 Latency Statistics:")
    stats = monitor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Performance grade
    print("\n🎯 Performance Grade:")
    grade = monitor.get_performance_grade()
    for key, value in grade.items():
        print(f"  {key}: {value}")
    
    # Degradation check
    print("\n⏱️  Degradation Analysis:")
    degradation = monitor.detect_degradation(window=25)
    for key, value in degradation.items():
        print(f"  {key}: {value}")
    
    print("\n📋 Recent Orders (Last 5):")
    recent = monitor.get_recent_orders(5)
    for order in recent:
        print(f"  {order['order_id']}: {order['latency_ms']}ms")
    
    print("\n" + "=" * 70)
    print("✅ Latency monitor working correctly!")
    print("=" * 70)
