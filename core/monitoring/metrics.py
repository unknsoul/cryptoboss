"""
Metrics Collection and Monitoring
Tracks key performance indicators in real-time
"""

import time
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading


class MetricsCollector:
    """
    Collects and aggregates trading metrics
    Thread-safe implementation for concurrent access
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.lock = threading.Lock()
        
        # Counters
        self.counters = defaultdict(int)
        
        # Gauges (current values)
        self.gauges = defaultdict(float)
        
        # Histograms (time-series data with timestamps)
        self.histograms = defaultdict(lambda: deque(maxlen=10000))
        
        # Timers (for latency tracking)
        self.timers = defaultdict(list)
        
        # Trade statistics
        self.trades = []
        self.pnl_history = deque(maxlen=10000)
        
        # API call statistics
        self.api_calls = defaultdict(lambda: {"count": 0, "errors": 0, "total_latency": 0.0})
        
    def increment(self, metric: str, value: int = 1):
        """Increment a counter"""
        with self.lock:
            self.counters[metric] += value
    
    def set_gauge(self, metric: str, value: float):
        """Set a gauge value"""
        with self.lock:
            self.gauges[metric] = value
    
    def record_histogram(self, metric: str, value: float):
        """Record a histogram value with timestamp"""
        with self.lock:
            self.histograms[metric].append({
                "value": value,
                "timestamp": datetime.now()
            })
    
    def record_timer(self, metric: str, duration_ms: float):
        """Record a timer value (latency)"""
        with self.lock:
            self.timers[metric].append(duration_ms)
            # Keep only last 1000 measurements
            if len(self.timers[metric]) > 1000:
                self.timers[metric] = self.timers[metric][-1000:]
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a trade"""
        with self.lock:
            trade_data['timestamp'] = datetime.now()
            self.trades.append(trade_data)
            
            # Update PnL history
            if 'pnl' in trade_data:
                self.pnl_history.append({
                    'pnl': trade_data['pnl'],
                    'timestamp': trade_data['timestamp']
                })
            
            # Update counters
            self.increment(f"trades_{trade_data.get('side', 'unknown').lower()}")
            if trade_data.get('pnl', 0) > 0:
                self.increment('trades_winning')
            else:
                self.increment('trades_losing')
    
    def record_api_call(self, endpoint: str, latency_ms: float, success: bool = True):
        """Record an API call"""
        with self.lock:
            self.api_calls[endpoint]["count"] += 1
            self.api_calls[endpoint]["total_latency"] += latency_ms
            if not success:
                self.api_calls[endpoint]["errors"] += 1
            
            self.record_timer(f"api_latency_{endpoint}", latency_ms)
    
    def get_counter(self, metric: str) -> int:
        """Get counter value"""
        with self.lock:
            return self.counters.get(metric, 0)
    
    def get_gauge(self, metric: str) -> float:
        """Get gauge value"""
        with self.lock:
            return self.gauges.get(metric, 0.0)
    
    def get_timer_stats(self, metric: str) -> Dict[str, float]:
        """Get timer statistics (min, max, avg, p95, p99)"""
        with self.lock:
            values = self.timers.get(metric, [])
            if not values:
                return {"count": 0}
            
            sorted_values = sorted(values)
            count = len(sorted_values)
            
            return {
                "count": count,
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "avg": sum(sorted_values) / count,
                "p50": sorted_values[int(count * 0.5)],
                "p95": sorted_values[int(count * 0.95)],
                "p99": sorted_values[int(count * 0.99)]
            }
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics"""
        with self.lock:
            total_trades = len(self.trades)
            if total_trades == 0:
                return {"total_trades": 0}
            
            winning_trades = self.counters.get('trades_winning', 0)
            losing_trades = self.counters.get('trades_losing', 0)
            
            pnl_values = [t.get('pnl', 0) for t in self.trades if 'pnl' in t]
            winning_pnl = [p for p in pnl_values if p > 0]
            losing_pnl = [p for p in pnl_values if p < 0]
            
            total_pnl = sum(pnl_values)
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
                "total_pnl": total_pnl,
                "avg_win": sum(winning_pnl) / len(winning_pnl) if winning_pnl else 0,
                "avg_loss": sum(losing_pnl) / len(losing_pnl) if losing_pnl else 0,
                "profit_factor": abs(sum(winning_pnl) / sum(losing_pnl)) if losing_pnl and sum(losing_pnl) != 0 else 0,
                "largest_win": max(winning_pnl) if winning_pnl else 0,
                "largest_loss": min(losing_pnl) if losing_pnl else 0,
                "buy_trades": self.counters.get('trades_buy', 0),
                "sell_trades": self.counters.get('trades_sell', 0)
            }
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API call statistics"""
        with self.lock:
            stats = {}
            for endpoint, data in self.api_calls.items():
                if data["count"] > 0:
                    stats[endpoint] = {
                        "total_calls": data["count"],
                        "errors": data["errors"],
                        "error_rate": data["errors"] / data["count"],
                        "avg_latency_ms": data["total_latency"] / data["count"]
                    }
            return stats
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics in one snapshot"""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "trading_stats": self.get_trading_stats(),
            "api_stats": self.get_api_stats(),
            "timestamp": datetime.now().isoformat()
        }
    
    def reset_counters(self):
        """Reset all counters (useful for periodic reporting)"""
        with self.lock:
            self.counters.clear()
    
    def cleanup_old_data(self):
        """Remove data older than retention period"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
            
            # Clean histograms
            for metric, values in self.histograms.items():
                while values and values[0]['timestamp'] < cutoff_time:
                    values.popleft()
            
            # Clean trades
            self.trades = [t for t in self.trades if t['timestamp'] > cutoff_time]


# Global metrics collector
_metrics_instance: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get or create global metrics collector"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


if __name__ == "__main__":
    # Test metrics collector
    metrics = get_metrics()
    
    # Record some data
    metrics.increment("total_trades")
    metrics.set_gauge("current_equity", 10500.50)
    metrics.record_timer("order_execution", 125.5)
    metrics.record_trade({
        "symbol": "BTCUSDT",
        "side": "BUY",
        "pnl": 50.0
    })
    metrics.record_api_call("/api/v1/order", 45.2, success=True)
    
    # Get stats
    print("Trading Stats:", metrics.get_trading_stats())
    print("API Stats:", metrics.get_api_stats())
    print("All Metrics:", metrics.get_all_metrics())
