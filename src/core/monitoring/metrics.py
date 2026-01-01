"""
Metrics Module
Provides metrics collection and tracking for the trading system.
"""

from typing import Dict, Optional, List
from collections import defaultdict
import time
import threading


class MetricsCollector:
    """
    Simple metrics collector for trading system monitoring.
    Tracks counters, gauges, and histograms.
    """
    
    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment(self, name: str, value: int = 1, labels: Dict = None):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Dict = None):
        """Set a gauge metric value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
    
    def observe(self, name: str, value: float, labels: Dict = None):
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)
            # Keep only last 1000 observations per metric
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
    
    def _make_key(self, name: str, labels: Dict = None) -> str:
        """Create a unique key for the metric."""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def get_counter(self, name: str, labels: Dict = None) -> int:
        """Get current counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)
    
    def get_gauge(self, name: str, labels: Dict = None) -> float:
        """Get current gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0.0)
    
    def get_histogram_stats(self, name: str, labels: Dict = None) -> Dict:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])
        
        if not values:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}
        
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values)
        }
    
    def get_all_metrics(self) -> Dict:
        """Get all collected metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: self.get_histogram_stats(k) for k in self._histograms}
            }
    
    # Convenience aliases
    def inc(self, name: str, value: int = 1, labels: Dict = None):
        """Alias for increment."""
        self.increment(name, value, labels)
    
    def set(self, name: str, value: float, labels: Dict = None):
        """Alias for set_gauge."""
        self.set_gauge(name, value, labels)


# Singleton instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """
    Get the singleton metrics collector instance.
    
    Returns:
        MetricsCollector instance
    """
    global _metrics
    
    if _metrics is None:
        _metrics = MetricsCollector()
    
    return _metrics


__all__ = ['get_metrics', 'MetricsCollector']
