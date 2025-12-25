"""
System Optimization - Enterprise Features #330, #335, #340, #345
Latency Monitor, Memory Profiler, CPU Tracker, Connection Pool.
"""

import logging
import time
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)


class LatencyMonitor:
    """
    Feature #330: Latency Monitor
    
    Monitors system and API latency.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize latency monitor.
        
        Args:
            window_size: Number of samples to keep
        """
        self.window_size = window_size
        self.latencies: Dict[str, deque] = {}
        
        logger.info("Latency Monitor initialized")
    
    def record(self, operation: str, latency_ms: float):
        """Record a latency measurement."""
        if operation not in self.latencies:
            self.latencies[operation] = deque(maxlen=self.window_size)
        
        self.latencies[operation].append({
            'latency_ms': latency_ms,
            'timestamp': datetime.now().isoformat()
        })
    
    def measure(self, operation: str):
        """Decorator to measure function latency."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    latency = (time.time() - start) * 1000
                    self.record(operation, latency)
            return wrapper
        return decorator
    
    def get_stats(self, operation: str) -> Dict:
        """Get latency statistics for an operation."""
        if operation not in self.latencies or not self.latencies[operation]:
            return {'samples': 0}
        
        values = [l['latency_ms'] for l in self.latencies[operation]]
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            'operation': operation,
            'samples': n,
            'avg_ms': round(sum(values) / n, 2),
            'min_ms': round(min(values), 2),
            'max_ms': round(max(values), 2),
            'p50_ms': round(sorted_values[n // 2], 2),
            'p95_ms': round(sorted_values[int(n * 0.95)], 2) if n >= 20 else None,
            'p99_ms': round(sorted_values[int(n * 0.99)], 2) if n >= 100 else None
        }
    
    def get_all_stats(self) -> Dict:
        """Get stats for all operations."""
        return {op: self.get_stats(op) for op in self.latencies}


class MemoryProfiler:
    """
    Feature #335: Memory Profiler
    
    Tracks memory usage over time.
    """
    
    def __init__(self, sample_interval: int = 60):
        """
        Initialize memory profiler.
        
        Args:
            sample_interval: Seconds between samples
        """
        self.sample_interval = sample_interval
        self.samples: List[Dict] = []
        self.running = False
        self._thread: Optional[threading.Thread] = None
        
        logger.info("Memory Profiler initialized")
    
    def get_current_usage(self) -> Dict:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            
            return {
                'rss_mb': round(mem_info.rss / (1024 * 1024), 2),
                'vms_mb': round(mem_info.vms / (1024 * 1024), 2),
                'percent': round(process.memory_percent(), 2),
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            return {'error': 'psutil not installed'}
    
    def sample(self):
        """Take a memory sample."""
        usage = self.get_current_usage()
        if 'error' not in usage:
            self.samples.append(usage)
            self.samples = self.samples[-1000:]
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        self.running = True
        
        def monitor():
            while self.running:
                self.sample()
                time.sleep(self.sample_interval)
        
        self._thread = threading.Thread(target=monitor, daemon=True)
        self._thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.running = False
    
    def get_trend(self) -> Dict:
        """Analyze memory usage trend."""
        if len(self.samples) < 10:
            return {'trend': 'insufficient_data'}
        
        recent = self.samples[-10:]
        older = self.samples[:10]
        
        recent_avg = sum(s['rss_mb'] for s in recent) / len(recent)
        older_avg = sum(s['rss_mb'] for s in older) / len(older)
        
        change_pct = (recent_avg - older_avg) / older_avg * 100 if older_avg > 0 else 0
        
        return {
            'current_mb': recent[-1]['rss_mb'],
            'avg_mb': round(recent_avg, 2),
            'change_pct': round(change_pct, 2),
            'trend': 'increasing' if change_pct > 10 else 'stable' if change_pct > -10 else 'decreasing',
            'samples': len(self.samples)
        }


class CPUUsageTracker:
    """
    Feature #340: CPU Usage Tracker
    
    Monitors CPU utilization.
    """
    
    def __init__(self):
        """Initialize CPU tracker."""
        self.samples: List[Dict] = []
        
        logger.info("CPU Usage Tracker initialized")
    
    def get_current_usage(self) -> Dict:
        """Get current CPU usage."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            process = psutil.Process(os.getpid())
            process_cpu = process.cpu_percent()
            
            return {
                'system_cpu_pct': cpu_percent,
                'process_cpu_pct': process_cpu,
                'cpu_count': psutil.cpu_count(),
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            return {'error': 'psutil not installed'}
    
    def sample(self):
        """Take a CPU sample."""
        usage = self.get_current_usage()
        if 'error' not in usage:
            self.samples.append(usage)
            self.samples = self.samples[-1000:]
    
    def get_stats(self) -> Dict:
        """Get CPU usage statistics."""
        if not self.samples:
            return {'samples': 0}
        
        process_values = [s['process_cpu_pct'] for s in self.samples]
        system_values = [s['system_cpu_pct'] for s in self.samples]
        
        return {
            'current': self.samples[-1] if self.samples else {},
            'avg_process_cpu': round(sum(process_values) / len(process_values), 2),
            'avg_system_cpu': round(sum(system_values) / len(system_values), 2),
            'max_process_cpu': round(max(process_values), 2),
            'max_system_cpu': round(max(system_values), 2),
            'samples': len(self.samples)
        }


class ConnectionPoolManager:
    """
    Feature #345: Connection Pool Manager
    
    Manages pooled connections for efficiency.
    """
    
    def __init__(
        self,
        pool_size: int = 10,
        max_idle_time: int = 300
    ):
        """
        Initialize connection pool.
        
        Args:
            pool_size: Maximum pool size
            max_idle_time: Max idle time before eviction (seconds)
        """
        self.pool_size = pool_size
        self.max_idle_time = max_idle_time
        
        self.pool: Dict[str, List[Dict]] = {}  # name -> connections
        self.in_use: Dict[str, int] = {}  # name -> count
        self._lock = threading.Lock()
        
        logger.info(f"Connection Pool initialized - Size: {pool_size}")
    
    def create_pool(self, name: str, factory: Callable):
        """Create a new connection pool."""
        with self._lock:
            self.pool[name] = []
            self.in_use[name] = 0
            
            # Pre-populate pool
            for _ in range(min(3, self.pool_size)):
                try:
                    conn = factory()
                    self.pool[name].append({
                        'connection': conn,
                        'created_at': datetime.now(),
                        'last_used': datetime.now()
                    })
                except Exception as e:
                    logger.error(f"Failed to create connection: {e}")
    
    def acquire(self, name: str) -> Optional[object]:
        """Acquire a connection from pool."""
        with self._lock:
            if name not in self.pool:
                return None
            
            pool = self.pool[name]
            
            # Find available connection
            now = datetime.now()
            for entry in pool:
                if 'in_use' not in entry or not entry['in_use']:
                    # Check idle time
                    idle = (now - entry['last_used']).total_seconds()
                    if idle > self.max_idle_time:
                        # Connection too old, skip
                        continue
                    
                    entry['in_use'] = True
                    entry['last_used'] = now
                    self.in_use[name] += 1
                    
                    return entry['connection']
            
            return None
    
    def release(self, name: str, connection: object):
        """Release a connection back to pool."""
        with self._lock:
            if name not in self.pool:
                return
            
            for entry in self.pool[name]:
                if entry['connection'] is connection:
                    entry['in_use'] = False
                    entry['last_used'] = datetime.now()
                    self.in_use[name] = max(0, self.in_use[name] - 1)
                    break
    
    def get_stats(self, name: Optional[str] = None) -> Dict:
        """Get pool statistics."""
        if name:
            pool = self.pool.get(name, [])
            return {
                'name': name,
                'total': len(pool),
                'in_use': self.in_use.get(name, 0),
                'available': len([e for e in pool if not e.get('in_use')])
            }
        
        return {
            pool_name: {
                'total': len(pool),
                'in_use': self.in_use.get(pool_name, 0)
            }
            for pool_name, pool in self.pool.items()
        }


# Singletons
_latency: Optional[LatencyMonitor] = None
_memory: Optional[MemoryProfiler] = None
_cpu: Optional[CPUUsageTracker] = None
_pool: Optional[ConnectionPoolManager] = None


def get_latency_monitor() -> LatencyMonitor:
    global _latency
    if _latency is None:
        _latency = LatencyMonitor()
    return _latency


def get_memory_profiler() -> MemoryProfiler:
    global _memory
    if _memory is None:
        _memory = MemoryProfiler()
    return _memory


def get_cpu_tracker() -> CPUUsageTracker:
    global _cpu
    if _cpu is None:
        _cpu = CPUUsageTracker()
    return _cpu


def get_connection_pool() -> ConnectionPoolManager:
    global _pool
    if _pool is None:
        _pool = ConnectionPoolManager()
    return _pool


if __name__ == '__main__':
    # Test latency monitor
    latency = LatencyMonitor()
    
    @latency.measure('test_op')
    def slow_function():
        time.sleep(0.1)
        return "done"
    
    for _ in range(10):
        slow_function()
    
    print(f"Latency stats: {latency.get_stats('test_op')}")
    
    # Test memory
    mem = MemoryProfiler()
    print(f"Memory: {mem.get_current_usage()}")
    
    # Test CPU
    cpu = CPUUsageTracker()
    print(f"CPU: {cpu.get_current_usage()}")
