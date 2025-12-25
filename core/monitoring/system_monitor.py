"""
System Resource Monitor
Tracks CPU, memory, and network performance.
"""
import logging
import psutil
import threading
import time
from typing import Dict, Optional
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Monitors system resources.
    
    Features:
    - CPU usage tracking
    - Memory usage monitoring
    - Disk space alerts
    - Network latency tracking
    """
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize system monitor.
        
        Args:
            check_interval: Monitoring interval in seconds
        """
        self.check_interval = check_interval
        self.is_running = False
        self.monitor_thread = None
        
        # Resource history
        self.cpu_history = deque(maxlen=60)  # Last hour
        self.memory_history = deque(maxlen=60)
        
        self.current_stats = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'disk_percent': 0,
            'process_memory_mb': 0
        }
        
        logger.info(f"System Monitor initialized (interval: {check_interval}s)")
    
    def start(self):
        """Start monitoring in background thread."""
        if self.is_running:
            logger.warning("System Monitor already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System Monitor started")
    
    def stop(self):
        """Stop monitoring."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System Monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                self._collect_metrics()
                self._check_thresholds()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
    
    def _collect_metrics(self):
        """Collect system metrics."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_history.append((datetime.now(), cpu_percent))
        
        # Memory
        memory = psutil.virtual_memory()
        self.memory_history.append((datetime.now(), memory.percent))
        
        # Disk
        disk = psutil.disk_usage('/')
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.current_stats = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / 1024 / 1024 / 1024,
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / 1024 / 1024 / 1024,
            'process_memory_mb': process_memory,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.debug(f"METRICS: CPU={cpu_percent:.1f}% MEM={memory.percent:.1f}% DISK={disk.percent:.1f}%")
    
    def _check_thresholds(self):
        """Check resource thresholds and alert if exceeded."""
        stats = self.current_stats
        
        # High CPU warning
        if stats['cpu_percent'] > 90:
            logger.warning(f"⚠️  HIGH CPU USAGE: {stats['cpu_percent']:.1f}%")
        
        # High memory warning
        if stats['memory_percent'] > 85:
            logger.warning(f"⚠️  HIGH MEMORY USAGE: {stats['memory_percent']:.1f}%")
        
        # Low disk space
        if stats['disk_percent'] > 90:
            logger.critical(f"🚨 LOW DISK SPACE: {stats['disk_percent']:.1f}% used")
        
        # Process memory leak detection (> 1GB)
        if stats['process_memory_mb'] > 1024:
            logger.warning(f"⚠️  Bot process using {stats['process_memory_mb']:.0f}MB memory")
    
    def get_stats(self) -> Dict:
        """Get current system statistics."""
        return self.current_stats.copy()
    
    def get_average_cpu(self, minutes: int = 5) -> float:
        """Get average CPU usage over last N minutes."""
        if not self.cpu_history:
            return 0.0
        
        recent = [cpu for ts, cpu in self.cpu_history if (datetime.now() - ts).seconds < minutes * 60]
        return sum(recent) / len(recent) if recent else 0.0
    
    def get_average_memory(self, minutes: int = 5) -> float:
        """Get average memory usage over last N minutes."""
        if not self.memory_history:
            return 0.0
        
        recent = [mem for ts, mem in self.memory_history if (datetime.now() - ts).seconds < minutes * 60]
        return sum(recent) / len(recent) if recent else 0.0


# Global monitor instance
_system_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """Get or create system monitor instance."""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
        _system_monitor.start()
    return _system_monitor
