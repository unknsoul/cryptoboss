"""
Deployment & DevOps - Enterprise Features #310, #315, #318, #325
Health Checks, Metrics, Config Reload, and Graceful Shutdown.
"""

import logging
import signal
import sys
import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)


class HealthCheckSystem:
    """
    Feature #310: Health Check Endpoint
    
    Provides health status for monitoring and load balancers.
    """
    
    def __init__(self):
        """Initialize health check system."""
        self.checks: Dict[str, Callable[[], Dict]] = {}
        self.last_check_time: Optional[datetime] = None
        self.check_interval = 30  # seconds
        self.cached_status: Optional[Dict] = None
        
        # Register default checks
        self._register_defaults()
        
        logger.info("Health Check System initialized")
    
    def _register_defaults(self):
        """Register default health checks."""
        self.register('system', self._check_system)
        self.register('memory', self._check_memory)
    
    def _check_system(self) -> Dict:
        """Check system health."""
        return {
            'status': 'healthy',
            'uptime_seconds': self._get_uptime(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_memory(self) -> Dict:
        """Check memory usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                'status': 'healthy' if mem.percent < 90 else 'degraded',
                'used_percent': mem.percent,
                'available_gb': round(mem.available / (1024**3), 2)
            }
        except ImportError:
            return {'status': 'unknown', 'reason': 'psutil not installed'}
    
    def _get_uptime(self) -> int:
        """Get process uptime in seconds."""
        try:
            import psutil
            p = psutil.Process(os.getpid())
            return int(time.time() - p.create_time())
        except:
            return 0
    
    def register(self, name: str, check_fn: Callable[[], Dict]):
        """Register a health check."""
        self.checks[name] = check_fn
        logger.debug(f"Registered health check: {name}")
    
    def check_all(self, force: bool = False) -> Dict:
        """
        Run all health checks.
        
        Args:
            force: Force check even if cached
            
        Returns:
            Overall health status
        """
        now = datetime.now()
        
        # Use cache if recent
        if not force and self.cached_status and self.last_check_time:
            if (now - self.last_check_time).total_seconds() < self.check_interval:
                return self.cached_status
        
        results = {}
        overall_healthy = True
        
        for name, check_fn in self.checks.items():
            try:
                result = check_fn()
                results[name] = result
                if result.get('status') not in ['healthy', 'ok']:
                    overall_healthy = False
            except Exception as e:
                results[name] = {'status': 'error', 'error': str(e)}
                overall_healthy = False
        
        status = {
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'timestamp': now.isoformat(),
            'checks': results
        }
        
        self.cached_status = status
        self.last_check_time = now
        
        return status
    
    def get_liveness(self) -> Dict:
        """Simple liveness check."""
        return {'status': 'alive', 'timestamp': datetime.now().isoformat()}
    
    def get_readiness(self) -> Dict:
        """Readiness check for accepting traffic."""
        status = self.check_all()
        return {
            'ready': status['status'] == 'healthy',
            'status': status['status']
        }


class MetricsExporter:
    """
    Feature #315: Metrics Exporter (Prometheus format)
    
    Exports metrics in Prometheus-compatible format.
    """
    
    def __init__(self, prefix: str = 'trading_bot'):
        """
        Initialize metrics exporter.
        
        Args:
            prefix: Metric name prefix
        """
        self.prefix = prefix
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.labels: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        
        logger.info(f"Metrics Exporter initialized - Prefix: {prefix}")
    
    def inc_counter(self, name: str, value: int = 1, labels: Optional[Dict] = None):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + value
            if labels:
                self.labels[key] = labels
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict] = None):
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self.gauges[key] = value
            if labels:
                self.labels[key] = labels
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict] = None):
        """Add observation to histogram."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self.histograms:
                self.histograms[key] = []
            self.histograms[key].append(value)
            self.histograms[key] = self.histograms[key][-1000:]  # Keep last 1000
            if labels:
                self.labels[key] = labels
    
    def _make_key(self, name: str, labels: Optional[Dict]) -> str:
        """Create unique key for metric."""
        if labels:
            label_str = ','.join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []
        
        # Counters
        for key, value in self.counters.items():
            name = key.split('{')[0]
            lines.append(f"# TYPE {self.prefix}_{name} counter")
            lines.append(f"{self.prefix}_{key} {value}")
        
        # Gauges
        for key, value in self.gauges.items():
            name = key.split('{')[0]
            lines.append(f"# TYPE {self.prefix}_{name} gauge")
            lines.append(f"{self.prefix}_{key} {value}")
        
        # Histograms (simplified - just count and sum)
        for key, values in self.histograms.items():
            name = key.split('{')[0]
            if values:
                lines.append(f"# TYPE {self.prefix}_{name} summary")
                lines.append(f"{self.prefix}_{name}_count{{{key.split('{')[1] if '{' in key else ''} {len(values)}")
                lines.append(f"{self.prefix}_{name}_sum{{{key.split('{')[1] if '{' in key else ''} {sum(values):.4f}")
        
        return '\n'.join(lines)
    
    def export_json(self) -> Dict:
        """Export metrics as JSON."""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': {k: {'count': len(v), 'sum': sum(v), 'avg': sum(v)/len(v) if v else 0}
                          for k, v in self.histograms.items()}
        }


class ConfigHotReload:
    """
    Feature #318: Config Hot-Reload
    
    Reloads configuration without restart.
    """
    
    def __init__(self, config_path: str = 'config.json'):
        """
        Initialize config reloader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Dict = {}
        self.last_modified: Optional[float] = None
        self.callbacks: List[Callable[[Dict], None]] = []
        self.watch_thread: Optional[threading.Thread] = None
        self.running = False
        
        logger.info(f"Config Hot-Reload initialized - Path: {config_path}")
    
    def load(self) -> Dict:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                self.last_modified = os.path.getmtime(self.config_path)
                logger.info(f"Config loaded: {len(self.config)} keys")
            else:
                self.config = {}
        except Exception as e:
            logger.error(f"Config load failed: {e}")
        
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def on_reload(self, callback: Callable[[Dict], None]):
        """Register reload callback."""
        self.callbacks.append(callback)
    
    def check_reload(self) -> bool:
        """Check and reload if file changed."""
        if not os.path.exists(self.config_path):
            return False
        
        current_mtime = os.path.getmtime(self.config_path)
        
        if self.last_modified is None or current_mtime > self.last_modified:
            old_config = self.config.copy()
            self.load()
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(self.config)
                except Exception as e:
                    logger.error(f"Config callback failed: {e}")
            
            logger.info("Config reloaded")
            return True
        
        return False
    
    def start_watching(self, interval: int = 5):
        """Start watching config file for changes."""
        self.running = True
        
        def watch():
            while self.running:
                self.check_reload()
                time.sleep(interval)
        
        self.watch_thread = threading.Thread(target=watch, daemon=True)
        self.watch_thread.start()
        logger.info(f"Config watcher started - Interval: {interval}s")
    
    def stop_watching(self):
        """Stop watching config file."""
        self.running = False


class GracefulShutdownHandler:
    """
    Feature #325: Graceful Shutdown Handler
    
    Handles clean shutdown on SIGTERM/SIGINT.
    """
    
    def __init__(self):
        """Initialize shutdown handler."""
        self.shutdown_callbacks: List[Callable[[], None]] = []
        self.is_shutting_down = False
        self.shutdown_timeout = 30  # seconds
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        logger.info("Graceful Shutdown Handler initialized")
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signal."""
        if self.is_shutting_down:
            logger.warning("Forced shutdown")
            sys.exit(1)
        
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self.shutdown()
    
    def register(self, callback: Callable[[], None], priority: int = 50):
        """
        Register shutdown callback.
        
        Args:
            callback: Function to call on shutdown
            priority: Lower = called first
        """
        self.shutdown_callbacks.append((priority, callback))
        self.shutdown_callbacks.sort(key=lambda x: x[0])
    
    def shutdown(self):
        """Execute graceful shutdown."""
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        logger.info(f"Running {len(self.shutdown_callbacks)} shutdown callbacks...")
        
        for priority, callback in self.shutdown_callbacks:
            try:
                logger.debug(f"Running shutdown callback (priority {priority})")
                callback()
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")
        
        logger.info("Graceful shutdown complete")
    
    def is_running(self) -> bool:
        """Check if system is still running (not shutting down)."""
        return not self.is_shutting_down


# Singletons
_health: Optional[HealthCheckSystem] = None
_metrics: Optional[MetricsExporter] = None
_config: Optional[ConfigHotReload] = None
_shutdown: Optional[GracefulShutdownHandler] = None


def get_health_check() -> HealthCheckSystem:
    global _health
    if _health is None:
        _health = HealthCheckSystem()
    return _health


def get_metrics() -> MetricsExporter:
    global _metrics
    if _metrics is None:
        _metrics = MetricsExporter()
    return _metrics


def get_config() -> ConfigHotReload:
    global _config
    if _config is None:
        _config = ConfigHotReload()
    return _config


def get_shutdown_handler() -> GracefulShutdownHandler:
    global _shutdown
    if _shutdown is None:
        _shutdown = GracefulShutdownHandler()
    return _shutdown


if __name__ == '__main__':
    # Test health checks
    health = HealthCheckSystem()
    status = health.check_all()
    print(f"Health: {status}")
    
    # Test metrics
    metrics = MetricsExporter()
    metrics.inc_counter('trades_total', labels={'side': 'buy'})
    metrics.set_gauge('equity', 10500)
    metrics.observe_histogram('trade_duration', 5.2)
    print(f"Metrics JSON: {metrics.export_json()}")
    
    # Test config
    config = ConfigHotReload()
    config.config = {'trading': {'enabled': True}}
    print(f"Config get: {config.get('trading.enabled')}")
