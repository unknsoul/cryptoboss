"""
Centralized Observability - Upgrade G

Structured logging with Loguru, metrics collection, health endpoint.

Features:
- Structured JSON logging
- Log levels configurable per component
- Prometheus metrics export
- Health check endpoint
- Log aggregation ready
"""

import sys
import json
import time
from typing import Dict, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
from functools import wraps
import threading

# Use loguru for structured logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class Metrics:
    """Simple metrics collector (Prometheus-compatible)."""
    
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()
    
    def inc(self, name: str, value: int = 1, labels: Dict = None):
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value
    
    def set(self, name: str, value: float, labels: Dict = None):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
    
    def observe(self, name: str, value: float, labels: Dict = None):
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
            # Keep only last 1000 observations
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
    
    def _make_key(self, name: str, labels: Dict = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_metrics(self) -> Dict:
        """Get all metrics in Prometheus format-like dict."""
        with self._lock:
            return {
                "counters": self._counters.copy(),
                "gauges": self._gauges.copy(),
                "histograms": {
                    k: {
                        "count": len(v),
                        "sum": sum(v),
                        "avg": sum(v) / len(v) if v else 0,
                        "min": min(v) if v else 0,
                        "max": max(v) if v else 0
                    }
                    for k, v in self._histograms.items()
                },
                "uptime_seconds": time.time() - self._start_time
            }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        for name, value in self._counters.items():
            lines.append(f"# TYPE {name.split('{')[0]} counter")
            lines.append(f"{name} {value}")
        
        for name, value in self._gauges.items():
            lines.append(f"# TYPE {name.split('{')[0]} gauge")
            lines.append(f"{name} {value}")
        
        return "\n".join(lines)


class ObservabilityManager:
    """
    Central observability for the trading system.
    
    Usage:
        obs = ObservabilityManager()
        obs.setup()
        
        # Structured logging
        obs.log("info", "Order placed", order_id="123", symbol="BTCUSDT")
        
        # Metrics
        obs.metrics.inc("orders_total", labels={"side": "buy"})
        obs.metrics.observe("order_latency_ms", 45.2)
        
        # Health check
        health = obs.get_health()
        
        # Timing decorator
        @obs.timed("strategy_signal")
        def generate_signal():
            ...
    """
    
    def __init__(
        self,
        service_name: str = "cryptoboss",
        log_level: str = "INFO",
        log_file: str = "logs/cryptoboss.log",
        json_logs: bool = True
    ):
        self.service_name = service_name
        self.log_level = log_level
        self.log_file = log_file
        self.json_logs = json_logs
        
        self.metrics = Metrics()
        self._health_checks: Dict[str, Callable] = {}
        self._component_status: Dict[str, str] = {}
        
        self._setup_done = False
    
    def setup(self):
        """Setup logging and metrics."""
        if self._setup_done:
            return
        
        # Ensure log directory exists
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            from loguru import logger as loguru_logger
            
            # Remove default handler
            loguru_logger.remove()
            
            # Add console handler
            if self.json_logs:
                loguru_logger.add(
                    sys.stderr,
                    format=self._json_format,
                    level=self.log_level,
                    serialize=False
                )
            else:
                loguru_logger.add(
                    sys.stderr,
                    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                    level=self.log_level
                )
            
            # Add file handler with rotation
            loguru_logger.add(
                self.log_file,
                rotation="100 MB",
                retention="7 days",
                compression="gz",
                level=self.log_level,
                serialize=self.json_logs
            )
            
            self._logger = loguru_logger
            
        except ImportError:
            # Fallback to standard logging
            import logging
            logging.basicConfig(
                level=getattr(logging, self.log_level),
                format='%(asctime)s | %(levelname)s | %(name)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(self.log_file)
                ]
            )
            self._logger = logging.getLogger(self.service_name)
        
        self._setup_done = True
        self.log("info", "Observability initialized", service=self.service_name)
    
    def _json_format(self, record):
        """Format log record as JSON."""
        log_dict = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "service": self.service_name,
            "module": record["name"],
            "function": record["function"],
            "line": record["line"]
        }
        
        # Add extra fields
        if record["extra"]:
            log_dict["extra"] = record["extra"]
        
        return json.dumps(log_dict, default=str) + "\n"
    
    def log(self, level: str, message: str, **kwargs):
        """Log with structured data."""
        if not self._setup_done:
            self.setup()
        
        # Add to metrics
        self.metrics.inc("logs_total", labels={"level": level})
        
        # Log message
        log_method = getattr(self._logger, level.lower(), self._logger.info)
        
        if kwargs:
            extra_str = " | " + " ".join(f"{k}={v}" for k, v in kwargs.items())
            log_method(message + extra_str)
        else:
            log_method(message)
    
    def timed(self, name: str):
        """Decorator to time function execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    self.metrics.inc(f"{name}_success_total")
                    return result
                except Exception as e:
                    self.metrics.inc(f"{name}_error_total")
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    self.metrics.observe(f"{name}_duration_ms", elapsed_ms)
            return wrapper
        return decorator
    
    def async_timed(self, name: str):
        """Decorator to time async function execution."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    self.metrics.inc(f"{name}_success_total")
                    return result
                except Exception as e:
                    self.metrics.inc(f"{name}_error_total")
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    self.metrics.observe(f"{name}_duration_ms", elapsed_ms)
            return wrapper
        return decorator
    
    def register_health_check(self, name: str, check: Callable[[], bool]):
        """Register a health check function."""
        self._health_checks[name] = check
    
    def set_component_status(self, component: str, status: str):
        """Set component status."""
        self._component_status[component] = status
        self.metrics.set("component_status", 1 if status == "healthy" else 0, 
                         labels={"component": component})
    
    def get_health(self) -> Dict:
        """Get system health status."""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "checks": {},
            "components": self._component_status.copy(),
            "metrics_summary": {
                "uptime_seconds": self.metrics.get_metrics()["uptime_seconds"]
            }
        }
        
        # Run health checks
        for name, check in self._health_checks.items():
            try:
                passed = check()
                health["checks"][name] = "pass" if passed else "fail"
                if not passed:
                    health["status"] = "unhealthy"
            except Exception as e:
                health["checks"][name] = f"error: {e}"
                health["status"] = "unhealthy"
        
        return health
    
    def get_metrics_endpoint(self) -> str:
        """Get metrics in Prometheus format for /metrics endpoint."""
        return self.metrics.export_prometheus()


# Singleton
_observability: Optional[ObservabilityManager] = None

def get_observability(service_name: str = "cryptoboss") -> ObservabilityManager:
    global _observability
    if _observability is None:
        _observability = ObservabilityManager(service_name=service_name)
        _observability.setup()
    return _observability


# Convenience functions
def log_info(message: str, **kwargs):
    get_observability().log("info", message, **kwargs)

def log_warning(message: str, **kwargs):
    get_observability().log("warning", message, **kwargs)

def log_error(message: str, **kwargs):
    get_observability().log("error", message, **kwargs)

def log_debug(message: str, **kwargs):
    get_observability().log("debug", message, **kwargs)
