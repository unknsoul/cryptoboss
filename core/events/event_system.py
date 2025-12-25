"""
Alert System & Event Handling - Features #232-248
"""
import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4

class AlertManager:
    """Feature #232: Central Alert Manager"""
    def __init__(self):
        self.alerts: List[Dict] = []
        self.handlers: List[Callable] = []
    
    def add_handler(self, handler: Callable):
        self.handlers.append(handler)
    
    def trigger(self, level: AlertLevel, message: str, data: Optional[Dict] = None):
        alert = {'level': level.name, 'message': message, 'data': data, 'time': datetime.now().isoformat()}
        self.alerts.append(alert)
        self.alerts = self.alerts[-1000:]
        for handler in self.handlers:
            try:
                handler(alert)
            except: pass

class PriceAlertSystem:
    """Feature #233: Price Alert System"""
    def __init__(self):
        self.alerts: List[Dict] = []
    
    def add_alert(self, symbol: str, price: float, direction: str, callback: Callable):
        self.alerts.append({'symbol': symbol, 'price': price, 'direction': direction, 'callback': callback, 'triggered': False})
    
    def check(self, symbol: str, current_price: float):
        for alert in self.alerts:
            if alert['symbol'] == symbol and not alert['triggered']:
                if (alert['direction'] == 'above' and current_price >= alert['price']) or \
                   (alert['direction'] == 'below' and current_price <= alert['price']):
                    alert['triggered'] = True
                    alert['callback'](symbol, current_price)

class IndicatorAlertSystem:
    """Feature #234: Indicator Alert System"""
    def __init__(self):
        self.thresholds: Dict[str, Dict] = {}
    
    def set_threshold(self, indicator: str, low: float, high: float, callback: Callable):
        self.thresholds[indicator] = {'low': low, 'high': high, 'callback': callback}
    
    def check(self, indicator: str, value: float):
        if indicator in self.thresholds:
            t = self.thresholds[indicator]
            if value <= t['low'] or value >= t['high']:
                t['callback'](indicator, value)

class VolumeAlertSystem:
    """Feature #235: Volume Spike Alert"""
    def __init__(self, spike_threshold: float = 3.0):
        self.threshold = spike_threshold
        self.avg_volume = 0
        self.callbacks: List[Callable] = []
    
    def update_avg(self, avg: float):
        self.avg_volume = avg
    
    def check(self, current_volume: float):
        if self.avg_volume > 0 and current_volume > self.avg_volume * self.threshold:
            for cb in self.callbacks:
                cb(current_volume, self.avg_volume)

class DrawdownAlert:
    """Feature #236: Drawdown Alert"""
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.peak = 0
        self.callbacks: List[Callable] = []
    
    def update(self, equity: float):
        self.peak = max(self.peak, equity)
        dd = (self.peak - equity) / self.peak if self.peak > 0 else 0
        if dd >= self.threshold:
            for cb in self.callbacks:
                cb(dd, equity)

class ConnectionAlert:
    """Feature #237: Connection Status Alert"""
    def __init__(self):
        self.connected = True
        self.last_ping = datetime.now()
        self.callbacks: List[Callable] = []
    
    def ping(self):
        self.last_ping = datetime.now()
        if not self.connected:
            self.connected = True
            for cb in self.callbacks:
                cb('reconnected')
    
    def check_timeout(self, timeout_seconds: int = 30):
        if (datetime.now() - self.last_ping).total_seconds() > timeout_seconds:
            if self.connected:
                self.connected = False
                for cb in self.callbacks:
                    cb('disconnected')

class TradeExecutionAlert:
    """Feature #238: Trade Execution Alert"""
    def __init__(self):
        self.callbacks: List[Callable] = []
    
    def on_fill(self, trade: Dict):
        for cb in self.callbacks:
            cb('filled', trade)
    
    def on_reject(self, order: Dict, reason: str):
        for cb in self.callbacks:
            cb('rejected', {'order': order, 'reason': reason})

class EventBus:
    """Feature #239: Central Event Bus"""
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, callback: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def publish(self, event_type: str, data: Dict):
        for callback in self.subscribers.get(event_type, []):
            try:
                callback(data)
            except: pass

class EventLogger:
    """Feature #240: Event Logger"""
    def __init__(self):
        self.events: List[Dict] = []
    
    def log(self, event_type: str, data: Dict):
        self.events.append({'type': event_type, 'data': data, 'time': datetime.now().isoformat()})
        self.events = self.events[-10000:]
    
    def get_events(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        filtered = [e for e in self.events if not event_type or e['type'] == event_type]
        return filtered[-limit:]

class EventAggregator:
    """Feature #241: Event Aggregator"""
    def __init__(self, window_size: int = 60):
        self.window = window_size
        self.events: List[Dict] = []
    
    def add(self, event_type: str):
        self.events.append({'type': event_type, 'time': datetime.now()})
        cutoff = datetime.now()
        self.events = [e for e in self.events if (cutoff - e['time']).seconds < self.window]
    
    def count(self, event_type: str) -> int:
        return sum(1 for e in self.events if e['type'] == event_type)

class StateManager:
    """Feature #242: State Manager"""
    def __init__(self):
        self.state: Dict[str, any] = {}
        self.history: List[Dict] = []
    
    def set(self, key: str, value):
        old = self.state.get(key)
        self.state[key] = value
        if old != value:
            self.history.append({'key': key, 'old': old, 'new': value, 'time': datetime.now().isoformat()})
    
    def get(self, key: str, default=None):
        return self.state.get(key, default)

class WorkflowEngine:
    """Feature #243: Workflow Engine"""
    def __init__(self):
        self.workflows: Dict[str, List[Callable]] = {}
    
    def define(self, name: str, steps: List[Callable]):
        self.workflows[name] = steps
    
    def execute(self, name: str, context: Dict) -> Dict:
        if name not in self.workflows:
            return {'error': 'Workflow not found'}
        for step in self.workflows[name]:
            context = step(context)
        return context

class ScheduledTaskRunner:
    """Feature #244: Scheduled Task Runner"""
    def __init__(self):
        self.tasks: List[Dict] = []
    
    def schedule(self, name: str, interval_seconds: int, task: Callable):
        self.tasks.append({'name': name, 'interval': interval_seconds, 'task': task, 'last_run': None})
    
    def tick(self):
        now = datetime.now()
        for task in self.tasks:
            if task['last_run'] is None or (now - task['last_run']).seconds >= task['interval']:
                task['task']()
                task['last_run'] = now

class RetryManager:
    """Feature #245: Retry Manager"""
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
    
    def execute(self, func: Callable, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
        return None

class CircuitBreaker:
    """Feature #246: Circuit Breaker Pattern"""
    def __init__(self, failure_threshold: int = 5, recovery_time: int = 60):
        self.threshold = failure_threshold
        self.recovery = recovery_time
        self.failures = 0
        self.state = 'closed'
        self.last_failure = None
    
    def record_failure(self):
        self.failures += 1
        self.last_failure = datetime.now()
        if self.failures >= self.threshold:
            self.state = 'open'
    
    def record_success(self):
        self.failures = 0
        self.state = 'closed'
    
    def is_open(self) -> bool:
        if self.state == 'open' and self.last_failure:
            if (datetime.now() - self.last_failure).seconds > self.recovery:
                self.state = 'half-open'
                return False
        return self.state == 'open'

class HealthMonitor:
    """Feature #247: Health Monitor"""
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.status: Dict[str, str] = {}
    
    def register(self, name: str, check: Callable):
        self.checks[name] = check
    
    def run_checks(self) -> Dict:
        for name, check in self.checks.items():
            try:
                self.status[name] = 'healthy' if check() else 'unhealthy'
            except:
                self.status[name] = 'error'
        return self.status

class MetricsCollector:
    """Feature #248: Metrics Collector"""
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        self.metrics[name] = self.metrics[name][-1000:]
    
    def get_avg(self, name: str) -> float:
        values = self.metrics.get(name, [])
        return sum(values) / len(values) if values else 0

# Factories
def get_alert_manager(): return AlertManager()
def get_price_alerts(): return PriceAlertSystem()
def get_indicator_alerts(): return IndicatorAlertSystem()
def get_volume_alerts(): return VolumeAlertSystem()
def get_dd_alerts(): return DrawdownAlert()
def get_connection_alerts(): return ConnectionAlert()
def get_trade_alerts(): return TradeExecutionAlert()
def get_event_bus(): return EventBus()
def get_event_logger(): return EventLogger()
def get_event_aggregator(): return EventAggregator()
def get_state_manager(): return StateManager()
def get_workflow_engine(): return WorkflowEngine()
def get_task_runner(): return ScheduledTaskRunner()
def get_retry_manager(): return RetryManager()
def get_circuit_breaker(): return CircuitBreaker()
def get_health_monitor(): return HealthMonitor()
def get_metrics_collector(): return MetricsCollector()
