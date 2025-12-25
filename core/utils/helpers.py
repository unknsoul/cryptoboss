"""
Utility Functions & Helpers - Features #36-45
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
import time
import hashlib
import random

logger = logging.getLogger(__name__)

class RateLimiter:
    """Feature #36: Rate Limiter"""
    def __init__(self, max_calls: int, period_seconds: int):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: List[datetime] = []
    
    def can_call(self) -> bool:
        now = datetime.now()
        self.calls = [c for c in self.calls if (now - c).total_seconds() < self.period]
        return len(self.calls) < self.max_calls
    
    def record_call(self):
        self.calls.append(datetime.now())

class ExponentialBackoff:
    """Feature #37: Exponential Backoff"""
    def __init__(self, base: float = 1.0, max_delay: float = 60.0, multiplier: float = 2.0):
        self.base = base
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.attempt = 0
    
    def get_delay(self) -> float:
        delay = min(self.base * (self.multiplier ** self.attempt), self.max_delay)
        self.attempt += 1
        return delay
    
    def reset(self):
        self.attempt = 0

class Throttler:
    """Feature #38: Action Throttler"""
    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self.last_action: Optional[datetime] = None
    
    def should_run(self) -> bool:
        if self.last_action is None:
            return True
        return (datetime.now() - self.last_action).total_seconds() >= self.min_interval
    
    def record_action(self):
        self.last_action = datetime.now()

class TimeWindow:
    """Feature #39: Time Window Manager"""
    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self.items: List[Dict] = []
    
    def add(self, item: Any):
        self.items.append({'item': item, 'time': datetime.now()})
        self._cleanup()
    
    def _cleanup(self):
        cutoff = datetime.now() - timedelta(seconds=self.window)
        self.items = [i for i in self.items if i['time'] > cutoff]
    
    def get_all(self) -> List[Any]:
        self._cleanup()
        return [i['item'] for i in self.items]

class MovingWindow:
    """Feature #40: Moving Window Calculator"""
    def __init__(self, size: int = 20):
        self.size = size
        self.values: List[float] = []
    
    def add(self, value: float):
        self.values.append(value)
        self.values = self.values[-self.size:]
    
    def average(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0
    
    def std_dev(self) -> float:
        if len(self.values) < 2:
            return 0
        avg = self.average()
        variance = sum((v - avg) ** 2 for v in self.values) / len(self.values)
        return variance ** 0.5

class CircularBuffer:
    """Feature #41: Circular Buffer"""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: List[Any] = []
        self.index = 0
    
    def add(self, item: Any):
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.index] = item
        self.index = (self.index + 1) % self.capacity
    
    def get_all(self) -> List[Any]:
        return self.buffer

class RetryDecorator:
    """Feature #42: Retry Decorator"""
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise e
                    time.sleep(self.delay)
            return None
        return wrapper

class TimeoutManager:
    """Feature #43: Timeout Manager"""
    def __init__(self, default_timeout: float = 10.0):
        self.default_timeout = default_timeout
        self.timeouts: Dict[str, float] = {}
    
    def set_timeout(self, key: str, timeout: float):
        self.timeouts[key] = timeout
    
    def get_timeout(self, key: str) -> float:
        return self.timeouts.get(key, self.default_timeout)

class UniqueIdGenerator:
    """Feature #44: Unique ID Generator"""
    def __init__(self, prefix: str = ''):
        self.prefix = prefix
        self.counter = 0
    
    def generate(self) -> str:
        self.counter += 1
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"{self.prefix}{timestamp}{self.counter:05d}"
    
    def generate_hash(self, data: str) -> str:
        return hashlib.md5(f"{data}{random.random()}".encode()).hexdigest()[:12]

class DataValidator:
    """Feature #45: Data Validator"""
    def __init__(self):
        self.rules: Dict[str, Callable] = {}
    
    def add_rule(self, field: str, validator: Callable):
        self.rules[field] = validator
    
    def validate(self, data: Dict) -> Dict:
        errors = {}
        for field, validator in self.rules.items():
            if field in data:
                if not validator(data[field]):
                    errors[field] = 'Validation failed'
            else:
                errors[field] = 'Missing field'
        return {'valid': len(errors) == 0, 'errors': errors}

# Factories
def get_rate_limiter(max_calls: int, period: int): return RateLimiter(max_calls, period)
def get_backoff(): return ExponentialBackoff()
def get_throttler(): return Throttler()
def get_time_window(seconds: int = 60): return TimeWindow(seconds)
def get_moving_window(size: int = 20): return MovingWindow(size)
def get_circular_buffer(capacity: int = 1000): return CircularBuffer(capacity)
def get_retry_decorator(): return RetryDecorator()
def get_timeout_manager(): return TimeoutManager()
def get_id_generator(): return UniqueIdGenerator()
def get_validator(): return DataValidator()
