"""
Infrastructure Utilities - Enterprise Features #7, #14, #22, #26
Retry, Rate Limiting, Caching, and Trade Replay.
"""

import logging
import time
import functools
import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class ExponentialBackoff:
    """
    Feature #7: Retry with Exponential Backoff
    
    Retries failed operations with exponential delay:
    delay = base_delay * (multiplier ^ attempt) + jitter
    """
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        max_retries: int = 5,
        jitter: float = 0.1
    ):
        """
        Initialize backoff configuration.
        
        Args:
            base_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            multiplier: Delay multiplier per attempt
            max_retries: Maximum retry attempts
            jitter: Random jitter factor (0-1)
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.max_retries = max_retries
        self.jitter = jitter
        
        self.retry_stats: Dict[str, Dict] = {}
        
        logger.info(f"Exponential Backoff initialized - Max {max_retries} retries, {max_delay}s max delay")
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        import random
        delay = min(self.base_delay * (self.multiplier ** attempt), self.max_delay)
        jitter_amount = delay * self.jitter * random.random()
        return delay + jitter_amount
    
    def retry(self, operation_name: str = "operation"):
        """
        Decorator for retry with backoff.
        
        Usage:
            @backoff.retry("fetch_data")
            def fetch_data():
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        result = func(*args, **kwargs)
                        self._record_success(operation_name)
                        return result
                    except Exception as e:
                        last_exception = e
                        self._record_failure(operation_name, str(e))
                        
                        if attempt < self.max_retries:
                            delay = self.calculate_delay(attempt)
                            logger.warning(f"{operation_name} failed (attempt {attempt + 1}), "
                                         f"retrying in {delay:.1f}s: {e}")
                            time.sleep(delay)
                        else:
                            logger.error(f"{operation_name} failed after {self.max_retries} retries: {e}")
                
                raise last_exception
            return wrapper
        return decorator
    
    def _record_success(self, operation: str):
        if operation not in self.retry_stats:
            self.retry_stats[operation] = {'success': 0, 'failures': 0}
        self.retry_stats[operation]['success'] += 1
    
    def _record_failure(self, operation: str, error: str):
        if operation not in self.retry_stats:
            self.retry_stats[operation] = {'success': 0, 'failures': 0, 'last_error': None}
        self.retry_stats[operation]['failures'] += 1
        self.retry_stats[operation]['last_error'] = error


class RateLimiter:
    """
    Feature #14: Smart Rate-Limiting System
    
    Token bucket rate limiter with burst support.
    """
    
    def __init__(
        self,
        requests_per_second: float = 10,
        burst_size: int = 20
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Steady-state request rate
            burst_size: Maximum burst capacity
        """
        self.rate = requests_per_second
        self.burst = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = threading.Lock()
        
        self.requests_made = 0
        self.requests_throttled = 0
        
        logger.info(f"Rate Limiter initialized - {requests_per_second}/s, burst: {burst_size}")
    
    def acquire(self, tokens: int = 1, block: bool = True) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            block: If True, wait for tokens; if False, return immediately
            
        Returns:
            True if tokens acquired, False if not available (non-blocking)
        """
        with self._lock:
            # Refill tokens
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            # Check availability
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.requests_made += 1
                return True
            
            if not block:
                self.requests_throttled += 1
                return False
        
        # Wait for tokens
        wait_time = (tokens - self.tokens) / self.rate
        self.requests_throttled += 1
        logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
        time.sleep(wait_time)
        
        with self._lock:
            self.tokens = 0
            self.requests_made += 1
            return True
    
    def rate_limit(self, tokens: int = 1):
        """Decorator for rate-limited functions."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.acquire(tokens)
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict:
        return {
            'requests_made': self.requests_made,
            'requests_throttled': self.requests_throttled,
            'current_tokens': round(self.tokens, 2),
            'rate': self.rate,
            'burst': self.burst
        }


class LRUCache:
    """
    Feature #22: In-Memory Caching Layer
    
    Thread-safe LRU cache with TTL support.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 300  # 5 minutes
    ):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum cache entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.expiry: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        
        self.hits = 0
        self.misses = 0
        
        logger.info(f"LRU Cache initialized - Max {max_size} entries, TTL: {default_ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check expiry
            if key in self.expiry and datetime.now() > self.expiry[key]:
                del self.cache[key]
                del self.expiry[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    oldest = next(iter(self.cache))
                    del self.cache[oldest]
                    if oldest in self.expiry:
                        del self.expiry[oldest]
            
            self.cache[key] = value
            self.expiry[key] = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
    
    def invalidate(self, key: str):
        """Remove key from cache."""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.expiry:
                del self.expiry[key]
    
    def clear(self):
        """Clear entire cache."""
        with self._lock:
            self.cache.clear()
            self.expiry.clear()
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True, default=str)
        return hashlib.md5(data.encode()).hexdigest()
    
    def cached(self, ttl: Optional[int] = None):
        """Decorator for cached functions."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                key = f"{func.__name__}:{self.cache_key(*args, **kwargs)}"
                result = self.get(key)
                if result is not None:
                    return result
                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                return result
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict:
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': round(hit_rate, 3),
            'size': len(self.cache),
            'max_size': self.max_size
        }


class TradeReplaySystem:
    """
    Feature #26: Trade Execution Replay System
    
    Records and replays trade executions for analysis.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize replay system.
        
        Args:
            max_history: Maximum events to store
        """
        self.max_history = max_history
        self.events: List[Dict] = []
        self._lock = threading.Lock()
        
        logger.info(f"Trade Replay System initialized - Max {max_history} events")
    
    def record_event(
        self,
        event_type: str,
        data: Dict,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a trade event.
        
        Args:
            event_type: Type of event (SIGNAL, ENTRY, EXIT, etc.)
            data: Event data
            timestamp: Event timestamp
        """
        event = {
            'timestamp': (timestamp or datetime.now()).isoformat(),
            'type': event_type,
            'data': data
        }
        
        with self._lock:
            self.events.append(event)
            if len(self.events) > self.max_history:
                self.events = self.events[-self.max_history:]
    
    def get_trade_timeline(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """Get filtered trade timeline."""
        filtered = self.events
        
        if start_time:
            start_str = start_time.isoformat()
            filtered = [e for e in filtered if e['timestamp'] >= start_str]
        
        if end_time:
            end_str = end_time.isoformat()
            filtered = [e for e in filtered if e['timestamp'] <= end_str]
        
        if event_types:
            filtered = [e for e in filtered if e['type'] in event_types]
        
        return filtered
    
    def replay(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Replay events in sequence.
        
        Args:
            start_idx: Starting index
            end_idx: Ending index
            callback: Function to call for each event
            
        Returns:
            List of replayed events
        """
        events = self.events[start_idx:end_idx]
        
        if callback:
            for event in events:
                callback(event)
        
        return events
    
    def get_trade_forensics(self, trade_id: str) -> Dict:
        """Get all events related to a specific trade."""
        related = [e for e in self.events if e['data'].get('trade_id') == trade_id]
        
        return {
            'trade_id': trade_id,
            'events': related,
            'event_count': len(related),
            'timeline': [{'type': e['type'], 'time': e['timestamp']} for e in related]
        }


# Singleton instances
_backoff: Optional[ExponentialBackoff] = None
_rate_limiter: Optional[RateLimiter] = None
_cache: Optional[LRUCache] = None
_replay: Optional[TradeReplaySystem] = None


def get_backoff() -> ExponentialBackoff:
    global _backoff
    if _backoff is None:
        _backoff = ExponentialBackoff()
    return _backoff


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def get_cache() -> LRUCache:
    global _cache
    if _cache is None:
        _cache = LRUCache()
    return _cache


def get_replay_system() -> TradeReplaySystem:
    global _replay
    if _replay is None:
        _replay = TradeReplaySystem()
    return _replay


if __name__ == '__main__':
    # Test backoff
    backoff = ExponentialBackoff(max_retries=3)
    
    @backoff.retry("test_operation")
    def flaky_function():
        import random
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "Success"
    
    try:
        result = flaky_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test rate limiter
    limiter = RateLimiter(requests_per_second=2, burst_size=5)
    for i in range(10):
        limiter.acquire()
        print(f"Request {i+1} at {time.time():.2f}")
    
    # Test cache
    cache = LRUCache(max_size=100)
    cache.set("key1", "value1")
    print(f"Cache get: {cache.get('key1')}")
    print(f"Cache stats: {cache.get_stats()}")
