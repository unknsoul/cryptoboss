"""
Integration & API Features - Features #15-25
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
import json
import hashlib
import time

logger = logging.getLogger(__name__)

class APIClient:
    """Feature #15: Generic API Client"""
    def __init__(self, base_url: str, api_key: str = ''):
        self.base_url = base_url
        self.api_key = api_key
        self.request_count = 0
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        self.request_count += 1
        return {'status': 'ok', 'endpoint': endpoint, 'mock': True}
    
    def post(self, endpoint: str, data: Dict) -> Dict:
        self.request_count += 1
        return {'status': 'ok', 'endpoint': endpoint, 'mock': True}

class APIRateLimiter:
    """Feature #16: API Rate Limiter"""
    def __init__(self, requests_per_second: float = 10):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.time()
    
    def acquire(self) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
        self.last_update = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
    
    def wait(self):
        while not self.acquire():
            time.sleep(0.1)

class APIResponseCache:
    """Feature #17: API Response Cache"""
    def __init__(self, ttl_seconds: int = 60):
        self.ttl = ttl_seconds
        self.cache: Dict[str, Dict] = {}
    
    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['time'] < self.ttl:
                return entry['data']
            del self.cache[key]
        return None
    
    def set(self, key: str, data: Dict):
        self.cache[key] = {'data': data, 'time': time.time()}

class WebhookServer:
    """Feature #18: Webhook Receiver"""
    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.received: List[Dict] = []
    
    def register(self, path: str, handler: Callable):
        self.handlers[path] = handler
    
    def receive(self, path: str, data: Dict):
        self.received.append({'path': path, 'data': data, 'time': datetime.now().isoformat()})
        if path in self.handlers:
            self.handlers[path](data)

class DataStreamer:
    """Feature #19: Data Streaming Manager"""
    def __init__(self):
        self.streams: Dict[str, Dict] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
    
    def subscribe(self, stream: str, callback: Callable):
        if stream not in self.callbacks:
            self.callbacks[stream] = []
        self.callbacks[stream].append(callback)
    
    def push(self, stream: str, data: Dict):
        for cb in self.callbacks.get(stream, []):
            cb(data)

class RequestLogger:
    """Feature #20: Request/Response Logger"""
    def __init__(self):
        self.logs: List[Dict] = []
    
    def log_request(self, method: str, url: str, data: Optional[Dict] = None):
        self.logs.append({'type': 'request', 'method': method, 'url': url, 'data': data, 'time': datetime.now().isoformat()})
    
    def log_response(self, status: int, data: Dict, latency_ms: float):
        self.logs.append({'type': 'response', 'status': status, 'latency_ms': latency_ms, 'time': datetime.now().isoformat()})
        self.logs = self.logs[-5000:]

class APIHealthChecker:
    """Feature #21: API Health Checker"""
    def __init__(self):
        self.endpoints: Dict[str, Dict] = {}
    
    def register(self, name: str, check_fn: Callable):
        self.endpoints[name] = {'check': check_fn, 'status': 'unknown', 'last_check': None}
    
    def check_all(self) -> Dict:
        results = {}
        for name, ep in self.endpoints.items():
            try:
                ep['status'] = 'healthy' if ep['check']() else 'unhealthy'
            except:
                ep['status'] = 'error'
            ep['last_check'] = datetime.now().isoformat()
            results[name] = ep['status']
        return results

class MessageQueue:
    """Feature #22: Message Queue"""
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queues: Dict[str, List] = {}
    
    def push(self, queue: str, message: Dict):
        if queue not in self.queues:
            self.queues[queue] = []
        self.queues[queue].append(message)
        self.queues[queue] = self.queues[queue][-self.max_size:]
    
    def pop(self, queue: str) -> Optional[Dict]:
        if queue in self.queues and self.queues[queue]:
            return self.queues[queue].pop(0)
        return None

class ServiceRegistry:
    """Feature #23: Service Registry"""
    def __init__(self):
        self.services: Dict[str, Dict] = {}
    
    def register(self, name: str, endpoint: str, metadata: Optional[Dict] = None):
        self.services[name] = {'endpoint': endpoint, 'metadata': metadata or {}, 'registered_at': datetime.now().isoformat()}
    
    def discover(self, name: str) -> Optional[Dict]:
        return self.services.get(name)
    
    def list_all(self) -> List[str]:
        return list(self.services.keys())

class ConfigManager:
    """Feature #24: Configuration Manager"""
    def __init__(self):
        self.config: Dict = {}
        self.defaults: Dict = {}
    
    def set_default(self, key: str, value):
        self.defaults[key] = value
    
    def set(self, key: str, value):
        self.config[key] = value
    
    def get(self, key: str, default=None):
        if key in self.config:
            return self.config[key]
        if key in self.defaults:
            return self.defaults[key]
        return default

class FeatureFlags:
    """Feature #25: Feature Flags"""
    def __init__(self):
        self.flags: Dict[str, bool] = {}
    
    def set(self, flag: str, enabled: bool):
        self.flags[flag] = enabled
    
    def is_enabled(self, flag: str) -> bool:
        return self.flags.get(flag, False)
    
    def list_all(self) -> Dict[str, bool]:
        return dict(self.flags)

# Factories
def get_api_client(url: str): return APIClient(url)
def get_rate_limiter(): return APIRateLimiter()
def get_response_cache(): return APIResponseCache()
def get_webhook_server(): return WebhookServer()
def get_data_streamer(): return DataStreamer()
def get_request_logger(): return RequestLogger()
def get_api_health(): return APIHealthChecker()
def get_message_queue(): return MessageQueue()
def get_service_registry(): return ServiceRegistry()
def get_config_manager(): return ConfigManager()
def get_feature_flags(): return FeatureFlags()
