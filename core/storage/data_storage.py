"""
Logging, Database & Utilities - Features #26-35
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os

logger = logging.getLogger(__name__)

class StructuredLogger:
    """Feature #26: Structured Logging"""
    def __init__(self, name: str):
        self.name = name
        self.logs: List[Dict] = []
    
    def log(self, level: str, message: str, **kwargs):
        entry = {'level': level, 'message': message, 'logger': self.name, 
                 'timestamp': datetime.now().isoformat(), **kwargs}
        self.logs.append(entry)
        self.logs = self.logs[-10000:]
    
    def info(self, message: str, **kwargs): self.log('INFO', message, **kwargs)
    def warning(self, message: str, **kwargs): self.log('WARNING', message, **kwargs)
    def error(self, message: str, **kwargs): self.log('ERROR', message, **kwargs)

class TradeLogger:
    """Feature #27: Trade-Specific Logger"""
    def __init__(self):
        self.trades: List[Dict] = []
    
    def log_entry(self, trade: Dict):
        self.trades.append({**trade, 'logged_at': datetime.now().isoformat()})
    
    def log_exit(self, trade_id: str, exit_price: float, pnl: float):
        for t in self.trades:
            if t.get('id') == trade_id:
                t['exit_price'] = exit_price
                t['pnl'] = pnl
                t['closed_at'] = datetime.now().isoformat()

class PerformanceLogger:
    """Feature #28: Performance Logger"""
    def __init__(self):
        self.snapshots: List[Dict] = []
    
    def snapshot(self, equity: float, pnl: float, positions: int):
        self.snapshots.append({'equity': equity, 'pnl': pnl, 'positions': positions, 
                               'time': datetime.now().isoformat()})
        self.snapshots = self.snapshots[-5000:]

class ErrorTracker:
    """Feature #29: Error Tracking System"""
    def __init__(self):
        self.errors: List[Dict] = []
    
    def track(self, error: Exception, context: Optional[Dict] = None):
        self.errors.append({'error': str(error), 'type': type(error).__name__, 
                           'context': context, 'time': datetime.now().isoformat()})
        self.errors = self.errors[-1000:]
    
    def get_recent(self, n: int = 10) -> List[Dict]:
        return self.errors[-n:]

class FileStorage:
    """Feature #30: File Storage Manager"""
    def __init__(self, base_path: str = 'data'):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_json(self, filename: str, data: Dict):
        path = os.path.join(self.base_path, filename)
        with open(path, 'w') as f:
            json.dump(data, f, default=str)
    
    def load_json(self, filename: str) -> Optional[Dict]:
        path = os.path.join(self.base_path, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

class DataVersioning:
    """Feature #31: Data Versioning"""
    def __init__(self):
        self.versions: Dict[str, List[Dict]] = {}
    
    def save_version(self, key: str, data: Dict):
        if key not in self.versions:
            self.versions[key] = []
        self.versions[key].append({'data': data, 'version': len(self.versions[key]) + 1,
                                   'created_at': datetime.now().isoformat()})
    
    def get_version(self, key: str, version: int = -1) -> Optional[Dict]:
        if key in self.versions and self.versions[key]:
            return self.versions[key][version]['data']
        return None

class CacheManager:
    """Feature #32: In-Memory Cache"""
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.access_order.append(key)

class TimeSeriesDB:
    """Feature #33: Time Series Database"""
    def __init__(self):
        self.series: Dict[str, List[Dict]] = {}
    
    def insert(self, name: str, value: float, timestamp: Optional[datetime] = None):
        if name not in self.series:
            self.series[name] = []
        self.series[name].append({'value': value, 'time': (timestamp or datetime.now()).isoformat()})
        self.series[name] = self.series[name][-10000:]
    
    def query(self, name: str, limit: int = 100) -> List[Dict]:
        return self.series.get(name, [])[-limit:]

class KeyValueStore:
    """Feature #34: Key-Value Store"""
    def __init__(self):
        self.store: Dict[str, Dict] = {}
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        self.store[key] = {'value': value, 'expires': datetime.now().timestamp() + ttl_seconds if ttl_seconds else None}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.store:
            entry = self.store[key]
            if entry['expires'] is None or datetime.now().timestamp() < entry['expires']:
                return entry['value']
            del self.store[key]
        return None

class DataPipeline:
    """Feature #35: Data Pipeline"""
    def __init__(self):
        self.stages: List[callable] = []
    
    def add_stage(self, transform: callable):
        self.stages.append(transform)
    
    def process(self, data: Any) -> Any:
        result = data
        for stage in self.stages:
            result = stage(result)
        return result

# Factories
def get_structured_logger(name: str): return StructuredLogger(name)
def get_trade_logger(): return TradeLogger()
def get_performance_logger(): return PerformanceLogger()
def get_error_tracker(): return ErrorTracker()
def get_file_storage(): return FileStorage()
def get_data_versioning(): return DataVersioning()
def get_cache_manager(): return CacheManager()
def get_timeseries_db(): return TimeSeriesDB()
def get_kv_store(): return KeyValueStore()
def get_data_pipeline(): return DataPipeline()
