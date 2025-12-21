
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List

class BaseExchange(ABC):
    """
    Abstract Base Class for Exchange Connectors
    Defines the standard interface for all exchange integrations.
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.callbacks: Dict[str, List[Callable]] = {}
        self.is_connected = False
        
    @abstractmethod
    def connect(self):
        """Establish connection to the exchange (WebSocket & REST)"""
        pass
        
    @abstractmethod
    def disconnect(self):
        """Close all connections"""
        pass
        
    @abstractmethod
    def subscribe_ticker(self, symbol: str):
        """Subscribe to real-time ticker updates"""
        pass
        
    @abstractmethod
    def subscribe_orderbook(self, symbol: str):
        """Subscribe to real-time orderbook updates"""
        pass
        
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Any:
        """Fetch historical candle data"""
        pass
        
    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Get latest ticker price"""
        pass
        
    def on(self, event: str, callback: Callable):
        """Register a callback for an event (ticker, trade, orderbook)"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
        
    def _emit(self, event: str, data: Any):
        """Emit data to registered listeners"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Error in callback for {event}: {e}")
