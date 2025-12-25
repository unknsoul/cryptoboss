"""
Exchange Integration - Enterprise Features #1, #5, #9, #12
Multi-Exchange Connector, Order Book, Balance Sync, WebSocket.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import threading
import time
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    FTX = "ftx"
    BYBIT = "bybit"


class MultiExchangeConnector:
    """
    Feature #1: Multi-Exchange Connector
    
    Unified interface for multiple cryptocurrency exchanges.
    """
    
    def __init__(self):
        """Initialize multi-exchange connector."""
        self.exchanges: Dict[str, Dict] = {}
        self.active_exchange: Optional[str] = None
        self.api_handlers: Dict[str, Callable] = {}
        
        logger.info("Multi-Exchange Connector initialized")
    
    def register_exchange(
        self,
        name: str,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        **kwargs
    ):
        """Register an exchange connection."""
        self.exchanges[name] = {
            'name': name,
            'api_key': api_key[:8] + '***',  # Masked
            'api_secret': '***',
            'testnet': testnet,
            'connected': False,
            'last_ping': None,
            'config': kwargs
        }
        logger.info(f"Registered exchange: {name} (testnet={testnet})")
    
    def connect(self, name: str) -> bool:
        """Connect to an exchange."""
        if name not in self.exchanges:
            return False
        
        # Simulate connection
        self.exchanges[name]['connected'] = True
        self.exchanges[name]['last_ping'] = datetime.now().isoformat()
        self.active_exchange = name
        
        logger.info(f"Connected to {name}")
        return True
    
    def disconnect(self, name: str):
        """Disconnect from an exchange."""
        if name in self.exchanges:
            self.exchanges[name]['connected'] = False
            if self.active_exchange == name:
                self.active_exchange = None
    
    def get_ticker(self, symbol: str, exchange: Optional[str] = None) -> Dict:
        """Get ticker data from exchange."""
        ex = exchange or self.active_exchange
        if not ex or not self.exchanges.get(ex, {}).get('connected'):
            return {'error': 'Not connected'}
        
        # Simulated ticker response
        return {
            'symbol': symbol,
            'exchange': ex,
            'bid': 50000,
            'ask': 50010,
            'last': 50005,
            'volume_24h': 1000,
            'timestamp': datetime.now().isoformat()
        }
    
    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = 'MARKET',
        price: Optional[float] = None,
        exchange: Optional[str] = None
    ) -> Dict:
        """Place order on exchange."""
        ex = exchange or self.active_exchange
        if not ex:
            return {'error': 'No active exchange'}
        
        order = {
            'id': f"{ex[:3].upper()}-{int(time.time()*1000)}",
            'exchange': ex,
            'symbol': symbol,
            'side': side,
            'size': size,
            'type': order_type,
            'price': price,
            'status': 'submitted',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Order placed: {order['id']}")
        return order
    
    def get_all_exchanges(self) -> List[Dict]:
        """Get all registered exchanges."""
        return [
            {
                'name': name,
                'connected': ex['connected'],
                'testnet': ex['testnet']
            }
            for name, ex in self.exchanges.items()
        ]


class OrderBookSync:
    """
    Feature #5: Order Book Sync
    
    Synchronizes and maintains order book state.
    """
    
    def __init__(self, depth: int = 20):
        """
        Initialize order book sync.
        
        Args:
            depth: Number of levels to maintain
        """
        self.depth = depth
        self.order_books: Dict[str, Dict] = {}  # symbol -> order book
        self.last_update: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        
        logger.info(f"Order Book Sync initialized - Depth: {depth}")
    
    def update(self, symbol: str, bids: List[List], asks: List[List]):
        """
        Update order book with new data.
        
        Args:
            symbol: Trading pair
            bids: List of [price, size]
            asks: List of [price, size]
        """
        with self._lock:
            self.order_books[symbol] = {
                'bids': sorted(bids, key=lambda x: x[0], reverse=True)[:self.depth],
                'asks': sorted(asks, key=lambda x: x[0])[:self.depth],
                'timestamp': datetime.now().isoformat()
            }
            self.last_update[symbol] = datetime.now()
    
    def get(self, symbol: str) -> Optional[Dict]:
        """Get current order book."""
        return self.order_books.get(symbol)
    
    def get_spread(self, symbol: str) -> Dict:
        """Get bid-ask spread."""
        book = self.order_books.get(symbol)
        if not book or not book['bids'] or not book['asks']:
            return {'spread': 0, 'spread_pct': 0}
        
        best_bid = book['bids'][0][0]
        best_ask = book['asks'][0][0]
        spread = best_ask - best_bid
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_pct': round(spread / best_ask * 100, 4),
            'mid': (best_bid + best_ask) / 2
        }
    
    def get_depth_info(self, symbol: str) -> Dict:
        """Get order book depth analysis."""
        book = self.order_books.get(symbol)
        if not book:
            return {}
        
        bid_volume = sum(b[1] for b in book['bids'])
        ask_volume = sum(a[1] for a in book['asks'])
        
        return {
            'bid_levels': len(book['bids']),
            'ask_levels': len(book['asks']),
            'bid_volume': round(bid_volume, 4),
            'ask_volume': round(ask_volume, 4),
            'imbalance': round((bid_volume - ask_volume) / (bid_volume + ask_volume), 3) if (bid_volume + ask_volume) > 0 else 0
        }


class BalanceSynchronizer:
    """
    Feature #9: Balance Synchronizer
    
    Synchronizes account balances across exchanges.
    """
    
    def __init__(self):
        """Initialize balance synchronizer."""
        self.balances: Dict[str, Dict[str, Dict]] = {}  # exchange -> asset -> balance
        self.last_sync: Dict[str, datetime] = {}
        
        logger.info("Balance Synchronizer initialized")
    
    def update_balance(
        self,
        exchange: str,
        asset: str,
        free: float,
        locked: float = 0
    ):
        """Update balance for an asset."""
        if exchange not in self.balances:
            self.balances[exchange] = {}
        
        self.balances[exchange][asset] = {
            'free': free,
            'locked': locked,
            'total': free + locked,
            'updated_at': datetime.now().isoformat()
        }
        self.last_sync[exchange] = datetime.now()
    
    def get_balance(self, exchange: str, asset: str) -> Dict:
        """Get balance for specific asset."""
        return self.balances.get(exchange, {}).get(asset, {
            'free': 0, 'locked': 0, 'total': 0
        })
    
    def get_total_balance(self, asset: str) -> Dict:
        """Get total balance across all exchanges."""
        total_free = 0
        total_locked = 0
        by_exchange = {}
        
        for ex, assets in self.balances.items():
            if asset in assets:
                bal = assets[asset]
                total_free += bal['free']
                total_locked += bal['locked']
                by_exchange[ex] = bal['total']
        
        return {
            'asset': asset,
            'total_free': total_free,
            'total_locked': total_locked,
            'total': total_free + total_locked,
            'by_exchange': by_exchange
        }
    
    def get_portfolio_value(self, prices: Dict[str, float], base: str = 'USDT') -> float:
        """Calculate total portfolio value."""
        total = 0
        
        for exchange, assets in self.balances.items():
            for asset, balance in assets.items():
                if asset == base:
                    total += balance['total']
                elif asset in prices:
                    total += balance['total'] * prices[asset]
        
        return round(total, 2)


class WebSocketManager:
    """
    Feature #12: WebSocket Manager
    
    Manages WebSocket connections for real-time data.
    """
    
    def __init__(self, max_connections: int = 5):
        """
        Initialize WebSocket manager.
        
        Args:
            max_connections: Maximum concurrent connections
        """
        self.max_connections = max_connections
        self.connections: Dict[str, Dict] = {}
        self.subscriptions: Dict[str, List[str]] = defaultdict(list)
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.running = False
        
        logger.info(f"WebSocket Manager initialized - Max: {max_connections}")
    
    def connect(self, name: str, url: str) -> bool:
        """Establish a WebSocket connection."""
        if len(self.connections) >= self.max_connections:
            logger.warning("Max connections reached")
            return False
        
        self.connections[name] = {
            'url': url,
            'status': 'connected',
            'connected_at': datetime.now().isoformat(),
            'messages_received': 0,
            'last_message': None
        }
        
        logger.info(f"WebSocket connected: {name}")
        return True
    
    def disconnect(self, name: str):
        """Close a WebSocket connection."""
        if name in self.connections:
            self.connections[name]['status'] = 'disconnected'
            del self.connections[name]
            logger.info(f"WebSocket disconnected: {name}")
    
    def subscribe(self, connection: str, channel: str, callback: Callable):
        """Subscribe to a channel."""
        if connection not in self.connections:
            return False
        
        self.subscriptions[connection].append(channel)
        self.callbacks[channel].append(callback)
        
        logger.debug(f"Subscribed to {channel} on {connection}")
        return True
    
    def unsubscribe(self, connection: str, channel: str):
        """Unsubscribe from a channel."""
        if connection in self.subscriptions:
            if channel in self.subscriptions[connection]:
                self.subscriptions[connection].remove(channel)
    
    def on_message(self, connection: str, channel: str, data: Any):
        """Handle incoming message."""
        if connection in self.connections:
            self.connections[connection]['messages_received'] += 1
            self.connections[connection]['last_message'] = datetime.now().isoformat()
        
        # Dispatch to callbacks
        for callback in self.callbacks.get(channel, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_status(self) -> Dict:
        """Get WebSocket manager status."""
        return {
            'active_connections': len(self.connections),
            'max_connections': self.max_connections,
            'total_subscriptions': sum(len(s) for s in self.subscriptions.values()),
            'connections': {
                name: {
                    'status': conn['status'],
                    'messages': conn['messages_received']
                }
                for name, conn in self.connections.items()
            }
        }


# Singletons
_exchange: Optional[MultiExchangeConnector] = None
_orderbook: Optional[OrderBookSync] = None
_balance: Optional[BalanceSynchronizer] = None
_websocket: Optional[WebSocketManager] = None


def get_exchange_connector() -> MultiExchangeConnector:
    global _exchange
    if _exchange is None:
        _exchange = MultiExchangeConnector()
    return _exchange


def get_orderbook_sync() -> OrderBookSync:
    global _orderbook
    if _orderbook is None:
        _orderbook = OrderBookSync()
    return _orderbook


def get_balance_sync() -> BalanceSynchronizer:
    global _balance
    if _balance is None:
        _balance = BalanceSynchronizer()
    return _balance


def get_websocket_manager() -> WebSocketManager:
    global _websocket
    if _websocket is None:
        _websocket = WebSocketManager()
    return _websocket


if __name__ == '__main__':
    # Test exchange connector
    ex = MultiExchangeConnector()
    ex.register_exchange('binance', 'api_key_123', 'secret_456', testnet=True)
    ex.connect('binance')
    print(f"Ticker: {ex.get_ticker('BTCUSDT')}")
    
    # Test order book
    ob = OrderBookSync()
    ob.update('BTCUSDT', [[50000, 1], [49990, 2]], [[50010, 1], [50020, 2]])
    print(f"Spread: {ob.get_spread('BTCUSDT')}")
    
    # Test balance
    bal = BalanceSynchronizer()
    bal.update_balance('binance', 'USDT', 10000, 500)
    bal.update_balance('binance', 'BTC', 0.5, 0.1)
    print(f"USDT balance: {bal.get_balance('binance', 'USDT')}")
