"""
Exchange Abstraction Layer
Supports multiple exchanges with unified interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ExchangeInterface(ABC):
    """Base interface for all exchanges."""
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data."""
        pass
    
    @abstractmethod
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book."""
        pass
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict:
        """Place an order."""
        pass
    
    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict:
        """Get account balance."""
        pass


class MockExchange(ExchangeInterface):
    """Mock exchange for testing."""
    
    def __init__(self):
        self.balance = {'USDT': 10000.0, 'BTC': 0.0}
        self.positions = []
        self.orders = []
        logger.info("MockExchange initialized")
    
    def get_ticker(self, symbol: str) -> Dict:
        return {
            'symbol': symbol,
            'price': 40000.0,
            'bid': 39995.0,
            'ask': 40005.0,
            'volume': 12345.67
        }
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        return {
            'bids': [(40000 - i * 10, 0.1 + i * 0.01) for i in range(limit)],
            'asks': [(40000 + i * 10, 0.1 + i * 0.01) for i in range(limit)]
        }
    
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict:
        order = {
            'order_id': f"MOCK_{len(self.orders) + 1}",
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'price': price or 40000.0,
            'status': 'FILLED'
        }
        self.orders.append(order)
        logger.info(f"Mock order placed: {order}")
        return order
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        return {'order_id': order_id, 'status': 'CANCELLED'}
    
    def get_positions(self) -> List[Dict]:
        return self.positions
    
    def get_balance(self) -> Dict:
        return self.balance


class BinanceExchange(ExchangeInterface):
    """Binance exchange implementation."""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Import binance client if available
        try:
            from binance.client import Client
            if api_key and api_secret:
                self.client = Client(api_key, api_secret)
                logger.info("Binance client initialized")
            else:
                logger.warning("Binance credentials not provided")
                self.client = None
        except ImportError:
            logger.warning("Binance library not installed")
            self.client = None
    
    def get_ticker(self, symbol: str) -> Dict:
        if not self.client:
            raise Exception("Binance client not initialized")
        
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return {
            'symbol': ticker['symbol'],
            'price': float(ticker['price']),
        }
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        if not self.client:
            raise Exception("Binance client not initialized")
        
        orderbook = self.client.get_order_book(symbol=symbol, limit=limit)
        return {
            'bids': [(float(b[0]), float(b[1])) for b in orderbook['bids']],
            'asks': [(float(a[0]), float(a[1])) for a in orderbook['asks']]
        }
    
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict:
        if not self.client:
            raise Exception("Binance client not initialized")
        
        if order_type == 'MARKET':
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
        else:  # LIMIT
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=str(price)
            )
        
        return order
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        if not self.client:
            raise Exception("Binance client not initialized")
        
        result = self.client.cancel_order(symbol=symbol, orderId=order_id)
        return result
    
    def get_positions(self) -> List[Dict]:
        if not self.client:
            return []
        
        # For futures
        positions = self.client.futures_position_information()
        return [p for p in positions if float(p['positionAmt']) != 0]
    
    def get_balance(self) -> Dict:
        if not self.client:
            return {}
        
        account = self.client.get_account()
        balances = {b['asset']: float(b['free']) for b in account['balances'] if float(b['free']) > 0}
        return balances


def get_exchange(exchange_type: str = 'mock', **kwargs) -> ExchangeInterface:
    """Factory function to get exchange instance."""
    if exchange_type == 'mock':
        return MockExchange()
    elif exchange_type == 'binance':
        return BinanceExchange(**kwargs)
    else:
        raise ValueError(f"Unknown exchange type: {exchange_type}")
