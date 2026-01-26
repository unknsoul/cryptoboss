"""
Binance Exchange Client - Real Exchange Integration

Wraps ccxt for unified exchange access.
Provides both REST and WebSocket functionality.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
import os

logger = logging.getLogger(__name__)

try:
    import ccxt
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("ccxt not installed. Install with: pip install ccxt")


class BinanceClient:
    """
    Unified Binance client for REST and WebSocket operations.
    
    Usage:
        client = BinanceClient(testnet=True)
        await client.connect()
        
        # REST operations
        balance = await client.get_balance()
        order = await client.create_order("BTC/USDT", "buy", "market", 0.001)
        
        # WebSocket
        client.subscribe_trades("BTC/USDT", on_trade_callback)
        
        await client.close()
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = True
    ):
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt library required. Install with: pip install ccxt")
        
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        self.testnet = testnet
        
        # Create exchange instance
        exchange_class = ccxt_async.binance
        
        config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        }
        
        if testnet:
            config['options']['testnet'] = True
            config['urls'] = {
                'api': {
                    'public': 'https://testnet.binance.vision/api/v3',
                    'private': 'https://testnet.binance.vision/api/v3',
                }
            }
        
        self.exchange = exchange_class(config)
        self._connected = False
        self._price_callbacks: Dict[str, List[Callable]] = {}
        self._prices: Dict[str, float] = {}
        
        logger.info(f"BinanceClient initialized (testnet={testnet})")
    
    async def connect(self):
        """Connect to exchange and load markets."""
        try:
            await self.exchange.load_markets()
            self._connected = True
            logger.info("Connected to Binance")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
    
    async def close(self):
        """Close connection."""
        await self.exchange.close()
        self._connected = False
        logger.info("Disconnected from Binance")
    
    # === REST API ===
    
    async def get_balance(self) -> Dict[str, float]:
        """Get account balances."""
        try:
            balance = await self.exchange.fetch_balance()
            return {
                currency: {
                    'free': data['free'],
                    'used': data['used'],
                    'total': data['total']
                }
                for currency, data in balance['total'].items()
                if data and (isinstance(data, (int, float)) and data > 0)
            }
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return {}
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker for a symbol."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            self._prices[symbol] = ticker['last']
            return ticker
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return {}
    
    async def get_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        if symbol in self._prices:
            return self._prices[symbol]
        ticker = await self.get_ticker(symbol)
        return ticker.get('last', 0)
    
    async def create_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        order_type: str,  # 'market' or 'limit'
        amount: float,
        price: float = None,
        params: Dict = None
    ) -> Dict:
        """Create an order."""
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params or {}
            )
            logger.info(f"Order created: {order['id']} - {side} {amount} {symbol}")
            return order
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an order."""
        try:
            result = await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order cancelled: {order_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    async def fetch_order(self, order_id: str, symbol: str) -> Dict:
        """Fetch order status."""
        try:
            return await self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            logger.error(f"Failed to fetch order {order_id}: {e}")
            raise
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders."""
        try:
            return await self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100
    ) -> List:
        """Get OHLCV candlestick data."""
        try:
            return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Failed to get OHLCV: {e}")
            return []
    
    # === Price Feed ===
    
    async def start_price_feed(self, symbols: List[str], interval: float = 1.0):
        """Start polling price feed for symbols."""
        logger.info(f"Starting price feed for {symbols}")
        
        while True:
            try:
                for symbol in symbols:
                    ticker = await self.get_ticker(symbol)
                    if ticker and 'last' in ticker:
                        price = ticker['last']
                        self._prices[symbol] = price
                        
                        # Notify callbacks
                        if symbol in self._price_callbacks:
                            for callback in self._price_callbacks[symbol]:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(symbol, price)
                                    else:
                                        callback(symbol, price)
                                except Exception as e:
                                    logger.error(f"Price callback error: {e}")
                
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Price feed error: {e}")
                await asyncio.sleep(5)
    
    def subscribe_price(self, symbol: str, callback: Callable):
        """Subscribe to price updates for a symbol."""
        if symbol not in self._price_callbacks:
            self._price_callbacks[symbol] = []
        self._price_callbacks[symbol].append(callback)
    
    def get_cached_price(self, symbol: str) -> Optional[float]:
        """Get cached price (no API call)."""
        return self._prices.get(symbol)


# Factory function
def create_binance_client(testnet: bool = True) -> BinanceClient:
    """Create a Binance client with environment credentials."""
    return BinanceClient(testnet=testnet)
