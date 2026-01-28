"""
Binance Exchange Client - Real Exchange Integration

Wraps ccxt for unified exchange access with proper testnet support.
Provides both REST and WebSocket functionality.

Testnet Configuration:
- REST: https://testnet.binance.vision/api
- WebSocket: wss://testnet.binance.vision/ws

Live Configuration:
- REST: https://api.binance.com/api
- WebSocket: wss://stream.binance.com:9443/ws

Required Environment Variables:
- BINANCE_API_KEY: Your API key
- BINANCE_API_SECRET: Your API secret
- BINANCE_TESTNET_ENABLED: Set to 'true' for testnet (optional, defaults to False)
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import ccxt
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("ccxt not installed. Install with: pip install ccxt")

# Import symbol validator for filtering fake assets
try:
    from src.core.symbol_whitelist import filter_balances, is_valid_symbol, get_symbol_validator
    SYMBOL_VALIDATOR_AVAILABLE = True
except ImportError:
    SYMBOL_VALIDATOR_AVAILABLE = False
    logger.warning("symbol_whitelist not available, balance filtering disabled")


class BinanceEnvironment(Enum):

    """Binance API environments."""
    TESTNET = "testnet"
    LIVE = "live"


@dataclass
class BinanceEndpoints:
    """Binance API endpoints for a specific environment."""
    rest_base: str
    ws_base: str
    environment: BinanceEnvironment
    
    @classmethod
    def testnet(cls) -> "BinanceEndpoints":
        return cls(
            rest_base="https://testnet.binance.vision",
            ws_base="wss://testnet.binance.vision/ws",
            environment=BinanceEnvironment.TESTNET
        )
    
    @classmethod
    def live(cls) -> "BinanceEndpoints":
        return cls(
            rest_base="https://api.binance.com",
            ws_base="wss://stream.binance.com:9443/ws",
            environment=BinanceEnvironment.LIVE
        )


class BinanceConfigError(Exception):
    """Configuration error for Binance client."""
    pass


class BinanceAuthError(Exception):
    """Authentication error for Binance client."""
    pass


def validate_binance_config(
    api_key: Optional[str],
    api_secret: Optional[str],
    require_keys: bool = True
) -> Dict[str, str]:
    """
    Validate Binance API configuration.
    
    Args:
        api_key: API key (or None to read from env)
        api_secret: API secret (or None to read from env)
        require_keys: If True, raise error for missing keys
        
    Returns:
        Dict with validated api_key and api_secret
        
    Raises:
        BinanceConfigError: If configuration is invalid
    """
    resolved_key = api_key or os.getenv("BINANCE_API_KEY")
    resolved_secret = api_secret or os.getenv("BINANCE_API_SECRET")
    
    if require_keys:
        if not resolved_key:
            raise BinanceConfigError(
                "Missing Binance API key. Set BINANCE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        if not resolved_secret:
            raise BinanceConfigError(
                "Missing Binance API secret. Set BINANCE_API_SECRET environment variable "
                "or pass api_secret parameter."
            )
    
    return {
        "api_key": resolved_key,
        "api_secret": resolved_secret
    }


def get_testnet_enabled() -> bool:
    """Check if testnet is enabled via environment variable."""
    value = os.getenv("BINANCE_TESTNET_ENABLED", "false").lower()
    return value in ("true", "1", "yes")


class BinanceClient:
    """
    Unified Binance client for REST and WebSocket operations.
    
    Supports both testnet and live environments with automatic endpoint selection.
    
    Usage:
        # Create client for testnet
        client = BinanceClient(testnet=True, api_key="...", api_secret="...")
        await client.connect()
        
        # REST operations
        balance = await client.get_balance()
        order = await client.create_order("BTC/USDT", "buy", "market", 0.001)
        
        # Cleanup
        await client.close()
    
    Environment Variables:
        BINANCE_API_KEY: API key
        BINANCE_API_SECRET: API secret
        BINANCE_TESTNET_ENABLED: Set to 'true' for testnet mode
    """
    
    # Binance API error codes
    ERROR_CODES = {
        -1000: "Unknown error",
        -1001: "Disconnected",
        -1002: "Unauthorized - Invalid API key",
        -1003: "Too many requests",
        -1013: "Invalid quantity",
        -1015: "Too many orders",
        -1021: "Timestamp outside recv window",
        -1022: "Invalid signature",
        -2010: "New order rejected",
        -2011: "Cancel order rejected",
        -2013: "Order does not exist",
        -2014: "API key format invalid",
        -2015: "Invalid API key, IP, or permissions",
    }
    
    AUTH_ERROR_CODES = {-1002, -1022, -2014, -2015}
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = None
    ):
        """
        Initialize Binance client.
        
        Args:
            api_key: API key (or reads from BINANCE_API_KEY env var)
            api_secret: API secret (or reads from BINANCE_API_SECRET env var)
            testnet: True for testnet, False for live. If None, reads from 
                     BINANCE_TESTNET_ENABLED env var (defaults to False)
        """
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt library required. Install with: pip install ccxt")
        
        # Determine environment
        if testnet is None:
            testnet = get_testnet_enabled()
        
        self.testnet = testnet
        self.endpoints = BinanceEndpoints.testnet() if testnet else BinanceEndpoints.live()
        
        # Validate and store credentials
        try:
            creds = validate_binance_config(api_key, api_secret, require_keys=True)
            self.api_key = creds["api_key"]
            self.api_secret = creds["api_secret"]
        except BinanceConfigError as e:
            logger.error(f"Configuration error: {e}")
            raise
        
        # Create exchange instance with proper config
        self._init_exchange()
        
        # State
        self._connected = False
        self._running = True
        self._price_callbacks: Dict[str, List[Callable]] = {}
        self._prices: Dict[str, float] = {}
        
        # Server time sync
        self._timestamp_offset: int = 0  # ms difference: server_time - local_time
        self._last_time_sync: Optional[datetime] = None
        self._server_time: Optional[int] = None
        
        # Startup tracking
        self._startup_complete = False
        self._startup_error: Optional[str] = None
        
        env_name = "TESTNET" if testnet else "LIVE"
        logger.info(f"BinanceClient initialized ({env_name})")
        logger.info(f"  REST endpoint: {self.endpoints.rest_base}")
        logger.info(f"  WebSocket endpoint: {self.endpoints.ws_base}")

    
    def _init_exchange(self):
        """Initialize the exchange client with proper config."""
        # Try python-binance first (better testnet support)
        try:
            from binance.client import Client as BinanceSDK
            self._use_binance_sdk = True
            
            self.exchange = BinanceSDK(self.api_key, self.api_secret)
            
            if self.testnet:
                # Override API URL for testnet - matching user's working script
                self.exchange.API_URL = "https://testnet.binance.vision/api"
            
            logger.info(f"Using python-binance SDK (testnet={self.testnet})")
            
        except ImportError:
            # Fall back to ccxt
            self._use_binance_sdk = False
            logger.info("python-binance not installed, falling back to ccxt")
            
            if not CCXT_AVAILABLE:
                raise ImportError("Neither python-binance nor ccxt is available")
            
            exchange_class = ccxt_async.binance
            
            config = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000,
                }
            }
            
            if self.testnet:
                config['options']['testnet'] = True
                config['options']['sandboxMode'] = True
                config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binance.vision/api/v3',
                        'private': 'https://testnet.binance.vision/api/v3',
                    }
                }
            
            self.exchange = exchange_class(config)

    
    def _parse_binance_error(self, error: Exception) -> tuple:
        """
        Parse Binance API error into code and message.
        
        Returns:
            Tuple of (error_code, error_message, is_auth_error)
        """
        error_str = str(error)
        error_code = None
        is_auth_error = False
        
        # Try to extract error code from ccxt exception
        if hasattr(error, 'code'):
            error_code = error.code
        
        # Try to parse from error message
        import re
        code_match = re.search(r'"code"\s*:\s*(-?\d+)', error_str)
        if code_match:
            error_code = int(code_match.group(1))
        
        if error_code:
            is_auth_error = error_code in self.AUTH_ERROR_CODES
            error_msg = self.ERROR_CODES.get(error_code, error_str)
        else:
            error_msg = error_str
            # Check for common auth error patterns
            if any(x in error_str.lower() for x in ['signature', 'api key', 'unauthorized', 'permission']):
                is_auth_error = True
        
        return error_code, error_msg, is_auth_error
    
    def _format_error(self, error: Exception, context: str = "") -> str:
        """Format error message with context and environment info."""
        code, msg, is_auth = self._parse_binance_error(error)
        env_name = "TESTNET" if self.testnet else "LIVE"
        
        parts = [f"[Binance {env_name}]"]
        if context:
            parts.append(f"{context}:")
        if code:
            parts.append(f"Error {code}:")
        parts.append(msg)
        
        if is_auth:
            parts.append(f"(Check your {'testnet' if self.testnet else 'live'} API credentials)")
        
        return " ".join(parts)
    
    async def sync_server_time(self) -> int:
        """
        Synchronize with exchange server time.
        
        Returns:
            Timestamp offset in milliseconds (server_time - local_time)
        """
        import time
        
        try:
            local_before = int(time.time() * 1000)
            
            if self._use_binance_sdk:
                # python-binance SDK (sync)
                server_time_response = self.exchange.get_server_time()
                server_time = server_time_response["serverTime"]
            else:
                # ccxt (async)
                server_time = await self.exchange.fetch_time()
            
            local_after = int(time.time() * 1000)
            
            # Estimate network latency (half of round trip)
            latency = (local_after - local_before) / 2
            local_time = local_before + int(latency)
            
            self._timestamp_offset = server_time - local_time
            self._server_time = server_time
            self._last_time_sync = datetime.now()
            
            # Set timestamp offset on the SDK client (matches user's working script)
            if self._use_binance_sdk:
                self.exchange.timestamp_offset = self._timestamp_offset
            
            logger.info(f"Server time synced. Offset: {self._timestamp_offset}ms")
            return self._timestamp_offset
            
        except Exception as e:
            logger.error(f"Failed to sync server time: {e}")
            # Continue with zero offset
            self._timestamp_offset = 0
            return 0

    
    def get_server_time(self) -> int:
        """Get estimated current server time in milliseconds."""
        import time
        local_now = int(time.time() * 1000)
        return local_now + self._timestamp_offset
    
    async def startup(self) -> Dict:
        """
        Complete startup sequence with fail-fast behavior.
        
        Sequence:
        1. Sync server time
        2. Load markets (ccxt only)
        3. Validate credentials (auth check)
        
        Returns:
            Dict with startup result including balances
            
        Raises:
            BinanceAuthError: If authentication fails
            Exception: If any startup step fails
        """
        env_name = "TESTNET" if self.testnet else "LIVE"
        logger.info(f"=== Starting Binance {env_name} Client ===")
        
        try:
            # Step 1: Sync server time
            logger.info("Step 1/3: Syncing server time...")
            await self.sync_server_time()
            
            # Step 2: Load markets (ccxt) or skip (python-binance)
            logger.info("Step 2/3: Loading exchange info...")
            if self._use_binance_sdk:
                # python-binance: get exchange info
                exchange_info = self.exchange.get_exchange_info()
                symbols_count = len(exchange_info.get('symbols', []))
                self._connected = True
                logger.info(f"  Loaded {symbols_count} symbols")
            else:
                # ccxt: load markets
                await self.exchange.load_markets()
                self._connected = True
                logger.info(f"  Loaded {len(self.exchange.markets)} markets")
            
            # Step 3: Auth check (fetch balance)
            logger.info("Step 3/3: Authenticating...")
            if self._use_binance_sdk:
                # python-binance: get account (matches user's working script)
                account = self.exchange.get_account()
                raw_balances = {}
                for asset in account.get("balances", []):
                    free = float(asset.get("free", 0))
                    locked = float(asset.get("locked", 0))
                    total = free + locked
                    if total > 0:
                        raw_balances[asset["asset"]] = total
            else:
                # ccxt
                balance = await self.exchange.fetch_balance()
                raw_balances = {
                    k: v for k, v in balance.get('total', {}).items()
                    if isinstance(v, (int, float)) and v > 0
                }
            
            # Apply symbol validation filter to remove fake/junk assets
            if SYMBOL_VALIDATOR_AVAILABLE:
                non_zero = filter_balances(raw_balances)
                filtered_count = len(raw_balances) - len(non_zero)
                if filtered_count > 0:
                    logger.info(f"  Filtered {filtered_count} non-tradable assets")
            else:
                non_zero = raw_balances
            
            logger.info(f"  Found {len(non_zero)} tradable assets with balance")
            
            self._startup_complete = True
            logger.info(f"=== Binance {env_name} Client Ready ===")
            
            return {
                "success": True,
                "environment": env_name.lower(),
                "testnet": self.testnet,
                "markets_loaded": symbols_count if self._use_binance_sdk else len(self.exchange.markets),
                "timestamp_offset_ms": self._timestamp_offset,
                "balances": non_zero,
                "rest_endpoint": self.endpoints.rest_base,
                "ws_endpoint": self.endpoints.ws_base
            }

            
        except Exception as e:
            code, msg, is_auth = self._parse_binance_error(e)
            error_msg = self._format_error(e, "Startup failed")
            self._startup_error = error_msg
            logger.error(error_msg)
            
            if is_auth:
                raise BinanceAuthError(error_msg)
            raise Exception(error_msg)
    
    async def connect(self):
        """
        Connect to exchange and load markets.
        For full startup sequence with auth check, use startup() instead.
        """
        try:
            logger.info(f"Connecting to Binance ({'testnet' if self.testnet else 'live'})...")
            await self.sync_server_time()
            
            if self._use_binance_sdk:
                # python-binance doesn't need explicit market loading
                pass
            else:
                await self.exchange.load_markets()
                
            self._connected = True
            logger.info(f"Connected successfully.")
        except Exception as e:
            error_msg = self._format_error(e, "Connection failed")
            logger.error(error_msg)
            raise BinanceAuthError(error_msg) if self._parse_binance_error(e)[2] else Exception(error_msg)
    
    async def close(self):
        """Close connection."""
        if not self._use_binance_sdk:
            await self.exchange.close()
        self._connected = False
        self._startup_complete = False
        logger.info("Disconnected from Binance")

    
    async def destroy(self):
        """
        Completely destroy the client and all state.
        Use when switching sessions or modes.
        """
        try:
            self._running = False
            self._prices.clear()
            self._price_callbacks.clear()

            
            if self._connected and not self._use_binance_sdk:
                await self.exchange.close()
            self._connected = False
            
            logger.info("BinanceClient destroyed - all state cleared")
        except Exception as e:
            logger.error(f"Error destroying client: {e}")
    
    async def validate_credentials(self) -> dict:
        """
        Validate API credentials by attempting to fetch account info.
        
        Returns:
            dict with: success, message, balances (if successful), testnet
        """
        try:
            # Sync server time first (critical for testnet)
            await self.sync_server_time()
            
            if self._use_binance_sdk:
                # python-binance: get account (matches user's working script)
                account = self.exchange.get_account()
                raw_balances = {}
                for asset in account.get("balances", []):
                    free = float(asset.get("free", 0))
                    locked = float(asset.get("locked", 0))
                    total = free + locked
                    if total > 0:
                        raw_balances[asset["asset"]] = total
                self._connected = True
            else:
                # ccxt
                if not self._connected:
                    await self.connect()
                
                balance = await self.exchange.fetch_balance()
                raw_balances = {
                    k: v for k, v in balance.get('total', {}).items()
                    if isinstance(v, (int, float)) and v > 0
                }
            
            # Apply symbol validation filter to remove fake/junk assets
            if SYMBOL_VALIDATOR_AVAILABLE:
                non_zero = filter_balances(raw_balances)
            else:
                non_zero = raw_balances
            
            env_name = "testnet" if self.testnet else "live"
            logger.info(f"Credentials validated ({env_name}). Found {len(non_zero)} tradable assets.")
            
            return {
                "success": True,
                "message": f"API KEY IS WORKING ({env_name.upper()})",
                "balances": non_zero,
                "testnet": self.testnet,
                "environment": self.endpoints.environment.value
            }

            
        except Exception as e:
            error_msg = self._format_error(e, "Credential validation failed")
            logger.error(error_msg)
            
            return {
                "success": False,
                "message": error_msg,
                "balances": {},
                "testnet": self.testnet,
                "environment": self.endpoints.environment.value
            }

    
    async def cancel_all_orders(self, symbol: str = None) -> List[Dict]:
        """Cancel all open orders."""
        cancelled = []
        try:
            orders = await self.get_open_orders(symbol)
            for order in orders:
                try:
                    result = await self.cancel_order(order['id'], order['symbol'])
                    cancelled.append(result)
                except Exception as e:
                    logger.error(f"Failed to cancel order {order['id']}: {e}")
            
            logger.info(f"Cancelled {len(cancelled)} orders")
            return cancelled
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return cancelled
    
    async def fetch_fresh_state(self) -> Dict:
        """
        Fetch fresh state from exchange (balances, orders, positions).
        Used when starting a new session.
        """
        try:
            if not self._connected:
                await self.connect()
            
            balance_task = self.exchange.fetch_balance()
            orders_task = self.exchange.fetch_open_orders()
            
            balance, orders = await asyncio.gather(balance_task, orders_task)
            
            balances = {
                k: {"free": v.get("free", 0), "used": v.get("used", 0), "total": v.get("total", 0)}
                for k, v in balance.items()
                if isinstance(v, dict) and v.get("total", 0) > 0
            }
            
            return {
                "balances": balances,
                "open_orders": orders,
                "positions": [],
                "fetched_at": datetime.now().isoformat(),
                "environment": self.endpoints.environment.value
            }
        except Exception as e:
            error_msg = self._format_error(e, "Failed to fetch fresh state")
            logger.error(error_msg)
            return {
                "balances": {},
                "open_orders": [],
                "positions": [],
                "error": error_msg,
                "environment": self.endpoints.environment.value
            }
    
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
            logger.error(self._format_error(e, "Failed to get balance"))
            return {}
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker for a symbol."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            self._prices[symbol] = ticker['last']
            return ticker
        except Exception as e:
            logger.error(self._format_error(e, f"Failed to get ticker for {symbol}"))
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
        side: str,
        order_type: str,
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
            error_msg = self._format_error(e, f"Failed to create order")
            logger.error(error_msg)
            raise Exception(error_msg)
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an order."""
        try:
            result = await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order cancelled: {order_id}")
            return result
        except Exception as e:
            error_msg = self._format_error(e, f"Failed to cancel order {order_id}")
            logger.error(error_msg)
            raise Exception(error_msg)
    
    async def fetch_order(self, order_id: str, symbol: str) -> Dict:
        """Fetch order status."""
        try:
            return await self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            error_msg = self._format_error(e, f"Failed to fetch order {order_id}")
            logger.error(error_msg)
            raise Exception(error_msg)
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders."""
        try:
            return await self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(self._format_error(e, "Failed to get open orders"))
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
            logger.error(self._format_error(e, "Failed to get OHLCV"))
            return []
    
    # === Price Feed ===
    
    async def start_price_feed(self, symbols: List[str], interval: float = 1.0):
        """Start polling price feed for symbols."""
        logger.info(f"Starting price feed for {symbols}")
        
        while self._running:
            try:
                for symbol in symbols:
                    ticker = await self.get_ticker(symbol)
                    if ticker and 'last' in ticker:
                        price = ticker['last']
                        self._prices[symbol] = price
                        
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
    
    def get_environment_info(self) -> Dict:
        """Get current environment information."""
        return {
            "testnet": self.testnet,
            "environment": self.endpoints.environment.value,
            "rest_endpoint": self.endpoints.rest_base,
            "ws_endpoint": self.endpoints.ws_base,
            "connected": self._connected
        }


# === Factory Functions ===

def create_binance_client(
    testnet: bool = None,
    api_key: str = None,
    api_secret: str = None
) -> BinanceClient:
    """
    Create a Binance client with proper configuration.
    
    Args:
        testnet: True for testnet, False for live. If None, reads from 
                 BINANCE_TESTNET_ENABLED env var
        api_key: API key (or reads from BINANCE_API_KEY env var)
        api_secret: API secret (or reads from BINANCE_API_SECRET env var)
    
    Returns:
        Configured BinanceClient instance
    
    Raises:
        BinanceConfigError: If configuration is invalid
    """
    return BinanceClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )


async def test_binance_connection(
    testnet: bool = None,
    api_key: str = None,
    api_secret: str = None
) -> Dict:
    """
    Test Binance connection and credentials.
    
    Returns:
        Dict with connection test results
    """
    client = None
    try:
        client = create_binance_client(testnet, api_key, api_secret)
        result = await client.validate_credentials()
        return result
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "balances": {}
        }
    finally:
        if client:
            await client.destroy()
