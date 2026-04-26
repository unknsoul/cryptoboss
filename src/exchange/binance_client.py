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
import aiohttp
from typing import Any, Callable, Dict, List, Optional
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

        # Retry + circuit breaker state
        self._error_count: int = 0
        self._circuit_open: bool = False
        self._circuit_open_at: Optional[datetime] = None
        self._retry_delays: List[float] = [0.5, 1.0, 2.0]
        self._circuit_error_threshold: int = 5
        self._circuit_reset_seconds: int = 30
        
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

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Convert a slash-formatted symbol to Binance REST format."""
        return symbol.replace("/", "").upper()

    @staticmethod
    def _restore_symbol(symbol: str) -> str:
        """Convert a Binance REST symbol into the slash-separated form used internally."""
        uppercase = symbol.upper()
        for quote in ("USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"):
            if uppercase.endswith(quote) and len(uppercase) > len(quote):
                return f"{uppercase[:-len(quote)]}/{quote}"
        return uppercase

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        """Best-effort float conversion for exchange payload values."""
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _format_quantity(value: float) -> float:
        """Round quantities to a practical exchange-safe precision."""
        return float(f"{value:.8f}")

    async def _sdk_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Run blocking python-binance SDK calls off the event loop."""
        method = getattr(self.exchange, method_name)
        return await asyncio.to_thread(method, *args, **kwargs)

    def _normalize_ccxt_balances(self, balance: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Normalize a ccxt balance payload into free/used/total buckets."""
        totals = balance.get("total", {}) if isinstance(balance, dict) else {}
        free_map = balance.get("free", {}) if isinstance(balance, dict) else {}
        used_map = balance.get("used", {}) if isinstance(balance, dict) else {}
        normalized: Dict[str, Dict[str, float]] = {}

        for currency, total in totals.items():
            total_value = self._coerce_float(total)
            if total_value <= 0:
                continue
            normalized[currency] = {
                "free": self._coerce_float(free_map.get(currency), total_value),
                "used": self._coerce_float(used_map.get(currency), 0.0),
                "total": total_value,
            }

        return normalized

    def _normalize_sdk_balances(self, account: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Normalize a python-binance account payload into free/used/total buckets."""
        normalized: Dict[str, Dict[str, float]] = {}

        for asset in account.get("balances", []):
            free = self._coerce_float(asset.get("free"))
            used = self._coerce_float(asset.get("locked"))
            total = free + used
            if total <= 0:
                continue
            normalized[str(asset.get("asset"))] = {
                "free": free,
                "used": used,
                "total": total,
            }

        return normalized

    @staticmethod
    def _flatten_balances(balance_map: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Reduce normalized balances down to asset totals."""
        return {
            currency: float(values.get("total", 0.0))
            for currency, values in balance_map.items()
            if float(values.get("total", 0.0)) > 0
        }

    def _normalize_sdk_order(
        self,
        order: Dict[str, Any],
        symbol: str,
        side: str,
        order_type: str,
        fallback_quantity: float,
        fallback_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Map python-binance order responses to the ccxt-like shape used elsewhere."""
        executed_qty = self._coerce_float(order.get("executedQty"), self._coerce_float(order.get("origQty"), fallback_quantity))
        quote_qty = self._coerce_float(order.get("cummulativeQuoteQty"))
        fills = order.get("fills", []) or []
        fee_cost = sum(self._coerce_float(fill.get("commission")) for fill in fills)
        avg_price = self._coerce_float(order.get("price"))

        if executed_qty > 0:
            if quote_qty > 0:
                avg_price = quote_qty / executed_qty
            elif fills:
                fill_notional = sum(
                    self._coerce_float(fill.get("price")) * self._coerce_float(fill.get("qty"))
                    for fill in fills
                )
                avg_price = fill_notional / executed_qty if fill_notional > 0 else avg_price

        if avg_price <= 0 and fallback_price is not None:
            avg_price = float(fallback_price)

        return {
            **order,
            "id": str(order.get("orderId", "")),
            "symbol": symbol,
            "side": str(order.get("side", side.upper())).lower(),
            "type": str(order.get("type", order_type.upper())).lower(),
            "price": avg_price,
            "average": avg_price,
            "filled": executed_qty,
            "fee": {"cost": fee_cost},
        }

    
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

    def _register_request_success(self):
        """Reset circuit breaker state after a successful request."""
        self._error_count = 0
        if self._circuit_open:
            logger.info("Circuit breaker closed after successful probe request")
        self._circuit_open = False
        self._circuit_open_at = None

    def _register_request_failure(self):
        """Track request failures and open the circuit when threshold is reached."""
        self._error_count += 1
        if self._error_count >= self._circuit_error_threshold and not self._circuit_open:
            self._circuit_open = True
            self._circuit_open_at = datetime.utcnow()
            logger.error(
                "Circuit breaker opened after %s consecutive errors",
                self._error_count,
            )

    def _circuit_breaker_check(self) -> bool:
        """Return True when calls are allowed, False when circuit is open."""
        if not self._circuit_open:
            return True

        if self._circuit_open_at is None:
            return False

        elapsed = (datetime.utcnow() - self._circuit_open_at).total_seconds()
        if elapsed >= self._circuit_reset_seconds:
            # Half-open probe: allow one request to test connectivity.
            self._circuit_open = False
            self._error_count = 0
            self._circuit_open_at = None
            logger.warning("Circuit breaker moving to half-open after cooldown")
            return True
        return False

    async def _request_with_retry(
        self,
        method,
        endpoint,
        max_retries=3,
        **kwargs,
    ):
        """Execute public REST request with retry and circuit breaker safeguards."""
        if not self._circuit_breaker_check():
            raise RuntimeError("Binance circuit breaker is open")

        retries = max(1, int(max_retries))
        url = f"{self.endpoints.rest_base}{endpoint}"
        params = kwargs.pop("params", None)
        timeout = float(kwargs.pop("timeout", 10))

        for attempt in range(retries):
            try:
                request_timeout = aiohttp.ClientTimeout(total=timeout)
                async with aiohttp.ClientSession(timeout=request_timeout) as session:
                    async with session.request(
                        method=method.upper(),
                        url=url,
                        params=params,
                        **kwargs,
                    ) as response:
                        if response.status >= 400:
                            body = await response.text()
                            raise Exception(f"HTTP {response.status}: {body}")

                        data = await response.json(content_type=None)
                        self._register_request_success()
                        return data

            except Exception as exc:
                self._register_request_failure()
                if attempt == retries - 1:
                    raise Exception(
                        self._format_error(exc, f"Request failed after {retries} attempts")
                    ) from exc

                delay_idx = min(attempt, len(self._retry_delays) - 1)
                delay = self._retry_delays[delay_idx]
                await asyncio.sleep(delay)
    
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
                # python-binance SDK
                server_time_response = await self._sdk_call("get_server_time")
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
                exchange_info = await self._sdk_call("get_exchange_info")
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
                account = await self._sdk_call("get_account")
                normalized_balances = self._normalize_sdk_balances(account)
            else:
                # ccxt
                balance = await self.exchange.fetch_balance()
                normalized_balances = self._normalize_ccxt_balances(balance)

            raw_balances = self._flatten_balances(normalized_balances)
            
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
                account = await self._sdk_call("get_account")
                normalized_balances = self._normalize_sdk_balances(account)
                self._connected = True
            else:
                # ccxt
                if not self._connected:
                    await self.connect()
                
                balance = await self.exchange.fetch_balance()
                normalized_balances = self._normalize_ccxt_balances(balance)

            raw_balances = self._flatten_balances(normalized_balances)
            
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

            if self._use_binance_sdk:
                account, orders = await asyncio.gather(
                    self._sdk_call("get_account"),
                    self._sdk_call("get_open_orders"),
                )
                balances = self._normalize_sdk_balances(account)
            else:
                balance, orders = await asyncio.gather(
                    self.exchange.fetch_balance(),
                    self.exchange.fetch_open_orders(),
                )
                balances = self._normalize_ccxt_balances(balance)
            
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
    
    async def get_balance(self) -> Dict[str, Dict[str, float]]:
        """Get account balances."""
        try:
            if self._use_binance_sdk:
                account = await self._sdk_call("get_account")
                return self._normalize_sdk_balances(account)

            balance = await self.exchange.fetch_balance()
            return self._normalize_ccxt_balances(balance)
        except Exception as e:
            logger.error(self._format_error(e, "Failed to get balance"))
            return {}
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker for a symbol."""
        try:
            if self._use_binance_sdk:
                symbol_clean = self._normalize_symbol(symbol)
                raw_ticker = await self._sdk_call("get_ticker", symbol=symbol_clean)
                ticker = {
                    **raw_ticker,
                    "symbol": symbol,
                    "last": self._coerce_float(raw_ticker.get("lastPrice"), self._coerce_float(raw_ticker.get("price"))),
                    "bid": self._coerce_float(raw_ticker.get("bidPrice")),
                    "ask": self._coerce_float(raw_ticker.get("askPrice")),
                    "high": self._coerce_float(raw_ticker.get("highPrice")),
                    "low": self._coerce_float(raw_ticker.get("lowPrice")),
                    "baseVolume": self._coerce_float(raw_ticker.get("volume")),
                    "percentage": self._coerce_float(raw_ticker.get("priceChangePercent")),
                }
            else:
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
            if self._use_binance_sdk:
                order_params: Dict[str, Any] = {
                    "symbol": self._normalize_symbol(symbol),
                    "side": side.upper(),
                    "type": order_type.upper(),
                    "quantity": self._format_quantity(amount),
                    "newOrderRespType": "FULL",
                }
                if price is not None and order_type.lower() != "market":
                    order_params["price"] = self._format_quantity(price)
                    order_params.setdefault("timeInForce", "GTC")
                if params:
                    order_params.update(params)
                raw_order = await self._sdk_call("create_order", **order_params)
                order = self._normalize_sdk_order(raw_order, symbol, side, order_type, amount, price)
            else:
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
            if self._use_binance_sdk:
                result = await self._sdk_call(
                    "cancel_order",
                    symbol=self._normalize_symbol(symbol),
                    orderId=order_id,
                )
                result = {
                    **result,
                    "id": str(result.get("orderId", order_id)),
                    "symbol": symbol,
                }
            else:
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
            if self._use_binance_sdk:
                raw_order = await self._sdk_call(
                    "get_order",
                    symbol=self._normalize_symbol(symbol),
                    orderId=order_id,
                )
                return self._normalize_sdk_order(
                    raw_order,
                    symbol,
                    str(raw_order.get("side", "")).lower(),
                    str(raw_order.get("type", "")).lower(),
                    self._coerce_float(raw_order.get("origQty")),
                    self._coerce_float(raw_order.get("price")),
                )
            return await self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            error_msg = self._format_error(e, f"Failed to fetch order {order_id}")
            logger.error(error_msg)
            raise Exception(error_msg)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders."""
        try:
            if self._use_binance_sdk:
                kwargs: Dict[str, Any] = {}
                if symbol:
                    kwargs["symbol"] = self._normalize_symbol(symbol)
                orders = await self._sdk_call("get_open_orders", **kwargs)
                return [
                    self._normalize_sdk_order(
                        order,
                        symbol or self._restore_symbol(str(order.get("symbol", ""))),
                        str(order.get("side", "")).lower(),
                        str(order.get("type", "")).lower(),
                        self._coerce_float(order.get("origQty")),
                        self._coerce_float(order.get("price")),
                    )
                    for order in orders
                ]
            return await self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(self._format_error(e, "Failed to get open orders"))
            return []

    async def get_exchange_info(self) -> Dict:
        """Get exchange metadata (symbols, filters, limits)."""
        return await self._request_with_retry("GET", "/api/v3/exchangeInfo")

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Get spot order book snapshot for a symbol."""
        symbol_clean = symbol.replace("/", "").upper()
        return await self._request_with_retry(
            "GET",
            "/api/v3/depth",
            params={"symbol": symbol_clean, "limit": int(limit)},
        )

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Get recent public trades for a symbol."""
        symbol_clean = symbol.replace("/", "").upper()
        return await self._request_with_retry(
            "GET",
            "/api/v3/trades",
            params={"symbol": symbol_clean, "limit": int(limit)},
        )
    
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100
    ) -> List:
        """Get OHLCV candlestick data."""
        try:
            if self._use_binance_sdk:
                raw_klines = await self._sdk_call(
                    "get_klines",
                    symbol=self._normalize_symbol(symbol),
                    interval=timeframe,
                    limit=limit,
                )
                return [
                    [
                        int(kline[0]),
                        self._coerce_float(kline[1]),
                        self._coerce_float(kline[2]),
                        self._coerce_float(kline[3]),
                        self._coerce_float(kline[4]),
                        self._coerce_float(kline[5]),
                    ]
                    for kline in raw_klines
                ]
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


# === MarketDataService ===

class MarketDataService:
    """
    WebSocket-based live price streaming service.
    
    Provides real-time price updates with proper event emission for frontend consumption.
    
    Usage:
        service = MarketDataService(exchange_account_id="acc-123", testnet=True)
        await service.start()
        service.subscribe("BTCUSDT", callback)
        # ... later
        service.unsubscribe("BTCUSDT", callback)
        await service.stop()
    
    Events emitted to callbacks:
        {
            "exchange_account_id": "acc-123",
            "symbol": "BTCUSDT",
            "price": 45000.50,
            "timestamp_ms": 1707161234567,
            "source": "TESTNET" | "LIVE"
        }
    """
    
    def __init__(
        self,
        exchange_account_id: str,
        testnet: bool = None,
        poll_interval: float = 1.0
    ):
        """
        Initialize MarketDataService.
        
        Args:
            exchange_account_id: Account ID for scoping price events
            testnet: True for testnet, False for live
            poll_interval: Seconds between price polls (WebSocket fallback)
        """
        if testnet is None:
            testnet = get_testnet_enabled()
        
        self.exchange_account_id = exchange_account_id
        self.testnet = testnet
        
        # CRITICAL: Always use MAINNET for prices (read-only), testnet for trading
        # Testnet does NOT have reliable market data
        self.price_endpoints = BinanceEndpoints.live()  # MAINNET for prices
        self.trading_endpoints = BinanceEndpoints.testnet() if testnet else BinanceEndpoints.live()
        self.poll_interval = poll_interval
        
        # Subscriptions: {symbol: [callback1, callback2, ...]}
        self._subscriptions: Dict[str, List[Callable]] = {}
        
        # Latest prices: {symbol: price}
        self._prices: Dict[str, float] = {}
        
        # State
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._ws_connection = None
        
        # Source tag - MAINNET for prices always
        self._source = "TESTNET" if testnet else "LIVE"  # Trading environment
        self._price_source = "LIVE"  # Price source always mainnet
        
        env_name = "TESTNET" if testnet else "LIVE"
        logger.info(f"MarketDataService initialized (trading: {env_name}, prices: MAINNET) for account {exchange_account_id}")
    
    async def start(self) -> bool:
        """
        Start the price streaming service.
        
        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("MarketDataService already running")
            return True
        
        try:
            self._running = True
            
            # Start price polling (WebSocket upgrade can be added later)
            self._poll_task = asyncio.create_task(self._price_poll_loop())
            
            logger.info(f"MarketDataService started for account {self.exchange_account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MarketDataService: {e}")
            self._running = False
            return False
    
    async def stop(self):
        """Stop the price streaming service."""
        self._running = False
        
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None
        
        logger.info(f"MarketDataService stopped for account {self.exchange_account_id}")
    
    def subscribe(self, symbol: str, callback: Callable):
        """
        Subscribe to price updates for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g. "BTCUSDT")
            callback: Function to call with price events
        """
        symbol = symbol.upper()
        
        if symbol not in self._subscriptions:
            self._subscriptions[symbol] = []
        
        if callback not in self._subscriptions[symbol]:
            self._subscriptions[symbol].append(callback)
            logger.info(f"Subscribed to {symbol} price updates")
    
    def unsubscribe(self, symbol: str, callback: Callable = None):
        """
        Unsubscribe from price updates.
        
        Args:
            symbol: Trading pair symbol
            callback: Specific callback to remove, or None to remove all
        """
        symbol = symbol.upper()
        
        if symbol not in self._subscriptions:
            return
        
        if callback is None:
            del self._subscriptions[symbol]
            logger.info(f"Unsubscribed all callbacks from {symbol}")
        elif callback in self._subscriptions[symbol]:
            self._subscriptions[symbol].remove(callback)
            if not self._subscriptions[symbol]:
                del self._subscriptions[symbol]
            logger.info(f"Unsubscribed callback from {symbol}")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest cached price for a symbol."""
        return self._prices.get(symbol.upper())
    
    def get_all_prices(self) -> Dict[str, Dict]:
        """
        Get all cached prices with metadata.
        
        Returns:
            Dict of {symbol: {price, timestamp_ms, source}}
        """
        import time
        now_ms = int(time.time() * 1000)
        
        return {
            symbol: {
                "price": price,
                "timestamp_ms": now_ms,
                "source": self._source
            }
            for symbol, price in self._prices.items()
        }
    
    async def _price_poll_loop(self):
        """Internal price polling loop."""
        import aiohttp
        import time
        
        while self._running:
            try:
                # Get symbols to poll
                symbols = list(self._subscriptions.keys())
                
                if not symbols:
                    await asyncio.sleep(self.poll_interval)
                    continue
                
                # Fetch prices from MAINNET Binance REST API (always mainnet for reliable prices)
                async with aiohttp.ClientSession() as session:
                    for symbol in symbols:
                        try:
                            # CRITICAL: Use price_endpoints (MAINNET) not trading_endpoints
                            url = f"{self.price_endpoints.rest_base}/api/v3/ticker/price?symbol={symbol}"
                            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    price = float(data.get("price", 0))
                                    
                                    if price > 0:
                                        self._prices[symbol] = price
                                        await self._emit_price_event(symbol, price)
                                        
                        except Exception as e:
                            logger.debug(f"Failed to fetch price for {symbol}: {e}")
                
                await asyncio.sleep(self.poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Price poll error: {e}")
                await asyncio.sleep(5)
    
    async def _emit_price_event(self, symbol: str, price: float):
        """Emit price event to all subscribers."""
        import time
        
        if symbol not in self._subscriptions:
            return
        
        event = {
            "exchange_account_id": self.exchange_account_id,
            "symbol": symbol,
            "price": price,
            "timestamp_ms": int(time.time() * 1000),
            "source": self._source
        }
        
        for callback in self._subscriptions[symbol]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Price callback error for {symbol}: {e}")


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
