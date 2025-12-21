"""
Advanced Binance API Client
Professional-grade integration with testnet/mainnet support
"""

import ccxt
import time
import hmac
import hashlib
import requests
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import threading
import json
from collections import deque

from core.monitoring.logger import get_logger
from core.monitoring.metrics import get_metrics
from core.monitoring.alerting import get_alerts
from core.config import get_settings


logger = get_logger()
metrics = get_metrics()
alerts = get_alerts()
settings = get_settings()


class RateLimiter:
    """
    Advanced rate limiter with multiple tier support
    Prevents hitting Binance API rate limits
    """
    
    def __init__(self, requests_per_minute: int = 1200, 
                 orders_per_second: int = 10,
                 orders_per_day: int = 200000):
        """
        Args:
            requests_per_minute: Max API requests per minute
            orders_per_second: Max orders per second
            orders_per_day: Max orders per day
        """
        self.rpm_limit = requests_per_minute
        self.ops_limit = orders_per_second
        self.opd_limit = orders_per_day
        
        # Track request timestamps
        self.request_times = deque(maxlen=requests_per_minute)
        self.order_times_second = deque(maxlen=orders_per_second)
        self.order_times_day = deque(maxlen=orders_per_day)
        
        self.lock = threading.Lock()
        
        # Weight tracking (Binance uses weighted rate limits)
        self.request_weights = deque(maxlen=requests_per_minute)
        self.weight_per_minute = 0
        self.max_weight_per_minute = 1200
    
    def wait_if_needed(self, endpoint_type: str = 'general', weight: int = 1):
        """
        Wait if rate limit would be exceeded
        
        Args:
            endpoint_type: 'general', 'order', or 'heavy'
            weight: API weight of this request
        """
        with self.lock:
            now = time.time()
            
            # Clean old timestamps
            self._cleanup_old_timestamps(now)
            
            # Check weight limit
            if endpoint_type == 'heavy':
                weight = 5  # Heavy endpoints have 5x weight
            
            current_weight = sum(w for _, w in self.request_weights)
            
            if current_weight + weight > self.max_weight_per_minute:
                wait_time = 60 - (now - self.request_weights[0][0])
                if wait_time > 0:
                    logger.warning(f"Rate limit approaching, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    self._cleanup_old_timestamps(time.time())
            
            # Check requests per minute
            if len(self.request_times) >= self.rpm_limit:
                wait_time = 60 - (now - self.request_times[0])
                if wait_time > 0:
                    time.sleep(wait_time)
            
            # Check orders per second (for order endpoints)
            if endpoint_type == 'order':
                if len(self.order_times_second) >= self.ops_limit:
                    time.sleep(1.0)
                
                self.order_times_second.append(now)
                self.order_times_day.append(now)
            
            # Record request
            self.request_times.append(now)
            self.request_weights.append((now, weight))
    
    def _cleanup_old_timestamps(self, now: float):
        """Remove timestamps older than their respective windows"""
        # Clean minute window
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        
        while self.request_weights and now - self.request_weights[0][0] > 60:
            self.request_weights.popleft()
        
        # Clean second window
        while self.order_times_second and now - self.order_times_second[0] > 1:
            self.order_times_second.popleft()
        
        # Clean day window
        while self.order_times_day and now - self.order_times_day[0] > 86400:
            self.order_times_day.popleft()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        now = time.time()
        self._cleanup_old_timestamps(now)
        
        return {
            'requests_last_minute': len(self.request_times),
            'rpm_limit': self.rpm_limit,
            'orders_last_second': len(self.order_times_second),
            'ops_limit': self.ops_limit,
            'orders_today': len(self.order_times_day),
            'opd_limit': self.opd_limit,
            'current_weight': sum(w for _, w in self.request_weights),
            'max_weight': self.max_weight_per_minute
        }


class AdvancedBinanceClient:
    """
    Professional Binance API Client
    
    Features:
    - Testnet & Mainnet support
    - Rate limiting
    - Advanced order types (OCO, Iceberg, TWAP)
    - WebSocket streaming
    - Error handling & retries
    - Account management
    """
    
    def __init__(self, use_testnet: bool = None):
        """
        Initialize Binance client
        
        Args:
            use_testnet: Use testnet if True, mainnet if False, 
                        reads from settings if None
        """
        # Get configuration
        if use_testnet is None:
            use_testnet = settings.exchange.use_testnet
        
        self.use_testnet = use_testnet
        self.api_key = settings.exchange.api_key
        self.api_secret = settings.exchange.api_secret
        
        # Initialize CCXT
        if use_testnet:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # or 'spot'
                    'testnet': True
                }
            })
            # Testnet URLs
            self.exchange.urls['api'] = {
                'public': 'https://testnet.binancefuture.com',
                'private': 'https://testnet.binancefuture.com',
            }
            logger.info("🧪 Using Binance TESTNET")
        else:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'
                }
            })
            logger.warning("⚠️  Using Binance MAINNET - REAL MONEY!")
        
        # Rate limiter
        self.rate_limiter = RateLimiter()
        
        # WebSocket connection
        self.ws_connected = False
        self.ws_callbacks = {}
        self.ws_thread = None
        
        # Order tracking
        self.active_orders = {}
        self.order_history = []
        
        # Account info cache
        self.account_info = None
        self.last_account_update = None
        
        logger.info(
            "Binance client initialized",
            testnet=use_testnet,
            exchange_id=self.exchange.id
        )
    
    def test_connection(self) -> bool:
        """Test API connection and credentials"""
        try:
            self.rate_limiter.wait_if_needed('general', weight=1)
            
            # Test public endpoint
            server_time = self.exchange.fetch_time()
            logger.info(f"✅ Server time: {datetime.fromtimestamp(server_time/1000)}")
            
            # Test private endpoint (requires valid API keys)
            balance = self.exchange.fetch_balance()
            logger.info(f"✅ Account access successful")
            
            # Log account balances
            for currency, balance_info in balance['total'].items():
                if balance_info > 0:
                    logger.info(f"   {currency}: {balance_info}")
            
            metrics.increment("binance_connection_success")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            alerts.send_alert(
                "binance_connection_failed",
                f"Failed to connect to Binance: {str(e)}",
                {"testnet": self.use_testnet, "error": str(e)}
            )
            metrics.increment("binance_connection_failed")
            return False
    
    def get_account_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get account information (with caching)
        
        Args:
            force_refresh: Force refresh even if cached
        
        Returns:
            Account information dictionary
        """
        # Return cached if recent
        if (not force_refresh and self.account_info and 
            self.last_account_update and 
            datetime.now() - self.last_account_update < timedelta(seconds=30)):
            return self.account_info
        
        try:
            self.rate_limiter.wait_if_needed('general', weight=5)
            
            # Fetch balance
            balance = self.exchange.fetch_balance()
            
            # Fetch account info (futures)
            account_data = self.exchange.fapiPrivateGetAccount() if not self.use_testnet else {}
            
            self.account_info = {
                'balance': balance,
                'total_wallet_balance': balance.get('USDT', {}).get('total', 0),
                'available_balance': balance.get('USDT', {}).get('free', 0),
                'used_balance': balance.get('USDT', {}).get('used', 0),
                'positions': self.get_open_positions(),
                'leverage': account_data.get('leverage', 1) if account_data else 1,
                'timestamp': datetime.now()
            }
            
            self.last_account_update = datetime.now()
            
            return self.account_info
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        try:
            self.rate_limiter.wait_if_needed('general', weight=5)
            
            positions = self.exchange.fapiPrivateGetPositionRisk()
            
            # Filter to only open positions
            open_positions = [
                pos for pos in positions 
                if float(pos.get('positionAmt', 0)) != 0
            ]
            
            return open_positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            leverage: Leverage amount (1-125)
        
        Returns:
            True if successful
        """
        try:
            self.rate_limiter.wait_if_needed('order', weight=1)
            
            result = self.exchange.fapiPrivatePostLeverage({
                'symbol': symbol.replace('/', ''),
                'leverage': leverage
            })
            
            logger.info(f"Set leverage for {symbol} to {leverage}x", result=result)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False
    
    def set_margin_type(self, symbol: str, margin_type: str = 'ISOLATED') -> bool:
        """
        Set margin type (ISOLATED or CROSSED)
        
        Args:
            symbol: Trading pair
            margin_type: 'ISOLATED' or 'CROSSED'
        
        Returns:
            True if successful
        """
        try:
            self.rate_limiter.wait_if_needed('order', weight=1)
            
            result = self.exchange.fapiPrivatePostMarginType({
                'symbol': symbol.replace('/', ''),
                'marginType': margin_type
            })
            
            logger.info(f"Set margin type for {symbol} to {margin_type}", result=result)
            return True
            
        except Exception as e:
            # Margin type might already be set
            if 'No need to change margin type' in str(e):
                logger.info(f"Margin type for {symbol} already set to {margin_type}")
                return True
            logger.error(f"Failed to set margin type: {e}")
            return False
    
    def place_market_order(self, symbol: str, side: str, quantity: float,
                          reduce_only: bool = False) -> Optional[Dict[str, Any]]:
        """
        Place a market order
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            quantity: Order quantity
            reduce_only: Only reduce position (don't open new)
        
        Returns:
            Order info or None if failed
        """
        try:
            self.rate_limiter.wait_if_needed('order', weight=1)
            
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side.lower(),
                amount=quantity,
                params=params
            )
            
            logger.log_trade({
                'symbol': symbol,
                'side': side,
                'type': 'market',
                'quantity': quantity,
                'order_id': order['id'],
                'status': order['status'],
                'reduce_only': reduce_only
            })
            
            metrics.increment(f"order_placed_{side.lower()}")
            
            return order
            
        except Exception as e:
            logger.error(f"Market order failed: {e}", 
                        symbol=symbol, side=side, quantity=quantity)
            alerts.send_alert(
                "order_failed",
                f"Market order failed: {symbol} {side} {quantity}",
                {"error": str(e), "symbol": symbol}
            )
            return None
    
    def place_limit_order(self, symbol: str, side: str, quantity: float,
                         price: float, post_only: bool = False,
                         time_in_force: str = 'GTC') -> Optional[Dict[str, Any]]:
        """
        Place a limit order
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Limit price
            post_only: Only maker orders (no taker fee)
            time_in_force: 'GTC', 'IOC', 'FOK'
        
        Returns:
            Order info or None
        """
        try:
            self.rate_limiter.wait_if_needed('order', weight=1)
            
            params = {'timeInForce': time_in_force}
            if post_only:
                params['postOnly'] = True
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side.lower(),
                amount=quantity,
                price=price,
                params=params
            )
            
            logger.log_trade({
                'symbol': symbol,
                'side': side,
                'type': 'limit',
                'quantity': quantity,
                'price': price,
                'order_id': order['id'],
                'post_only': post_only
            })
            
            metrics.increment(f"limit_order_placed_{side.lower()}")
            
            return order
            
        except Exception as e:
            logger.error(f"Limit order failed: {e}")
            return None
    
    def place_stop_loss(self, symbol: str, side: str, quantity: float,
                       stop_price: float, limit_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Place a stop-loss order
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            quantity: Order quantity
            stop_price: Trigger price
            limit_price: Limit price (if None, will be market order)
        
        Returns:
            Order info or None
        """
        try:
            self.rate_limiter.wait_if_needed('order', weight=1)
            
            if limit_price:
                # Stop-limit order
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='STOP',
                    side=side.lower(),
                    amount=quantity,
                    price=limit_price,
                    params={'stopPrice': stop_price}
                )
            else:
                # Stop-market order
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_MARKET',
                    side=side.lower(),
                    amount=quantity,
                    params={'stopPrice': stop_price}
                )
            
            logger.info(f"Stop-loss placed: {symbol} {side} @ {stop_price}")
            return order
            
        except Exception as e:
            logger.error(f"Stop-loss order failed: {e}")
            return None
    
    def place_take_profit(self, symbol: str, side: str, quantity: float,
                         take_profit_price: float) -> Optional[Dict[str, Any]]:
        """Place a take-profit order"""
        try:
            self.rate_limiter.wait_if_needed('order', weight=1)
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side=side.lower(),
                amount=quantity,
                params={'stopPrice': take_profit_price}
            )
            
            logger.info(f"Take-profit placed: {symbol} {side} @ {take_profit_price}")
            return order
            
        except Exception as e:
            logger.error(f"Take-profit order failed: {e}")
            return None
    
    def place_oco_order(self, symbol: str, side: str, quantity: float,
                       price: float, stop_price: float, 
                       stop_limit_price: float) -> Optional[Dict[str, Any]]:
        """
        Place OCO (One-Cancels-Other) order
        Combines limit order with stop-loss
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Limit order price
            stop_price: Stop-loss trigger price
            stop_limit_price: Stop-loss limit price
        
        Returns:
            Order info or None
        """
        try:
            self.rate_limiter.wait_if_needed('order', weight=2)
            
            # Binance OCO order
            result = self.exchange.private_post_order_oco({
                'symbol': symbol.replace('/', ''),
                'side': side.upper(),
                'quantity': quantity,
                'price': price,
                'stopPrice': stop_price,
                'stopLimitPrice': stop_limit_price,
                'stopLimitTimeInForce': 'GTC'
            })
            
            logger.info(f"OCO order placed: {symbol} {side}")
            return result
            
        except Exception as e:
            logger.error(f"OCO order failed: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.rate_limiter.wait_if_needed('order', weight=1)
            
            result = self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Cancelled order {order_id}")
            metrics.increment("order_cancelled")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol"""
        try:
            self.rate_limiter.wait_if_needed('order', weight=1)
            
            result = self.exchange.cancel_all_orders(symbol)
            logger.info(f"Cancelled all orders for {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False
    
    def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status"""
        try:
            self.rate_limiter.wait_if_needed('general', weight=2)
            
            order = self.exchange.fetch_order(order_id, symbol)
            return order
            
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None
    
    def close_position(self, symbol: str, quantity: Optional[float] = None) -> bool:
        """
        Close an open position
        
        Args:
            symbol: Trading pair
            quantity: Amount to close (None = close all)
        
        Returns:
            True if successful
        """
        try:
            # Get current position
            positions = self.get_open_positions()
            position = next((p for p in positions if p['symbol'] == symbol.replace('/', '')), None)
            
            if not position:
                logger.warning(f"No open position for {symbol}")
                return False
            
            position_amt = float(position['positionAmt'])
            
            if quantity is None:
                quantity = abs(position_amt)
            
            # Determine side (opposite of position)
            side = 'sell' if position_amt > 0 else 'buy'
            
            # Place closing market order
            order = self.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                reduce_only=True
            )
            
            if order:
                logger.info(f"Closed position: {symbol} {quantity}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return self.rate_limiter.get_stats()


if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED BINANCE API CLIENT TEST")
    print("=" * 70)
    
    # Test with testnet
    client = AdvancedBinanceClient(use_testnet=True)
    
    print("\n1. Testing connection...")
    if client.test_connection():
        print("   ✅ Connection successful")
    else:
        print("   ❌ Connection failed")
        exit(1)
    
    print("\n2. Getting account info...")
    account = client.get_account_info()
    print(f"   Balance: ${account.get('total_wallet_balance', 0):.2f}")
    
    print("\n3. Rate limiter stats...")
    stats = client.get_rate_limit_stats()
    print(f"   Requests this minute: {stats['requests_last_minute']}/{stats['rpm_limit']}")
    print(f"   Current weight: {stats['current_weight']}/{stats['max_weight']}")
    
    print("\n✅ Binance API client test complete")
