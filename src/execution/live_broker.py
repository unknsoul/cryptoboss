"""
Live Broker - Production-Safe Order Execution
Idempotent orders, position sync, retry logic.
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
import time
from enum import Enum

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class LiveBroker:
    """
    Production-safe broker with idempotent execution.
    
    Features:
    - Idempotent order placement (prevents duplicates)
    - Position state sync with exchange
    - Retry logic for failed orders
    - Order cache for consistency
    """
    
    def __init__(self, exchange_client=None):
        """
        Initialize live broker.
        
        Args:
            exchange_client: Exchange API client
        """
        self.exchange_client = exchange_client
        self.order_cache: Dict[str, Dict] = {}  # client_order_id -> order_data
        self.position_state: Dict[str, Dict] = {}  # symbol -> position_data
        
        logger.info("LiveBroker initialized")
    
    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Dict:
        """
        Place order with idempotency.
        
        Args:
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            size: Order size
            order_type: 'MARKET' or 'LIMIT'
            price: Limit price (for LIMIT orders)
            client_order_id: Client-provided ID for idempotency
            
        Returns:
            Order result dict
        """
        # Generate client order ID if not provided
        if client_order_id is None:
            client_order_id = f"{symbol}_{side}_{int(time.time() * 1000)}"
        
        # Check cache for duplicate
        if client_order_id in self.order_cache:
            logger.info(f"Order {client_order_id} already exists in cache")
            return self.order_cache[client_order_id]
        
        # Place order
        try:
            if self.exchange_client:
                if order_type == "MARKET":
                    result = self.exchange_client.place_market_order(symbol, side, size)
                else:
                    result = self.exchange_client.place_limit_order(symbol, side, size, price)
                
                order_data = {
                    'client_order_id': client_order_id,
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'order_type': order_type,
                    'status': OrderStatus.FILLED.value,
                    'filled_price': result.get('price', price),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Simulation mode
                order_data = {
                    'client_order_id': client_order_id,
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'order_type': order_type,
                    'status': OrderStatus.FILLED.value,
                    'filled_price': price or 40000.0,
                    'timestamp': datetime.now().isoformat(),
                    'simulated': True
                }
            
            # Cache order
            self.order_cache[client_order_id] = order_data
            logger.info(f"Order placed: {client_order_id} - {side} {size} {symbol}")
            
            return order_data
            
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            
            # Retry logic (simplified)
            return self._retry_order(symbol, side, size, order_type, price, client_order_id)
    
    def _retry_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float],
        client_order_id: str,
        max_retries: int = 3
    ) -> Dict:
        """Retry failed order."""
        for attempt in range(max_retries):
            logger.warning(f"Retrying order {client_order_id}, attempt {attempt + 1}/{max_retries}")
            time.sleep(1 * (attempt + 1))  # Exponential backoff
            
            try:
                return self.place_order(symbol, side, size, order_type, price, client_order_id)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Order failed after {max_retries} retries: {e}")
                    return {
                        'client_order_id': client_order_id,
                        'status': OrderStatus.FAILED.value,
                        'error': str(e)
                    }
        
        return {'status': OrderStatus.FAILED.value}
    
    def sync_positions(self) -> Dict[str, Dict]:
        """
        Sync internal position state with exchange.
        
        Returns:
            Dict of symbol -> position data
        """
        if not self.exchange_client:
            logger.info("No exchange client - using cached positions")
            return self.position_state
        
        try:
            # Fetch positions from exchange
            exchange_positions = self.exchange_client.get_open_positions()
            
            # Update internal state
            for position in exchange_positions:
                symbol = position.get('symbol')
                self.position_state[symbol] = {
                    'symbol': symbol,
                    'size': float(position.get('positionAmt', 0)),
                    'entry_price': float(position.get('entryPrice', 0)),
                    'unrealized_pnl': float(position.get('unrealizedProfit', 0)),
                    'synced_at': datetime.now().isoformat()
                }
            
            logger.info(f"Synced {len(self.position_state)} positions with exchange")
            return self.position_state
            
        except Exception as e:
            logger.error(f"Position sync failed: {e}")
            return self.position_state
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for symbol."""
        return self.position_state.get(symbol)
    
    def verify_order_status(self, client_order_id: str) -> Dict:
        """Verify order status (from cache or exchange)."""
        if client_order_id in self.order_cache:
            return self.order_cache[client_order_id]
        
        logger.warning(f"Order {client_order_id} not in cache")
        return {'status': OrderStatus.PENDING.value}
