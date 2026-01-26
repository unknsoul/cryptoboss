"""
Exchange State Persistence - Upgrade A

Persists exchange-level state:
- Open order IDs
- WebSocket subscription offsets
- Position reconciliation data

Enables crash recovery without orphaned orders or double-fills.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class OpenOrder:
    """Tracked open order."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    filled_quantity: float
    status: str  # 'pending', 'open', 'partially_filled'
    strategy_id: str
    created_at: str
    exchange: str = "binance"


@dataclass
class WebSocketSubscription:
    """Tracked WebSocket subscription."""
    subscription_id: str
    stream_type: str  # 'trade', 'kline', 'depth', 'ticker'
    symbol: str
    interval: Optional[str]  # For klines
    last_event_time: int  # Unix timestamp ms
    active: bool


class ExchangeStateManager:
    """
    Persists and recovers exchange-level state.
    
    Usage:
        esm = ExchangeStateManager()
        
        # Track open order
        esm.save_open_order(order)
        
        # On restart - get all open orders for reconciliation
        open_orders = esm.get_open_orders()
        for order in open_orders:
            status = exchange.get_order_status(order.order_id)
            if status == 'filled':
                esm.mark_order_filled(order.order_id)
    """
    
    def __init__(self, db_path: str = "data/exchange_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        logger.info(f"ExchangeStateManager initialized: {db_path}")
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Open orders table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS open_orders (
                    order_id TEXT PRIMARY KEY,
                    client_order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    filled_quantity REAL DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    strategy_id TEXT,
                    created_at TEXT NOT NULL,
                    exchange TEXT DEFAULT 'binance',
                    updated_at TEXT
                )
            """)
            
            # WebSocket subscriptions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ws_subscriptions (
                    subscription_id TEXT PRIMARY KEY,
                    stream_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT,
                    last_event_time INTEGER DEFAULT 0,
                    active BOOLEAN DEFAULT 1,
                    created_at TEXT
                )
            """)
            
            # Position snapshots for reconciliation
            conn.execute("""
                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_entry_price REAL,
                    unrealized_pnl REAL,
                    snapshot_time TEXT NOT NULL,
                    exchange TEXT DEFAULT 'binance'
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON open_orders(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON open_orders(status)")
            conn.commit()
    
    # === Open Orders ===
    
    def save_open_order(self, order: OpenOrder) -> bool:
        """Save an open order for tracking."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO open_orders 
                        (order_id, client_order_id, symbol, side, order_type, 
                         quantity, price, filled_quantity, status, strategy_id, 
                         created_at, exchange, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        order.order_id, order.client_order_id, order.symbol,
                        order.side, order.order_type, order.quantity, order.price,
                        order.filled_quantity, order.status, order.strategy_id,
                        order.created_at, order.exchange, datetime.now().isoformat()
                    ))
                    conn.commit()
                logger.debug(f"Saved open order: {order.order_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to save order {order.order_id}: {e}")
                return False
    
    def get_open_orders(self, symbol: str = None, strategy_id: str = None) -> List[OpenOrder]:
        """Get all open orders, optionally filtered."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                query = "SELECT * FROM open_orders WHERE status IN ('pending', 'open', 'partially_filled')"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                if strategy_id:
                    query += " AND strategy_id = ?"
                    params.append(strategy_id)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [OpenOrder(
                    order_id=row['order_id'],
                    client_order_id=row['client_order_id'],
                    symbol=row['symbol'],
                    side=row['side'],
                    order_type=row['order_type'],
                    quantity=row['quantity'],
                    price=row['price'],
                    filled_quantity=row['filled_quantity'],
                    status=row['status'],
                    strategy_id=row['strategy_id'],
                    created_at=row['created_at'],
                    exchange=row['exchange']
                ) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    def update_order_status(self, order_id: str, status: str, filled_quantity: float = None):
        """Update order status after reconciliation."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    if filled_quantity is not None:
                        conn.execute("""
                            UPDATE open_orders 
                            SET status = ?, filled_quantity = ?, updated_at = ?
                            WHERE order_id = ?
                        """, (status, filled_quantity, datetime.now().isoformat(), order_id))
                    else:
                        conn.execute("""
                            UPDATE open_orders 
                            SET status = ?, updated_at = ?
                            WHERE order_id = ?
                        """, (status, datetime.now().isoformat(), order_id))
                    conn.commit()
                logger.debug(f"Updated order {order_id} status to {status}")
            except Exception as e:
                logger.error(f"Failed to update order {order_id}: {e}")
    
    def mark_order_filled(self, order_id: str, filled_quantity: float = None):
        """Mark order as filled."""
        self.update_order_status(order_id, 'filled', filled_quantity)
    
    def mark_order_cancelled(self, order_id: str):
        """Mark order as cancelled."""
        self.update_order_status(order_id, 'cancelled')
    
    def remove_order(self, order_id: str):
        """Remove order from tracking (after confirmed fill/cancel)."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM open_orders WHERE order_id = ?", (order_id,))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to remove order {order_id}: {e}")
    
    # === WebSocket Subscriptions ===
    
    def save_subscription(self, sub: WebSocketSubscription):
        """Save WebSocket subscription for recovery."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO ws_subscriptions
                        (subscription_id, stream_type, symbol, interval, 
                         last_event_time, active, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sub.subscription_id, sub.stream_type, sub.symbol,
                        sub.interval, sub.last_event_time, sub.active,
                        datetime.now().isoformat()
                    ))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to save subscription: {e}")
    
    def get_active_subscriptions(self) -> List[WebSocketSubscription]:
        """Get all active subscriptions for re-subscription on restart."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM ws_subscriptions WHERE active = 1"
                )
                rows = cursor.fetchall()
                
                return [WebSocketSubscription(
                    subscription_id=row['subscription_id'],
                    stream_type=row['stream_type'],
                    symbol=row['symbol'],
                    interval=row['interval'],
                    last_event_time=row['last_event_time'],
                    active=bool(row['active'])
                ) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get subscriptions: {e}")
            return []
    
    def update_subscription_offset(self, subscription_id: str, last_event_time: int):
        """Update last event time for a subscription."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE ws_subscriptions 
                        SET last_event_time = ?
                        WHERE subscription_id = ?
                    """, (last_event_time, subscription_id))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to update subscription offset: {e}")
    
    def deactivate_subscription(self, subscription_id: str):
        """Mark subscription as inactive."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE ws_subscriptions SET active = 0 WHERE subscription_id = ?",
                        (subscription_id,)
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to deactivate subscription: {e}")
    
    # === Position Snapshots ===
    
    def save_position_snapshot(self, symbol: str, quantity: float, 
                                avg_entry_price: float, unrealized_pnl: float):
        """Save position snapshot for reconciliation."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO position_snapshots 
                        (symbol, quantity, avg_entry_price, unrealized_pnl, snapshot_time)
                        VALUES (?, ?, ?, ?, ?)
                    """, (symbol, quantity, avg_entry_price, unrealized_pnl, 
                          datetime.now().isoformat()))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to save position snapshot: {e}")
    
    def get_latest_position_snapshot(self, symbol: str) -> Optional[Dict]:
        """Get latest position snapshot for a symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM position_snapshots 
                    WHERE symbol = ? 
                    ORDER BY snapshot_time DESC LIMIT 1
                """, (symbol,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'symbol': row['symbol'],
                        'quantity': row['quantity'],
                        'avg_entry_price': row['avg_entry_price'],
                        'unrealized_pnl': row['unrealized_pnl'],
                        'snapshot_time': row['snapshot_time']
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get position snapshot: {e}")
            return None
    
    # === Reconciliation ===
    
    async def reconcile_orders(self, exchange_client) -> Dict:
        """
        Reconcile tracked orders with exchange state.
        
        Call this on startup to sync local state with exchange.
        
        Returns:
            Dict with reconciliation results
        """
        results = {
            'reconciled': 0,
            'filled': 0,
            'cancelled': 0,
            'still_open': 0,
            'errors': []
        }
        
        open_orders = self.get_open_orders()
        logger.info(f"Reconciling {len(open_orders)} tracked orders...")
        
        for order in open_orders:
            try:
                # Query exchange for order status
                exchange_order = await exchange_client.fetch_order(
                    order.order_id, order.symbol
                )
                
                if exchange_order['status'] == 'closed':
                    # Order was filled while we were down
                    self.mark_order_filled(
                        order.order_id, 
                        exchange_order.get('filled', order.quantity)
                    )
                    results['filled'] += 1
                    logger.info(f"Order {order.order_id} was filled")
                    
                elif exchange_order['status'] == 'canceled':
                    self.mark_order_cancelled(order.order_id)
                    results['cancelled'] += 1
                    logger.info(f"Order {order.order_id} was cancelled")
                    
                else:
                    # Still open - update filled quantity
                    self.update_order_status(
                        order.order_id,
                        exchange_order['status'],
                        exchange_order.get('filled', 0)
                    )
                    results['still_open'] += 1
                
                results['reconciled'] += 1
                
            except Exception as e:
                error_msg = f"Failed to reconcile order {order.order_id}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        logger.info(f"Reconciliation complete: {results}")
        return results
    
    async def recover_subscriptions(self, ws_client) -> int:
        """
        Re-subscribe to all active WebSocket streams.
        
        Returns:
            Number of subscriptions recovered
        """
        subs = self.get_active_subscriptions()
        recovered = 0
        
        for sub in subs:
            try:
                if sub.stream_type == 'trade':
                    await ws_client.subscribe_trades(sub.symbol)
                elif sub.stream_type == 'kline':
                    await ws_client.subscribe_klines(sub.symbol, sub.interval)
                elif sub.stream_type == 'depth':
                    await ws_client.subscribe_depth(sub.symbol)
                elif sub.stream_type == 'ticker':
                    await ws_client.subscribe_ticker(sub.symbol)
                
                recovered += 1
                logger.info(f"Recovered subscription: {sub.stream_type}@{sub.symbol}")
                
            except Exception as e:
                logger.error(f"Failed to recover subscription {sub.subscription_id}: {e}")
        
        return recovered


# Singleton
_exchange_state: Optional[ExchangeStateManager] = None

def get_exchange_state() -> ExchangeStateManager:
    global _exchange_state
    if _exchange_state is None:
        _exchange_state = ExchangeStateManager()
    return _exchange_state
