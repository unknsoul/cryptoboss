"""
Order Management - Enterprise Features #101, #105, #108, #112
Smart Routing, Queue Management, Partial Fills, and Order Expiry.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum
from collections import deque
import threading
import uuid

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


class SmartOrderRouter:
    """
    Feature #101: Smart Order Router
    
    Routes orders to optimal execution venue based on conditions.
    """
    
    def __init__(self):
        """Initialize smart order router."""
        self.venues: Dict[str, Dict] = {}
        self.routing_rules: List[Callable] = []
        self.executed_orders: List[Dict] = []
        
        logger.info("Smart Order Router initialized")
    
    def add_venue(self, name: str, config: Dict):
        """Add an execution venue."""
        self.venues[name] = {
            'name': name,
            'config': config,
            'priority': config.get('priority', 50),
            'max_size': config.get('max_size', float('inf')),
            'min_size': config.get('min_size', 0),
            'latency_ms': config.get('latency_ms', 100)
        }
        logger.info(f"Added venue: {name}")
    
    def route_order(
        self,
        side: str,
        size: float,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        urgency: str = 'normal'
    ) -> Dict:
        """
        Route order to optimal venue.
        
        Args:
            side: BUY/SELL
            size: Order size
            price: Target price
            order_type: Order type
            urgency: low/normal/high
            
        Returns:
            Routing decision
        """
        if not self.venues:
            return {'venue': 'default', 'reason': 'No venues configured'}
        
        candidates = []
        
        for name, venue in self.venues.items():
            if venue['min_size'] <= size <= venue['max_size']:
                score = 100 - venue['priority']
                
                # Prefer low latency for urgent orders
                if urgency == 'high':
                    score -= venue['latency_ms'] / 10
                
                candidates.append((name, score, venue))
        
        if not candidates:
            return {'venue': 'default', 'reason': 'No suitable venue'}
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        
        decision = {
            'venue': best[0],
            'score': round(best[1], 2),
            'latency_ms': best[2]['latency_ms'],
            'alternatives': [c[0] for c in candidates[1:3]],
            'order': {
                'side': side,
                'size': size,
                'price': price,
                'type': order_type.value
            }
        }
        
        self.executed_orders.append(decision)
        return decision
    
    def split_order(self, size: float, max_per_venue: float) -> List[Dict]:
        """Split large order across multiple venues."""
        chunks = []
        remaining = size
        
        sorted_venues = sorted(
            self.venues.items(),
            key=lambda x: x[1]['priority']
        )
        
        for name, venue in sorted_venues:
            if remaining <= 0:
                break
            
            chunk_size = min(remaining, max_per_venue, venue['max_size'])
            if chunk_size >= venue['min_size']:
                chunks.append({
                    'venue': name,
                    'size': chunk_size
                })
                remaining -= chunk_size
        
        return chunks


class OrderQueueManager:
    """
    Feature #105: Order Queue Manager
    
    Manages order queue with priority and throttling.
    """
    
    def __init__(
        self,
        max_orders_per_second: int = 10,
        max_queue_size: int = 100
    ):
        """
        Initialize order queue.
        
        Args:
            max_orders_per_second: Rate limit
            max_queue_size: Maximum queue size
        """
        self.max_rate = max_orders_per_second
        self.max_size = max_queue_size
        
        self.queue: deque = deque()
        self.processing: List[Dict] = []
        self.completed: List[Dict] = []
        self.last_process_time = datetime.now()
        self._lock = threading.Lock()
        
        logger.info(f"Order Queue initialized - Rate: {max_orders_per_second}/s")
    
    def enqueue(self, order: Dict, priority: int = 5) -> str:
        """
        Add order to queue.
        
        Args:
            order: Order data
            priority: Priority (1=highest, 10=lowest)
            
        Returns:
            Queue ID
        """
        with self._lock:
            if len(self.queue) >= self.max_size:
                raise Exception("Order queue full")
            
            queue_id = str(uuid.uuid4())[:8]
            
            entry = {
                'id': queue_id,
                'order': order,
                'priority': priority,
                'status': 'queued',
                'queued_at': datetime.now(),
                'processed_at': None
            }
            
            # Insert by priority
            inserted = False
            for i, existing in enumerate(self.queue):
                if priority < existing['priority']:
                    self.queue.insert(i, entry)
                    inserted = True
                    break
            
            if not inserted:
                self.queue.append(entry)
            
            return queue_id
    
    def process_next(self) -> Optional[Dict]:
        """Process next order in queue."""
        with self._lock:
            # Rate limiting
            now = datetime.now()
            elapsed = (now - self.last_process_time).total_seconds()
            
            if elapsed < 1.0 / self.max_rate:
                return None
            
            if not self.queue:
                return None
            
            entry = self.queue.popleft()
            entry['status'] = 'processing'
            entry['processed_at'] = now
            
            self.processing.append(entry)
            self.last_process_time = now
            
            return entry
    
    def complete_order(self, queue_id: str, result: Dict):
        """Mark order as completed."""
        with self._lock:
            for i, entry in enumerate(self.processing):
                if entry['id'] == queue_id:
                    entry['status'] = 'completed'
                    entry['result'] = result
                    self.completed.append(entry)
                    self.processing.pop(i)
                    break
            
            self.completed = self.completed[-100:]
    
    def get_queue_status(self) -> Dict:
        """Get queue status."""
        return {
            'queued': len(self.queue),
            'processing': len(self.processing),
            'completed': len(self.completed),
            'max_size': self.max_size
        }


class PartialFillHandler:
    """
    Feature #108: Partial Fill Handler
    
    Handles partially filled orders and residual management.
    """
    
    def __init__(self, min_residual: float = 0.0001):
        """
        Initialize partial fill handler.
        
        Args:
            min_residual: Minimum residual to keep tracking
        """
        self.min_residual = min_residual
        self.active_orders: Dict[str, Dict] = {}
        self.fill_history: List[Dict] = []
        
        logger.info("Partial Fill Handler initialized")
    
    def create_order(self, order_id: str, size: float, side: str, price: float) -> Dict:
        """Create tracked order."""
        order = {
            'id': order_id,
            'original_size': size,
            'filled_size': 0,
            'remaining_size': size,
            'side': side,
            'avg_fill_price': 0,
            'fills': [],
            'created_at': datetime.now().isoformat(),
            'status': OrderStatus.SUBMITTED.value
        }
        
        self.active_orders[order_id] = order
        return order
    
    def record_fill(self, order_id: str, fill_size: float, fill_price: float) -> Dict:
        """
        Record a fill for an order.
        
        Args:
            order_id: Order ID
            fill_size: Size filled
            fill_price: Fill price
            
        Returns:
            Updated order status
        """
        if order_id not in self.active_orders:
            return {'error': 'Order not found'}
        
        order = self.active_orders[order_id]
        
        # Update fill data
        fill = {
            'size': fill_size,
            'price': fill_price,
            'timestamp': datetime.now().isoformat()
        }
        order['fills'].append(fill)
        
        # Update totals
        old_total = order['filled_size'] * order['avg_fill_price']
        new_total = old_total + (fill_size * fill_price)
        order['filled_size'] += fill_size
        order['avg_fill_price'] = new_total / order['filled_size'] if order['filled_size'] > 0 else 0
        order['remaining_size'] = order['original_size'] - order['filled_size']
        
        # Update status
        if order['remaining_size'] <= self.min_residual:
            order['status'] = OrderStatus.FILLED.value
            del self.active_orders[order_id]
            self.fill_history.append(order)
        else:
            order['status'] = OrderStatus.PARTIAL.value
        
        return order
    
    def get_unfilled_orders(self) -> List[Dict]:
        """Get all orders with remaining size."""
        return [
            {
                'id': oid,
                'remaining': o['remaining_size'],
                'filled_pct': round(o['filled_size'] / o['original_size'] * 100, 1)
            }
            for oid, o in self.active_orders.items()
        ]
    
    def cancel_residual(self, order_id: str) -> Optional[Dict]:
        """Cancel remaining unfilled portion."""
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            order['status'] = OrderStatus.CANCELLED.value
            self.fill_history.append(order)
            return order
        return None


class OrderExpiryManager:
    """
    Feature #112: Order Expiry Manager
    
    Manages time-based order expiration.
    """
    
    def __init__(self, default_ttl_minutes: int = 60):
        """
        Initialize expiry manager.
        
        Args:
            default_ttl_minutes: Default order TTL
        """
        self.default_ttl = timedelta(minutes=default_ttl_minutes)
        self.tracked_orders: Dict[str, Dict] = {}
        self.expired_orders: List[Dict] = []
        
        logger.info(f"Order Expiry Manager initialized - TTL: {default_ttl_minutes}m")
    
    def track_order(
        self,
        order_id: str,
        order_data: Dict,
        ttl_minutes: Optional[int] = None
    ):
        """Track an order with expiry."""
        ttl = timedelta(minutes=ttl_minutes) if ttl_minutes else self.default_ttl
        
        self.tracked_orders[order_id] = {
            'order': order_data,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + ttl,
            'ttl_minutes': ttl.total_seconds() / 60
        }
    
    def check_expired(self) -> List[Dict]:
        """Check and return expired orders."""
        now = datetime.now()
        expired = []
        
        for order_id in list(self.tracked_orders.keys()):
            entry = self.tracked_orders[order_id]
            
            if now >= entry['expires_at']:
                entry['expired_at'] = now.isoformat()
                expired.append({
                    'order_id': order_id,
                    **entry
                })
                self.expired_orders.append(entry)
                del self.tracked_orders[order_id]
        
        self.expired_orders = self.expired_orders[-100:]
        return expired
    
    def extend_ttl(self, order_id: str, additional_minutes: int) -> bool:
        """Extend order TTL."""
        if order_id in self.tracked_orders:
            self.tracked_orders[order_id]['expires_at'] += timedelta(minutes=additional_minutes)
            return True
        return False
    
    def cancel_tracking(self, order_id: str) -> bool:
        """Stop tracking an order."""
        if order_id in self.tracked_orders:
            del self.tracked_orders[order_id]
            return True
        return False
    
    def get_expiring_soon(self, within_minutes: int = 5) -> List[Dict]:
        """Get orders expiring soon."""
        cutoff = datetime.now() + timedelta(minutes=within_minutes)
        
        return [
            {
                'order_id': oid,
                'expires_in_seconds': (entry['expires_at'] - datetime.now()).total_seconds()
            }
            for oid, entry in self.tracked_orders.items()
            if entry['expires_at'] <= cutoff
        ]


# Singletons
_router: Optional[SmartOrderRouter] = None
_queue: Optional[OrderQueueManager] = None
_partial_handler: Optional[PartialFillHandler] = None
_expiry: Optional[OrderExpiryManager] = None


def get_order_router() -> SmartOrderRouter:
    global _router
    if _router is None:
        _router = SmartOrderRouter()
    return _router


def get_order_queue() -> OrderQueueManager:
    global _queue
    if _queue is None:
        _queue = OrderQueueManager()
    return _queue


def get_partial_handler() -> PartialFillHandler:
    global _partial_handler
    if _partial_handler is None:
        _partial_handler = PartialFillHandler()
    return _partial_handler


def get_expiry_manager() -> OrderExpiryManager:
    global _expiry
    if _expiry is None:
        _expiry = OrderExpiryManager()
    return _expiry


if __name__ == '__main__':
    # Test router
    router = SmartOrderRouter()
    router.add_venue('binance', {'priority': 1, 'latency_ms': 50})
    router.add_venue('coinbase', {'priority': 2, 'latency_ms': 100})
    
    decision = router.route_order('BUY', 0.1, 50000, urgency='high')
    print(f"Routing: {decision}")
    
    # Test queue
    queue = OrderQueueManager()
    qid = queue.enqueue({'side': 'BUY', 'size': 0.1}, priority=1)
    print(f"Queued: {qid}")
    
    # Test partial fill
    handler = PartialFillHandler()
    handler.create_order('ORD001', 1.0, 'BUY', 50000)
    result = handler.record_fill('ORD001', 0.3, 50010)
    print(f"Partial fill: {result}")
