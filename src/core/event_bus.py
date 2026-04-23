"""
Event Bus - Real-Time Event-Driven Architecture

The central nervous system of the trading bot.
All components communicate through events, not direct calls.

Event Types:
- MarketData: Price ticks, OHLCV candles
- OrderEvents: Placed, Filled, Cancelled, Rejected
- StrategyEvents: Signal generated, Position changed
- RiskEvents: Limit breached, Circuit breaker triggered
- SystemEvents: Startup, Shutdown, Error

Benefits:
- Loose coupling between components
- Easy to add new components
- Natural async processing
- Event replay for debugging
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import threading
import queue

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types in the system."""
    # Market Data
    PRICE_TICK = "price_tick"
    OHLCV_UPDATED = "ohlcv_updated"
    CANDLE_CLOSE = "candle_close"
    ORDERBOOK_UPDATE = "orderbook_update"
    
    # Orders
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_PARTIAL_FILL = "order_partial_fill"
    
    # Strategy
    SIGNAL_GENERATED = "signal_generated"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    
    # Risk
    RISK_LIMIT_BREACH = "risk_limit_breach"
    CIRCUIT_BREAKER = "circuit_breaker"
    EMERGENCY_STOP = "emergency_stop"
    
    # System
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    HEARTBEAT = "heartbeat"


@dataclass
class Event:
    """Base event class."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data
        }


@dataclass
class PriceTick(Event):
    """Price tick event."""
    event_type: EventType = EventType.PRICE_TICK
    symbol: str = ""
    price: float = 0.0
    volume: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.PRICE_TICK
        self.data = {"symbol": self.symbol, "price": self.price, "volume": self.volume}


@dataclass
class OrderEvent(Event):
    """Order-related event."""
    event_type: EventType = EventType.ORDER_PLACED
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0
    strategy_id: str = ""


class EventBus:
    """
    Central event bus for the trading system.
    
    Usage:
        bus = EventBus()
        
        # Subscribe to events
        def on_price_tick(event):
            print(f"Price: {event.data['price']}")
        
        bus.subscribe(EventType.PRICE_TICK, on_price_tick)
        
        # Publish events
        bus.publish(PriceTick(symbol="BTCUSDT", price=65000))
        
        # Start processing
        bus.start()
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._event_history: List[Event] = []
        self._max_history = 1000
        
        # Async support
        self._async_subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        
        logger.info("EventBus initialized")
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Subscribe to an event type with a callback."""
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value}: {callback.__name__}")
    
    def subscribe_async(self, event_type: EventType, callback: Callable[[Event], Any]):
        """Subscribe with an async callback."""
        self._async_subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from an event type."""
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
        if callback in self._async_subscribers[event_type]:
            self._async_subscribers[event_type].remove(callback)
    
    def publish(self, event: Event):
        """Publish an event to all subscribers."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("Event queue full, dropping oldest events")
            self._queue.get()
            self._queue.put_nowait(event)
    
    def publish_sync(self, event: Event):
        """Publish and process event synchronously (use for critical events)."""
        self._dispatch(event)
    
    def _dispatch(self, event: Event):
        """Dispatch event to all subscribers."""
        # Record history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Sync subscribers
        for callback in self._subscribers.get(event.event_type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event handler {callback.__name__}: {e}")
        
        # Async subscribers
        for callback in self._async_subscribers.get(event.event_type, []):
            try:
                asyncio.create_task(callback(event))
            except Exception as e:
                logger.error(f"Error in async event handler {callback.__name__}: {e}")
    
    def _process_loop(self):
        """Main event processing loop."""
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                self._dispatch(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def start(self):
        """Start the event processing thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        
        self.publish(Event(
            event_type=EventType.SYSTEM_STARTUP,
            source="EventBus",
            data={"message": "Event bus started"}
        ))
        
        logger.info("EventBus started")
    
    def stop(self):
        """Stop the event processing thread."""
        self._running = False
        
        self.publish_sync(Event(
            event_type=EventType.SYSTEM_SHUTDOWN,
            source="EventBus",
            data={"message": "Event bus stopping"}
        ))
        
        if self._thread:
            self._thread.join(timeout=5.0)
        
        logger.info("EventBus stopped")
    
    def get_history(self, event_type: EventType = None, limit: int = 100) -> List[Event]:
        """Get recent event history."""
        history = self._event_history
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        return history[-limit:]
    
    def clear_history(self):
        """Clear event history."""
        self._event_history.clear()
    
    def get_stats(self) -> Dict:
        """Get event bus statistics."""
        return {
            "queue_size": self._queue.qsize(),
            "history_size": len(self._event_history),
            "subscriber_count": sum(len(s) for s in self._subscribers.values()),
            "running": self._running
        }


# Convenience functions for common events
def emit_price_tick(bus: EventBus, symbol: str, price: float, volume: float = 0.0):
    """Emit a price tick event."""
    bus.publish(PriceTick(symbol=symbol, price=price, volume=volume))


def emit_order_filled(bus: EventBus, order_id: str, symbol: str, side: str, 
                      quantity: float, price: float, strategy_id: str = ""):
    """Emit an order filled event."""
    bus.publish(Event(
        event_type=EventType.ORDER_FILLED,
        source="ExecutionRouter",
        data={
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "strategy_id": strategy_id
        }
    ))


def emit_signal(bus: EventBus, strategy_id: str, symbol: str, signal: str, 
                confidence: float, metadata: Dict = None):
    """Emit a strategy signal event."""
    bus.publish(Event(
        event_type=EventType.SIGNAL_GENERATED,
        source=strategy_id,
        data={
            "strategy_id": strategy_id,
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "metadata": metadata or {}
        }
    ))


def emit_risk_breach(bus: EventBus, limit_type: str, current_value: float, 
                     limit_value: float, action_taken: str = ""):
    """Emit a risk limit breach event."""
    bus.publish(Event(
        event_type=EventType.RISK_LIMIT_BREACH,
        source="RiskGuardian",
        data={
            "limit_type": limit_type,
            "current_value": current_value,
            "limit_value": limit_value,
            "action_taken": action_taken
        }
    ))


# Singleton instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global EventBus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
