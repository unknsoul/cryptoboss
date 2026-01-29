"""
WebSocket Streaming Manager - Real-time Updates

Provides real-time streaming of:
- Trade decisions
- Price updates
- Risk state changes
- System events

v11.0 - Production-Grade Platform Upgrade
"""

import logging
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Callable
from enum import Enum
from fastapi import WebSocket, WebSocketDisconnect
import uuid

logger = logging.getLogger(__name__)


class StreamChannel(str, Enum):
    """Available streaming channels."""
    DECISIONS = "decisions"          # Trade decision stream
    PRICES = "prices"                # Price updates
    RISK = "risk"                    # Risk state changes
    EVENTS = "events"                # System events
    POSITIONS = "positions"          # Position updates
    HEALTH = "health"                # System health
    ALERTS = "alerts"                # Alerts and notifications
    ALL = "all"                      # All channels


@dataclass
class StreamMessage:
    """Message to be streamed to clients."""
    channel: StreamChannel
    event_type: str
    data: Dict
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def to_json(self) -> str:
        return json.dumps({
            'channel': self.channel.value,
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'message_id': self.message_id,
        })


@dataclass
class ClientConnection:
    """Represents a connected WebSocket client."""
    connection_id: str
    websocket: WebSocket
    subscribed_channels: Set[StreamChannel]
    connected_at: datetime
    last_message_at: Optional[datetime] = None
    message_count: int = 0
    symbols: Set[str] = field(default_factory=set)  # For filtered subscriptions
    
    async def send(self, message: StreamMessage) -> bool:
        """Send a message to this client."""
        try:
            # Check if client is subscribed to this channel
            if StreamChannel.ALL not in self.subscribed_channels:
                if message.channel not in self.subscribed_channels:
                    return True  # Not subscribed, skip
            
            # Check symbol filter for price channel
            if message.channel == StreamChannel.PRICES and self.symbols:
                symbol = message.data.get('symbol', '')
                if symbol and symbol not in self.symbols:
                    return True  # Not subscribed to this symbol
            
            await self.websocket.send_text(message.to_json())
            self.last_message_at = datetime.utcnow()
            self.message_count += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send to client {self.connection_id}: {e}")
            return False


class WebSocketManager:
    """
    WebSocket Manager - Handles real-time streaming to connected clients.
    
    Features:
    - Multiple streaming channels
    - Client subscription management
    - Broadcast and targeted messaging
    - Connection lifecycle management
    - Message buffering for reconnection
    
    Usage:
        manager = WebSocketManager()
        
        # Connect client
        await manager.connect(websocket, channels=["decisions", "risk"])
        
        # Broadcast to all clients
        await manager.broadcast_decision(decision)
        
        # Targeted broadcast
        await manager.broadcast("decisions", "new_decision", {"symbol": "BTC/USDT"})
    """
    
    def __init__(
        self,
        max_connections: int = 1000,
        message_buffer_size: int = 100,
        heartbeat_interval_seconds: int = 30
    ):
        """
        Initialize WebSocketManager.
        
        Args:
            max_connections: Maximum concurrent connections
            message_buffer_size: Recent messages to buffer for reconnection
            heartbeat_interval_seconds: Interval for heartbeat messages
        """
        self._connections: Dict[str, ClientConnection] = {}
        self._max_connections = max_connections
        self._message_buffer: List[StreamMessage] = []
        self._message_buffer_size = message_buffer_size
        self._heartbeat_interval = heartbeat_interval_seconds
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            'total_connections': 0,
            'total_disconnections': 0,
            'total_messages_sent': 0,
            'total_messages_failed': 0,
        }
        
        logger.info(f"WebSocketManager initialized: max_connections={max_connections}")
    
    async def connect(
        self,
        websocket: WebSocket,
        channels: List[str] = None,
        symbols: List[str] = None
    ) -> str:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: FastAPI WebSocket instance
            channels: Channels to subscribe to (default: all)
            symbols: Symbols to filter for (for price channel)
            
        Returns:
            Connection ID
        """
        # Check connection limit
        if len(self._connections) >= self._max_connections:
            await websocket.close(code=1013, reason="Max connections reached")
            return None
        
        # Accept connection
        await websocket.accept()
        
        # Generate connection ID
        connection_id = str(uuid.uuid4())
        
        # Parse channels
        subscribed_channels = set()
        if channels:
            for ch in channels:
                try:
                    subscribed_channels.add(StreamChannel(ch))
                except ValueError:
                    logger.warning(f"Unknown channel: {ch}")
        else:
            subscribed_channels.add(StreamChannel.ALL)
        
        # Create connection
        client = ClientConnection(
            connection_id=connection_id,
            websocket=websocket,
            subscribed_channels=subscribed_channels,
            connected_at=datetime.utcnow(),
            symbols=set(symbols) if symbols else set()
        )
        
        self._connections[connection_id] = client
        self._stats['total_connections'] += 1
        
        logger.info(f"Client connected: {connection_id} (channels: {[c.value for c in subscribed_channels]})")
        
        # Send welcome message
        await client.send(StreamMessage(
            channel=StreamChannel.EVENTS,
            event_type="connected",
            data={
                'connection_id': connection_id,
                'subscribed_channels': [c.value for c in subscribed_channels],
                'server_time': datetime.utcnow().isoformat(),
            }
        ))
        
        # Send buffered messages if any
        for msg in self._message_buffer[-10:]:  # Last 10 messages
            if msg.channel in subscribed_channels or StreamChannel.ALL in subscribed_channels:
                await client.send(msg)
        
        return connection_id
    
    async def disconnect(self, connection_id: str) -> None:
        """Disconnect a client."""
        if connection_id in self._connections:
            client = self._connections[connection_id]
            try:
                await client.websocket.close()
            except Exception:
                pass
            del self._connections[connection_id]
            self._stats['total_disconnections'] += 1
            logger.info(f"Client disconnected: {connection_id}")
    
    async def broadcast(
        self,
        channel: StreamChannel,
        event_type: str,
        data: Dict,
        exclude: Set[str] = None
    ) -> int:
        """
        Broadcast a message to all subscribed clients.
        
        Args:
            channel: Channel to broadcast on
            event_type: Type of event
            data: Event data
            exclude: Connection IDs to exclude
            
        Returns:
            Number of clients successfully sent to
        """
        message = StreamMessage(
            channel=channel,
            event_type=event_type,
            data=data
        )
        
        # Buffer message
        self._buffer_message(message)
        
        # Send to all clients
        exclude = exclude or set()
        success_count = 0
        failed_ids = []
        
        for conn_id, client in self._connections.items():
            if conn_id in exclude:
                continue
            
            if await client.send(message):
                success_count += 1
                self._stats['total_messages_sent'] += 1
            else:
                failed_ids.append(conn_id)
                self._stats['total_messages_failed'] += 1
        
        # Clean up failed connections
        for conn_id in failed_ids:
            await self.disconnect(conn_id)
        
        return success_count
    
    async def broadcast_decision(self, decision: Any) -> int:
        """
        Broadcast a trade decision.
        
        Args:
            decision: TradeDecision object or dict
            
        Returns:
            Number of clients sent to
        """
        data = decision.to_dict() if hasattr(decision, 'to_dict') else decision
        return await self.broadcast(
            channel=StreamChannel.DECISIONS,
            event_type="new_decision",
            data=data
        )
    
    async def broadcast_price(self, symbol: str, price: float, volume: float = 0) -> int:
        """Broadcast a price update."""
        return await self.broadcast(
            channel=StreamChannel.PRICES,
            event_type="price_update",
            data={
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'timestamp': datetime.utcnow().isoformat(),
            }
        )
    
    async def broadcast_risk_state(self, risk_state: Dict) -> int:
        """Broadcast risk state update."""
        return await self.broadcast(
            channel=StreamChannel.RISK,
            event_type="risk_update",
            data=risk_state
        )
    
    async def broadcast_position(self, position: Dict) -> int:
        """Broadcast position update."""
        return await self.broadcast(
            channel=StreamChannel.POSITIONS,
            event_type="position_update",
            data=position
        )
    
    async def broadcast_alert(self, alert_type: str, message: str, severity: str = "info") -> int:
        """Broadcast an alert."""
        return await self.broadcast(
            channel=StreamChannel.ALERTS,
            event_type="alert",
            data={
                'alert_type': alert_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.utcnow().isoformat(),
            }
        )
    
    async def broadcast_health(self, health: Dict) -> int:
        """Broadcast system health update."""
        return await self.broadcast(
            channel=StreamChannel.HEALTH,
            event_type="health_update",
            data=health
        )
    
    async def send_to_client(self, connection_id: str, message: StreamMessage) -> bool:
        """Send a message to a specific client."""
        client = self._connections.get(connection_id)
        if client:
            return await client.send(message)
        return False
    
    def update_subscription(
        self,
        connection_id: str,
        add_channels: List[StreamChannel] = None,
        remove_channels: List[StreamChannel] = None,
        symbols: List[str] = None
    ) -> bool:
        """Update a client's subscription."""
        client = self._connections.get(connection_id)
        if not client:
            return False
        
        if add_channels:
            client.subscribed_channels.update(add_channels)
        
        if remove_channels:
            client.subscribed_channels -= set(remove_channels)
        
        if symbols is not None:
            client.symbols = set(symbols)
        
        return True
    
    async def start_heartbeat(self) -> None:
        """Start the heartbeat task."""
        if self._running:
            return
        
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("WebSocket heartbeat started")
    
    async def stop_heartbeat(self) -> None:
        """Stop the heartbeat task."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        logger.info("WebSocket heartbeat stopped")
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to clients."""
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            
            if not self._connections:
                continue
            
            # Send heartbeat
            await self.broadcast(
                channel=StreamChannel.EVENTS,
                event_type="heartbeat",
                data={
                    'server_time': datetime.utcnow().isoformat(),
                    'active_connections': len(self._connections),
                }
            )
    
    def _buffer_message(self, message: StreamMessage) -> None:
        """Buffer a message for reconnection."""
        self._message_buffer.append(message)
        
        # Trim buffer
        if len(self._message_buffer) > self._message_buffer_size:
            self._message_buffer = self._message_buffer[-self._message_buffer_size:]
    
    def get_connection_count(self) -> int:
        """Get current connection count."""
        return len(self._connections)
    
    def get_connections(self) -> List[Dict]:
        """Get info about all connections."""
        return [
            {
                'connection_id': c.connection_id,
                'connected_at': c.connected_at.isoformat(),
                'channels': [ch.value for ch in c.subscribed_channels],
                'symbols': list(c.symbols),
                'message_count': c.message_count,
                'last_message_at': c.last_message_at.isoformat() if c.last_message_at else None,
            }
            for c in self._connections.values()
        ]
    
    def get_stats(self) -> Dict:
        """Get manager statistics."""
        return {
            **self._stats,
            'active_connections': len(self._connections),
            'buffer_size': len(self._message_buffer),
            'channels': [c.value for c in StreamChannel],
        }


# Singleton instance
_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocketManager instance."""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager


# FastAPI integration helpers
async def websocket_endpoint(websocket: WebSocket, channels: str = None, symbols: str = None):
    """
    FastAPI WebSocket endpoint handler.
    
    Usage in routes.py:
        @app.websocket("/ws")
        async def websocket_route(websocket: WebSocket, channels: str = None):
            await websocket_endpoint(websocket, channels)
    """
    manager = get_websocket_manager()
    
    # Parse channels and symbols
    channel_list = channels.split(",") if channels else None
    symbol_list = symbols.split(",") if symbols else None
    
    connection_id = await manager.connect(websocket, channel_list, symbol_list)
    
    if not connection_id:
        return
    
    try:
        while True:
            # Wait for client messages
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                # Handle subscription updates
                if message.get('action') == 'subscribe':
                    manager.update_subscription(
                        connection_id,
                        add_channels=[StreamChannel(c) for c in message.get('channels', [])]
                    )
                elif message.get('action') == 'unsubscribe':
                    manager.update_subscription(
                        connection_id,
                        remove_channels=[StreamChannel(c) for c in message.get('channels', [])]
                    )
                elif message.get('action') == 'set_symbols':
                    manager.update_subscription(
                        connection_id,
                        symbols=message.get('symbols', [])
                    )
                elif message.get('action') == 'ping':
                    await manager.send_to_client(
                        connection_id,
                        StreamMessage(
                            channel=StreamChannel.EVENTS,
                            event_type="pong",
                            data={'server_time': datetime.utcnow().isoformat()}
                        )
                    )
                    
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client {connection_id}")
            except Exception as e:
                logger.error(f"Error processing client message: {e}")
                
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(connection_id)
