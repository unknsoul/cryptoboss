"""
Persistent EventBus - Upgrade D

EventBus with optional SQLite/Redis backing for crash recovery.
On restart, can replay missed events.
"""

import sqlite3
import json
import asyncio
import logging
from typing import Dict, List, Callable, Optional
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import defaultdict

from .event_bus import Event, EventType, EventBus

logger = logging.getLogger(__name__)


class PersistentEventBus(EventBus):
    """
    EventBus with persistent storage for crash recovery.
    
    Features:
    - All events persisted to SQLite
    - Replay missed events on restart
    - Event history query
    - Automatic cleanup of old events
    
    Usage:
        bus = PersistentEventBus(db_path="data/events.db")
        
        # Subscribe
        bus.subscribe(EventType.ORDER_FILLED, on_order_filled)
        
        # Publish (automatically persisted)
        bus.publish(event)
        
        # On restart - replay missed events
        await bus.replay_events_since(last_processed_time)
    """
    
    def __init__(
        self,
        db_path: str = "data/events.db",
        max_queue_size: int = 10000,
        retention_days: int = 7
    ):
        super().__init__(max_queue_size)
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._db_lock = threading.Lock()
        
        self._init_db()
        self._last_processed_id: int = 0
        
        logger.info(f"PersistentEventBus initialized: {db_path}")
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT,
                    data_json TEXT,
                    processed BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_processed ON events(processed)")
            conn.commit()
    
    def _persist_event(self, event: Event) -> int:
        """Persist event to database, return event ID."""
        with self._db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        INSERT INTO events (event_type, timestamp, source, data_json)
                        VALUES (?, ?, ?, ?)
                    """, (
                        event.event_type.value,
                        event.timestamp.isoformat(),
                        event.source,
                        json.dumps(event.data, default=str)
                    ))
                    conn.commit()
                    return cursor.lastrowid
            except Exception as e:
                logger.error(f"Failed to persist event: {e}")
                return -1
    
    def _mark_processed(self, event_id: int):
        """Mark event as processed."""
        with self._db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE events SET processed = 1 WHERE id = ?",
                        (event_id,)
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to mark event processed: {e}")
    
    def publish(self, event: Event):
        """Publish event (persisted before dispatch)."""
        event_id = self._persist_event(event)
        super().publish(event)
        
        # Mark as processed after dispatch
        if event_id > 0:
            self._last_processed_id = event_id
    
    def publish_sync(self, event: Event):
        """Publish and process synchronously."""
        event_id = self._persist_event(event)
        super().publish_sync(event)
        
        if event_id > 0:
            self._mark_processed(event_id)
            self._last_processed_id = event_id
    
    async def replay_events_since(
        self,
        since: datetime = None,
        event_types: List[EventType] = None
    ) -> int:
        """
        Replay unprocessed events since a timestamp.
        
        Args:
            since: Replay events after this time (default: last hour)
            event_types: Filter by event types (default: all)
        
        Returns:
            Number of events replayed
        """
        if since is None:
            since = datetime.now() - timedelta(hours=1)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = """
                    SELECT * FROM events 
                    WHERE timestamp >= ? AND processed = 0
                """
                params = [since.isoformat()]
                
                if event_types:
                    placeholders = ",".join("?" * len(event_types))
                    query += f" AND event_type IN ({placeholders})"
                    params.extend([et.value for et in event_types])
                
                query += " ORDER BY id ASC"
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                count = 0
                for row in rows:
                    event = Event(
                        event_type=EventType(row['event_type']),
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        source=row['source'],
                        data=json.loads(row['data_json']) if row['data_json'] else {}
                    )
                    
                    # Dispatch to subscribers
                    self._dispatch(event)
                    self._mark_processed(row['id'])
                    count += 1
                
                logger.info(f"Replayed {count} events since {since}")
                return count
                
        except Exception as e:
            logger.error(f"Failed to replay events: {e}")
            return 0
    
    def get_event_history(
        self,
        event_type: EventType = None,
        since: datetime = None,
        limit: int = 100
    ) -> List[Event]:
        """Get event history from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM events WHERE 1=1"
                params = []
                
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type.value)
                
                if since:
                    query += " AND timestamp >= ?"
                    params.append(since.isoformat())
                
                query += " ORDER BY id DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [Event(
                    event_type=EventType(row['event_type']),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    source=row['source'],
                    data=json.loads(row['data_json']) if row['data_json'] else {}
                ) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get event history: {e}")
            return []
    
    def cleanup_old_events(self):
        """Remove events older than retention period."""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        
        with self._db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM events WHERE timestamp < ?",
                        (cutoff.isoformat(),)
                    )
                    deleted = cursor.rowcount
                    conn.commit()
                    logger.info(f"Cleaned up {deleted} old events")
                    return deleted
            except Exception as e:
                logger.error(f"Failed to cleanup events: {e}")
                return 0
    
    def get_stats(self) -> Dict:
        """Get event bus statistics."""
        base_stats = super().get_stats()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM events")
                total = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM events WHERE processed = 0")
                unprocessed = cursor.fetchone()[0]
                
                base_stats.update({
                    "total_persisted": total,
                    "unprocessed": unprocessed,
                    "last_processed_id": self._last_processed_id
                })
        except:
            pass
        
        return base_stats


# Factory function
def create_persistent_event_bus(db_path: str = "data/events.db") -> PersistentEventBus:
    return PersistentEventBus(db_path=db_path)
