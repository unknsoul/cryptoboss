"""
State Manager - Crash-Proof State Persistence

The "Black Box" that ensures no state is lost on restart.
Every strategy state change is persisted before execution.

Architecture:
    Strategy -> StateManager.save() -> Database -> Exchange API
    On Restart: Database -> StateManager.load() -> Strategy Resume

Supports:
    - SQLite (local, default)
    - PostgreSQL (production)
    - Redis (high-speed caching layer)
"""

import json
import sqlite3
import logging
from typing import Any, Dict, List, Optional, Type
from datetime import datetime
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from abc import ABC, abstractmethod
import threading

logger = logging.getLogger(__name__)


class StateBackend(ABC):
    """Abstract state storage backend."""
    
    @abstractmethod
    def save(self, key: str, state: Dict) -> bool:
        pass
    
    @abstractmethod
    def load(self, key: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        pass


class SQLiteBackend(StateBackend):
    """SQLite-based state persistence."""
    
    def __init__(self, db_path: str = "data/state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    schema_version INTEGER DEFAULT 1
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON state(key)")
            conn.commit()
        self._migrate_schema()
    
    def _migrate_schema(self):
        """Run schema migrations if needed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if schema_version column exists
                cursor = conn.execute("PRAGMA table_info(state)")
                columns = [row[1] for row in cursor.fetchall()]
                if "schema_version" not in columns:
                    conn.execute("ALTER TABLE state ADD COLUMN schema_version INTEGER DEFAULT 1")
                    conn.commit()
                    logger.info("Schema migration: added schema_version column")
        except Exception as e:
            logger.warning(f"Schema migration check failed: {e}")
    
    def save(self, key: str, state: Dict) -> bool:
        """Save state atomically using BEGIN IMMEDIATE transaction."""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute("""
                        INSERT OR REPLACE INTO state (key, state_json, updated_at, version, schema_version)
                        VALUES (?, ?, ?, COALESCE(
                            (SELECT version + 1 FROM state WHERE key = ?), 1
                        ), 1)
                    """, (key, json.dumps(state, default=str), datetime.now().isoformat(), key))
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                finally:
                    conn.close()
                return True
            except Exception as e:
                logger.error(f"Failed to save state {key}: {e}")
                return False
    
    def load(self, key: str) -> Optional[Dict]:
        """Load state. Returns None on corruption instead of crashing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT state_json FROM state WHERE key = ?", (key,))
                row = cursor.fetchone()
                if not row:
                    return None
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError as je:
                    logger.error(f"Corrupted state data for key {key}: {je}")
                    return None
        except Exception as e:
            logger.error(f"Failed to load state {key}: {e}")
            return None
    
    def backup_state(self) -> Optional[str]:
        """Create a backup of the state database."""
        import shutil
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.db_path.parent / f"state_backup_{timestamp}.db"
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"State backup created: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Failed to create state backup: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM state WHERE key = ?", (key,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete state {key}: {e}")
            return False
    
    def list_keys(self, prefix: str = "") -> List[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT key FROM state WHERE key LIKE ?",
                    (f"{prefix}%",)
                )
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            return []


@dataclass
class StrategyState:
    """Base state for all strategies."""
    strategy_id: str
    strategy_type: str
    symbol: str
    status: str  # 'active', 'paused', 'stopped'
    created_at: str
    updated_at: str
    capital_allocated: float
    current_pnl: float
    custom_state: Dict  # Strategy-specific state


class StateManager:
    """
    Central state management for all strategies.
    
    Usage:
        state_mgr = StateManager()
        
        # Save strategy state (do this BEFORE sending orders)
        state_mgr.save_strategy_state("dca_btc_1", {
            "active_deal": {...},
            "safety_orders_used": 3,
            ...
        })
        
        # On restart, recover all states
        states = state_mgr.load_all_active_strategies()
        for state in states:
            strategy = recreate_strategy(state)
            strategy.resume()
    """
    
    def __init__(self, backend: StateBackend = None):
        self.backend = backend or SQLiteBackend()
        self._cache: Dict[str, Dict] = {}
        logger.info("StateManager initialized with crash-proof persistence")
    
    def save_strategy_state(
        self,
        strategy_id: str,
        strategy_type: str,
        symbol: str,
        status: str,
        capital_allocated: float,
        current_pnl: float,
        custom_state: Dict
    ) -> bool:
        """
        Save strategy state. MUST be called before any order execution.
        
        Args:
            strategy_id: Unique identifier for the strategy instance
            strategy_type: Type of strategy ('dca', 'grid', 'market_making')
            symbol: Trading symbol
            status: Current status
            capital_allocated: Capital in this strategy
            current_pnl: Realized P&L so far
            custom_state: Strategy-specific state dict
        
        Returns:
            True if saved successfully
        """
        state = StrategyState(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            symbol=symbol,
            status=status,
            created_at=self._cache.get(strategy_id, {}).get('created_at', datetime.now().isoformat()),
            updated_at=datetime.now().isoformat(),
            capital_allocated=capital_allocated,
            current_pnl=current_pnl,
            custom_state=custom_state
        )
        
        key = f"strategy:{strategy_id}"
        state_dict = asdict(state)
        
        success = self.backend.save(key, state_dict)
        if success:
            self._cache[strategy_id] = state_dict
            logger.debug(f"Saved state for {strategy_id}")
        
        return success
    
    def load_strategy_state(self, strategy_id: str) -> Optional[StrategyState]:
        """Load a specific strategy's state."""
        key = f"strategy:{strategy_id}"
        data = self.backend.load(key)
        
        if data:
            return StrategyState(**data)
        return None
    
    def load_all_active_strategies(self) -> List[StrategyState]:
        """Load all active strategy states for recovery after restart."""
        keys = self.backend.list_keys("strategy:")
        states = []
        
        for key in keys:
            data = self.backend.load(key)
            if data and data.get('status') in ('active', 'paused'):
                states.append(StrategyState(**data))
        
        logger.info(f"Recovered {len(states)} active strategies from persistent storage")
        return states
    
    def mark_strategy_stopped(self, strategy_id: str) -> bool:
        """Mark a strategy as stopped (completed or cancelled)."""
        state = self.load_strategy_state(strategy_id)
        if state:
            return self.save_strategy_state(
                strategy_id=state.strategy_id,
                strategy_type=state.strategy_type,
                symbol=state.symbol,
                status='stopped',
                capital_allocated=state.capital_allocated,
                current_pnl=state.current_pnl,
                custom_state=state.custom_state
            )
        return False
    
    def save_order(self, order_id: str, order_data: Dict) -> bool:
        """Save an order for tracking."""
        key = f"order:{order_id}"
        order_data['saved_at'] = datetime.now().isoformat()
        return self.backend.save(key, order_data)
    
    def load_pending_orders(self) -> List[Dict]:
        """Load all orders that may need reconciliation."""
        keys = self.backend.list_keys("order:")
        orders = []
        for key in keys:
            data = self.backend.load(key)
            if data and data.get('status') in ('pending', 'open', 'partially_filled'):
                orders.append(data)
        return orders
    
    def get_portfolio_snapshot(self) -> Dict:
        """Get snapshot of all strategy states for monitoring."""
        strategies = self.load_all_active_strategies()
        
        total_capital = sum(s.capital_allocated for s in strategies)
        total_pnl = sum(s.current_pnl for s in strategies)
        
        return {
            'total_strategies': len(strategies),
            'total_capital_allocated': total_capital,
            'total_pnl': total_pnl,
            'strategies': [asdict(s) for s in strategies],
            'snapshot_time': datetime.now().isoformat()
        }


# Singleton instance for global access
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get the global StateManager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
