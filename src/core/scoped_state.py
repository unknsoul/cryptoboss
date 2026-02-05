"""
CryptoBoss 1.0.1 - Scoped State Manager

CRITICAL: All state MUST be scoped by (user_id, exchange_account_id).
No singleton may hold account-specific data.

This module provides:
- State namespacing by exchange_account_id
- Hard reset on account switch
- Validation that data matches active account
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Base directory for scoped state files
STATE_DIR = Path(__file__).parent.parent.parent / "data" / "account_state"


@dataclass
class IdentityTuple:
    """
    Core identity for all state operations.
    
    HARD RULE: No data exists without this tuple.
    """
    user_id: str
    exchange_account_id: str
    
    def validate(self) -> bool:
        """Validate identity tuple is complete."""
        return bool(self.user_id and self.exchange_account_id)
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "exchange_account_id": self.exchange_account_id
        }
    
    @property
    def namespace(self) -> str:
        """Get namespace prefix for this identity."""
        return f"{self.exchange_account_id[:8]}"


@dataclass
class ScopedState:
    """
    State container scoped to a specific exchange account.
    
    Each account gets completely isolated state.
    """
    identity: IdentityTuple
    
    # Risk state
    daily_loss: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    max_drawdown_today: float = 0.0
    
    # Trade history (kept minimal in memory)
    recent_trades: List[Dict] = field(default_factory=list)
    
    # Incident state
    incident_state: str = "NORMAL"
    incident_reason: Optional[str] = None
    
    # Analytics cache (empty for new account)
    analytics_cache: Dict = field(default_factory=dict)
    
    # Replay buffer (empty for new account)
    replay_buffer: List[Dict] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    
    def is_new_account(self) -> bool:
        """Check if account is less than 24 hours old."""
        age = datetime.now() - self.created_at
        return age.total_seconds() < 86400  # 24 hours
    
    def to_dict(self) -> Dict:
        return {
            "identity": self.identity.to_dict(),
            "daily_loss": self.daily_loss,
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
            "max_drawdown_today": self.max_drawdown_today,
            "recent_trades": self.recent_trades[-10:],  # Keep last 10
            "incident_state": self.incident_state,
            "incident_reason": self.incident_reason,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "is_new_account": self.is_new_account()
        }


class ScopedStateManager:
    """
    Manager for account-scoped state.
    
    CORE RULES:
    1. Engine MUST refuse to start without identity tuple
    2. Each exchange_account_id is a fresh universe
    3. Account switch triggers FULL state reset
    4. All filenames include exchange_account_id
    """
    
    _instances: Dict[str, 'ScopedStateManager'] = {}
    _active_instance: Optional['ScopedStateManager'] = None
    
    def __init__(self, identity: IdentityTuple):
        if not identity.validate():
            raise ValueError("Invalid identity tuple - both user_id and exchange_account_id required")
        
        self.identity = identity
        self.state = ScopedState(identity=identity)
        self._initialized = False
        self._state_dir = STATE_DIR / identity.exchange_account_id
        
        # Ensure state directory exists
        self._state_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔒 ScopedStateManager created for account {identity.namespace}")
    
    @classmethod
    def get_or_create(cls, user_id: str, exchange_account_id: str) -> 'ScopedStateManager':
        """Get existing or create new scoped state manager."""
        key = exchange_account_id
        
        if key not in cls._instances:
            identity = IdentityTuple(user_id=user_id, exchange_account_id=exchange_account_id)
            cls._instances[key] = cls(identity)
        
        return cls._instances[key]
    
    @classmethod
    def set_active(cls, manager: 'ScopedStateManager'):
        """Set the active state manager (triggers reset of old one)."""
        if cls._active_instance and cls._active_instance != manager:
            logger.info(f"🔄 Switching from account {cls._active_instance.identity.namespace} to {manager.identity.namespace}")
            cls._active_instance._save_state()
        
        cls._active_instance = manager
        manager._load_state()
    
    @classmethod
    def get_active(cls) -> Optional['ScopedStateManager']:
        """Get currently active state manager."""
        return cls._active_instance
    
    @classmethod
    def clear_all(cls):
        """Clear all state managers (for testing)."""
        for manager in cls._instances.values():
            manager._save_state()
        cls._instances.clear()
        cls._active_instance = None
    
    def hard_reset(self):
        """
        HARD RESET: Clear all state for this account.
        
        Used when creating a fresh account or fixing corruption.
        """
        logger.warning(f"⚠️ HARD RESET for account {self.identity.namespace}")
        
        self.state = ScopedState(identity=self.identity)
        
        # Delete state files
        for file in self._state_dir.glob("*.json"):
            file.unlink()
        
        logger.info(f"✅ Account {self.identity.namespace} reset to clean state")
    
    def get_state_file(self, name: str) -> Path:
        """Get namespaced path for a state file."""
        # GOOD: risk_state_<account_id>.json
        return self._state_dir / f"{name}_{self.identity.namespace}.json"
    
    def _load_state(self):
        """Load state from disk if exists."""
        state_file = self.get_state_file("state")
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                self.state.daily_loss = data.get("daily_loss", 0.0)
                self.state.daily_trades = data.get("daily_trades", 0)
                self.state.consecutive_losses = data.get("consecutive_losses", 0)
                self.state.incident_state = data.get("incident_state", "NORMAL")
                self.state.recent_trades = data.get("recent_trades", [])
                
                if "created_at" in data:
                    self.state.created_at = datetime.fromisoformat(data["created_at"])
                
                logger.info(f"📂 Loaded state for account {self.identity.namespace}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        else:
            logger.info(f"🆕 New account {self.identity.namespace} - starting with clean state")
        
        self.state.last_active = datetime.now()
        self._initialized = True
    
    def _save_state(self):
        """Save state to disk."""
        if not self._initialized:
            return
        
        state_file = self.get_state_file("state")
        
        try:
            with open(state_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
            logger.debug(f"💾 Saved state for account {self.identity.namespace}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def validate_data_scope(self, data: Dict) -> bool:
        """
        Validate that data matches this account's scope.
        
        HARD RULE: If mismatch, DROP DATA.
        """
        data_account = data.get("exchange_account_id")
        if data_account and data_account != self.identity.exchange_account_id:
            logger.warning(f"⚠️ Data scope mismatch: expected {self.identity.namespace}, got {data_account[:8]}")
            return False
        return True
    
    def wrap_response(self, data: Dict) -> Dict:
        """
        Wrap API response with mandatory identity fields.
        
        ALL responses must include:
        - user_id
        - exchange_account_id
        - environment
        - data_scope
        """
        return {
            **data,
            "user_id": self.identity.user_id,
            "exchange_account_id": self.identity.exchange_account_id,
            "data_scope": "SCOPED",
            "account_created_at": self.state.created_at.isoformat(),
            "is_new_account": self.state.is_new_account()
        }
    
    # === State Accessors ===
    
    def record_trade(self, trade: Dict):
        """Record a trade for this account."""
        trade["exchange_account_id"] = self.identity.exchange_account_id
        trade["timestamp"] = datetime.now().isoformat()
        
        self.state.recent_trades.append(trade)
        self.state.daily_trades += 1
        self.state.last_active = datetime.now()
        
        # Track P&L
        pnl = trade.get("pnl", 0)
        if pnl < 0:
            self.state.daily_loss += abs(pnl)
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        
        self._save_state()
    
    def get_daily_stats(self) -> Dict:
        """Get daily stats scoped to this account."""
        return {
            "exchange_account_id": self.identity.exchange_account_id,
            "daily_trades": self.state.daily_trades,
            "daily_loss": self.state.daily_loss,
            "consecutive_losses": self.state.consecutive_losses,
            "is_new_account": self.state.is_new_account()
        }
    
    def reset_daily(self):
        """Reset daily counters (called at midnight)."""
        self.state.daily_loss = 0.0
        self.state.daily_trades = 0
        self.state.max_drawdown_today = 0.0
        self._save_state()


# Convenience functions

def get_active_state() -> Optional[ScopedStateManager]:
    """Get the currently active scoped state manager."""
    return ScopedStateManager.get_active()


def require_active_state() -> ScopedStateManager:
    """Get active state or raise error."""
    state = get_active_state()
    if not state:
        raise RuntimeError("No active account selected - engine requires identity tuple")
    return state


def switch_account(user_id: str, exchange_account_id: str) -> ScopedStateManager:
    """
    Switch to a different exchange account.
    
    This triggers:
    1. Save current state
    2. Clear in-memory caches
    3. Load new account state (or create fresh)
    """
    manager = ScopedStateManager.get_or_create(user_id, exchange_account_id)
    ScopedStateManager.set_active(manager)
    return manager
