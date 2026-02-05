"""
CryptoBoss - Bot Instance Model

FUNDAMENTAL AXIOM: There is NO such thing as a global bot state.

Every exchange_account_id gets its own COMPLETE, ISOLATED bot instance.
Bot instances NEVER share:
- Memory
- Files
- Database rows
- WebSocket connections
- Price cache

If two accounts share data, the system is BROKEN.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class BotInstanceState(str, Enum):
    """Bot instance lifecycle state."""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    DESTROYED = "DESTROYED"


@dataclass
class AccountIdentity:
    """Identity binding for a bot instance."""
    user_id: str
    exchange_account_id: str
    environment: str  # LIVE or TESTNET
    
    def validate(self) -> bool:
        return bool(self.user_id and self.exchange_account_id and self.environment)
    
    @property
    def storage_path(self) -> Path:
        """Get storage directory for this account."""
        return Path("data") / f"user_{self.user_id}" / f"account_{self.exchange_account_id}"


@dataclass
class TradingState:
    """
    ALL trading state owned by ONE exchange_account_id.
    
    This is NOT shared. Each bot instance has its own copy.
    """
    # Balances
    balances: Dict[str, float] = field(default_factory=dict)
    
    # Positions
    positions: List[Dict] = field(default_factory=list)
    
    # Trade History (for THIS account only)
    trade_history: List[Dict] = field(default_factory=list)
    
    # Risk State
    daily_loss: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    max_drawdown_today: float = 0.0
    
    # Analytics
    total_pnl: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_trade_at: Optional[datetime] = None
    
    def reset(self):
        """Reset to empty state (for new account)."""
        self.balances = {}
        self.positions = []
        self.trade_history = []
        self.daily_loss = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.max_drawdown_today = 0.0
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.total_trades = 0
        self.last_trade_at = None
    
    def to_dict(self) -> Dict:
        return {
            "balances": self.balances,
            "positions": self.positions,
            "trade_history": self.trade_history[-100:],  # Last 100 trades
            "daily_loss": self.daily_loss,
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
            "max_drawdown_today": self.max_drawdown_today,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "created_at": self.created_at.isoformat(),
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None
        }


@dataclass
class PriceCache:
    """
    Price cache owned by ONE bot instance.
    
    RULES:
    - Empty on bot start
    - Cleared on account switch
    - No price shown until first valid tick
    """
    prices: Dict[str, Dict] = field(default_factory=dict)
    last_update: Optional[datetime] = None
    
    def clear(self):
        """Clear all cached prices."""
        self.prices = {}
        self.last_update = None
    
    def update(self, symbol: str, price: float, timestamp: datetime):
        self.prices[symbol] = {
            "price": price,
            "timestamp": timestamp.isoformat(),
            "age_ms": 0
        }
        self.last_update = timestamp


class BotInstance:
    """
    One COMPLETE trading engine bound to exactly ONE exchange_account_id.
    
    RULES:
    - One exchange_account_id = one bot instance
    - Bot instances NEVER share memory
    - Bot instances NEVER share files
    - Bot instances NEVER share database rows
    
    FORBIDDEN:
    - Singleton engines
    - Global caches
    - Static state
    """
    
    def __init__(self, identity: AccountIdentity):
        if not identity.validate():
            raise ValueError("Invalid account identity")
        
        self.identity = identity
        self.state = BotInstanceState.STOPPED
        
        # === OWN state (NOT shared) ===
        self.trading_state = TradingState()
        self.price_cache = PriceCache()
        
        # === OWN connections (NOT shared) ===
        self._ws_connection = None
        self._exchange_client = None
        
        # === OWN storage path ===
        self._storage_path = identity.storage_path
        
        # Create storage directory
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🤖 BotInstance created for {identity.exchange_account_id[:8]}...")
    
    def start(self) -> bool:
        """
        Start this bot instance.
        
        Steps:
        1. Initialize EMPTY state containers
        2. Load persisted state (if exists for THIS account)
        3. Connect to exchange
        4. Start price stream
        """
        if self.state == BotInstanceState.RUNNING:
            logger.warning("Bot instance already running")
            return False
        
        self.state = BotInstanceState.STARTING
        logger.info(f"🚀 Starting bot instance for {self.identity.exchange_account_id[:8]}...")
        
        # Step 1: Load state for THIS account only
        self._load_state()
        
        # Step 2: Clear price cache (empty on start)
        self.price_cache.clear()
        
        # Step 3: Connect to exchange (TODO: implement)
        self._connect_exchange()
        
        # Step 4: Start price stream (TODO: implement)
        self._start_price_stream()
        
        self.state = BotInstanceState.RUNNING
        logger.info(f"✅ Bot instance running for {self.identity.exchange_account_id[:8]}")
        return True
    
    def stop(self) -> bool:
        """
        Stop this bot instance.
        
        Steps:
        1. Flush state to storage
        2. Close websockets
        3. Disconnect from exchange
        """
        if self.state != BotInstanceState.RUNNING:
            logger.warning("Bot instance not running")
            return False
        
        self.state = BotInstanceState.STOPPING
        logger.info(f"⏹️ Stopping bot instance for {self.identity.exchange_account_id[:8]}...")
        
        # Step 1: Save state
        self._save_state()
        
        # Step 2: Close websockets
        self._stop_price_stream()
        
        # Step 3: Disconnect exchange
        self._disconnect_exchange()
        
        self.state = BotInstanceState.STOPPED
        logger.info(f"✅ Bot instance stopped for {self.identity.exchange_account_id[:8]}")
        return True
    
    def destroy(self):
        """
        DESTROY all in-memory objects.
        
        Called on account switch.
        """
        logger.warning(f"🗑️ Destroying bot instance for {self.identity.exchange_account_id[:8]}...")
        
        # Stop if running
        if self.state == BotInstanceState.RUNNING:
            self.stop()
        
        # Clear all state
        self.trading_state.reset()
        self.price_cache.clear()
        
        # Close connections
        self._ws_connection = None
        self._exchange_client = None
        
        self.state = BotInstanceState.DESTROYED
        logger.info(f"✅ Bot instance destroyed for {self.identity.exchange_account_id[:8]}")
    
    # === State Persistence ===
    
    def _load_state(self):
        """Load state from THIS account's storage."""
        state_file = self._storage_path / "trading_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                self.trading_state.balances = data.get("balances", {})
                self.trading_state.positions = data.get("positions", [])
                self.trading_state.trade_history = data.get("trade_history", [])
                self.trading_state.daily_loss = data.get("daily_loss", 0.0)
                self.trading_state.daily_trades = data.get("daily_trades", 0)
                self.trading_state.total_pnl = data.get("total_pnl", 0.0)
                self.trading_state.total_trades = data.get("total_trades", 0)
                
                logger.info(f"📂 Loaded state for {self.identity.exchange_account_id[:8]}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                self.trading_state.reset()
        else:
            # NEW ACCOUNT = empty state
            logger.info(f"🆕 New account {self.identity.exchange_account_id[:8]} - empty state")
            self.trading_state.reset()
    
    def _save_state(self):
        """Save state to THIS account's storage."""
        state_file = self._storage_path / "trading_state.json"
        
        try:
            with open(state_file, 'w') as f:
                json.dump(self.trading_state.to_dict(), f, indent=2)
            logger.debug(f"💾 Saved state for {self.identity.exchange_account_id[:8]}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    # === Exchange Connection (placeholders) ===
    
    def _connect_exchange(self):
        """Connect to exchange for THIS account."""
        logger.debug("Connecting to exchange...")
        # TODO: Implement actual exchange connection
    
    def _disconnect_exchange(self):
        """Disconnect from exchange."""
        logger.debug("Disconnecting from exchange...")
        self._exchange_client = None
    
    def _start_price_stream(self):
        """Start price websocket for THIS account."""
        logger.debug("Starting price stream...")
        # TODO: Implement price stream
    
    def _stop_price_stream(self):
        """Stop price websocket."""
        logger.debug("Stopping price stream...")
        self._ws_connection = None
    
    # === Trading Operations ===
    
    def record_trade(self, trade: Dict):
        """Record a trade for THIS account."""
        trade["exchange_account_id"] = self.identity.exchange_account_id
        trade["timestamp"] = datetime.now().isoformat()
        
        self.trading_state.trade_history.append(trade)
        self.trading_state.total_trades += 1
        self.trading_state.daily_trades += 1
        self.trading_state.last_trade_at = datetime.now()
        
        # Update PnL
        pnl = trade.get("pnl", 0)
        self.trading_state.total_pnl += pnl
        if pnl < 0:
            self.trading_state.daily_loss += abs(pnl)
            self.trading_state.consecutive_losses += 1
        else:
            self.trading_state.consecutive_losses = 0
        
        # Auto-save
        self._save_state()
    
    def get_dashboard_data(self) -> Dict:
        """Get dashboard data for THIS account only."""
        return {
            "exchange_account_id": self.identity.exchange_account_id,
            "environment": self.identity.environment,
            "state": self.state.value,
            "trading": self.trading_state.to_dict(),
            "prices": self.price_cache.prices,
            "timestamp": datetime.now().isoformat()
        }


class BotInstanceManager:
    """
    Manages bot instances by exchange_account_id.
    
    RULES:
    - One instance per exchange_account_id
    - switch_instance() = STOP → DESTROY → START
    - NO global state
    """
    
    _instances: Dict[str, BotInstance] = {}
    _active_instance: Optional[BotInstance] = None
    
    @classmethod
    def get_instance(cls, exchange_account_id: str) -> Optional[BotInstance]:
        """Get existing instance (does NOT create)."""
        return cls._instances.get(exchange_account_id)
    
    @classmethod
    def get_active(cls) -> Optional[BotInstance]:
        """Get currently active instance."""
        return cls._active_instance
    
    @classmethod
    def start_instance(cls, user_id: str, exchange_account_id: str, environment: str) -> BotInstance:
        """
        Start a NEW bot instance.
        
        If instance exists, returns it.
        Otherwise creates new empty instance.
        """
        if exchange_account_id in cls._instances:
            instance = cls._instances[exchange_account_id]
            if instance.state != BotInstanceState.RUNNING:
                instance.start()
            return instance
        
        # Create NEW instance
        identity = AccountIdentity(
            user_id=user_id,
            exchange_account_id=exchange_account_id,
            environment=environment.upper()
        )
        
        instance = BotInstance(identity)
        instance.start()
        
        cls._instances[exchange_account_id] = instance
        cls._active_instance = instance
        
        return instance
    
    @classmethod
    def stop_instance(cls, exchange_account_id: str):
        """Stop a bot instance."""
        instance = cls._instances.get(exchange_account_id)
        if instance:
            instance.stop()
    
    @classmethod
    def switch_instance(cls, user_id: str, exchange_account_id: str, environment: str) -> BotInstance:
        """
        Switch to a different account.
        
        MANDATORY BEHAVIOR:
        1. STOP current bot instance
        2. DESTROY all its memory
        3. START a brand-new bot instance
        4. LOAD state ONLY for selected exchange_account_id
        """
        logger.info(f"🔄 Switching to account {exchange_account_id[:8]}...")
        
        # Step 1 & 2: Stop and destroy current instance
        if cls._active_instance:
            old_id = cls._active_instance.identity.exchange_account_id
            cls._active_instance.destroy()
            del cls._instances[old_id]
            logger.info(f"🗑️ Destroyed previous instance {old_id[:8]}")
        
        # Step 3 & 4: Start new instance
        new_instance = cls.start_instance(user_id, exchange_account_id, environment)
        cls._active_instance = new_instance
        
        logger.info(f"✅ Switched to account {exchange_account_id[:8]}")
        return new_instance
    
    @classmethod
    def destroy_all(cls):
        """Destroy all instances (for shutdown)."""
        for instance in list(cls._instances.values()):
            instance.destroy()
        cls._instances.clear()
        cls._active_instance = None


# === Convenience Functions ===

def get_active_bot() -> Optional[BotInstance]:
    """Get the currently active bot instance."""
    return BotInstanceManager.get_active()


def require_active_bot() -> BotInstance:
    """Get active bot or raise error."""
    bot = get_active_bot()
    if not bot:
        raise RuntimeError("No active bot instance")
    return bot


def switch_account(user_id: str, exchange_account_id: str, environment: str) -> BotInstance:
    """Switch to a different account (triggers full reset)."""
    return BotInstanceManager.switch_instance(user_id, exchange_account_id, environment)
