"""
CryptoBoss 1.0.1 - Engine Lifecycle Manager

ABSOLUTE RULES:
1. NO DATA may exist without exchange_account_id
2. Engine MUST refuse to start without identity
3. NEW ACCOUNT means ZERO history
4. ACCOUNT SWITCH means FULL RESET

This module enforces strict engine lifecycle control.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EngineState(str, Enum):
    """Engine state machine."""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


@dataclass
class EngineIdentity:
    """
    Required identity for engine operation.
    
    HARD RULE: Engine MUST NOT operate without valid identity.
    """
    user_id: str
    exchange_account_id: str
    environment: str  # LIVE, TESTNET
    
    def validate(self) -> bool:
        """Validate identity is complete."""
        if not self.user_id:
            return False
        if not self.exchange_account_id:
            return False
        if self.environment not in ("LIVE", "TESTNET", "live", "testnet"):
            return False
        return True
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "exchange_account_id": self.exchange_account_id,
            "environment": self.environment
        }


class EngineLifecycle:
    """
    Strict engine lifecycle controller.
    
    STARTUP REQUIREMENTS:
    - user_id provided
    - exchange_account_id provided
    - environment validated
    
    ENGINE START RULES:
    - If exchange_account_id is NEW → create empty state
    - Do NOT load any previous files
    - Initialize balances, history, replay as empty
    
    ACCOUNT SWITCH RULES:
    - STOP engine
    - DESTROY all in-memory objects
    - CLEAR caches
    - RESTART engine with new exchange_account_id
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.state = EngineState.STOPPED
        self.identity: Optional[EngineIdentity] = None
        self.start_time: Optional[datetime] = None
        
        # Registered components that need reset
        self._components: Dict[str, Any] = {}
        self._reset_callbacks: list[Callable] = []
        
        # Price feed reference
        self._price_feed = None
        
        self._initialized = True
        logger.info("🔧 EngineLifecycle initialized")
    
    def register_component(self, name: str, component: Any):
        """Register a component for lifecycle management."""
        self._components[name] = component
        logger.debug(f"Registered component: {name}")
    
    def register_reset_callback(self, callback: Callable):
        """Register a callback to be called on reset."""
        self._reset_callbacks.append(callback)
    
    def start(self, user_id: str, exchange_account_id: str, environment: str) -> bool:
        """
        Start the engine with required identity.
        
        HARD RULE: Will NOT start without valid identity.
        """
        if self.state == EngineState.RUNNING:
            logger.warning("Engine already running - call switch_account instead")
            return False
        
        # Create and validate identity
        identity = EngineIdentity(
            user_id=user_id,
            exchange_account_id=exchange_account_id,
            environment=environment.upper()
        )
        
        if not identity.validate():
            logger.error("❌ Engine refused to start: invalid identity")
            self.state = EngineState.ERROR
            return False
        
        self.state = EngineState.STARTING
        self.identity = identity
        
        logger.info(f"🚀 Engine starting for account {exchange_account_id[:8]}... ({environment})")
        
        # Initialize empty state for new account
        self._initialize_empty_state()
        
        # Start price feed
        self._start_price_feed()
        
        self.state = EngineState.RUNNING
        self.start_time = datetime.now()
        
        logger.info(f"✅ Engine started successfully")
        return True
    
    def stop(self) -> bool:
        """
        Stop the engine and destroy all in-memory objects.
        """
        if self.state != EngineState.RUNNING:
            logger.warning("Engine not running")
            return False
        
        self.state = EngineState.STOPPING
        logger.info("⏹️ Engine stopping...")
        
        # Stop price feed
        self._stop_price_feed()
        
        # Destroy all components
        self._destroy_all_components()
        
        # Clear identity
        self.identity = None
        self.start_time = None
        
        self.state = EngineState.STOPPED
        logger.info("✅ Engine stopped")
        return True
    
    def switch_account(self, user_id: str, exchange_account_id: str, environment: str) -> bool:
        """
        Switch to a different account.
        
        THIS TRIGGERS:
        1. STOP engine
        2. DESTROY all in-memory objects
        3. CLEAR caches
        4. RESTART engine with new exchange_account_id
        """
        logger.info(f"🔄 Switching account to {exchange_account_id[:8]}...")
        
        # Step 1: Stop engine
        if self.state == EngineState.RUNNING:
            self.stop()
        
        # Step 2 & 3: Destroy and clear (done in stop())
        
        # Step 4: Start with new identity
        return self.start(user_id, exchange_account_id, environment)
    
    def _initialize_empty_state(self):
        """
        Initialize empty state for account.
        
        NEW ACCOUNT = ZERO history
        - Empty risk state
        - Empty trade history
        - Zero daily counters
        - Empty replay buffer
        - Empty analytics cache
        - Empty price cache
        """
        logger.info("📋 Initializing empty state...")
        
        # Import here to avoid circular imports
        try:
            from src.core.scoped_state import ScopedStateManager
            
            manager = ScopedStateManager.get_or_create(
                self.identity.user_id,
                self.identity.exchange_account_id
            )
            manager.hard_reset()
            ScopedStateManager.set_active(manager)
            
            logger.info("✅ Empty state initialized via ScopedStateManager")
        except ImportError:
            logger.warning("ScopedStateManager not available")
        
        # Run all reset callbacks
        for callback in self._reset_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Reset callback error: {e}")
    
    def _destroy_all_components(self):
        """Destroy all registered components."""
        logger.info(f"🗑️ Destroying {len(self._components)} components...")
        
        for name, component in list(self._components.items()):
            try:
                if hasattr(component, 'destroy'):
                    component.destroy()
                elif hasattr(component, 'close'):
                    component.close()
                elif hasattr(component, 'reset'):
                    component.reset()
            except Exception as e:
                logger.error(f"Error destroying {name}: {e}")
        
        self._components.clear()
        
        # Run all reset callbacks
        for callback in self._reset_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Reset callback error: {e}")
    
    def _start_price_feed(self):
        """Start price feed for current environment."""
        try:
            from src.core.live_price_feed import LivePriceFeed, get_price_feed
            
            self._price_feed = get_price_feed()
            self._price_feed.start(
                environment=self.identity.environment,
                exchange_account_id=self.identity.exchange_account_id
            )
            logger.info("📈 Price feed started")
        except ImportError:
            logger.warning("LivePriceFeed not available")
        except Exception as e:
            logger.error(f"Price feed start error: {e}")
    
    def _stop_price_feed(self):
        """Stop price feed."""
        if self._price_feed:
            try:
                self._price_feed.stop()
                logger.info("📉 Price feed stopped")
            except Exception as e:
                logger.error(f"Price feed stop error: {e}")
            self._price_feed = None
    
    def get_status(self) -> Dict:
        """Get current engine status."""
        return {
            "state": self.state.value,
            "identity": self.identity.to_dict() if self.identity else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "components_count": len(self._components)
        }
    
    def require_running(self):
        """Require engine to be running. Raises if not."""
        if self.state != EngineState.RUNNING:
            raise RuntimeError(f"Engine not running (state: {self.state.value})")
        if not self.identity:
            raise RuntimeError("Engine running but no identity set")


# Singleton accessor
def get_engine_lifecycle() -> EngineLifecycle:
    return EngineLifecycle()
