"""
Session Manager - Clean Session Lifecycle Management

This module implements the core session lifecycle system:
- Unique session_id generation on mode/API changes
- Session-scoped state containers
- Explicit reset and archive functions
- Mode switching with proper shutdown/startup sequences

Core Principles:
- Mode switching creates a new session
- No session may reuse another session's state
- All state is scoped to session_id
"""

import uuid
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    """Trading modes supported by the system."""
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


@dataclass
class ApiConfig:
    """API configuration for exchange connection."""
    mode: TradingMode
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    is_validated: bool = False
    validated_at: Optional[str] = None
    
    def is_configured(self) -> bool:
        """Check if API credentials are configured (for non-paper modes)."""
        if self.mode == TradingMode.PAPER:
            # PAPER mode is deprecated - still return True but prefer TESTNET
            logger.warning("PAPER mode is deprecated. Use TESTNET for testing.")
            return True
        return bool(self.api_key and self.api_secret)


@dataclass
class SessionState:
    """
    All mutable state scoped to a single session.
    This is completely reset on session change.
    """
    # Core identifiers
    session_id: str
    mode: TradingMode
    created_at: str
    
    # Exchange state (freshly fetched each session)
    balances: Dict[str, float] = field(default_factory=dict)
    open_positions: List[Dict] = field(default_factory=list)
    open_orders: List[Dict] = field(default_factory=list)
    
    # Trading state
    trades_this_session: List[Dict] = field(default_factory=list)
    decisions_this_session: List[Dict] = field(default_factory=list)
    proposals_cache: List[Dict] = field(default_factory=list)
    
    # Status
    is_running: bool = False
    last_price_update: Optional[str] = None
    last_exchange_sync: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def empty(cls, session_id: str, mode: TradingMode) -> "SessionState":
        """Create an empty session state."""
        return cls(
            session_id=session_id,
            mode=mode,
            created_at=datetime.now().isoformat()
        )


class SessionManager:
    """
    Manages session lifecycle and state isolation.
    
    Ensures:
    - Unique session_id per mode/API configuration
    - Complete state isolation between sessions
    - Proper shutdown/startup sequences
    - Session archival for audit
    """
    
    def __init__(self, archive_dir: Optional[Path] = None):
        self._current_session: Optional[SessionState] = None
        self._api_config: Optional[ApiConfig] = None
        self._archive_dir = archive_dir or Path("data/sessions")
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        self._shutdown_callbacks: List[callable] = []
        self._startup_callbacks: List[callable] = []
        
        logger.info("SessionManager initialized")
    
    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._current_session.session_id if self._current_session else None
    
    @property
    def mode(self) -> Optional[TradingMode]:
        """Get current trading mode."""
        return self._current_session.mode if self._current_session else None
    
    @property
    def state(self) -> Optional[SessionState]:
        """Get current session state (read-only access)."""
        return self._current_session
    
    @property
    def api_config(self) -> Optional[ApiConfig]:
        """Get current API configuration."""
        return self._api_config
    
    @property
    def is_active(self) -> bool:
        """Check if there is an active session."""
        return self._current_session is not None
    
    def register_shutdown_callback(self, callback: callable):
        """Register a function to call during shutdown."""
        self._shutdown_callbacks.append(callback)
    
    def register_startup_callback(self, callback: callable):
        """Register a function to call during startup."""
        self._startup_callbacks.append(callback)
    
    def generate_session_id(self) -> str:
        """Generate a new unique session ID."""
        return str(uuid.uuid4())
    
    async def start_session(
        self, 
        mode: TradingMode,
        api_config: Optional[ApiConfig] = None
    ) -> SessionState:
        """
        Start a new session with the specified mode.
        
        Args:
            mode: Trading mode (paper/testnet/live)
            api_config: Optional API configuration for exchange modes
            
        Returns:
            New SessionState
        """
        # If there's an existing session, archive and shutdown first
        if self._current_session:
            await self.end_session()
        
        # Generate new session ID
        session_id = self.generate_session_id()
        
        # Create empty state
        self._current_session = SessionState.empty(session_id, mode)
        self._api_config = api_config or ApiConfig(mode=mode)
        
        logger.info(f"Started new session: {session_id[:8]}... (mode={mode.value})")
        
        # Run startup callbacks
        for callback in self._startup_callbacks:
            try:
                if callable(callback):
                    result = callback(self._current_session)
                    if hasattr(result, '__await__'):
                        await result
            except Exception as e:
                logger.error(f"Startup callback error: {e}")
        
        return self._current_session
    
    async def end_session(self) -> Optional[str]:
        """
        End the current session.
        
        Returns:
            Path to archived session file, or None
        """
        if not self._current_session:
            return None
        
        session_id = self._current_session.session_id
        
        # Run shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                if callable(callback):
                    result = callback()
                    if hasattr(result, '__await__'):
                        await result
            except Exception as e:
                logger.error(f"Shutdown callback error: {e}")
        
        # Archive the session
        archive_path = self._archive_session()
        
        # Clear state
        self._current_session = None
        self._api_config = None
        
        logger.info(f"Ended session: {session_id[:8]}...")
        
        return str(archive_path) if archive_path else None
    
    async def switch_mode(
        self,
        new_mode: TradingMode,
        api_config: Optional[ApiConfig] = None
    ) -> SessionState:
        """
        Switch to a new mode, creating a fresh session.
        
        This is the primary method for mode switching, ensuring:
        1. Current session is properly shut down
        2. State is archived
        3. New session is created with clean state
        
        Args:
            new_mode: New trading mode
            api_config: API configuration for the new mode
            
        Returns:
            New SessionState
        """
        old_mode = self.mode
        
        logger.info(f"Switching mode: {old_mode} -> {new_mode}")
        
        # End current session (archives and clears)
        if self._current_session:
            await self.end_session()
        
        # Start new session
        return await self.start_session(new_mode, api_config)
    
    def reset_session_state(self):
        """
        Reset current session state without ending the session.
        Use for clearing data while keeping the same session_id.
        """
        if not self._current_session:
            return
        
        session_id = self._current_session.session_id
        mode = self._current_session.mode
        
        # Reset to empty state while keeping identity
        self._current_session = SessionState.empty(session_id, mode)
        
        logger.info(f"Reset session state: {session_id[:8]}...")
    
    def update_balances(self, balances: Dict[str, float]):
        """Update balances in current session."""
        if self._current_session:
            self._current_session.balances = balances
            self._current_session.last_exchange_sync = datetime.now().isoformat()
    
    def update_positions(self, positions: List[Dict]):
        """Update positions in current session."""
        if self._current_session:
            self._current_session.open_positions = positions
    
    def update_orders(self, orders: List[Dict]):
        """Update orders in current session."""
        if self._current_session:
            self._current_session.open_orders = orders
    
    def add_trade(self, trade: Dict):
        """Add a trade to current session."""
        if self._current_session:
            self._current_session.trades_this_session.append(trade)
    
    def add_decision(self, decision: Dict):
        """Add a decision to current session."""
        if self._current_session:
            self._current_session.decisions_this_session.append(decision)
    
    def validate_session_id(self, session_id: str) -> bool:
        """
        Validate that a session_id matches the current session.
        Used to reject stale requests.
        """
        if not self._current_session:
            return False
        return self._current_session.session_id == session_id
    
    def _archive_session(self) -> Optional[Path]:
        """Archive current session to disk."""
        if not self._current_session:
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = self._current_session.session_id[:8]
            mode = self._current_session.mode.value
            
            filename = f"session_{timestamp}_{mode}_{session_id}.json"
            archive_path = self._archive_dir / filename
            
            with open(archive_path, 'w') as f:
                json.dump(self._current_session.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Archived session to: {archive_path}")
            return archive_path
            
        except Exception as e:
            logger.error(f"Failed to archive session: {e}")
            return None
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get sanitized session info for API responses."""
        if not self._current_session:
            return {
                "active": False,
                "session_id": None,
                "mode": None
            }
        
        return {
            "active": True,
            "session_id": self._current_session.session_id,
            "mode": self._current_session.mode.value,
            "created_at": self._current_session.created_at,
            "is_running": self._current_session.is_running,
            "api_configured": self._api_config.is_configured() if self._api_config else False,
            "api_validated": self._api_config.is_validated if self._api_config else False
        }


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def reset_session_manager():
    """Reset the global session manager (for testing)."""
    global _session_manager
    _session_manager = None
