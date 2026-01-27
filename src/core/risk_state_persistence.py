"""
Risk State Persistence - Live Readiness Component

Persists critical risk state across restarts:
- Daily/weekly drawdown
- Consecutive losses
- Kill switch status
- Last market context and bias
- Trade budgets

Ensures trading is blocked until state sync completes.
"""

import json
import logging
import os
import tempfile
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PersistedRiskState:
    """Complete risk state for persistence."""
    # Drawdown tracking
    daily_drawdown: float = 0.0
    daily_drawdown_date: str = ""  # YYYY-MM-DD
    weekly_drawdown: float = 0.0
    weekly_start_date: str = ""  # YYYY-MM-DD (Monday)
    
    # Loss tracking
    consecutive_losses: int = 0
    losses_today: int = 0
    losses_this_week: int = 0
    
    # Kill switch
    kill_switch_active: bool = False
    kill_switch_reason: Optional[str] = None
    kill_switch_activated_at: Optional[str] = None
    
    # Context state
    last_market_context: str = "ranging"
    last_bias: str = "neutral"
    
    # Trade counts
    trades_today: int = 0
    trades_this_week: int = 0
    
    # Budgets (remaining)
    budget_trades_day: int = 10
    budget_trades_context: int = 3
    budget_losses_bias: int = 2
    
    # Metadata
    last_save_time: str = ""
    version: int = 2
    checksum: str = ""
    
    def compute_checksum(self) -> str:
        """Compute checksum for integrity verification."""
        data = asdict(self)
        data.pop('checksum', None)
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def validate_checksum(self) -> bool:
        """Validate stored checksum matches computed."""
        expected = self.compute_checksum()
        return self.checksum == expected


class RiskStatePersistence:
    """
    Manages persistence of risk state across restarts.
    
    Features:
    - Atomic writes (write to temp, then rename)
    - Checksum validation
    - Automatic date-based resets
    - State versioning
    
    Usage:
        persistence = RiskStatePersistence()
        
        # Load on startup (blocks until complete)
        state = persistence.load_state()
        
        # Save periodically
        persistence.save_state(state)
    """
    
    def __init__(
        self,
        state_file: str = "data/risk_state.json",
        backup_dir: str = "data/risk_backups"
    ):
        self.state_file = Path(state_file)
        self.backup_dir = Path(backup_dir)
        
        # Ensure directories exist
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self._loaded = False
        self._current_state: Optional[PersistedRiskState] = None
        
        logger.info(f"RiskStatePersistence initialized: {self.state_file}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if state has been loaded."""
        return self._loaded
    
    @property
    def state(self) -> Optional[PersistedRiskState]:
        """Get current state (None if not loaded)."""
        return self._current_state
    
    def load_state(self) -> PersistedRiskState:
        """
        Load risk state from disk.
        
        Returns default state if file doesn't exist.
        Raises exception if file is corrupted.
        """
        if not self.state_file.exists():
            logger.info("No existing risk state, creating default")
            state = self._create_default_state()
            self._current_state = state
            self._loaded = True
            self.save_state(state)  # Save initial state
            return state
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            # Validate version
            version = data.get('version', 1)
            if version < 2:
                logger.warning(f"Migrating state from v{version} to v2")
                data = self._migrate_state(data, version)
            
            state = PersistedRiskState(**data)
            
            # Validate checksum
            if not state.validate_checksum():
                raise ValueError("Risk state checksum mismatch - file may be corrupted")
            
            # Apply date-based resets
            state = self._apply_date_resets(state)
            
            self._current_state = state
            self._loaded = True
            
            logger.info(
                f"Risk state loaded: drawdown={state.daily_drawdown:.2f}, "
                f"losses={state.consecutive_losses}, "
                f"kill_switch={state.kill_switch_active}"
            )
            
            return state
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse risk state: {e}")
            self._backup_corrupted_file()
            raise RuntimeError(f"Risk state file corrupted: {e}")
        except Exception as e:
            logger.error(f"Failed to load risk state: {e}")
            raise
    
    def save_state(self, state: PersistedRiskState) -> bool:
        """
        Atomically save risk state to disk.
        
        Uses write-to-temp-then-rename pattern for safety.
        """
        try:
            # Update metadata
            state.last_save_time = datetime.now().isoformat()
            state.checksum = state.compute_checksum()
            
            # Write to temp file
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.json',
                dir=str(self.state_file.parent)
            )
            
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(asdict(state), f, indent=2)
                
                # Atomic rename
                os.replace(temp_path, self.state_file)
                
            except Exception:
                # Clean up temp file on failure
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
            
            self._current_state = state
            logger.debug(f"Risk state saved: {state.last_save_time}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")
            return False
    
    def update_drawdown(self, pnl: float) -> PersistedRiskState:
        """Update drawdown with new P&L."""
        if not self._loaded:
            raise RuntimeError("State not loaded")
        
        state = self._current_state
        
        # Update daily
        state.daily_drawdown += pnl
        
        # Update weekly
        state.weekly_drawdown += pnl
        
        if pnl < 0:
            state.consecutive_losses += 1
            state.losses_today += 1
            state.losses_this_week += 1
        else:
            state.consecutive_losses = 0  # Reset on win
        
        self.save_state(state)
        return state
    
    def record_trade(self) -> PersistedRiskState:
        """Record a trade execution."""
        if not self._loaded:
            raise RuntimeError("State not loaded")
        
        state = self._current_state
        state.trades_today += 1
        state.trades_this_week += 1
        state.budget_trades_day = max(0, state.budget_trades_day - 1)
        state.budget_trades_context = max(0, state.budget_trades_context - 1)
        
        self.save_state(state)
        return state
    
    def activate_kill_switch(self, reason: str) -> PersistedRiskState:
        """Activate kill switch."""
        if not self._loaded:
            raise RuntimeError("State not loaded")
        
        state = self._current_state
        state.kill_switch_active = True
        state.kill_switch_reason = reason
        state.kill_switch_activated_at = datetime.now().isoformat()
        
        logger.warning(f"KILL SWITCH ACTIVATED: {reason}")
        self.save_state(state)
        return state
    
    def deactivate_kill_switch(self) -> PersistedRiskState:
        """Deactivate kill switch."""
        if not self._loaded:
            raise RuntimeError("State not loaded")
        
        state = self._current_state
        state.kill_switch_active = False
        state.kill_switch_reason = None
        state.kill_switch_activated_at = None
        
        logger.info("Kill switch deactivated")
        self.save_state(state)
        return state
    
    def update_context(self, context: str, bias: str) -> PersistedRiskState:
        """Update last known context and bias."""
        if not self._loaded:
            raise RuntimeError("State not loaded")
        
        state = self._current_state
        state.last_market_context = context
        state.last_bias = bias
        
        self.save_state(state)
        return state
    
    def reset_context_budget(self) -> PersistedRiskState:
        """Reset context-specific budget (on context change)."""
        if not self._loaded:
            raise RuntimeError("State not loaded")
        
        state = self._current_state
        state.budget_trades_context = 3  # Reset to default
        
        self.save_state(state)
        return state
    
    def reset_bias_budget(self) -> PersistedRiskState:
        """Reset bias-specific budget (on bias change)."""
        if not self._loaded:
            raise RuntimeError("State not loaded")
        
        state = self._current_state
        state.budget_losses_bias = 2  # Reset to default
        
        self.save_state(state)
        return state
    
    def _create_default_state(self) -> PersistedRiskState:
        """Create default initial state."""
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        
        return PersistedRiskState(
            daily_drawdown_date=today.isoformat(),
            weekly_start_date=monday.isoformat(),
            last_save_time=datetime.now().isoformat()
        )
    
    def _apply_date_resets(self, state: PersistedRiskState) -> PersistedRiskState:
        """Apply automatic resets based on date changes."""
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        
        # Daily reset
        if state.daily_drawdown_date != today.isoformat():
            logger.info("New day detected, resetting daily metrics")
            state.daily_drawdown = 0.0
            state.daily_drawdown_date = today.isoformat()
            state.trades_today = 0
            state.losses_today = 0
            state.budget_trades_day = 10  # Reset daily budget
        
        # Weekly reset
        if state.weekly_start_date != monday.isoformat():
            logger.info("New week detected, resetting weekly metrics")
            state.weekly_drawdown = 0.0
            state.weekly_start_date = monday.isoformat()
            state.trades_this_week = 0
            state.losses_this_week = 0
        
        return state
    
    def _migrate_state(self, data: Dict, from_version: int) -> Dict:
        """Migrate state from older versions."""
        if from_version < 2:
            # Add new fields from v2
            data.setdefault('losses_today', 0)
            data.setdefault('losses_this_week', 0)
            data.setdefault('trades_this_week', 0)
            data.setdefault('budget_trades_day', 10)
            data.setdefault('budget_trades_context', 3)
            data.setdefault('budget_losses_bias', 2)
            data['version'] = 2
        
        return data
    
    def _backup_corrupted_file(self):
        """Backup corrupted state file for investigation."""
        if self.state_file.exists():
            backup_name = f"corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = self.backup_dir / backup_name
            self.state_file.rename(backup_path)
            logger.warning(f"Corrupted state backed up to: {backup_path}")


# Import needed for date math
from datetime import timedelta


# Singleton instance
_risk_persistence: Optional[RiskStatePersistence] = None


def get_risk_persistence() -> RiskStatePersistence:
    """Get global RiskStatePersistence instance."""
    global _risk_persistence
    if _risk_persistence is None:
        _risk_persistence = RiskStatePersistence()
    return _risk_persistence
