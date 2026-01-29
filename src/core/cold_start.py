"""
Cold Start Visibility - v10.4-TRUST-GRADE

Expose startup and restart safety logic to the operator.
Prevents "is it working?" confusion during initial cache warming and sync.

Features:
- Explicit startup states (WARMING_UP, SYNCING, READY)
- Progress tracking
- Trading block until READY status
"""

import logging
import time
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class StartupState(Enum):
    """System startup phases."""
    INITIALIZING = "initializing"       # Booting core services
    CONNECTING = "connecting"           # Establishing exchange/DB connections
    SYNCING_DATA = "syncing_data"       # Fetching historical data/open orders
    WARMING_CACHE = "warming_cache"     # calculating initial indicators
    READY_TO_TRADE = "ready_to_trade"   # Fully operational
    FAILED = "failed"                   # Startup failed


@dataclass
class StartupStep:
    name: str
    state: StartupState
    progress: float = 0.0  # 0.0 to 1.0
    details: str = ""
    error: Optional[str] = None


class ColdStartManager:
    """
    Manages the system startup sequence and reports progress.
    """
    
    def __init__(self):
        self._current_state = StartupState.INITIALIZING
        self._steps: Dict[str, StartupStep] = {}
        self._start_time = time.time()
        self._estimated_completion_time = 0.0
        
    def set_state(self, state: StartupState):
        self._current_state = state
        logger.info(f"STARTUP STATE: {state.value.upper()}")
        
    def add_step(self, name: str, state_phase: StartupState):
        self._steps[name] = StartupStep(name, state_phase)
        
    def update_step(self, name: str, progress: float, details: str = ""):
        if name in self._steps:
            self._steps[name].progress = progress
            self._steps[name].details = details
            
            # Auto-update global state if all steps in a phase are done? 
            # (Keeping it simple for now)
            
    def complete_step(self, name: str):
        if name in self._steps:
            self._steps[name].progress = 1.0
            self._steps[name].details = "Complete"
            logger.info(f"Startup Step Complete: {name}")

    def fail_startup(self, reason: str):
        self._current_state = StartupState.FAILED
        logger.critical(f"STARTUP FAILED: {reason}")
        
    def is_ready(self) -> bool:
        return self._current_state == StartupState.READY_TO_TRADE
    
    def get_progress_summary(self) -> Dict:
        """Get full visibility payload for UI."""
        total_steps = len(self._steps)
        if total_steps == 0:
            total_progress = 0.0
        else:
            total_progress = sum(s.progress for s in self._steps.values()) / total_steps
            
        return {
            'state': self._current_state.value,
            'total_progress': total_progress,
            'steps': {name: {
                'state': s.state.value,
                'progress': s.progress,
                'details': s.details,
                'error': s.error
            } for name, s in self._steps.items()},
            'uptime_seconds': time.time() - self._start_time
        }

# Singleton
_cold_start: Optional[ColdStartManager] = None

def get_cold_start_manager() -> ColdStartManager:
    global _cold_start
    if _cold_start is None:
        _cold_start = ColdStartManager()
    return _cold_start
