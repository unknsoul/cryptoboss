"""
Context State Machine - Live Readiness Component

Finite state machine for market context with:
- Valid state transitions
- Minimum state duration (cooldowns)
- Transition event emission
- State persistence support

This replaces stateless context classification with
deterministic, bounded state transitions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, List, Callable
import json

logger = logging.getLogger(__name__)


class ContextState(Enum):
    """Valid context states for the state machine."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    NO_TRADE = "no_trade"


# Valid state transitions (from -> [to...])
VALID_TRANSITIONS: Dict[ContextState, List[ContextState]] = {
    ContextState.TRENDING_UP: [
        ContextState.RANGING,
        ContextState.HIGH_VOLATILITY,
        ContextState.TRENDING_DOWN,  # Reversal
    ],
    ContextState.TRENDING_DOWN: [
        ContextState.RANGING,
        ContextState.HIGH_VOLATILITY,
        ContextState.TRENDING_UP,  # Reversal
    ],
    ContextState.RANGING: [
        ContextState.TRENDING_UP,
        ContextState.TRENDING_DOWN,
        ContextState.HIGH_VOLATILITY,
    ],
    ContextState.HIGH_VOLATILITY: [
        ContextState.NO_TRADE,
        ContextState.RANGING,  # Recovery
    ],
    ContextState.NO_TRADE: [
        ContextState.RANGING,  # Only recovery path
    ],
}


@dataclass
class ContextStateSnapshot:
    """Immutable snapshot of current state machine status."""
    current_state: ContextState
    entered_at: datetime
    transition_count_24h: int
    last_transition: Optional[datetime]
    is_locked: bool
    lock_expires: Optional[datetime]
    time_in_state: timedelta
    
    def to_dict(self) -> Dict:
        return {
            'current_state': self.current_state.value,
            'entered_at': self.entered_at.isoformat(),
            'transition_count_24h': self.transition_count_24h,
            'last_transition': self.last_transition.isoformat() if self.last_transition else None,
            'is_locked': self.is_locked,
            'lock_expires': self.lock_expires.isoformat() if self.lock_expires else None,
            'time_in_state_seconds': self.time_in_state.total_seconds()
        }


class ContextTransitionEvent:
    """Event emitted on state transitions."""
    def __init__(
        self,
        from_state: ContextState,
        to_state: ContextState,
        timestamp: datetime,
        reason: str,
        was_blocked: bool = False
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.timestamp = timestamp
        self.reason = reason
        self.was_blocked = was_blocked
    
    def __repr__(self):
        status = "BLOCKED" if self.was_blocked else "SUCCESS"
        return f"ContextTransition({self.from_state.value} -> {self.to_state.value} [{status}]: {self.reason})"


class ContextStateMachine:
    """
    Finite state machine for market context.
    
    Provides deterministic, bounded state transitions with:
    - Valid transition enforcement
    - Minimum state duration (default: 2 hours)
    - Transition cooldown (default: 30 minutes)
    - Maximum transitions per 24h (default: 6)
    - Event emission on transitions
    
    Usage:
        machine = ContextStateMachine()
        
        # Attempt transition
        success, event = machine.transition_to(
            ContextState.TRENDING_UP,
            reason="Strong uptrend detected"
        )
        
        if not success:
            logger.info(f"Transition blocked: {event.reason}")
    """
    
    def __init__(
        self,
        initial_state: ContextState = ContextState.RANGING,
        min_state_duration_hours: float = 2.0,
        transition_cooldown_minutes: float = 30.0,
        max_transitions_per_24h: int = 6,
        event_callback: Optional[Callable[[ContextTransitionEvent], None]] = None
    ):
        self._current_state = initial_state
        self._entered_at = datetime.now()
        self._last_transition: Optional[datetime] = None
        self._transition_history: List[ContextTransitionEvent] = []
        
        # Configuration
        self.min_state_duration = timedelta(hours=min_state_duration_hours)
        self.transition_cooldown = timedelta(minutes=transition_cooldown_minutes)
        self.max_transitions_per_24h = max_transitions_per_24h
        self.event_callback = event_callback
        
        logger.info(
            f"ContextStateMachine initialized: "
            f"state={initial_state.value}, "
            f"min_duration={min_state_duration_hours}h, "
            f"cooldown={transition_cooldown_minutes}m, "
            f"max_transitions={max_transitions_per_24h}/24h"
        )
    
    @property
    def current_state(self) -> ContextState:
        """Get current state."""
        return self._current_state
    
    @property
    def time_in_state(self) -> timedelta:
        """Get time spent in current state."""
        return datetime.now() - self._entered_at
    
    @property
    def is_locked(self) -> bool:
        """Check if state is locked (in cooldown)."""
        if self._last_transition is None:
            return False
        
        elapsed = datetime.now() - self._last_transition
        return elapsed < self.transition_cooldown
    
    @property
    def lock_expires(self) -> Optional[datetime]:
        """Get when current lock expires."""
        if not self.is_locked:
            return None
        return self._last_transition + self.transition_cooldown
    
    def get_transitions_24h(self) -> int:
        """Count transitions in last 24 hours."""
        cutoff = datetime.now() - timedelta(hours=24)
        return sum(
            1 for event in self._transition_history
            if event.timestamp > cutoff and not event.was_blocked
        )
    
    def get_snapshot(self) -> ContextStateSnapshot:
        """Get immutable snapshot of current state."""
        return ContextStateSnapshot(
            current_state=self._current_state,
            entered_at=self._entered_at,
            transition_count_24h=self.get_transitions_24h(),
            last_transition=self._last_transition,
            is_locked=self.is_locked,
            lock_expires=self.lock_expires,
            time_in_state=self.time_in_state
        )
    
    def can_transition_to(self, target_state: ContextState) -> tuple[bool, str]:
        """
        Check if transition to target state is allowed.
        
        Returns: (allowed, reason)
        """
        now = datetime.now()
        
        # Same state - no transition needed
        if target_state == self._current_state:
            return True, "Already in target state"
        
        # Check valid transitions
        valid_targets = VALID_TRANSITIONS.get(self._current_state, [])
        if target_state not in valid_targets:
            return False, f"Invalid transition: {self._current_state.value} -> {target_state.value}"

        # Check max transitions before time-based gates so hard limits win
        if self.get_transitions_24h() >= self.max_transitions_per_24h:
            return False, f"Max transitions ({self.max_transitions_per_24h}/24h) reached"
        
        # Check minimum state duration
        if self.time_in_state < self.min_state_duration:
            remaining = self.min_state_duration - self.time_in_state
            return False, f"Min duration not met: {remaining.total_seconds():.0f}s remaining"
        
        # Check cooldown
        if self.is_locked:
            remaining = self.lock_expires - now
            return False, f"Transition cooldown: {remaining.total_seconds():.0f}s remaining"
        
        return True, "Transition allowed"
    
    def transition_to(
        self,
        target_state: ContextState,
        reason: str = "",
        force: bool = False
    ) -> tuple[bool, ContextTransitionEvent]:
        """
        Attempt to transition to target state.
        
        Args:
            target_state: State to transition to
            reason: Human-readable reason for transition
            force: If True, bypass all checks (emergency use only)
            
        Returns:
            (success, event) - event contains details
        """
        now = datetime.now()
        
        # Same state check
        if target_state == self._current_state:
            event = ContextTransitionEvent(
                from_state=self._current_state,
                to_state=target_state,
                timestamp=now,
                reason="Already in target state",
                was_blocked=False
            )
            return True, event
        
        # Check if transition is allowed
        if not force:
            allowed, block_reason = self.can_transition_to(target_state)
            
            if not allowed:
                event = ContextTransitionEvent(
                    from_state=self._current_state,
                    to_state=target_state,
                    timestamp=now,
                    reason=block_reason,
                    was_blocked=True
                )
                self._transition_history.append(event)
                self._emit_event(event)
                
                logger.warning(
                    f"Transition BLOCKED: {self._current_state.value} -> {target_state.value} "
                    f"({block_reason})"
                )
                
                return False, event
        
        # Execute transition
        old_state = self._current_state
        self._current_state = target_state
        self._entered_at = now
        self._last_transition = now
        
        event = ContextTransitionEvent(
            from_state=old_state,
            to_state=target_state,
            timestamp=now,
            reason=reason or f"Transition to {target_state.value}",
            was_blocked=False
        )
        self._transition_history.append(event)
        self._emit_event(event)
        
        logger.info(
            f"Transition SUCCESS: {old_state.value} -> {target_state.value} ({reason})"
        )
        
        return True, event
    
    def force_no_trade(self, reason: str) -> ContextTransitionEvent:
        """
        Force immediate transition to NO_TRADE state.
        
        Use for emergency situations only.
        """
        logger.warning(f"FORCE NO_TRADE: {reason}")
        _, event = self.transition_to(
            ContextState.NO_TRADE,
            reason=f"FORCED: {reason}",
            force=True
        )
        return event
    
    def get_recent_events(self, hours: int = 24) -> List[ContextTransitionEvent]:
        """Get transition events from last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [e for e in self._transition_history if e.timestamp > cutoff]
    
    def _emit_event(self, event: ContextTransitionEvent):
        """Emit event to callback if registered."""
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def to_dict(self) -> Dict:
        """Serialize state machine for persistence."""
        return {
            'current_state': self._current_state.value,
            'entered_at': self._entered_at.isoformat(),
            'last_transition': self._last_transition.isoformat() if self._last_transition else None,
            'transition_count_24h': self.get_transitions_24h(),
            'config': {
                'min_state_duration_hours': self.min_state_duration.total_seconds() / 3600,
                'transition_cooldown_minutes': self.transition_cooldown.total_seconds() / 60,
                'max_transitions_per_24h': self.max_transitions_per_24h
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict, event_callback: Optional[Callable] = None) -> 'ContextStateMachine':
        """Restore state machine from persisted data."""
        config = data.get('config', {})
        
        machine = cls(
            initial_state=ContextState(data['current_state']),
            min_state_duration_hours=config.get('min_state_duration_hours', 2.0),
            transition_cooldown_minutes=config.get('transition_cooldown_minutes', 30.0),
            max_transitions_per_24h=config.get('max_transitions_per_24h', 6),
            event_callback=event_callback
        )
        
        machine._entered_at = datetime.fromisoformat(data['entered_at'])
        if data.get('last_transition'):
            machine._last_transition = datetime.fromisoformat(data['last_transition'])
        
        return machine


# Singleton instance
_context_state_machine: Optional[ContextStateMachine] = None


def get_context_state_machine() -> ContextStateMachine:
    """Get global ContextStateMachine instance."""
    global _context_state_machine
    if _context_state_machine is None:
        _context_state_machine = ContextStateMachine()
    return _context_state_machine


def reset_context_state_machine():
    """Reset global instance (for testing)."""
    global _context_state_machine
    _context_state_machine = None
