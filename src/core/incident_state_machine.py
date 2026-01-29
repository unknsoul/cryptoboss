"""
Incident State Machine - v10.2-OPERATOR-GRADE

Explicit failure handling with 4 states:
- NORMAL: Full trading capability
- DEGRADED: Reduced trading (smaller size, limited strategies)
- INCIDENT_FREEZE: No new trades, can manage existing positions
- HALTED: Complete stop, requires manual recovery

Non-Negotiable Rules:
- INCIDENT_FREEZE blocks all new trades
- Open positions may be managed but not increased
- Manual operator action required to exit INCIDENT_FREEZE
- No automated recovery from INCIDENT_FREEZE or HALTED
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class IncidentState(Enum):
    """Incident states - ordered by severity."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    INCIDENT_FREEZE = "incident_freeze"
    HALTED = "halted"


# State severity order (for transition validation)
STATE_SEVERITY = {
    IncidentState.NORMAL: 0,
    IncidentState.DEGRADED: 1,
    IncidentState.INCIDENT_FREEZE: 2,
    IncidentState.HALTED: 3,
}


@dataclass
class IncidentEvent:
    """Record of a state transition."""
    timestamp: datetime
    from_state: IncidentState
    to_state: IncidentState
    reason: str
    triggered_by: str  # "system" or operator_id
    auto_recoverable: bool
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'from_state': self.from_state.value,
            'to_state': self.to_state.value,
            'reason': self.reason,
            'triggered_by': self.triggered_by,
            'auto_recoverable': self.auto_recoverable,
            'context': self.context
        }


@dataclass
class IncidentStateSnapshot:
    """Current incident state snapshot."""
    state: IncidentState
    since: datetime
    reason: str
    triggered_by: str
    auto_recoverable: bool
    incident_count_today: int
    time_in_state_seconds: float
    
    def to_dict(self) -> Dict:
        return {
            'state': self.state.value,
            'since': self.since.isoformat(),
            'reason': self.reason,
            'triggered_by': self.triggered_by,
            'auto_recoverable': self.auto_recoverable,
            'incident_count_today': self.incident_count_today,
            'time_in_state_seconds': self.time_in_state_seconds
        }


class IncidentStateMachine:
    """
    Incident State Machine - Explicit failure handling.
    
    Manages system state through incidents with clear rules:
    - Escalation can happen automatically (NORMAL → DEGRADED → INCIDENT_FREEZE)
    - De-escalation from INCIDENT_FREEZE requires operator action
    - HALTED state always requires manual recovery
    
    Usage:
        ism = IncidentStateMachine()
        
        # Check current state
        state = ism.get_state()
        
        # System triggers incident
        ism.trigger_incident_freeze("Exchange API timeout threshold exceeded")
        
        # Operator resolves incident
        ism.resolve_incident("admin", "Exchange restored, verified connectivity")
    """
    
    def __init__(
        self,
        log_dir: str = "logs/incidents",
        operator_control = None  # Lazy import to avoid circular dependency
    ):
        self._current_state = IncidentState.NORMAL
        self._state_since = datetime.utcnow()
        self._state_reason = "System initialized"
        self._state_triggered_by = "system"
        self._auto_recoverable = True
        
        self._timeline: List[IncidentEvent] = []
        self._lock = threading.RLock()
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._operator_control = operator_control
        
        # Callbacks for state changes
        self._on_state_change_callbacks: List[callable] = []
        
        logger.info("IncidentStateMachine initialized in NORMAL state")
    
    def get_state(self) -> IncidentState:
        """Get current incident state."""
        return self._current_state
    
    def get_snapshot(self) -> IncidentStateSnapshot:
        """Get full state snapshot."""
        with self._lock:
            time_in_state = (datetime.utcnow() - self._state_since).total_seconds()
            incident_count = self._count_incidents_today()
            
            return IncidentStateSnapshot(
                state=self._current_state,
                since=self._state_since,
                reason=self._state_reason,
                triggered_by=self._state_triggered_by,
                auto_recoverable=self._auto_recoverable,
                incident_count_today=incident_count,
                time_in_state_seconds=time_in_state
            )
    
    def transition_to(
        self, 
        new_state: IncidentState, 
        reason: str,
        triggered_by: str = "system",
        auto_recoverable: bool = False
    ) -> Tuple[bool, str]:
        """
        Transition to a new state.
        
        Args:
            new_state: Target state
            reason: Why this transition is happening
            triggered_by: "system" or operator_id
            auto_recoverable: Can this state auto-recover (only for DEGRADED)
            
        Returns:
            (success, message)
        """
        with self._lock:
            old_state = self._current_state
            
            # Validate transition
            valid, msg = self._validate_transition(old_state, new_state, triggered_by)
            if not valid:
                logger.warning(f"Invalid state transition blocked: {old_state} -> {new_state}: {msg}")
                return False, msg
            
            # Create event
            event = IncidentEvent(
                timestamp=datetime.utcnow(),
                from_state=old_state,
                to_state=new_state,
                reason=reason,
                triggered_by=triggered_by,
                auto_recoverable=auto_recoverable,
                context={
                    'previous_state_duration_seconds': (datetime.utcnow() - self._state_since).total_seconds()
                }
            )
            
            # Update state
            self._current_state = new_state
            self._state_since = datetime.utcnow()
            self._state_reason = reason
            self._state_triggered_by = triggered_by
            self._auto_recoverable = auto_recoverable
            
            # Record event
            self._timeline.append(event)
            self._persist_event(event)
            
            # Notify operator control if entering HALTED
            if new_state == IncidentState.HALTED:
                self._notify_operator_control_halt(reason)
            
            # Fire callbacks
            self._fire_state_change_callbacks(old_state, new_state, reason)
            
            logger.warning(
                f"INCIDENT STATE CHANGE: {old_state.value} -> {new_state.value} ({reason})",
                extra={'from_state': old_state.value, 'to_state': new_state.value, 'triggered_by': triggered_by}
            )
            
            return True, f"Transitioned to {new_state.value}"
    
    def trigger_degraded(self, reason: str) -> Tuple[bool, str]:
        """
        Trigger DEGRADED state (reduced trading).
        
        This can auto-recover to NORMAL when conditions improve.
        """
        return self.transition_to(
            IncidentState.DEGRADED, 
            reason, 
            triggered_by="system",
            auto_recoverable=True
        )
    
    def trigger_incident_freeze(self, reason: str) -> Tuple[bool, str]:
        """
        Trigger INCIDENT_FREEZE state.
        
        Blocks all new trades. Existing positions can be managed.
        Requires operator intervention to resolve.
        """
        return self.transition_to(
            IncidentState.INCIDENT_FREEZE,
            reason,
            triggered_by="system",
            auto_recoverable=False  # Requires manual resolution
        )
    
    def trigger_halt(self, reason: str) -> Tuple[bool, str]:
        """
        Trigger HALTED state.
        
        Complete trading stop. Requires manual recovery.
        """
        return self.transition_to(
            IncidentState.HALTED,
            reason,
            triggered_by="system",
            auto_recoverable=False
        )
    
    def resolve_incident(self, operator_id: str, reason: str) -> Tuple[bool, str]:
        """
        Resolve an incident and return to NORMAL state.
        
        Only operators can resolve INCIDENT_FREEZE or HALTED states.
        
        Args:
            operator_id: ID of the operator resolving the incident
            reason: Explanation of resolution
            
        Returns:
            (success, message)
        """
        with self._lock:
            if self._current_state == IncidentState.NORMAL:
                return False, "No incident to resolve"
            
            if self._current_state == IncidentState.DEGRADED and self._auto_recoverable:
                return False, "DEGRADED state is auto-recoverable, no manual resolution needed"
            
            return self.transition_to(
                IncidentState.NORMAL,
                f"Resolved by {operator_id}: {reason}",
                triggered_by=operator_id,
                auto_recoverable=True
            )
    
    def auto_recover(self) -> Tuple[bool, str]:
        """
        Attempt automatic recovery from DEGRADED state.
        
        Only works if current state is DEGRADED and marked auto-recoverable.
        Called by health monitors when conditions improve.
        
        Returns:
            (success, message)
        """
        with self._lock:
            if self._current_state != IncidentState.DEGRADED:
                return False, f"Cannot auto-recover from {self._current_state.value}"
            
            if not self._auto_recoverable:
                return False, "Current DEGRADED state requires manual resolution"
            
            return self.transition_to(
                IncidentState.NORMAL,
                "Automatic recovery - conditions improved",
                triggered_by="system",
                auto_recoverable=True
            )
    
    def can_open_new_positions(self) -> bool:
        """Check if new positions can be opened."""
        return self._current_state in [IncidentState.NORMAL, IncidentState.DEGRADED]
    
    def can_manage_existing_positions(self) -> bool:
        """Check if existing positions can be managed (close, reduce)."""
        return self._current_state != IncidentState.HALTED
    
    def get_size_multiplier(self) -> float:
        """
        Get position sizing multiplier for current state.
        
        Returns:
            1.0 for NORMAL, 0.5 for DEGRADED, 0.0 for others
        """
        if self._current_state == IncidentState.NORMAL:
            return 1.0
        elif self._current_state == IncidentState.DEGRADED:
            return 0.5
        else:
            return 0.0
    
    def get_timeline(self, hours: int = 24) -> List[Dict]:
        """Get incident timeline for specified period."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with self._lock:
            return [
                event.to_dict()
                for event in self._timeline
                if event.timestamp >= cutoff
            ]
    
    def register_state_change_callback(self, callback: callable) -> None:
        """Register a callback for state changes."""
        self._on_state_change_callbacks.append(callback)
    
    def _validate_transition(
        self, 
        from_state: IncidentState, 
        to_state: IncidentState,
        triggered_by: str
    ) -> Tuple[bool, str]:
        """Validate if a transition is allowed."""
        # Same state is a no-op
        if from_state == to_state:
            return False, "Already in this state"
        
        # De-escalation from INCIDENT_FREEZE or HALTED requires operator
        if from_state in [IncidentState.INCIDENT_FREEZE, IncidentState.HALTED]:
            if to_state == IncidentState.NORMAL and triggered_by == "system":
                return False, "De-escalation from INCIDENT_FREEZE/HALTED requires operator action"
        
        # Cannot skip directly from HALTED to NORMAL via system
        if from_state == IncidentState.HALTED and to_state != IncidentState.NORMAL:
            if triggered_by == "system":
                return False, "HALTED state requires complete resolution to NORMAL"
        
        return True, "Transition allowed"
    
    def _count_incidents_today(self) -> int:
        """Count number of incident state entries today."""
        today = datetime.utcnow().date()
        count = 0
        for event in self._timeline:
            if event.timestamp.date() == today:
                if event.to_state in [IncidentState.INCIDENT_FREEZE, IncidentState.HALTED]:
                    count += 1
        return count
    
    def _notify_operator_control_halt(self, reason: str) -> None:
        """Notify operator control that manual recovery is required."""
        if self._operator_control:
            try:
                self._operator_control.set_requires_manual_recovery(reason)
            except Exception as e:
                logger.error(f"Failed to notify operator control: {e}")
    
    def _fire_state_change_callbacks(
        self, 
        old_state: IncidentState, 
        new_state: IncidentState,
        reason: str
    ) -> None:
        """Fire all registered state change callbacks."""
        for callback in self._on_state_change_callbacks:
            try:
                callback(old_state, new_state, reason)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    def _persist_event(self, event: IncidentEvent) -> None:
        """Persist event to disk."""
        try:
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = self._log_dir / f"incidents_{date_str}.jsonl"
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist incident event: {e}")


# Singleton instance
_incident_state_machine: Optional[IncidentStateMachine] = None


def get_incident_state_machine() -> IncidentStateMachine:
    """Get global IncidentStateMachine instance."""
    global _incident_state_machine
    if _incident_state_machine is None:
        _incident_state_machine = IncidentStateMachine()
    return _incident_state_machine


def reset_incident_state_machine() -> None:
    """Reset the singleton (for testing)."""
    global _incident_state_machine
    _incident_state_machine = None
