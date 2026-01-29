"""
Operator Control Layer - v10.2-OPERATOR-GRADE

Explicit human-in-the-loop control for production trading.
Provides manual pause, resume, and recovery capabilities with full audit trail.

Non-Negotiable Rules:
- Operator cannot override risk or capital veto
- Operator cannot force trade execution
- Resume requires system health validation
- All actions are logged with timestamp and reason
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class OperatorAction(Enum):
    """Types of operator actions."""
    MANUAL_PAUSE = "manual_pause"
    MANUAL_RESUME = "manual_resume"
    MANUAL_RECOVER = "manual_recover"
    ACKNOWLEDGE_INCIDENT = "acknowledge_incident"
    ACKNOWLEDGE_METRICS_RESET = "acknowledge_metrics_reset"


@dataclass
class OperatorActionLog:
    """Record of an operator action."""
    timestamp: datetime
    action: OperatorAction
    operator_id: str
    reason: str
    success: bool
    failure_reason: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'action': self.action.value,
            'operator_id': self.operator_id,
            'reason': self.reason,
            'success': self.success,
            'failure_reason': self.failure_reason,
            'context': self.context
        }


@dataclass 
class OperatorState:
    """Current operator control state."""
    is_paused: bool = False
    paused_at: Optional[datetime] = None
    paused_by: Optional[str] = None
    pause_reason: Optional[str] = None
    requires_manual_recovery: bool = False
    recovery_required_reason: Optional[str] = None
    last_action: Optional[OperatorActionLog] = None


class OperatorControlLayer:
    """
    Operator Control Layer - Human-in-the-loop control.
    
    Provides controlled manual intervention without allowing unsafe overrides.
    
    Usage:
        operator = OperatorControlLayer()
        
        # Pause trading
        success, msg = operator.pause("admin", "Investigating anomaly")
        
        # Resume (requires health check)
        success, msg = operator.resume("admin", "Issue resolved")
        
        # Get audit log
        logs = operator.get_action_log(hours=24)
    """
    
    def __init__(
        self,
        log_dir: str = "logs/operator",
        health_checker: callable = None,
        max_log_entries: int = 10000
    ):
        self._state = OperatorState()
        self._action_log: List[OperatorActionLog] = []
        self._lock = threading.RLock()
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._health_checker = health_checker
        self._max_log_entries = max_log_entries
        
        logger.info("OperatorControlLayer initialized")
    
    def pause(self, operator_id: str, reason: str) -> Tuple[bool, str]:
        """
        Manually pause all trading activity.
        
        Args:
            operator_id: Identifier of the operator (e.g., username, API key ID)
            reason: Human-readable reason for pause
            
        Returns:
            (success, message)
        """
        with self._lock:
            if self._state.is_paused:
                return False, "System is already paused"
            
            self._state.is_paused = True
            self._state.paused_at = datetime.utcnow()
            self._state.paused_by = operator_id
            self._state.pause_reason = reason
            
            log_entry = OperatorActionLog(
                timestamp=datetime.utcnow(),
                action=OperatorAction.MANUAL_PAUSE,
                operator_id=operator_id,
                reason=reason,
                success=True,
                context={'previous_state': 'running'}
            )
            self._record_action(log_entry)
            
            logger.warning(
                f"OPERATOR PAUSE: {operator_id} - {reason}",
                extra={'operator_id': operator_id, 'action': 'pause'}
            )
            
            return True, "Trading paused successfully"
    
    def resume(self, operator_id: str, reason: str) -> Tuple[bool, str]:
        """
        Resume trading after manual pause.
        
        Requires system health validation before resuming.
        Cannot resume if system requires manual recovery.
        
        Args:
            operator_id: Identifier of the operator
            reason: Reason for resuming
            
        Returns:
            (success, message)
        """
        with self._lock:
            # Check if paused
            if not self._state.is_paused:
                return False, "System is not paused"
            
            # Check if manual recovery required
            if self._state.requires_manual_recovery:
                return False, f"Manual recovery required: {self._state.recovery_required_reason}"
            
            # Health check before resume
            if self._health_checker:
                try:
                    health_ok, health_msg = self._health_checker()
                    if not health_ok:
                        log_entry = OperatorActionLog(
                            timestamp=datetime.utcnow(),
                            action=OperatorAction.MANUAL_RESUME,
                            operator_id=operator_id,
                            reason=reason,
                            success=False,
                            failure_reason=f"Health check failed: {health_msg}",
                            context={'health_status': 'failed'}
                        )
                        self._record_action(log_entry)
                        return False, f"Cannot resume: Health check failed - {health_msg}"
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    return False, f"Cannot resume: Health check error - {str(e)}"
            
            # Resume
            pause_duration = datetime.utcnow() - self._state.paused_at if self._state.paused_at else timedelta(0)
            
            self._state.is_paused = False
            self._state.paused_at = None
            self._state.paused_by = None
            self._state.pause_reason = None
            
            log_entry = OperatorActionLog(
                timestamp=datetime.utcnow(),
                action=OperatorAction.MANUAL_RESUME,
                operator_id=operator_id,
                reason=reason,
                success=True,
                context={'pause_duration_seconds': pause_duration.total_seconds()}
            )
            self._record_action(log_entry)
            
            logger.info(
                f"OPERATOR RESUME: {operator_id} - {reason} (paused for {pause_duration})",
                extra={'operator_id': operator_id, 'action': 'resume'}
            )
            
            return True, f"Trading resumed successfully (was paused for {pause_duration})"
    
    def recover_from_halt(self, operator_id: str, reason: str) -> Tuple[bool, str]:
        """
        Recover from a HALTED state that requires manual intervention.
        
        This is for severe incidents that blocked all activity.
        Requires explicit acknowledgment that the issue is resolved.
        
        Args:
            operator_id: Identifier of the operator
            reason: Detailed explanation of how issue was resolved
            
        Returns:
            (success, message)
        """
        with self._lock:
            if not self._state.requires_manual_recovery:
                return False, "System does not require manual recovery"
            
            # Health check before recovery
            if self._health_checker:
                try:
                    health_ok, health_msg = self._health_checker()
                    if not health_ok:
                        log_entry = OperatorActionLog(
                            timestamp=datetime.utcnow(),
                            action=OperatorAction.MANUAL_RECOVER,
                            operator_id=operator_id,
                            reason=reason,
                            success=False,
                            failure_reason=f"Health check failed: {health_msg}",
                            context={'health_status': 'failed'}
                        )
                        self._record_action(log_entry)
                        return False, f"Cannot recover: Health check failed - {health_msg}"
                except Exception as e:
                    logger.error(f"Health check error during recovery: {e}")
                    return False, f"Cannot recover: Health check error - {str(e)}"
            
            original_reason = self._state.recovery_required_reason
            
            self._state.requires_manual_recovery = False
            self._state.recovery_required_reason = None
            self._state.is_paused = False
            self._state.paused_at = None
            self._state.paused_by = None
            self._state.pause_reason = None
            
            log_entry = OperatorActionLog(
                timestamp=datetime.utcnow(),
                action=OperatorAction.MANUAL_RECOVER,
                operator_id=operator_id,
                reason=reason,
                success=True,
                context={'original_halt_reason': original_reason}
            )
            self._record_action(log_entry)
            
            logger.warning(
                f"OPERATOR RECOVERY: {operator_id} - {reason} (was halted: {original_reason})",
                extra={'operator_id': operator_id, 'action': 'recover'}
            )
            
            return True, "System recovered successfully"
    
    def set_requires_manual_recovery(self, reason: str) -> None:
        """
        Set the system to require manual recovery.
        Called by incident state machine or other safety systems.
        
        Args:
            reason: Why manual recovery is needed
        """
        with self._lock:
            self._state.requires_manual_recovery = True
            self._state.recovery_required_reason = reason
            self._state.is_paused = True
            self._state.paused_at = datetime.utcnow()
            self._state.paused_by = "SYSTEM"
            self._state.pause_reason = f"Auto-halt: {reason}"
            
            logger.critical(
                f"MANUAL RECOVERY REQUIRED: {reason}",
                extra={'action': 'auto_halt', 'reason': reason}
            )
    
    def acknowledge_incident(self, operator_id: str, incident_id: str, notes: str) -> Tuple[bool, str]:
        """
        Acknowledge an incident (required for metrics reset).
        
        Args:
            operator_id: Identifier of the operator
            incident_id: ID of the incident being acknowledged
            notes: Operator notes about the incident
            
        Returns:
            (success, message)
        """
        with self._lock:
            log_entry = OperatorActionLog(
                timestamp=datetime.utcnow(),
                action=OperatorAction.ACKNOWLEDGE_INCIDENT,
                operator_id=operator_id,
                reason=notes,
                success=True,
                context={'incident_id': incident_id}
            )
            self._record_action(log_entry)
            
            logger.info(
                f"INCIDENT ACKNOWLEDGED: {operator_id} - incident {incident_id}: {notes}",
                extra={'operator_id': operator_id, 'incident_id': incident_id}
            )
            
            return True, f"Incident {incident_id} acknowledged"
    
    def is_paused(self) -> bool:
        """Check if trading is currently paused."""
        return self._state.is_paused
    
    def requires_manual_recovery(self) -> bool:
        """Check if system requires manual recovery."""
        return self._state.requires_manual_recovery
    
    def get_state(self) -> Dict[str, Any]:
        """Get current operator control state."""
        with self._lock:
            return {
                'is_paused': self._state.is_paused,
                'paused_at': self._state.paused_at.isoformat() if self._state.paused_at else None,
                'paused_by': self._state.paused_by,
                'pause_reason': self._state.pause_reason,
                'requires_manual_recovery': self._state.requires_manual_recovery,
                'recovery_required_reason': self._state.recovery_required_reason
            }
    
    def get_action_log(self, hours: int = 24) -> List[Dict]:
        """
        Get operator action log for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of action log entries as dictionaries
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with self._lock:
            return [
                log.to_dict() 
                for log in self._action_log 
                if log.timestamp >= cutoff
            ]
    
    def _record_action(self, log_entry: OperatorActionLog) -> None:
        """Record an action to the log."""
        self._action_log.append(log_entry)
        self._state.last_action = log_entry
        
        # Trim log if too large
        if len(self._action_log) > self._max_log_entries:
            self._action_log = self._action_log[-self._max_log_entries:]
        
        # Persist to disk
        self._persist_action(log_entry)
    
    def _persist_action(self, log_entry: OperatorActionLog) -> None:
        """Persist action to disk for audit trail."""
        try:
            date_str = log_entry.timestamp.strftime("%Y-%m-%d")
            log_file = self._log_dir / f"operator_actions_{date_str}.jsonl"
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist operator action: {e}")


# Singleton instance
_operator_control: Optional[OperatorControlLayer] = None


def get_operator_control(health_checker: callable = None) -> OperatorControlLayer:
    """Get global OperatorControlLayer instance."""
    global _operator_control
    if _operator_control is None:
        _operator_control = OperatorControlLayer(health_checker=health_checker)
    return _operator_control


def reset_operator_control() -> None:
    """Reset the singleton (for testing)."""
    global _operator_control
    _operator_control = None
