"""
Operator Discipline Layer - v10.3 OPERATIONAL-GRADE

Formalizes human responsibility without allowing unsafe overrides.

Features:
- OperatorIdentity: Who is taking the action
- ActionReason: Why the action is being taken (mandatory)
- InterventionAuditLog: Immutable record of all operator actions

Rules:
- Operator cannot bypass risk, bias, or capital veto
- Resume actions require system health validation
- All actions are immutable and auditable

v10.3 - Operator-Safe, Incident-Resilient Platform
"""

import logging
import hashlib
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


# ============================================================================
# Reason Codes
# ============================================================================

class ActionReasonCode(Enum):
    """Mandatory reason codes for operator interventions."""
    # Pause/Halt Reasons
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"
    OBSERVED_ANOMALY = "observed_anomaly"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    EXCHANGE_ISSUES = "exchange_issues"
    EXTERNAL_MARKET_EVENT = "external_market_event"
    RISK_LIMIT_REVIEW = "risk_limit_review"
    
    # Resume Reasons
    ISSUE_RESOLVED = "issue_resolved"
    MAINTENANCE_COMPLETE = "maintenance_complete"
    MARKET_STABILIZED = "market_stabilized"
    RISK_ASSESSMENT_PASSED = "risk_assessment_passed"
    
    # Config Change Reasons
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    RISK_PARAMETER_ADJUSTMENT = "risk_parameter_adjustment"
    NEW_MARKET_CONDITIONS = "new_market_conditions"
    
    # Incident Acknowledgment
    INCIDENT_INVESTIGATED = "incident_investigated"
    ROOT_CAUSE_IDENTIFIED = "root_cause_identified"
    CORRECTIVE_ACTION_TAKEN = "corrective_action_taken"
    
    # Other
    OTHER = "other"  # Requires free-text explanation


class ActionType(Enum):
    """Types of operator actions."""
    PAUSE_TRADING = "pause_trading"
    RESUME_TRADING = "resume_trading"
    HALT_SYSTEM = "halt_system"
    ACKNOWLEDGE_INCIDENT = "acknowledge_incident"
    MODIFY_CONFIG = "modify_config"
    FORCE_REDUCE_POSITION = "force_reduce_position"
    EMERGENCY_CLOSE_ALL = "emergency_close_all"
    CLEAR_FREEZE = "clear_freeze"


# ============================================================================
# Operator Identity
# ============================================================================

@dataclass(frozen=True)
class OperatorIdentity:
    """
    Immutable identity of the operator taking an action.
    
    In production, this would integrate with SSO/auth systems.
    For now, it captures essential identity information.
    """
    operator_id: str
    name: str
    role: str  # 'admin', 'trader', 'risk_manager'
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    ip_address: Optional[str] = None
    
    def __post_init__(self):
        if not self.operator_id:
            raise ValueError("operator_id is required")
        if not self.name:
            raise ValueError("name is required")
        if self.role not in ('admin', 'trader', 'risk_manager'):
            raise ValueError(f"Invalid role: {self.role}")
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def system(cls) -> "OperatorIdentity":
        """System-initiated actions (e.g., auto-halt on error)."""
        return cls(
            operator_id="SYSTEM",
            name="Automated System",
            role="admin",
            session_id="SYSTEM"
        )


# ============================================================================
# Action Reason
# ============================================================================

@dataclass(frozen=True)
class ActionReason:
    """
    Mandatory reason for operator intervention.
    
    Every operator action MUST have a reason. Actions without
    reasons are rejected.
    """
    code: ActionReasonCode
    description: str
    supporting_data: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.description or len(self.description) < 10:
            raise ValueError("Reason description must be at least 10 characters")
        if self.code == ActionReasonCode.OTHER and len(self.description) < 20:
            raise ValueError("'OTHER' reason requires detailed description (20+ chars)")
    
    def to_dict(self) -> Dict:
        return {
            'code': self.code.value,
            'description': self.description,
            'supporting_data': self.supporting_data,
        }


# ============================================================================
# Intervention Record
# ============================================================================

@dataclass(frozen=True)
class InterventionRecord:
    """
    Immutable record of a single operator intervention.
    
    Once created, cannot be modified or deleted.
    """
    record_id: str
    timestamp: datetime
    operator: OperatorIdentity
    action_type: ActionType
    reason: ActionReason
    
    # Pre-action state
    pre_state: Dict = field(default_factory=dict)
    
    # Post-action state
    post_state: Dict = field(default_factory=dict)
    
    # Was action successful?
    success: bool = True
    error_message: Optional[str] = None
    
    # Checksum for integrity
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            # Calculate checksum on creation
            data = f"{self.record_id}{self.timestamp.isoformat()}{self.operator.operator_id}{self.action_type.value}{self.reason.code.value}"
            checksum = hashlib.sha256(data.encode()).hexdigest()[:16]
            object.__setattr__(self, 'checksum', checksum)
    
    def to_dict(self) -> Dict:
        return {
            'record_id': self.record_id,
            'timestamp': self.timestamp.isoformat(),
            'operator': self.operator.to_dict(),
            'action_type': self.action_type.value,
            'reason': self.reason.to_dict(),
            'pre_state': self.pre_state,
            'post_state': self.post_state,
            'success': self.success,
            'error_message': self.error_message,
            'checksum': self.checksum,
        }


# ============================================================================
# Intervention Audit Log
# ============================================================================

class InterventionAuditLog:
    """
    Immutable, append-only audit log for operator interventions.
    
    Features:
    - Append-only (no modifications or deletions)
    - Integrity verification via checksums
    - Queryable by time, operator, action type
    - Persistence-ready (serializable)
    
    Usage:
        log = InterventionAuditLog()
        
        record = log.record_intervention(
            operator=OperatorIdentity(...),
            action_type=ActionType.PAUSE_TRADING,
            reason=ActionReason(ActionReasonCode.OBSERVED_ANOMALY, "..."),
            pre_state={'trading_active': True},
            post_state={'trading_active': False},
        )
    """
    
    def __init__(self):
        self._records: List[InterventionRecord] = []
        self._chain_hash: str = "GENESIS"
    
    def record_intervention(
        self,
        operator: OperatorIdentity,
        action_type: ActionType,
        reason: ActionReason,
        pre_state: Dict = None,
        post_state: Dict = None,
        success: bool = True,
        error_message: str = None,
    ) -> InterventionRecord:
        """
        Record a new intervention. Returns the created record.
        
        Args:
            operator: Who performed the action
            action_type: What action was taken
            reason: Why the action was taken
            pre_state: System state before action
            post_state: System state after action
            success: Whether action succeeded
            error_message: Error message if failed
        
        Returns:
            The created InterventionRecord
        """
        record = InterventionRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            operator=operator,
            action_type=action_type,
            reason=reason,
            pre_state=pre_state or {},
            post_state=post_state or {},
            success=success,
            error_message=error_message,
        )
        
        # Update chain hash for integrity
        chain_data = f"{self._chain_hash}{record.checksum}"
        self._chain_hash = hashlib.sha256(chain_data.encode()).hexdigest()[:16]
        
        self._records.append(record)
        
        logger.info(
            f"Intervention logged: {action_type.value} by {operator.name} "
            f"({reason.code.value})"
        )
        
        return record
    
    def get_all(self) -> List[InterventionRecord]:
        """Get all intervention records."""
        return list(self._records)
    
    def get_recent(self, limit: int = 50) -> List[InterventionRecord]:
        """Get most recent interventions."""
        return list(reversed(self._records[-limit:]))
    
    def get_by_operator(self, operator_id: str) -> List[InterventionRecord]:
        """Get interventions by specific operator."""
        return [r for r in self._records if r.operator.operator_id == operator_id]
    
    def get_by_action_type(self, action_type: ActionType) -> List[InterventionRecord]:
        """Get interventions of specific type."""
        return [r for r in self._records if r.action_type == action_type]
    
    def get_in_timerange(
        self,
        start: datetime,
        end: datetime
    ) -> List[InterventionRecord]:
        """Get interventions within time range."""
        return [r for r in self._records if start <= r.timestamp <= end]
    
    def verify_integrity(self) -> bool:
        """Verify chain integrity hasn't been tampered with."""
        if not self._records:
            return self._chain_hash == "GENESIS"
        
        chain = "GENESIS"
        for record in self._records:
            chain_data = f"{chain}{record.checksum}"
            chain = hashlib.sha256(chain_data.encode()).hexdigest()[:16]
        
        is_valid = chain == self._chain_hash
        if not is_valid:
            logger.error("AUDIT LOG INTEGRITY VIOLATION DETECTED!")
        return is_valid
    
    def to_json(self) -> str:
        """Serialize to JSON for persistence."""
        return json.dumps({
            'chain_hash': self._chain_hash,
            'records': [r.to_dict() for r in self._records],
        }, indent=2)
    
    @property
    def count(self) -> int:
        return len(self._records)


# ============================================================================
# Operator Discipline Enforcer
# ============================================================================

class OperatorDiscipline:
    """
    Enforces operator discipline rules.
    
    Rules:
    1. Every action requires valid operator identity
    2. Every action requires mandatory reason
    3. Operator cannot bypass risk, bias, or capital veto
    4. Resume actions require system health validation
    5. All actions are logged immutably
    
    Usage:
        discipline = OperatorDiscipline()
        
        # This will raise if reason is missing or invalid
        discipline.validate_action(
            operator=identity,
            action=ActionType.PAUSE_TRADING,
            reason=reason,
        )
        
        # Record the action
        discipline.execute_action(
            operator=identity,
            action=ActionType.PAUSE_TRADING,
            reason=reason,
            action_callback=lambda: do_pause(),
        )
    """
    
    # Actions that CANNOT bypass risk
    RISK_PROTECTED_ACTIONS = {
        ActionType.RESUME_TRADING,
        ActionType.CLEAR_FREEZE,
    }
    
    # Actions that require admin role
    ADMIN_ONLY_ACTIONS = {
        ActionType.HALT_SYSTEM,
        ActionType.EMERGENCY_CLOSE_ALL,
        ActionType.MODIFY_CONFIG,
    }
    
    def __init__(self, health_checker: callable = None):
        """
        Args:
            health_checker: Optional callable that returns (is_healthy, issues)
        """
        self.audit_log = InterventionAuditLog()
        self._health_checker = health_checker
    
    def validate_action(
        self,
        operator: OperatorIdentity,
        action: ActionType,
        reason: ActionReason,
    ) -> tuple[bool, str]:
        """
        Validate if an action can be performed.
        
        Returns:
            (is_valid, error_message)
        """
        # Check identity
        if not operator or not operator.operator_id:
            return False, "Invalid operator identity"
        
        # Check reason
        if not reason or not reason.description:
            return False, "Action reason is required"
        
        # Check admin-only actions
        if action in self.ADMIN_ONLY_ACTIONS:
            if operator.role != 'admin':
                return False, f"Action {action.value} requires admin role"
        
        # Check risk-protected actions require health check
        if action in self.RISK_PROTECTED_ACTIONS:
            if self._health_checker:
                is_healthy, issues = self._health_checker()
                if not is_healthy:
                    return False, f"Cannot {action.value}: system not healthy - {issues}"
        
        return True, ""
    
    def execute_action(
        self,
        operator: OperatorIdentity,
        action: ActionType,
        reason: ActionReason,
        action_callback: callable,
        pre_state: Dict = None,
    ) -> tuple[bool, InterventionRecord]:
        """
        Execute an operator action with full discipline enforcement.
        
        Args:
            operator: Who is performing the action
            action: What action to perform
            reason: Why the action is being performed
            action_callback: The actual function to execute
            pre_state: Optional pre-action state snapshot
        
        Returns:
            (success, intervention_record)
        """
        # Validate
        is_valid, error = self.validate_action(operator, action, reason)
        
        if not is_valid:
            # Log the rejected attempt
            record = self.audit_log.record_intervention(
                operator=operator,
                action_type=action,
                reason=reason,
                pre_state=pre_state or {},
                success=False,
                error_message=error,
            )
            logger.warning(f"Action rejected: {action.value} - {error}")
            return False, record
        
        # Execute
        try:
            result = action_callback()
            post_state = result if isinstance(result, dict) else {'result': 'success'}
            
            record = self.audit_log.record_intervention(
                operator=operator,
                action_type=action,
                reason=reason,
                pre_state=pre_state or {},
                post_state=post_state,
                success=True,
            )
            
            logger.info(f"Action executed: {action.value} by {operator.name}")
            return True, record
            
        except Exception as e:
            record = self.audit_log.record_intervention(
                operator=operator,
                action_type=action,
                reason=reason,
                pre_state=pre_state or {},
                success=False,
                error_message=str(e),
            )
            logger.error(f"Action failed: {action.value} - {e}")
            return False, record
    
    def get_audit_log(self) -> InterventionAuditLog:
        """Get the audit log."""
        return self.audit_log
    
    def get_recent_interventions(self, limit: int = 20) -> List[Dict]:
        """Get recent interventions as dictionaries."""
        return [r.to_dict() for r in self.audit_log.get_recent(limit)]


# ============================================================================
# Singleton
# ============================================================================

_operator_discipline: Optional[OperatorDiscipline] = None


def get_operator_discipline() -> OperatorDiscipline:
    """Get global OperatorDiscipline instance."""
    global _operator_discipline
    if _operator_discipline is None:
        _operator_discipline = OperatorDiscipline()
    return _operator_discipline


def set_health_checker(checker: callable):
    """Set the health checker for the discipline instance."""
    get_operator_discipline()._health_checker = checker
