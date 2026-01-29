"""
Decision Drift Guard - v10.3-OPERATIONAL-GRADE

Detects divergence between expected and actual behavior.

Features:
- Compare live decisions vs replay/shadow decisions
- Config checksum validation
- ML feature statistics monitoring
- Automatic freeze on unexplained drift

Rules:
- Alert on divergence beyond threshold
- Freeze trading on unexplained drift
- Log affected modules and timestamps

v10.3 - Operator-Safe, Incident-Resilient Platform
"""

import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Drift Types
# ============================================================================

class DriftType(Enum):
    """Types of drift that can be detected."""
    DECISION_DIVERGENCE = "decision_divergence"  # Live vs replay mismatch
    CONFIG_MISMATCH = "config_mismatch"          # Config changed unexpectedly
    FEATURE_DRIFT = "feature_drift"               # ML features out of bounds
    TIMING_DRIFT = "timing_drift"                 # Processing time anomaly
    OUTPUT_DRIFT = "output_drift"                 # Unexpected output patterns


class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    INFO = "info"           # Logged only
    WARNING = "warning"     # Alert but continue
    CRITICAL = "critical"   # Freeze trading


# ============================================================================
# Drift Event
# ============================================================================

@dataclass
class DriftEvent:
    """Record of a detected drift event."""
    event_id: str
    timestamp: datetime
    drift_type: DriftType
    severity: DriftSeverity
    module: str
    description: str
    
    # Comparison values
    expected_value: Any = None
    actual_value: Any = None
    divergence_score: float = 0.0
    
    # Context
    context: Dict = field(default_factory=dict)
    
    # Resolution
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'drift_type': self.drift_type.value,
            'severity': self.severity.value,
            'module': self.module,
            'description': self.description,
            'divergence_score': self.divergence_score,
            'resolved': self.resolved,
        }


# ============================================================================
# Config Checksum
# ============================================================================

class ConfigChecksum:
    """
    Tracks configuration integrity via checksums.
    
    Validates that live configuration hasn't changed unexpectedly.
    """
    
    def __init__(self):
        self._sealed_configs: Dict[str, str] = {}  # config_name -> checksum
        self._seal_time: Optional[datetime] = None
        self._is_sealed: bool = False
    
    def seal_config(self, name: str, config: Dict) -> str:
        """
        Seal a configuration (calculate and store its checksum).
        
        Args:
            name: Configuration name (e.g., 'strategy', 'risk')
            config: Configuration dictionary
            
        Returns:
            The calculated checksum
        """
        checksum = self._calculate_checksum(config)
        self._sealed_configs[name] = checksum
        self._seal_time = datetime.utcnow()
        self._is_sealed = True
        
        logger.info(f"Config sealed: {name} = {checksum[:16]}...")
        return checksum
    
    def validate_config(self, name: str, config: Dict) -> Tuple[bool, str]:
        """
        Validate a configuration against its sealed checksum.
        
        Args:
            name: Configuration name
            config: Current configuration to validate
            
        Returns:
            (is_valid, message)
        """
        if not self._is_sealed:
            return True, "No configs sealed - validation skipped"
        
        if name not in self._sealed_configs:
            return True, f"Config '{name}' not sealed - validation skipped"
        
        current_checksum = self._calculate_checksum(config)
        expected_checksum = self._sealed_configs[name]
        
        if current_checksum != expected_checksum:
            return False, (
                f"CONFIG MISMATCH: '{name}' has changed! "
                f"Expected {expected_checksum[:16]}..., got {current_checksum[:16]}..."
            )
        
        return True, "Config valid"
    
    def get_all_checksums(self) -> Dict[str, str]:
        """Get all sealed config checksums."""
        return dict(self._sealed_configs)
    
    def is_sealed(self) -> bool:
        """Check if configs have been sealed."""
        return self._is_sealed
    
    def _calculate_checksum(self, config: Dict) -> str:
        """Calculate SHA256 checksum of config."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()


# ============================================================================
# Decision Drift Guard
# ============================================================================

class DecisionDriftGuard:
    """
    Detects divergence between expected and actual behavior.
    
    Features:
    - Live vs shadow decision comparison
    - Config checksum validation on each cycle
    - ML feature bounds checking
    - Automatic freeze trigger on critical drift
    
    Usage:
        guard = DecisionDriftGuard()
        
        # Seal configs at start
        guard.seal_config('strategy', strategy_config)
        guard.seal_config('risk', risk_config)
        
        # On each decision cycle
        drift = guard.check_decision_drift(live_decision, shadow_decision)
        config_ok = guard.validate_all_configs(current_configs)
        
        if guard.should_freeze():
            incident_state_machine.trigger_incident_freeze("Drift detected")
    """
    
    # Thresholds
    DECISION_DIVERGENCE_THRESHOLD = 0.1  # 10% divergence triggers warning
    CRITICAL_DIVERGENCE_THRESHOLD = 0.3  # 30% divergence triggers freeze
    MAX_DRIFT_EVENTS_BEFORE_FREEZE = 5   # 5 warnings within window triggers freeze
    DRIFT_WINDOW_MINUTES = 30            # Time window for counting drift events
    
    def __init__(self, incident_state_machine=None):
        """
        Args:
            incident_state_machine: Optional reference to trigger freeze
        """
        self._config_checksum = ConfigChecksum()
        self._drift_events: deque = deque(maxlen=1000)
        self._incident_sm = incident_state_machine
        self._freeze_triggered = False
        self._last_check_time: Optional[datetime] = None
        
        # Feature bounds for ML drift detection
        self._feature_bounds: Dict[str, Tuple[float, float]] = {}
    
    def seal_config(self, name: str, config: Dict) -> str:
        """Seal a configuration for integrity checking."""
        return self._config_checksum.seal_config(name, config)
    
    def validate_config(self, name: str, config: Dict) -> Tuple[bool, str]:
        """Validate a configuration against sealed checksum."""
        is_valid, message = self._config_checksum.validate_config(name, config)
        
        if not is_valid:
            self._record_drift_event(
                drift_type=DriftType.CONFIG_MISMATCH,
                severity=DriftSeverity.CRITICAL,
                module=f"config.{name}",
                description=message,
            )
        
        return is_valid, message
    
    def validate_all_configs(self, configs: Dict[str, Dict]) -> Tuple[bool, List[str]]:
        """
        Validate all provided configs against sealed checksums.
        
        Args:
            configs: Dict of config_name -> config_dict
            
        Returns:
            (all_valid, list_of_errors)
        """
        errors = []
        all_valid = True
        
        for name, config in configs.items():
            is_valid, message = self.validate_config(name, config)
            if not is_valid:
                all_valid = False
                errors.append(message)
        
        return all_valid, errors
    
    def check_decision_drift(
        self,
        live_decision: Dict,
        shadow_decision: Dict
    ) -> Tuple[bool, float, str]:
        """
        Compare live decision vs shadow/replay decision.
        
        Args:
            live_decision: The actual decision made in live trading
            shadow_decision: Decision from shadow/replay calculation
            
        Returns:
            (is_aligned, divergence_score, message)
        """
        divergence = self._calculate_decision_divergence(live_decision, shadow_decision)
        
        if divergence > self.CRITICAL_DIVERGENCE_THRESHOLD:
            self._record_drift_event(
                drift_type=DriftType.DECISION_DIVERGENCE,
                severity=DriftSeverity.CRITICAL,
                module="decision_pipeline",
                description=f"Critical decision divergence: {divergence:.2%}",
                expected_value=shadow_decision,
                actual_value=live_decision,
                divergence_score=divergence,
            )
            return False, divergence, f"CRITICAL: Decision divergence {divergence:.2%}"
        
        elif divergence > self.DECISION_DIVERGENCE_THRESHOLD:
            self._record_drift_event(
                drift_type=DriftType.DECISION_DIVERGENCE,
                severity=DriftSeverity.WARNING,
                module="decision_pipeline",
                description=f"Decision divergence warning: {divergence:.2%}",
                expected_value=shadow_decision,
                actual_value=live_decision,
                divergence_score=divergence,
            )
            return True, divergence, f"WARNING: Decision divergence {divergence:.2%}"
        
        return True, divergence, "Decisions aligned"
    
    def set_feature_bounds(self, feature_name: str, min_val: float, max_val: float):
        """Set expected bounds for an ML feature."""
        self._feature_bounds[feature_name] = (min_val, max_val)
    
    def check_feature_bounds(self, features: Dict[str, float]) -> List[str]:
        """
        Check if ML features are within expected bounds.
        
        Args:
            features: Dict of feature_name -> value
            
        Returns:
            List of out-of-bounds feature names
        """
        out_of_bounds = []
        
        for name, value in features.items():
            if name in self._feature_bounds:
                min_val, max_val = self._feature_bounds[name]
                if value < min_val or value > max_val:
                    out_of_bounds.append(name)
                    self._record_drift_event(
                        drift_type=DriftType.FEATURE_DRIFT,
                        severity=DriftSeverity.WARNING,
                        module=f"ml.feature.{name}",
                        description=f"Feature '{name}' out of bounds: {value} not in [{min_val}, {max_val}]",
                        expected_value=(min_val, max_val),
                        actual_value=value,
                    )
        
        return out_of_bounds
    
    def should_freeze(self) -> bool:
        """
        Check if trading should be frozen due to drift.
        
        Freeze is triggered if:
        - Any CRITICAL drift event occurred
        - Too many WARNING events within the time window
        """
        if self._freeze_triggered:
            return True
        
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=self.DRIFT_WINDOW_MINUTES)
        
        recent_events = [
            e for e in self._drift_events
            if e.timestamp >= window_start
        ]
        
        # Check for critical events
        critical_events = [e for e in recent_events if e.severity == DriftSeverity.CRITICAL]
        if critical_events:
            self._freeze_triggered = True
            logger.error(f"DRIFT FREEZE: {len(critical_events)} critical drift event(s)")
            return True
        
        # Check for too many warnings
        warning_events = [e for e in recent_events if e.severity == DriftSeverity.WARNING]
        if len(warning_events) >= self.MAX_DRIFT_EVENTS_BEFORE_FREEZE:
            self._freeze_triggered = True
            logger.error(f"DRIFT FREEZE: {len(warning_events)} warnings in {self.DRIFT_WINDOW_MINUTES}min window")
            return True
        
        return False
    
    def trigger_freeze_if_needed(self) -> bool:
        """
        Check drift and trigger incident freeze if needed.
        
        Returns:
            True if freeze was triggered
        """
        if self.should_freeze() and self._incident_sm:
            self._incident_sm.trigger_incident_freeze(
                f"Decision drift detected - {len(self.get_recent_events())} drift events"
            )
            return True
        return False
    
    def get_recent_events(self, limit: int = 50) -> List[DriftEvent]:
        """Get recent drift events."""
        return list(self._drift_events)[-limit:]
    
    def get_drift_summary(self) -> Dict:
        """Get summary of drift status."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=self.DRIFT_WINDOW_MINUTES)
        
        recent = [e for e in self._drift_events if e.timestamp >= window_start]
        
        return {
            'freeze_triggered': self._freeze_triggered,
            'total_events': len(self._drift_events),
            'recent_events': len(recent),
            'critical_count': sum(1 for e in recent if e.severity == DriftSeverity.CRITICAL),
            'warning_count': sum(1 for e in recent if e.severity == DriftSeverity.WARNING),
            'config_sealed': self._config_checksum.is_sealed(),
            'config_checksums': self._config_checksum.get_all_checksums(),
        }
    
    def reset_freeze(self, operator_id: str, reason: str):
        """
        Reset drift freeze (requires operator action).
        
        Args:
            operator_id: Operator resetting the freeze
            reason: Why the freeze is being reset
        """
        if not operator_id or len(reason) < 10:
            raise ValueError("Valid operator_id and reason required to reset drift freeze")
        
        self._freeze_triggered = False
        logger.info(f"Drift freeze reset by {operator_id}: {reason}")
    
    def _record_drift_event(
        self,
        drift_type: DriftType,
        severity: DriftSeverity,
        module: str,
        description: str,
        expected_value: Any = None,
        actual_value: Any = None,
        divergence_score: float = 0.0,
        context: Dict = None,
    ):
        """Record a drift event."""
        import uuid
        
        event = DriftEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            drift_type=drift_type,
            severity=severity,
            module=module,
            description=description,
            expected_value=expected_value,
            actual_value=actual_value,
            divergence_score=divergence_score,
            context=context or {},
        )
        
        self._drift_events.append(event)
        
        log_level = logging.WARNING if severity == DriftSeverity.WARNING else logging.ERROR
        logger.log(log_level, f"DRIFT: [{severity.value}] {description}")
    
    def _calculate_decision_divergence(
        self,
        live: Dict,
        shadow: Dict
    ) -> float:
        """
        Calculate divergence score between two decisions.
        
        Returns a value 0.0-1.0 where 0.0 = identical, 1.0 = completely different.
        """
        if not live or not shadow:
            return 1.0 if (live or shadow) else 0.0
        
        # Key fields to compare
        key_fields = ['status', 'direction', 'symbol', 'action']
        numeric_fields = ['confidence_score', 'position_size', 'entry_price']
        
        divergence = 0.0
        field_count = 0
        
        # Check key fields (exact match required)
        for field in key_fields:
            if field in live or field in shadow:
                field_count += 1
                if live.get(field) != shadow.get(field):
                    divergence += 1.0
        
        # Check numeric fields (relative difference)
        for field in numeric_fields:
            if field in live and field in shadow:
                field_count += 1
                live_val = float(live.get(field, 0) or 0)
                shadow_val = float(shadow.get(field, 0) or 0)
                
                if live_val == 0 and shadow_val == 0:
                    continue
                
                max_val = max(abs(live_val), abs(shadow_val))
                if max_val > 0:
                    diff = abs(live_val - shadow_val) / max_val
                    divergence += min(diff, 1.0)
        
        return divergence / max(field_count, 1)


# ============================================================================
# Singleton
# ============================================================================

_drift_guard: Optional[DecisionDriftGuard] = None


def get_drift_guard() -> DecisionDriftGuard:
    """Get global DecisionDriftGuard instance."""
    global _drift_guard
    if _drift_guard is None:
        _drift_guard = DecisionDriftGuard()
    return _drift_guard


def set_drift_guard_incident_sm(incident_sm):
    """Set the incident state machine reference."""
    get_drift_guard()._incident_sm = incident_sm
