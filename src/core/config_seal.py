"""
Live Configuration Guard - v10.3-OPERATIONAL-GRADE

Prevents silent configuration drift in live trading.

Rules:
- Live configs are immutable once sealed
- Config checksum validated on each cycle
- Mismatch triggers safe halt

v10.3 - Operator-Safe, Incident-Resilient Platform
"""

import logging
import hashlib
import json
import copy
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Seal Status
# ============================================================================

class SealStatus(Enum):
    """Status of configuration seal."""
    UNSEALED = "unsealed"       # Not yet sealed, can be modified
    SEALED = "sealed"           # Sealed, immutable
    VIOLATED = "violated"       # Seal was violated


# ============================================================================
# Sealed Configuration
# ============================================================================

@dataclass
class SealedConfig:
    """A sealed, immutable configuration snapshot."""
    name: str
    checksum: str
    sealed_at: datetime
    sealed_by: str
    config_snapshot: Dict
    status: SealStatus = SealStatus.SEALED
    violation_detected: Optional[datetime] = None
    violation_details: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'checksum': self.checksum[:16] + '...',
            'sealed_at': self.sealed_at.isoformat(),
            'sealed_by': self.sealed_by,
            'status': self.status.value,
            'violation_detected': self.violation_detected.isoformat() if self.violation_detected else None,
        }


# ============================================================================
# Live Configuration Guard
# ============================================================================

class LiveConfigGuard:
    """
    Enforces configuration immutability during live trading.
    
    Once a configuration is sealed:
    - No modifications are allowed
    - Checksum is validated on each access/cycle
    - Any mismatch triggers an alert and optional halt
    
    Usage:
        guard = LiveConfigGuard()
        
        # Seal configs before going live
        guard.seal('strategy', strategy_config, operator_id='admin')
        guard.seal('risk', risk_config, operator_id='admin')
        
        # Validate on each cycle
        is_valid, errors = guard.validate_all()
        
        if not is_valid:
            # Trigger halt
            pass
    """
    
    def __init__(self, incident_state_machine=None):
        """
        Args:
            incident_state_machine: Optional reference to trigger halt on violation
        """
        self._sealed_configs: Dict[str, SealedConfig] = {}
        self._incident_sm = incident_state_machine
        self._validation_count = 0
        self._violation_count = 0
        self._auto_halt_on_violation = True
    
    def seal(
        self,
        name: str,
        config: Dict,
        operator_id: str = "system"
    ) -> SealedConfig:
        """
        Seal a configuration, making it immutable.
        
        Args:
            name: Configuration name (e.g., 'strategy', 'risk')
            config: The configuration dictionary to seal
            operator_id: Who is sealing the config
            
        Returns:
            The sealed configuration object
        """
        if name in self._sealed_configs:
            existing = self._sealed_configs[name]
            if existing.status == SealStatus.SEALED:
                raise ValueError(f"Config '{name}' is already sealed. Cannot reseal.")
        
        checksum = self._calculate_checksum(config)
        
        sealed = SealedConfig(
            name=name,
            checksum=checksum,
            sealed_at=datetime.utcnow(),
            sealed_by=operator_id,
            config_snapshot=copy.deepcopy(config),
            status=SealStatus.SEALED,
        )
        
        self._sealed_configs[name] = sealed
        
        logger.info(
            f"Config sealed: {name} by {operator_id} (checksum: {checksum[:16]}...)",
            extra={'config_name': name, 'operator': operator_id}
        )
        
        return sealed
    
    def validate(self, name: str, current_config: Dict) -> Tuple[bool, str]:
        """
        Validate a configuration against its sealed version.
        
        Args:
            name: Configuration name
            current_config: The current config to validate
            
        Returns:
            (is_valid, message)
        """
        self._validation_count += 1
        
        if name not in self._sealed_configs:
            return True, f"Config '{name}' not sealed - validation skipped"
        
        sealed = self._sealed_configs[name]
        
        if sealed.status == SealStatus.VIOLATED:
            return False, f"Config '{name}' was previously violated"
        
        current_checksum = self._calculate_checksum(current_config)
        
        if current_checksum != sealed.checksum:
            # Violation detected!
            self._violation_count += 1
            sealed.status = SealStatus.VIOLATED
            sealed.violation_detected = datetime.utcnow()
            sealed.violation_details = (
                f"Checksum mismatch: expected {sealed.checksum[:16]}..., "
                f"got {current_checksum[:16]}..."
            )
            
            logger.error(
                f"CONFIG SEAL VIOLATION: {name}",
                extra={
                    'config_name': name,
                    'expected': sealed.checksum[:16],
                    'actual': current_checksum[:16],
                }
            )
            
            # Trigger halt if configured
            if self._auto_halt_on_violation and self._incident_sm:
                self._incident_sm.trigger_halt(
                    f"Configuration seal violation detected: {name}"
                )
            
            return False, sealed.violation_details
        
        return True, "Config valid"
    
    def validate_all(self) -> Tuple[bool, List[str]]:
        """
        Validate all sealed configurations.
        
        Note: This requires the caller to provide current configs.
        This method validates internal consistency only.
        
        Returns:
            (all_valid, list_of_errors)
        """
        errors = []
        all_valid = True
        
        for name, sealed in self._sealed_configs.items():
            if sealed.status == SealStatus.VIOLATED:
                all_valid = False
                errors.append(f"Config '{name}' has a pending violation")
        
        return all_valid, errors
    
    def validate_configs(self, configs: Dict[str, Dict]) -> Tuple[bool, List[str]]:
        """
        Validate multiple configurations at once.
        
        Args:
            configs: Dict of config_name -> current_config
            
        Returns:
            (all_valid, list_of_errors)
        """
        errors = []
        all_valid = True
        
        for name, config in configs.items():
            is_valid, message = self.validate(name, config)
            if not is_valid:
                all_valid = False
                errors.append(message)
        
        return all_valid, errors
    
    def get_sealed_config(self, name: str) -> Optional[Dict]:
        """
        Get the sealed (original) version of a config.
        
        Returns a deep copy to prevent modification.
        """
        if name not in self._sealed_configs:
            return None
        return copy.deepcopy(self._sealed_configs[name].config_snapshot)
    
    def get_seal_status(self) -> Dict:
        """Get status of all seals."""
        return {
            'sealed_count': len(self._sealed_configs),
            'validation_count': self._validation_count,
            'violation_count': self._violation_count,
            'auto_halt_enabled': self._auto_halt_on_violation,
            'configs': {
                name: sealed.to_dict()
                for name, sealed in self._sealed_configs.items()
            }
        }
    
    def is_sealed(self, name: str) -> bool:
        """Check if a config is sealed."""
        return name in self._sealed_configs and \
               self._sealed_configs[name].status == SealStatus.SEALED
    
    def has_violations(self) -> bool:
        """Check if any violations have been detected."""
        return self._violation_count > 0
    
    def unseal(self, name: str, operator_id: str, reason: str) -> bool:
        """
        Unseal a configuration (requires operator action).
        
        Args:
            name: Configuration name
            operator_id: Operator unsealing
            reason: Why the config is being unsealed
            
        Returns:
            True if successful
        """
        if name not in self._sealed_configs:
            return False
        
        if not operator_id or not reason or len(reason) < 10:
            raise ValueError("Valid operator_id and reason required to unseal")
        
        del self._sealed_configs[name]
        
        logger.warning(
            f"Config unsealed: {name} by {operator_id} - {reason}",
            extra={'config_name': name, 'operator': operator_id}
        )
        
        return True
    
    def set_auto_halt(self, enabled: bool):
        """Enable or disable auto-halt on violation."""
        self._auto_halt_on_violation = enabled
    
    def _calculate_checksum(self, config: Dict) -> str:
        """Calculate SHA256 checksum of config."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()


# ============================================================================
# Singleton
# ============================================================================

_config_guard: Optional[LiveConfigGuard] = None


def get_config_guard() -> LiveConfigGuard:
    """Get global LiveConfigGuard instance."""
    global _config_guard
    if _config_guard is None:
        _config_guard = LiveConfigGuard()
    return _config_guard


def set_config_guard_incident_sm(incident_sm):
    """Set the incident state machine reference."""
    get_config_guard()._incident_sm = incident_sm
