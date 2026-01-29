"""
Live Config Guard - v10.2-OPERATOR-GRADE

Prevents accidental configuration drift in live trading.
Ensures configuration immutability when running in LIVE mode.

Non-Negotiable Rules:
- Configs are immutable in LIVE mode
- Any config change requires restart
- Config checksum logged on startup
- Mismatch triggers safe halt
"""

import logging
import hashlib
import json
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import yaml
import threading

logger = logging.getLogger(__name__)


@dataclass
class ConfigSnapshot:
    """Snapshot of configuration at a point in time."""
    timestamp: datetime
    checksum: str
    mode: str
    config_files: Dict[str, str]  # filename -> checksum
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'checksum': self.checksum,
            'mode': self.mode,
            'config_files': self.config_files
        }


class ConfigGuard:
    """
    Live Config Guard - Configuration immutability enforcement.
    
    Computes a checksum of all configuration at startup and monitors
    for any drift. In LIVE mode, any configuration change triggers
    a safe halt.
    
    Usage:
        guard = ConfigGuard(mode="live", config_dir="configs")
        
        # On startup
        checksum = guard.compute_checksum()
        logger.info(f"Config checksum: {checksum}")
        
        # Periodic validation
        is_valid, reason = guard.validate_immutability()
        if not is_valid:
            guard.trigger_safe_halt(reason)
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        mode: str = "paper",
        config_dir: str = "configs",
        halt_callback: callable = None
    ):
        self._config = config or {}
        self._mode = mode.lower()
        self._config_dir = Path(config_dir)
        self._halt_callback = halt_callback
        self._lock = threading.RLock()
        
        # Store startup snapshot
        self._startup_snapshot: Optional[ConfigSnapshot] = None
        self._last_check: Optional[datetime] = None
        self._drift_detected = False
        
        # Initialize snapshot on creation
        self._initialize_snapshot()
        
        logger.info(
            f"ConfigGuard initialized in {self._mode.upper()} mode, "
            f"checksum={self._startup_snapshot.checksum if self._startup_snapshot else 'N/A'}"
        )
    
    def _initialize_snapshot(self) -> None:
        """Create initial configuration snapshot."""
        checksum = self._compute_config_checksum()
        file_checksums = self._compute_file_checksums()
        
        self._startup_snapshot = ConfigSnapshot(
            timestamp=datetime.utcnow(),
            checksum=checksum,
            mode=self._mode,
            config_files=file_checksums
        )
    
    def _compute_config_checksum(self) -> str:
        """Compute checksum of configuration dictionary."""
        # Sort keys for deterministic hash
        config_str = json.dumps(self._config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _compute_file_checksums(self) -> Dict[str, str]:
        """Compute checksums of all config files."""
        checksums = {}
        
        if not self._config_dir.exists():
            return checksums
        
        for ext in ['*.yaml', '*.yml', '*.json']:
            for config_file in self._config_dir.glob(ext):
                try:
                    with open(config_file, 'rb') as f:
                        content = f.read()
                        checksum = hashlib.sha256(content).hexdigest()[:16]
                        checksums[config_file.name] = checksum
                except Exception as e:
                    logger.error(f"Failed to compute checksum for {config_file}: {e}")
        
        return checksums
    
    def compute_checksum(self) -> str:
        """
        Compute current configuration checksum.
        
        Returns:
            16-character hex checksum
        """
        with self._lock:
            return self._compute_config_checksum()
    
    def get_startup_checksum(self) -> str:
        """Get the checksum computed at startup."""
        return self._startup_snapshot.checksum if self._startup_snapshot else ""
    
    def get_startup_snapshot(self) -> Dict:
        """Get full startup snapshot."""
        if self._startup_snapshot:
            return self._startup_snapshot.to_dict()
        return {}
    
    def validate_immutability(self) -> Tuple[bool, str]:
        """
        Validate that configuration has not changed since startup.
        
        In PAPER mode, this only logs warnings.
        In LIVE mode, any change is a critical error.
        
        Returns:
            (is_valid, reason)
        """
        with self._lock:
            self._last_check = datetime.utcnow()
            
            if not self._startup_snapshot:
                return True, "No startup snapshot to compare"
            
            # Check in-memory config
            current_checksum = self._compute_config_checksum()
            if current_checksum != self._startup_snapshot.checksum:
                self._drift_detected = True
                reason = f"In-memory config drift: startup={self._startup_snapshot.checksum}, current={current_checksum}"
                
                if self.is_live_mode():
                    logger.critical(f"CONFIG DRIFT IN LIVE MODE: {reason}")
                    return False, reason
                else:
                    logger.warning(f"Config drift detected (PAPER mode): {reason}")
            
            # Check file checksums
            current_files = self._compute_file_checksums()
            for filename, startup_checksum in self._startup_snapshot.config_files.items():
                current = current_files.get(filename)
                if current != startup_checksum:
                    self._drift_detected = True
                    reason = f"Config file changed: {filename} (startup={startup_checksum}, current={current})"
                    
                    if self.is_live_mode():
                        logger.critical(f"CONFIG FILE DRIFT IN LIVE MODE: {reason}")
                        return False, reason
                    else:
                        logger.warning(f"Config file drift (PAPER mode): {reason}")
            
            return True, "Configuration unchanged"
    
    def detect_drift(self) -> bool:
        """
        Check if any configuration drift has been detected.
        
        Returns:
            True if drift detected, False otherwise
        """
        return self._drift_detected
    
    def is_live_mode(self) -> bool:
        """Check if running in live mode."""
        return self._mode == "live"
    
    def trigger_safe_halt(self, reason: str) -> None:
        """
        Trigger a safe halt due to configuration drift.
        
        Args:
            reason: Why the halt was triggered
        """
        logger.critical(
            f"CONFIG GUARD TRIGGERING SAFE HALT: {reason}",
            extra={'action': 'safe_halt', 'reason': reason}
        )
        
        if self._halt_callback:
            try:
                self._halt_callback(reason)
            except Exception as e:
                logger.error(f"Halt callback failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current config guard status."""
        with self._lock:
            return {
                'mode': self._mode,
                'is_live': self.is_live_mode(),
                'startup_checksum': self.get_startup_checksum(),
                'current_checksum': self._compute_config_checksum(),
                'drift_detected': self._drift_detected,
                'last_check': self._last_check.isoformat() if self._last_check else None,
                'startup_time': self._startup_snapshot.timestamp.isoformat() if self._startup_snapshot else None
            }
    
    def update_config_reference(self, new_config: Dict[str, Any]) -> None:
        """
        Update the config reference (only allowed in PAPER mode).
        
        In LIVE mode, this will trigger a safe halt.
        
        Args:
            new_config: New configuration dictionary
        """
        if self.is_live_mode():
            self.trigger_safe_halt("Attempted config update in LIVE mode")
            return
        
        with self._lock:
            self._config = new_config
            logger.info("Config reference updated (PAPER mode)")


# Singleton instance
_config_guard: Optional[ConfigGuard] = None


def get_config_guard(
    config: Dict[str, Any] = None,
    mode: str = None,
    halt_callback: callable = None
) -> ConfigGuard:
    """Get global ConfigGuard instance."""
    global _config_guard
    if _config_guard is None:
        _config_guard = ConfigGuard(
            config=config or {},
            mode=mode or "paper",
            halt_callback=halt_callback
        )
    return _config_guard


def reset_config_guard() -> None:
    """Reset the singleton (for testing)."""
    global _config_guard
    _config_guard = None
