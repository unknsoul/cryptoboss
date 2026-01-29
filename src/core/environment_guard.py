"""
Environment Truth Guard - v10.4-TRUST-GRADE

Ensures absolute clarity between TESTNET, LIVE, and PAPER modes across the system.
Prevents accidental real-money trading in test environments and vice versa.

Rules:
- Backend emits immutable environment signature
- All data objects include environment tag
- Frontend blocks mixed-environment rendering
"""

import logging
import hashlib
import json
import os
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class EnvironmentMode(Enum):
    """
    Strict environment modes.
    """
    LIVE = "live"       # Real money, real exchange
    TESTNET = "testnet" # Fake money, real testnet exchange
    PAPER = "paper"     # Fake money, simulated exchange (local)


@dataclass(frozen=True)
class EnvironmentSignature:
    """
    Immutable signature generated at system startup.
    Must be included in all critical response payloads.
    """
    mode: EnvironmentMode
    exchange_id: str
    exchange_url: str
    startup_timestamp: str
    config_checksum: str
    system_id: str
    
    @property
    def is_live(self) -> bool:
        return self.mode == EnvironmentMode.LIVE

    def to_dict(self) -> Dict:
        return {
            'mode': self.mode.value,
            'exchange_id': self.exchange_id,
            'exchange_url': self.exchange_url,
            'startup_timestamp': self.startup_timestamp,
            'config_checksum': self.config_checksum,
            'system_id': self.system_id,
            'signature_hash': self._generate_hash()
        }
    
    def _generate_hash(self) -> str:
        """Generate integrity hash of the signature itself."""
        payload = f"{self.mode.value}|{self.exchange_id}|{self.exchange_url}|{self.startup_timestamp}|{self.config_checksum}|{self.system_id}"
        return hashlib.sha256(payload.encode()).hexdigest()


class EnvironmentGuard:
    """
    Guardian of environment truth.
    Singleton that enforces environment consistency.
    """
    
    def __init__(self):
        self._signature: Optional[EnvironmentSignature] = None
        self._initialized = False
    
    def initialize(
        self, 
        mode: str, 
        exchange_id: str, 
        exchange_url: str,
        config: Dict
    ) -> EnvironmentSignature:
        """
        Initialize the environment guard. Can only be called once per process.
        """
        if self._initialized:
            raise RuntimeError("EnvironmentGuard already initialized - cannot change mode at runtime")
            
        try:
            env_mode = EnvironmentMode(mode.lower())
        except ValueError:
            raise ValueError(f"Invalid environment mode: {mode}. Must be one of {[e.value for e in EnvironmentMode]}")
            
        # Calculate config checksum
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_checksum = hashlib.sha256(config_str.encode()).hexdigest()
        
        # specific check for LIVE safety
        if env_mode == EnvironmentMode.LIVE:
            if "testnet" in exchange_url.lower():
                raise ValueError("CRITICAL: Configured for LIVE but using TESTNET URL")
            # In a real scenario, we'd check for "live" specific URL patterns or API key prefixes if possible safely
            
        if env_mode == EnvironmentMode.TESTNET:
            if "testnet" not in exchange_url.lower() and "sandbox" not in exchange_url.lower():
                 logger.warning(f"CAUTION: Configured for TESTNET but URL '{exchange_url}' does not look like standard testnet")

        self._signature = EnvironmentSignature(
            mode=env_mode,
            exchange_id=exchange_id,
            exchange_url=exchange_url,
            startup_timestamp=datetime.utcnow().isoformat(),
            config_checksum=config_checksum,
            system_id=hashlib.md5(f"{os.getpid()}-{datetime.utcnow()}".encode()).hexdigest()[:8]
        )
        
        self._initialized = True
        logger.info(f"EnvironmentGuard Initialized: MODE={env_mode.value.upper()} | {exchange_id}")
        
        if env_mode == EnvironmentMode.LIVE:
            logger.critical("🛑 SYSTEM RUNNING IN LIVE TRADING MODE - REAL CAPITAL AT RISK 🛑")
            
        return self._signature

    def get_signature(self) -> EnvironmentSignature:
        """Get the current immutable environment signature."""
        if not self._initialized:
            raise RuntimeError("EnvironmentGuard not initialized")
        return self._signature
    
    def validate_action(self, target_mode: str) -> bool:
        """
        Validate if an action intended for `target_mode` is allowed in current env.
        """
        if not self._initialized:
            return False
            
        current_mode = self._signature.mode.value
        if target_mode.lower() != current_mode:
            logger.error(f"Environment Mismatch: Action intended for {target_mode} blocked in {current_mode}")
            return False
        return True

    def verify_payload_signature(self, payload: Dict) -> bool:
        """
        Verify incoming data has matching environment signature.
        """
        if 'environment_signature' not in payload:
            return False
            
        sig = payload['environment_signature']
        # Simple check: mode must match
        if sig.get('mode') != self._signature.mode.value:
            return False
            
        return True

# Singleton
_env_guard: Optional[EnvironmentGuard] = None

def get_environment_guard() -> EnvironmentGuard:
    global _env_guard
    if _env_guard is None:
        _env_guard = EnvironmentGuard()
    return _env_guard
