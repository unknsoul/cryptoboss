"""
Cold Start Controller - Live Readiness Component

Prevents unsafe trading after startup:
- Risk state must load successfully
- Market data must be sufficient
- Context and bias must stabilize
- Exchange must be healthy
- Minimum warm-up time must elapse

Trading is blocked until ALL checks pass.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ColdStartPhase(Enum):
    """Cold start phases."""
    INITIALIZING = "initializing"      # Loading components
    SYNCING_STATE = "syncing_state"    # Loading persisted state
    SYNCING_DATA = "syncing_data"      # Fetching market data
    STABILIZING = "stabilizing"        # Waiting for context/bias stabilization
    READY = "ready"                    # All checks passed
    FAILED = "failed"                  # Startup failure


@dataclass
class WarmUpCheck:
    """Individual warm-up check result."""
    name: str
    passed: bool
    message: str
    required: bool = True


@dataclass
class ColdStartStatus:
    """Current cold start status."""
    phase: ColdStartPhase
    started_at: datetime
    elapsed: timedelta
    
    # Individual checks
    checks: List[WarmUpCheck]
    passed_checks: int
    total_required: int
    
    # Status
    is_ready: bool
    is_failed: bool
    failure_reason: Optional[str]
    
    # Remaining time
    warm_up_remaining: timedelta
    
    def to_dict(self) -> Dict:
        return {
            'phase': self.phase.value,
            'started_at': self.started_at.isoformat(),
            'elapsed_seconds': self.elapsed.total_seconds(),
            'checks': [
                {'name': c.name, 'passed': c.passed, 'message': c.message, 'required': c.required}
                for c in self.checks
            ],
            'passed_checks': self.passed_checks,
            'total_required': self.total_required,
            'is_ready': self.is_ready,
            'is_failed': self.is_failed,
            'failure_reason': self.failure_reason,
            'warm_up_remaining_seconds': self.warm_up_remaining.total_seconds()
        }


class ColdStartController:
    """
    Controls cold start and warm-up sequence.
    
    Requirements for trading readiness:
    1. Risk state loaded successfully
    2. At least 100 candles of market data
    3. Market context classified successfully
    4. Bias stabilized (no flip during warm-up)
    5. Exchange health score > 0.8
    6. Minimum 5 minute warm-up period
    
    If any required check fails after timeout -> FAILED state.
    
    Usage:
        controller = ColdStartController()
        
        # Progress through phases
        controller.on_state_loaded()
        controller.on_data_synced(candle_count=150)
        controller.on_context_classified("trending_up")
        controller.on_bias_determined("long_only")
        controller.on_exchange_healthy(score=0.9)
        
        # Check if ready
        status = controller.get_status()
        if not status.is_ready:
            logger.info(f"Warm-up: {status.phase.value}")
            return
    """
    
    # Configuration
    MIN_WARM_UP_MINUTES = 5
    STARTUP_TIMEOUT_MINUTES = 10
    MIN_CANDLE_COUNT = 100
    MIN_EXCHANGE_HEALTH = 0.8
    
    def __init__(
        self,
        min_warm_up_minutes: float = 5.0,
        startup_timeout_minutes: float = 10.0,
        on_ready_callback: Optional[Callable[[], None]] = None,
        on_failed_callback: Optional[Callable[[str], None]] = None
    ):
        self.min_warm_up = timedelta(minutes=min_warm_up_minutes)
        self.startup_timeout = timedelta(minutes=startup_timeout_minutes)
        
        self._started_at = datetime.now()
        self._phase = ColdStartPhase.INITIALIZING
        
        # Check states
        self._state_loaded = False
        self._data_synced = False
        self._candle_count = 0
        self._context_classified = False
        self._context_value: Optional[str] = None
        self._bias_determined = False
        self._bias_value: Optional[str] = None
        self._bias_stable = False
        self._bias_first_seen: Optional[datetime] = None
        self._exchange_healthy = False
        self._exchange_score = 0.0
        
        # Failure handling
        self._is_failed = False
        self._failure_reason: Optional[str] = None
        
        # Callbacks
        self._on_ready = on_ready_callback
        self._on_failed = on_failed_callback
        
        logger.info(
            f"ColdStartController initialized: "
            f"warm_up={min_warm_up_minutes}m, timeout={startup_timeout_minutes}m"
        )
    
    @property
    def is_ready(self) -> bool:
        """Check if system is ready for trading."""
        return self._phase == ColdStartPhase.READY
    
    @property
    def is_failed(self) -> bool:
        """Check if startup has failed."""
        return self._is_failed
    
    def on_state_loaded(self, success: bool = True, error: Optional[str] = None):
        """Called when risk state loading completes."""
        self._state_loaded = success
        
        if success:
            logger.info("Cold start: Risk state loaded ✓")
            self._advance_phase()
        else:
            self._fail(f"Risk state load failed: {error}")
    
    def on_data_synced(self, candle_count: int):
        """Called when market data sync completes."""
        self._candle_count = candle_count
        self._data_synced = candle_count >= self.MIN_CANDLE_COUNT
        
        if self._data_synced:
            logger.info(f"Cold start: Data synced ({candle_count} candles) ✓")
            self._advance_phase()
        else:
            logger.warning(
                f"Cold start: Insufficient data ({candle_count} < {self.MIN_CANDLE_COUNT})"
            )
    
    def on_context_classified(self, context: str):
        """Called when market context is classified."""
        self._context_classified = True
        self._context_value = context
        
        logger.info(f"Cold start: Context classified ({context}) ✓")
        self._advance_phase()
    
    def on_bias_determined(self, bias: str):
        """Called when directional bias is determined."""
        now = datetime.now()
        
        if not self._bias_determined:
            # First bias determination
            self._bias_determined = True
            self._bias_value = bias
            self._bias_first_seen = now
            logger.info(f"Cold start: Initial bias ({bias})")
        elif bias != self._bias_value:
            # Bias changed during warm-up - reset stability
            self._bias_value = bias
            self._bias_first_seen = now
            self._bias_stable = False
            logger.warning(f"Cold start: Bias changed to {bias}, resetting stability")
        
        # Check stability (no flip for 2 minutes)
        if self._bias_first_seen:
            elapsed = now - self._bias_first_seen
            if elapsed >= timedelta(minutes=2):
                self._bias_stable = True
                logger.info(f"Cold start: Bias stable ({bias}) ✓")
                self._advance_phase()
    
    def on_exchange_healthy(self, score: float):
        """Called when exchange health check completes."""
        self._exchange_score = score
        self._exchange_healthy = score >= self.MIN_EXCHANGE_HEALTH
        
        if self._exchange_healthy:
            logger.info(f"Cold start: Exchange healthy ({score:.2f}) ✓")
            self._advance_phase()
        else:
            logger.warning(
                f"Cold start: Exchange health low ({score:.2f} < {self.MIN_EXCHANGE_HEALTH})"
            )
    
    def get_status(self) -> ColdStartStatus:
        """Get current cold start status."""
        now = datetime.now()
        elapsed = now - self._started_at
        
        # Build check list
        checks = [
            WarmUpCheck(
                name="risk_state_loaded",
                passed=self._state_loaded,
                message="Risk state loaded" if self._state_loaded else "Waiting for state load",
                required=True
            ),
            WarmUpCheck(
                name="market_data_synced",
                passed=self._data_synced,
                message=f"{self._candle_count} candles" if self._data_synced else f"Need {self.MIN_CANDLE_COUNT} candles",
                required=True
            ),
            WarmUpCheck(
                name="context_classified",
                passed=self._context_classified,
                message=self._context_value or "Not classified",
                required=True
            ),
            WarmUpCheck(
                name="bias_stable",
                passed=self._bias_stable,
                message=f"{self._bias_value} stable" if self._bias_stable else "Waiting for stability",
                required=True
            ),
            WarmUpCheck(
                name="exchange_healthy",
                passed=self._exchange_healthy,
                message=f"Score: {self._exchange_score:.2f}" if self._exchange_healthy else "Health check pending",
                required=True
            )
        ]
        
        # Check minimum warm-up time
        warm_up_remaining = max(timedelta(0), self.min_warm_up - elapsed)
        warm_up_complete = warm_up_remaining.total_seconds() <= 0
        
        checks.append(WarmUpCheck(
            name="min_warm_up_elapsed",
            passed=warm_up_complete,
            message=f"{elapsed.seconds}s elapsed" if warm_up_complete else f"{warm_up_remaining.seconds}s remaining",
            required=True
        ))
        
        # Count passed required checks
        required_checks = [c for c in checks if c.required]
        passed_required = sum(1 for c in required_checks if c.passed)
        
        # Check for timeout
        if elapsed > self.startup_timeout and not self.is_ready and not self._is_failed:
            failed_checks = [c.name for c in required_checks if not c.passed]
            self._fail(f"Startup timeout: {', '.join(failed_checks)} not ready")
        
        # Determine if ready
        is_ready = (
            passed_required == len(required_checks) and
            not self._is_failed
        )
        
        # Update phase if ready
        if is_ready and self._phase != ColdStartPhase.READY:
            self._phase = ColdStartPhase.READY
            logger.info("Cold start complete: READY FOR TRADING ✓")
            if self._on_ready:
                self._on_ready()
        
        return ColdStartStatus(
            phase=self._phase,
            started_at=self._started_at,
            elapsed=elapsed,
            checks=checks,
            passed_checks=passed_required,
            total_required=len(required_checks),
            is_ready=is_ready and self._phase == ColdStartPhase.READY,
            is_failed=self._is_failed,
            failure_reason=self._failure_reason,
            warm_up_remaining=warm_up_remaining
        )
    
    def reset(self):
        """Reset cold start controller for new startup."""
        self._started_at = datetime.now()
        self._phase = ColdStartPhase.INITIALIZING
        self._state_loaded = False
        self._data_synced = False
        self._candle_count = 0
        self._context_classified = False
        self._context_value = None
        self._bias_determined = False
        self._bias_value = None
        self._bias_stable = False
        self._bias_first_seen = None
        self._exchange_healthy = False
        self._exchange_score = 0.0
        self._is_failed = False
        self._failure_reason = None
        
        logger.info("ColdStartController reset")
    
    def _advance_phase(self):
        """Advance to next appropriate phase."""
        if self._is_failed:
            return
        
        if not self._state_loaded:
            self._phase = ColdStartPhase.SYNCING_STATE
        elif not self._data_synced:
            self._phase = ColdStartPhase.SYNCING_DATA
        elif not self._bias_stable:
            self._phase = ColdStartPhase.STABILIZING
        else:
            # Final check will be done in get_status()
            self._phase = ColdStartPhase.STABILIZING
    
    def _fail(self, reason: str):
        """Enter failed state."""
        self._is_failed = True
        self._failure_reason = reason
        self._phase = ColdStartPhase.FAILED
        
        logger.error(f"COLD START FAILED: {reason}")
        
        if self._on_failed:
            self._on_failed(reason)


# Singleton instance
_cold_start_controller: Optional[ColdStartController] = None


def get_cold_start_controller() -> ColdStartController:
    """Get global ColdStartController instance."""
    global _cold_start_controller
    if _cold_start_controller is None:
        _cold_start_controller = ColdStartController()
    return _cold_start_controller


def reset_cold_start_controller():
    """Reset global instance."""
    global _cold_start_controller
    if _cold_start_controller:
        _cold_start_controller.reset()
