"""
Exchange Health Monitor - Live Readiness Component

Monitors exchange connectivity and performance:
- API latency
- Order rejection rate
- Partial fill ratio
- WebSocket lag

Disables execution on degradation, requires manual recovery.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Deque
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class HealthLevel(Enum):
    """Exchange health levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class OrderMetrics:
    """Metrics for a single order."""
    order_id: str
    timestamp: datetime
    latency_ms: float
    was_rejected: bool
    was_partial_fill: bool
    fill_ratio: float  # 0.0 to 1.0


@dataclass
class ExchangeHealthSnapshot:
    """Current health status snapshot."""
    timestamp: datetime
    health_level: HealthLevel
    health_score: float  # 0.0 to 1.0
    
    # Metrics
    avg_latency_ms: float
    order_rejection_rate: float
    partial_fill_ratio: float
    websocket_lag_ms: float
    
    # Failure tracking
    consecutive_failures: int
    last_successful_order: Optional[datetime]
    
    # Status
    execution_allowed: bool
    requires_manual_recovery: bool
    issues: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'health_level': self.health_level.value,
            'health_score': self.health_score,
            'avg_latency_ms': self.avg_latency_ms,
            'order_rejection_rate': self.order_rejection_rate,
            'partial_fill_ratio': self.partial_fill_ratio,
            'websocket_lag_ms': self.websocket_lag_ms,
            'consecutive_failures': self.consecutive_failures,
            'execution_allowed': self.execution_allowed,
            'requires_manual_recovery': self.requires_manual_recovery,
            'issues': self.issues
        }


class ExchangeHealthMonitor:
    """
    Monitors exchange health and controls execution permission.
    
    Thresholds:
        - API Latency: Warning > 500ms, Critical > 2000ms
        - Rejection Rate: Warning > 10%, Critical > 30%
        - Partial Fill: Warning > 20%, Critical > 50%
        - WS Lag: Warning > 1000ms, Critical > 5000ms
    
    Response:
        - Warning: Log + reduce position sizes by 50%
        - Critical: Disable execution + alert + require manual recovery
    
    Usage:
        monitor = ExchangeHealthMonitor()
        
        # Record order result
        monitor.record_order_result(
            order_id="123",
            latency_ms=150,
            was_rejected=False,
            fill_ratio=1.0
        )
        
        # Check health before trading
        health = monitor.get_health()
        if not health.execution_allowed:
            logger.error(f"Execution blocked: {health.issues}")
    """
    
    # Thresholds
    LATENCY_WARNING_MS = 500
    LATENCY_CRITICAL_MS = 2000
    
    REJECTION_WARNING_PCT = 10.0
    REJECTION_CRITICAL_PCT = 30.0
    
    PARTIAL_WARNING_PCT = 20.0
    PARTIAL_CRITICAL_PCT = 50.0
    
    WS_LAG_WARNING_MS = 1000
    WS_LAG_CRITICAL_MS = 5000
    
    # Recovery
    RECOVERY_COOLDOWN_MINUTES = 10
    MAX_CONSECUTIVE_FAILURES = 5
    
    def __init__(self, window_size: int = 20):
        self._order_history: Deque[OrderMetrics] = deque(maxlen=window_size)
        self._websocket_lag_ms: float = 0.0
        self._consecutive_failures: int = 0
        self._last_successful_order: Optional[datetime] = None
        self._requires_manual_recovery: bool = False
        self._recovery_requested_at: Optional[datetime] = None
        
        logger.info(f"ExchangeHealthMonitor initialized (window_size={window_size})")
    
    def record_order_result(
        self,
        order_id: str,
        latency_ms: float,
        was_rejected: bool,
        fill_ratio: float = 1.0
    ):
        """
        Record the result of an order attempt.
        
        Args:
            order_id: Exchange order ID
            latency_ms: Time from submit to response
            was_rejected: True if order was rejected
            fill_ratio: Ratio of filled amount (0.0 to 1.0)
        """
        metrics = OrderMetrics(
            order_id=order_id,
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            was_rejected=was_rejected,
            was_partial_fill=fill_ratio < 1.0,
            fill_ratio=fill_ratio
        )
        
        self._order_history.append(metrics)
        
        if was_rejected:
            self._consecutive_failures += 1
            logger.warning(
                f"Order rejected: {order_id}, "
                f"consecutive_failures={self._consecutive_failures}"
            )
        else:
            self._consecutive_failures = 0
            self._last_successful_order = datetime.now()
        
        # Check if we should enter critical state
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            self._requires_manual_recovery = True
            logger.error(
                f"CRITICAL: {self._consecutive_failures} consecutive failures, "
                f"execution disabled, manual recovery required"
            )
    
    def update_websocket_lag(self, lag_ms: float):
        """Update current WebSocket lag measurement."""
        self._websocket_lag_ms = lag_ms
        
        if lag_ms > self.WS_LAG_CRITICAL_MS:
            logger.warning(f"WebSocket lag critical: {lag_ms:.0f}ms")
    
    def record_api_error(self, error_type: str):
        """Record a general API error."""
        self._consecutive_failures += 1
        
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            self._requires_manual_recovery = True
            logger.error(
                f"CRITICAL: API errors ({error_type}), "
                f"execution disabled"
            )
    
    def get_health(self) -> ExchangeHealthSnapshot:
        """
        Get current health status.
        
        Evaluates all metrics and returns comprehensive status.
        """
        now = datetime.now()
        issues: List[str] = []
        
        # Calculate metrics from history
        if not self._order_history:
            return ExchangeHealthSnapshot(
                timestamp=now,
                health_level=HealthLevel.UNKNOWN,
                health_score=0.5,
                avg_latency_ms=0.0,
                order_rejection_rate=0.0,
                partial_fill_ratio=0.0,
                websocket_lag_ms=self._websocket_lag_ms,
                consecutive_failures=self._consecutive_failures,
                last_successful_order=self._last_successful_order,
                execution_allowed=not self._requires_manual_recovery,
                requires_manual_recovery=self._requires_manual_recovery,
                issues=["No order history"]
            )
        
        # Recent orders for calculations
        recent = list(self._order_history)
        
        # Latency
        latencies = [o.latency_ms for o in recent]
        avg_latency = sum(latencies) / len(latencies)
        
        if avg_latency > self.LATENCY_CRITICAL_MS:
            issues.append(f"CRITICAL: Latency {avg_latency:.0f}ms > {self.LATENCY_CRITICAL_MS}ms")
        elif avg_latency > self.LATENCY_WARNING_MS:
            issues.append(f"WARNING: Latency {avg_latency:.0f}ms > {self.LATENCY_WARNING_MS}ms")
        
        # Rejection rate
        rejections = sum(1 for o in recent if o.was_rejected)
        rejection_rate = (rejections / len(recent)) * 100
        
        if rejection_rate > self.REJECTION_CRITICAL_PCT:
            issues.append(f"CRITICAL: Rejection rate {rejection_rate:.1f}% > {self.REJECTION_CRITICAL_PCT}%")
        elif rejection_rate > self.REJECTION_WARNING_PCT:
            issues.append(f"WARNING: Rejection rate {rejection_rate:.1f}% > {self.REJECTION_WARNING_PCT}%")
        
        # Partial fills
        partial_fills = sum(1 for o in recent if o.was_partial_fill)
        partial_rate = (partial_fills / len(recent)) * 100
        
        if partial_rate > self.PARTIAL_CRITICAL_PCT:
            issues.append(f"CRITICAL: Partial fill rate {partial_rate:.1f}% > {self.PARTIAL_CRITICAL_PCT}%")
        elif partial_rate > self.PARTIAL_WARNING_PCT:
            issues.append(f"WARNING: Partial fill rate {partial_rate:.1f}% > {self.PARTIAL_WARNING_PCT}%")
        
        # WebSocket lag
        if self._websocket_lag_ms > self.WS_LAG_CRITICAL_MS:
            issues.append(f"CRITICAL: WS lag {self._websocket_lag_ms:.0f}ms > {self.WS_LAG_CRITICAL_MS}ms")
        elif self._websocket_lag_ms > self.WS_LAG_WARNING_MS:
            issues.append(f"WARNING: WS lag {self._websocket_lag_ms:.0f}ms > {self.WS_LAG_WARNING_MS}ms")
        
        # Determine health level
        has_critical = any("CRITICAL" in i for i in issues)
        has_warning = any("WARNING" in i for i in issues)
        
        if self._requires_manual_recovery or has_critical:
            health_level = HealthLevel.CRITICAL
        elif has_warning:
            health_level = HealthLevel.WARNING
        else:
            health_level = HealthLevel.HEALTHY
        
        # Calculate health score (0.0 to 1.0)
        health_score = self._calculate_health_score(
            avg_latency, rejection_rate, partial_rate, self._websocket_lag_ms
        )
        
        # Execution allowed
        execution_allowed = (
            health_level != HealthLevel.CRITICAL and
            not self._requires_manual_recovery
        )
        
        return ExchangeHealthSnapshot(
            timestamp=now,
            health_level=health_level,
            health_score=health_score,
            avg_latency_ms=avg_latency,
            order_rejection_rate=rejection_rate,
            partial_fill_ratio=partial_rate,
            websocket_lag_ms=self._websocket_lag_ms,
            consecutive_failures=self._consecutive_failures,
            last_successful_order=self._last_successful_order,
            execution_allowed=execution_allowed,
            requires_manual_recovery=self._requires_manual_recovery,
            issues=issues
        )
    
    def request_recovery(self) -> bool:
        """
        Request recovery from critical state.
        
        Must wait for cooldown period before execution is re-enabled.
        
        Returns: True if recovery initiated
        """
        if not self._requires_manual_recovery:
            return True  # Already healthy
        
        now = datetime.now()
        
        # Check cooldown
        if self._recovery_requested_at:
            elapsed = now - self._recovery_requested_at
            if elapsed < timedelta(minutes=self.RECOVERY_COOLDOWN_MINUTES):
                remaining = timedelta(minutes=self.RECOVERY_COOLDOWN_MINUTES) - elapsed
                logger.info(f"Recovery cooldown: {remaining.seconds}s remaining")
                return False
        
        # Complete recovery
        self._requires_manual_recovery = False
        self._consecutive_failures = 0
        self._recovery_requested_at = None
        
        logger.info("Exchange health recovery complete, execution re-enabled")
        return True
    
    def initiate_recovery_cooldown(self):
        """Start recovery cooldown period."""
        self._recovery_requested_at = datetime.now()
        logger.info(
            f"Recovery initiated, {self.RECOVERY_COOLDOWN_MINUTES} minute cooldown started"
        )
    
    def force_critical(self, reason: str):
        """Force entry into critical state."""
        self._requires_manual_recovery = True
        logger.error(f"FORCED CRITICAL STATE: {reason}")
    
    def _calculate_health_score(
        self,
        latency: float,
        rejection_rate: float,
        partial_rate: float,
        ws_lag: float
    ) -> float:
        """Calculate normalized health score."""
        scores = []
        
        # Latency score
        if latency <= self.LATENCY_WARNING_MS:
            scores.append(1.0)
        elif latency <= self.LATENCY_CRITICAL_MS:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # Rejection score
        if rejection_rate <= self.REJECTION_WARNING_PCT:
            scores.append(1.0)
        elif rejection_rate <= self.REJECTION_CRITICAL_PCT:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # Partial fill score
        if partial_rate <= self.PARTIAL_WARNING_PCT:
            scores.append(1.0)
        elif partial_rate <= self.PARTIAL_CRITICAL_PCT:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # WS lag score
        if ws_lag <= self.WS_LAG_WARNING_MS:
            scores.append(1.0)
        elif ws_lag <= self.WS_LAG_CRITICAL_MS:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        return sum(scores) / len(scores)
    
    def get_size_multiplier(self) -> float:
        """
        Get position size multiplier based on health.
        
        Returns: 1.0 for healthy, 0.5 for warning, 0.0 for critical
        """
        health = self.get_health()
        
        if health.health_level == HealthLevel.CRITICAL:
            return 0.0
        elif health.health_level == HealthLevel.WARNING:
            return 0.5
        else:
            return 1.0


# Singleton instance
_exchange_monitor: Optional[ExchangeHealthMonitor] = None


def get_exchange_monitor() -> ExchangeHealthMonitor:
    """Get global ExchangeHealthMonitor instance."""
    global _exchange_monitor
    if _exchange_monitor is None:
        _exchange_monitor = ExchangeHealthMonitor()
    return _exchange_monitor
