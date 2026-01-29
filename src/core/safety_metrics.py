"""
Safety Metrics Collector - v10.2-OPERATOR-GRADE

Collects and tracks safety-focused metrics that are displayed
BEFORE profit metrics on the dashboard.

Non-Negotiable Rules:
- Safety metrics displayed before profit metrics
- Metrics reset only on operator acknowledgement
- All safety events are recorded with context
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SafetyEvent:
    """Record of a safety-related event."""
    timestamp: datetime
    event_type: str
    reason: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'reason': self.reason,
            'context': self.context
        }


@dataclass
class SafetyMetrics:
    """Aggregated safety metrics."""
    # Core safety rates
    no_trade_rate: float = 0.0              # % of periods with no trade
    permission_rejection_rate: float = 0.0  # % of proposals rejected by permission filter
    capital_veto_rate: float = 0.0          # % of proposals vetoed by capital governor
    
    # Incident counts
    exchange_degradation_count: int = 0     # Number of times exchange became degraded
    incident_freeze_count: int = 0          # Number of INCIDENT_FREEZE states
    halt_count: int = 0                     # Number of HALTED states
    
    # Time tracking
    last_reset: Optional[datetime] = None
    reset_by: Optional[str] = None
    period_start: Optional[datetime] = None
    
    # Raw counts for rate calculation
    total_candles: int = 0
    no_trade_candles: int = 0
    total_proposals: int = 0
    permission_rejections: int = 0
    capital_vetoes: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'no_trade_rate': round(self.no_trade_rate, 4),
            'permission_rejection_rate': round(self.permission_rejection_rate, 4),
            'capital_veto_rate': round(self.capital_veto_rate, 4),
            'exchange_degradation_count': self.exchange_degradation_count,
            'incident_freeze_count': self.incident_freeze_count,
            'halt_count': self.halt_count,
            'last_reset': self.last_reset.isoformat() if self.last_reset else None,
            'reset_by': self.reset_by,
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'total_candles': self.total_candles,
            'total_proposals': self.total_proposals
        }


class SafetyMetricsCollector:
    """
    Safety Metrics Collector - Tracks system safety indicators.
    
    Collects metrics that indicate how well the system is protecting
    capital and following safety rules.
    
    Usage:
        collector = SafetyMetricsCollector()
        
        # Record events
        collector.record_no_trade("Cold start protection active")
        collector.record_permission_rejection("Exceeded daily loss limit")
        collector.record_capital_veto("NO_TRADE context")
        
        # Get metrics
        metrics = collector.get_metrics()
        
        # Reset requires operator acknowledgment
        collector.acknowledge_and_reset("admin")
    """
    
    def __init__(
        self,
        log_dir: str = "logs/safety",
        max_events: int = 10000
    ):
        self._metrics = SafetyMetrics(
            period_start=datetime.utcnow()
        )
        self._events: deque = deque(maxlen=max_events)
        self._lock = threading.RLock()
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("SafetyMetricsCollector initialized")
    
    def record_candle_processed(self, had_trade: bool) -> None:
        """
        Record that a candle was processed.
        
        Args:
            had_trade: Whether a trade was executed on this candle
        """
        with self._lock:
            self._metrics.total_candles += 1
            if not had_trade:
                self._metrics.no_trade_candles += 1
            
            self._update_rates()
    
    def record_no_trade(self, reason: str, context: Dict = None) -> None:
        """
        Record a no-trade period.
        
        Args:
            reason: Why no trade was taken
            context: Additional context
        """
        with self._lock:
            event = SafetyEvent(
                timestamp=datetime.utcnow(),
                event_type='no_trade',
                reason=reason,
                context=context or {}
            )
            self._events.append(event)
            self._persist_event(event)
    
    def record_proposal(self) -> None:
        """Record that a trade proposal was generated."""
        with self._lock:
            self._metrics.total_proposals += 1
    
    def record_permission_rejection(self, reason: str, context: Dict = None) -> None:
        """
        Record a permission filter rejection.
        
        Args:
            reason: Why proposal was rejected
            context: Additional context
        """
        with self._lock:
            self._metrics.permission_rejections += 1
            self._update_rates()
            
            event = SafetyEvent(
                timestamp=datetime.utcnow(),
                event_type='permission_rejection',
                reason=reason,
                context=context or {}
            )
            self._events.append(event)
            self._persist_event(event)
            
            logger.debug(f"Permission rejection recorded: {reason}")
    
    def record_capital_veto(self, reason: str = None, context: Dict = None) -> None:
        """
        Record a capital governor veto.
        
        Args:
            reason: Why proposal was vetoed
            context: Additional context
        """
        with self._lock:
            self._metrics.capital_vetoes += 1
            self._update_rates()
            
            event = SafetyEvent(
                timestamp=datetime.utcnow(),
                event_type='capital_veto',
                reason=reason or "Capital allocation veto",
                context=context or {}
            )
            self._events.append(event)
            self._persist_event(event)
            
            logger.debug(f"Capital veto recorded: {reason}")
    
    def record_exchange_degradation(self, reason: str, context: Dict = None) -> None:
        """
        Record exchange entering degraded state.
        
        Args:
            reason: Why exchange degraded
            context: Additional context
        """
        with self._lock:
            self._metrics.exchange_degradation_count += 1
            
            event = SafetyEvent(
                timestamp=datetime.utcnow(),
                event_type='exchange_degradation',
                reason=reason,
                context=context or {}
            )
            self._events.append(event)
            self._persist_event(event)
            
            logger.warning(f"Exchange degradation recorded: {reason}")
    
    def record_incident_freeze(self, reason: str, context: Dict = None) -> None:
        """
        Record entering INCIDENT_FREEZE state.
        
        Args:
            reason: Why incident freeze occurred
            context: Additional context
        """
        with self._lock:
            self._metrics.incident_freeze_count += 1
            
            event = SafetyEvent(
                timestamp=datetime.utcnow(),
                event_type='incident_freeze',
                reason=reason,
                context=context or {}
            )
            self._events.append(event)
            self._persist_event(event)
            
            logger.error(f"Incident freeze recorded: {reason}")
    
    def record_halt(self, reason: str, context: Dict = None) -> None:
        """
        Record entering HALTED state.
        
        Args:
            reason: Why halt occurred
            context: Additional context
        """
        with self._lock:
            self._metrics.halt_count += 1
            
            event = SafetyEvent(
                timestamp=datetime.utcnow(),
                event_type='halt',
                reason=reason,
                context=context or {}
            )
            self._events.append(event)
            self._persist_event(event)
            
            logger.critical(f"Halt recorded: {reason}")
    
    def get_metrics(self) -> SafetyMetrics:
        """Get current safety metrics."""
        with self._lock:
            return SafetyMetrics(
                no_trade_rate=self._metrics.no_trade_rate,
                permission_rejection_rate=self._metrics.permission_rejection_rate,
                capital_veto_rate=self._metrics.capital_veto_rate,
                exchange_degradation_count=self._metrics.exchange_degradation_count,
                incident_freeze_count=self._metrics.incident_freeze_count,
                halt_count=self._metrics.halt_count,
                last_reset=self._metrics.last_reset,
                reset_by=self._metrics.reset_by,
                period_start=self._metrics.period_start,
                total_candles=self._metrics.total_candles,
                no_trade_candles=self._metrics.no_trade_candles,
                total_proposals=self._metrics.total_proposals,
                permission_rejections=self._metrics.permission_rejections,
                capital_vetoes=self._metrics.capital_vetoes
            )
    
    def get_metrics_dict(self) -> Dict:
        """Get metrics as dictionary."""
        return self.get_metrics().to_dict()
    
    def get_recent_events(self, count: int = 50, event_type: str = None) -> List[Dict]:
        """
        Get recent safety events.
        
        Args:
            count: Number of events to return
            event_type: Filter by event type (optional)
            
        Returns:
            List of event dictionaries
        """
        with self._lock:
            events = list(self._events)
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            return [e.to_dict() for e in events[-count:]]
    
    def acknowledge_and_reset(self, operator_id: str) -> bool:
        """
        Acknowledge metrics and reset counters.
        
        REQUIRES operator acknowledgment - metrics cannot be auto-reset.
        
        Args:
            operator_id: ID of operator acknowledging the reset
            
        Returns:
            True if reset successful
        """
        with self._lock:
            old_metrics = self.get_metrics_dict()
            
            # Log the acknowledgment
            event = SafetyEvent(
                timestamp=datetime.utcnow(),
                event_type='metrics_reset',
                reason=f"Acknowledged by {operator_id}",
                context={'previous_metrics': old_metrics}
            )
            self._events.append(event)
            self._persist_event(event)
            
            # Reset metrics
            self._metrics = SafetyMetrics(
                last_reset=datetime.utcnow(),
                reset_by=operator_id,
                period_start=datetime.utcnow()
            )
            
            logger.info(
                f"Safety metrics reset by {operator_id}",
                extra={'operator_id': operator_id, 'previous_metrics': old_metrics}
            )
            
            return True
    
    def _update_rates(self) -> None:
        """Update calculated rates."""
        if self._metrics.total_candles > 0:
            self._metrics.no_trade_rate = (
                self._metrics.no_trade_candles / self._metrics.total_candles
            )
        
        if self._metrics.total_proposals > 0:
            self._metrics.permission_rejection_rate = (
                self._metrics.permission_rejections / self._metrics.total_proposals
            )
            self._metrics.capital_veto_rate = (
                self._metrics.capital_vetoes / self._metrics.total_proposals
            )
    
    def _persist_event(self, event: SafetyEvent) -> None:
        """Persist event to disk."""
        try:
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = self._log_dir / f"safety_events_{date_str}.jsonl"
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist safety event: {e}")


# Singleton instance
_safety_metrics: Optional[SafetyMetricsCollector] = None


def get_safety_metrics() -> SafetyMetricsCollector:
    """Get global SafetyMetricsCollector instance."""
    global _safety_metrics
    if _safety_metrics is None:
        _safety_metrics = SafetyMetricsCollector()
    return _safety_metrics


def reset_safety_metrics() -> None:
    """Reset the singleton (for testing)."""
    global _safety_metrics
    _safety_metrics = None
