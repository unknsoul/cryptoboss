"""
Drift Detection Engine - v10.2-OPERATOR-GRADE

Detects divergence between expected and actual system behavior.
Compares live decisions against deterministic replay to catch:
- Non-deterministic behavior
- Logic bugs
- Environmental differences
- Data inconsistencies

Non-Negotiable Rules:
- Alert if divergence exceeds threshold
- Log divergence reason and affected modules
- No silent drift - all mismatches are recorded
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Record of a detected drift between live and expected behavior."""
    timestamp: datetime
    decision_type: str  # 'context', 'bias', 'permission', 'trade'
    live_result: str
    expected_result: str
    divergence_score: float  # 0.0 to 1.0
    affected_modules: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    severity: str = "warning"  # 'info', 'warning', 'critical'
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'decision_type': self.decision_type,
            'live_result': self.live_result,
            'expected_result': self.expected_result,
            'divergence_score': self.divergence_score,
            'affected_modules': self.affected_modules,
            'context': self.context,
            'severity': self.severity
        }


@dataclass
class DriftMetrics:
    """Aggregated drift metrics."""
    total_comparisons: int = 0
    total_divergences: int = 0
    drift_rate: float = 0.0
    last_divergence: Optional[datetime] = None
    divergences_by_type: Dict[str, int] = field(default_factory=dict)
    affected_modules_count: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'total_comparisons': self.total_comparisons,
            'total_divergences': self.total_divergences,
            'drift_rate': self.drift_rate,
            'last_divergence': self.last_divergence.isoformat() if self.last_divergence else None,
            'divergences_by_type': self.divergences_by_type,
            'affected_modules_count': self.affected_modules_count
        }


class DriftDetectionEngine:
    """
    Drift Detection Engine - Behavioral consistency verification.
    
    Compares live trading decisions against replay/expected decisions
    to detect any divergence from deterministic behavior.
    
    Usage:
        detector = DriftDetectionEngine()
        
        # After each decision, compare with expected
        alert = detector.compare_decision(
            live_decision="LONG_BIAS",
            expected_decision="LONG_BIAS",
            decision_type="bias",
            context={'symbol': 'BTC/USDT'}
        )
        
        if alert:
            # Handle drift detected
            logger.warning(f"Drift detected: {alert}")
        
        # Get drift metrics
        metrics = detector.get_metrics()
    """
    
    def __init__(
        self,
        threshold: float = 0.01,  # 1% divergence threshold
        log_dir: str = "logs/drift",
        max_alerts: int = 1000,
        window_size: int = 10000  # Comparisons to track
    ):
        self._threshold = threshold
        self._alerts: deque = deque(maxlen=max_alerts)
        self._comparisons: deque = deque(maxlen=window_size)
        self._lock = threading.RLock()
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self._metrics = DriftMetrics()
        
        # Module importance weights for severity calculation
        self._module_weights = {
            'context': 0.8,
            'bias': 0.7,
            'permission': 0.9,
            'capital': 0.9,
            'trade': 1.0,
            'risk': 1.0
        }
        
        logger.info(f"DriftDetectionEngine initialized with threshold={threshold}")
    
    def compare_decision(
        self,
        live_decision: str,
        expected_decision: str,
        decision_type: str,
        context: Dict[str, Any] = None,
        affected_modules: List[str] = None
    ) -> Optional[DriftAlert]:
        """
        Compare a live decision against expected decision.
        
        Args:
            live_decision: The actual decision made in live trading
            expected_decision: What the decision should have been (from replay)
            decision_type: Type of decision ('context', 'bias', 'permission', 'trade')
            context: Additional context for debugging
            affected_modules: List of modules involved in this decision
            
        Returns:
            DriftAlert if divergence detected, None otherwise
        """
        with self._lock:
            # Record comparison
            is_match = live_decision == expected_decision
            self._comparisons.append({
                'timestamp': datetime.utcnow(),
                'type': decision_type,
                'match': is_match
            })
            
            self._metrics.total_comparisons += 1
            
            if is_match:
                return None
            
            # Divergence detected
            self._metrics.total_divergences += 1
            self._metrics.last_divergence = datetime.utcnow()
            
            # Update divergence by type
            if decision_type not in self._metrics.divergences_by_type:
                self._metrics.divergences_by_type[decision_type] = 0
            self._metrics.divergences_by_type[decision_type] += 1
            
            # Calculate divergence score
            divergence_score = self._calculate_divergence_score(
                live_decision, 
                expected_decision, 
                decision_type
            )
            
            # Determine affected modules
            if affected_modules is None:
                affected_modules = [decision_type]
            
            # Update affected modules count
            for module in affected_modules:
                if module not in self._metrics.affected_modules_count:
                    self._metrics.affected_modules_count[module] = 0
                self._metrics.affected_modules_count[module] += 1
            
            # Determine severity
            severity = self._calculate_severity(divergence_score, decision_type, affected_modules)
            
            # Create alert
            alert = DriftAlert(
                timestamp=datetime.utcnow(),
                decision_type=decision_type,
                live_result=live_decision,
                expected_result=expected_decision,
                divergence_score=divergence_score,
                affected_modules=affected_modules,
                context=context or {},
                severity=severity
            )
            
            self._alerts.append(alert)
            self._persist_alert(alert)
            
            # Update drift rate
            self._update_drift_rate()
            
            logger.warning(
                f"DRIFT DETECTED [{severity.upper()}]: {decision_type} - "
                f"live='{live_decision}' expected='{expected_decision}' "
                f"(score={divergence_score:.3f})",
                extra={
                    'decision_type': decision_type,
                    'divergence_score': divergence_score,
                    'severity': severity
                }
            )
            
            return alert
    
    def get_drift_rate(self) -> float:
        """
        Get current drift rate (divergences / comparisons).
        
        Returns:
            Float between 0.0 and 1.0
        """
        with self._lock:
            return self._metrics.drift_rate
    
    def is_drifting(self) -> bool:
        """Check if current drift rate exceeds threshold."""
        return self.get_drift_rate() > self._threshold
    
    def get_recent_alerts(self, count: int = 50) -> List[Dict]:
        """Get most recent drift alerts."""
        with self._lock:
            alerts = list(self._alerts)[-count:]
            return [a.to_dict() for a in alerts]
    
    def get_alerts_by_type(self, decision_type: str, hours: int = 24) -> List[Dict]:
        """Get drift alerts for a specific decision type."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with self._lock:
            return [
                a.to_dict() 
                for a in self._alerts 
                if a.decision_type == decision_type and a.timestamp >= cutoff
            ]
    
    def get_metrics(self) -> Dict:
        """Get aggregated drift metrics."""
        with self._lock:
            return self._metrics.to_dict()
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set the drift rate threshold for alerts.
        
        Args:
            threshold: Float between 0.0 and 1.0 (e.g., 0.01 = 1%)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self._threshold = threshold
        logger.info(f"Drift threshold updated to {threshold}")
    
    def get_threshold(self) -> float:
        """Get current drift threshold."""
        return self._threshold
    
    def reset_metrics(self) -> None:
        """Reset drift metrics (requires operator acknowledgment in production)."""
        with self._lock:
            self._metrics = DriftMetrics()
            self._comparisons.clear()
            logger.info("Drift metrics reset")
    
    def _calculate_divergence_score(
        self, 
        live: str, 
        expected: str, 
        decision_type: str
    ) -> float:
        """
        Calculate a divergence score based on how different the results are.
        
        Returns a score between 0.0 (identical) and 1.0 (completely different).
        """
        if live == expected:
            return 0.0
        
        # For boolean-like decisions (approved/rejected)
        if decision_type in ['permission', 'trade']:
            # Complete mismatch for approve/reject
            return 1.0
        
        # For categorical decisions (context, bias)
        # Partial score if categories are "close" (e.g., LONG_BIAS vs NEUTRAL)
        live_lower = live.lower()
        expected_lower = expected.lower()
        
        # If one contains the other, partial match
        if live_lower in expected_lower or expected_lower in live_lower:
            return 0.5
        
        # Direction mismatch is more severe
        if ('long' in live_lower and 'short' in expected_lower) or \
           ('short' in live_lower and 'long' in expected_lower):
            return 1.0
        
        # Default divergence
        return 0.7
    
    def _calculate_severity(
        self, 
        divergence_score: float, 
        decision_type: str,
        affected_modules: List[str]
    ) -> str:
        """Calculate alert severity based on divergence and importance."""
        # Weight by module importance
        max_weight = max(
            self._module_weights.get(m, 0.5) 
            for m in affected_modules
        )
        
        weighted_score = divergence_score * max_weight
        
        if weighted_score >= 0.8:
            return 'critical'
        elif weighted_score >= 0.5:
            return 'warning'
        else:
            return 'info'
    
    def _update_drift_rate(self) -> None:
        """Update the rolling drift rate."""
        if len(self._comparisons) == 0:
            self._metrics.drift_rate = 0.0
            return
        
        mismatches = sum(1 for c in self._comparisons if not c['match'])
        self._metrics.drift_rate = mismatches / len(self._comparisons)
    
    def _persist_alert(self, alert: DriftAlert) -> None:
        """Persist alert to disk."""
        try:
            date_str = alert.timestamp.strftime("%Y-%m-%d")
            log_file = self._log_dir / f"drift_alerts_{date_str}.jsonl"
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(alert.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist drift alert: {e}")


# Singleton instance
_drift_detector: Optional[DriftDetectionEngine] = None


def get_drift_detector() -> DriftDetectionEngine:
    """Get global DriftDetectionEngine instance."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetectionEngine()
    return _drift_detector


def reset_drift_detector() -> None:
    """Reset the singleton (for testing)."""
    global _drift_detector
    _drift_detector = None
