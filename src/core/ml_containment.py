"""
ML Output Containment - v10.0 Component

Ensures machine learning models are treated as feature providers only:
- ML outputs are features, NOT proposals
- ML cannot generate EntryProposal objects
- ML influence is explicitly logged

Critical for maintaining deterministic, explainable trading decisions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class MLOutputType(Enum):
    """Types of ML outputs (all are features, none are proposals)."""
    PREDICTION = "prediction"           # Price/direction prediction
    CONFIDENCE = "confidence"           # Model confidence score
    FEATURE = "feature"                 # Derived feature value
    EMBEDDING = "embedding"             # Feature embedding
    ANOMALY_SCORE = "anomaly_score"     # Anomaly detection score


@dataclass
class MLFeatureOutput:
    """
    ML model output - treated as FEATURE ONLY.
    
    This is NOT a trade proposal and cannot be used to
    directly generate trading decisions.
    """
    model_id: str
    model_version: str
    output_type: MLOutputType
    feature_name: str
    value: float
    confidence: float  # Model's confidence in this output
    timestamp: datetime
    
    # Metadata for logging
    input_features: Dict[str, float] = field(default_factory=dict)
    computation_time_ms: float = 0.0
    
    # CRITICAL: This is NEVER a proposal
    is_proposal: bool = field(default=False, init=False)
    
    def to_dict(self) -> Dict:
        return {
            'model_id': self.model_id,
            'model_version': self.model_version,
            'output_type': self.output_type.value,
            'feature_name': self.feature_name,
            'value': self.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'is_proposal': self.is_proposal,  # Always False
            'computation_time_ms': self.computation_time_ms
        }


class MLContainmentError(Exception):
    """Raised when ML attempts to violate containment rules."""
    pass


@dataclass
class MLInfluenceLog:
    """Logged when ML output influences a decision."""
    timestamp: datetime
    decision_id: str
    model_id: str
    feature_name: str
    feature_value: float
    influence_weight: float
    decision_component: str  # Which component used this feature
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'decision_id': self.decision_id,
            'model_id': self.model_id,
            'feature_name': self.feature_name,
            'feature_value': self.feature_value,
            'influence_weight': self.influence_weight,
            'decision_component': self.decision_component
        }


class MLContainmentManager:
    """
    Manages ML output containment and influence logging.
    
    Rules:
    1. ML outputs are treated as features ONLY
    2. ML cannot generate EntryProposal objects
    3. ML influence must be explicitly logged
    4. Any attempt to use ML for direct trading is blocked
    
    Usage:
        containment = MLContainmentManager()
        
        # Register ML output as feature
        feature = containment.register_feature(
            model_id="price_predictor_v3",
            feature_name="predicted_direction",
            value=0.75,
            confidence=0.82
        )
        
        # Use in scoring (logged)
        containment.log_influence(
            decision_id="trade_123",
            feature=feature,
            influence_weight=0.15,
            decision_component="context_fit"
        )
    """
    
    # Blocked patterns that suggest proposal generation
    BLOCKED_FEATURE_NAMES = [
        'entry_signal',
        'trade_action',
        'buy_signal',
        'sell_signal',
        'entry_price',
        'position_size',
        'stop_loss',
        'take_profit'
    ]
    
    def __init__(self, max_ml_influence: float = 0.30):
        """
        Initialize containment manager.
        
        Args:
            max_ml_influence: Maximum allowed ML influence on any decision (0.0-1.0)
        """
        self.max_ml_influence = max_ml_influence
        self._registered_features: Dict[str, MLFeatureOutput] = {}
        self._influence_log: List[MLInfluenceLog] = []
        self._violation_count: int = 0
        
        logger.info(
            f"MLContainmentManager initialized (max_influence={max_ml_influence})"
        )
    
    def register_feature(
        self,
        model_id: str,
        feature_name: str,
        value: float,
        confidence: float,
        output_type: MLOutputType = MLOutputType.PREDICTION,
        model_version: str = "1.0",
        input_features: Optional[Dict[str, float]] = None
    ) -> MLFeatureOutput:
        """
        Register an ML output as a feature.
        
        This validates the output is not attempting to be a proposal.
        """
        # Check for blocked patterns
        feature_lower = feature_name.lower()
        for blocked in self.BLOCKED_FEATURE_NAMES:
            if blocked in feature_lower:
                self._violation_count += 1
                logger.error(
                    f"ML CONTAINMENT VIOLATION: Feature name '{feature_name}' "
                    f"contains blocked pattern '{blocked}'"
                )
                raise MLContainmentError(
                    f"Feature '{feature_name}' violates containment rules. "
                    f"ML cannot generate trading signals directly."
                )
        
        # Create feature output
        feature = MLFeatureOutput(
            model_id=model_id,
            model_version=model_version,
            output_type=output_type,
            feature_name=feature_name,
            value=value,
            confidence=confidence,
            timestamp=datetime.now(),
            input_features=input_features or {}
        )
        
        # Store for reference
        key = f"{model_id}:{feature_name}"
        self._registered_features[key] = feature
        
        logger.debug(
            f"ML feature registered: {model_id}/{feature_name} = {value:.4f}"
        )
        
        return feature
    
    def log_influence(
        self,
        decision_id: str,
        feature: MLFeatureOutput,
        influence_weight: float,
        decision_component: str
    ) -> MLInfluenceLog:
        """
        Log when an ML feature influences a decision.
        
        This provides explainability for ML impact on trading.
        """
        # Validate influence weight
        if influence_weight > self.max_ml_influence:
            logger.warning(
                f"ML influence {influence_weight:.2f} exceeds max {self.max_ml_influence:.2f}, "
                f"clamping to max"
            )
            influence_weight = self.max_ml_influence
        
        log_entry = MLInfluenceLog(
            timestamp=datetime.now(),
            decision_id=decision_id,
            model_id=feature.model_id,
            feature_name=feature.feature_name,
            feature_value=feature.value,
            influence_weight=influence_weight,
            decision_component=decision_component
        )
        
        self._influence_log.append(log_entry)
        
        logger.debug(
            f"ML influence logged: {feature.model_id} -> {decision_component} "
            f"(weight={influence_weight:.2f})"
        )
        
        return log_entry
    
    def get_feature(self, model_id: str, feature_name: str) -> Optional[MLFeatureOutput]:
        """Get a registered feature by model and name."""
        key = f"{model_id}:{feature_name}"
        return self._registered_features.get(key)
    
    def get_all_features(self) -> List[MLFeatureOutput]:
        """Get all registered features."""
        return list(self._registered_features.values())
    
    def get_influence_report(
        self,
        decision_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get influence log entries.
        
        Args:
            decision_id: Filter by specific decision
            limit: Maximum entries to return
        """
        logs = self._influence_log
        
        if decision_id:
            logs = [l for l in logs if l.decision_id == decision_id]
        
        return [l.to_dict() for l in logs[-limit:]]
    
    def get_total_ml_influence(self, decision_id: str) -> float:
        """Calculate total ML influence on a specific decision."""
        total = sum(
            l.influence_weight
            for l in self._influence_log
            if l.decision_id == decision_id
        )
        return min(total, 1.0)
    
    def validate_no_proposal(self, obj: Any) -> bool:
        """
        Validate that an object is not a proposal.
        
        Returns True if valid (not a proposal), raises if violation.
        """
        # Check for proposal-like attributes
        proposal_attributes = ['entry_price', 'stop_loss', 'take_profit', 'direction', 'size']
        
        if hasattr(obj, '__dict__'):
            obj_attrs = set(obj.__dict__.keys())
            matches = obj_attrs.intersection(proposal_attributes)
            
            if len(matches) >= 3:  # Has 3+ proposal attributes
                self._violation_count += 1
                raise MLContainmentError(
                    f"Object appears to be a proposal (has {matches}). "
                    f"ML cannot generate proposals."
                )
        
        return True
    
    def get_violation_count(self) -> int:
        """Get total containment violations detected."""
        return self._violation_count
    
    def clear_features(self):
        """Clear all registered features (for new trading session)."""
        self._registered_features.clear()
        logger.info("ML features cleared for new session")


# Singleton instance
_ml_containment: Optional[MLContainmentManager] = None


def get_ml_containment(max_influence: float = 0.30) -> MLContainmentManager:
    """Get global MLContainmentManager instance."""
    global _ml_containment
    if _ml_containment is None:
        _ml_containment = MLContainmentManager(max_ml_influence=max_influence)
    return _ml_containment
