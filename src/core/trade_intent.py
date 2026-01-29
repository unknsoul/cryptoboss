"""
Trade Intent - Immutable Trade Request Contract

A TradeIntent is the ONLY way strategies can request trades.
It is:
- Immutable once created
- Produced ONLY by strategies
- Consumed ONLY by the ExecutionFlowOrchestrator
- Never directly converted to orders

v11.0 - Production-Grade Platform Upgrade
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
import hashlib
import json

logger = logging.getLogger(__name__)


class IntentDirection(str, Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


class IntentPriority(str, Enum):
    """Intent priority level."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class IntentStatus(str, Enum):
    """Intent lifecycle status."""
    CREATED = "created"
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TradeIntent:
    """
    Immutable Trade Intent - The only way to request a trade.
    
    This is the formal contract between strategies and the execution pipeline.
    Strategies produce TradeIntents, the pipeline consumes them.
    
    Key Properties:
    - Immutable: Cannot be modified after creation
    - Identifiable: Unique intent_id for tracking
    - Traceable: Full context captured at creation time
    - Auditable: Hashable for integrity verification
    
    Example:
        intent = TradeIntent.create(
            symbol="BTC/USDT",
            direction=IntentDirection.LONG,
            strategy_id="momentum_breakout",
            confidence=0.85,
            reasoning="Price broke resistance with volume confirmation"
        )
    """
    
    # === Core Identity (Required, no defaults) ===
    intent_id: str
    timestamp: datetime
    symbol: str
    direction: IntentDirection
    strategy_id: str
    confidence: float  # 0.0 - 1.0
    reasoning: str  # Human-readable explanation
    
    # === Optional with defaults ===
    strategy_version: str = "1.0"
    signal_strength: float = 0.0  # Raw signal strength from strategy
    
    # === Market Context at Creation ===
    market_regime: str = "unknown"
    directional_bias: str = "neutral"
    volatility_regime: str = "normal"
    
    # === Entry Parameters (Suggested, not final) ===
    suggested_entry: Optional[float] = None
    suggested_stop: Optional[float] = None
    suggested_target: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    
    # === ML Features (Advisory Only) ===
    ml_probability: Optional[float] = None
    ml_confidence: Optional[float] = None
    ml_model_id: Optional[str] = None
    ml_features: tuple = field(default_factory=tuple)  # Immutable tuple of (key, value) pairs
    
    # === Risk Context at Creation ===
    risk_state: str = "unknown"
    portfolio_heat: float = 0.0
    
    # === Metadata ===
    priority: IntentPriority = IntentPriority.NORMAL
    ttl_seconds: int = 300  # Time-to-live before expiration
    tags: tuple = field(default_factory=tuple)  # Immutable tags
    
    # === Integrity ===
    checksum: str = ""
    
    @classmethod
    def create(
        cls,
        symbol: str,
        direction: IntentDirection,
        strategy_id: str,
        confidence: float,
        reasoning: str,
        **kwargs
    ) -> "TradeIntent":
        """
        Factory method to create a TradeIntent with proper initialization.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            direction: Trade direction (long/short)
            strategy_id: Unique strategy identifier
            confidence: Confidence level (0.0 - 1.0)
            reasoning: Human-readable explanation
            **kwargs: Additional optional parameters
            
        Returns:
            Immutable TradeIntent instance
        """
        # Generate unique ID
        intent_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Validate confidence
        confidence = max(0.0, min(1.0, confidence))
        
        # Convert ml_features dict to tuple if provided
        ml_features = kwargs.pop('ml_features', {})
        if isinstance(ml_features, dict):
            ml_features = tuple(sorted(ml_features.items()))
        
        # Convert tags list to tuple if provided
        tags = kwargs.pop('tags', [])
        if isinstance(tags, list):
            tags = tuple(tags)
        
        # Create intent without checksum first
        intent = cls(
            intent_id=intent_id,
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            strategy_id=strategy_id,
            confidence=confidence,
            reasoning=reasoning,
            ml_features=ml_features,
            tags=tags,
            **kwargs
        )
        
        # Calculate checksum (requires creating new instance due to frozen)
        checksum = cls._calculate_checksum(intent)
        
        # Return new instance with checksum
        return cls(
            intent_id=intent_id,
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            strategy_id=strategy_id,
            confidence=confidence,
            reasoning=reasoning,
            ml_features=ml_features,
            tags=tags,
            checksum=checksum,
            **kwargs
        )
    
    @staticmethod
    def _calculate_checksum(intent: "TradeIntent") -> str:
        """Calculate integrity checksum for the intent."""
        data = {
            'intent_id': intent.intent_id,
            'symbol': intent.symbol,
            'direction': intent.direction.value if isinstance(intent.direction, IntentDirection) else intent.direction,
            'strategy_id': intent.strategy_id,
            'confidence': intent.confidence,
            'timestamp': intent.timestamp.isoformat(),
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def verify_integrity(self) -> bool:
        """Verify the intent hasn't been tampered with."""
        if not self.checksum:
            return True  # No checksum to verify
        expected = self._calculate_checksum(self)
        return self.checksum == expected
    
    def is_expired(self) -> bool:
        """Check if the intent has expired."""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl_seconds
    
    def get_ml_features_dict(self) -> Dict[str, float]:
        """Get ML features as a dictionary."""
        return dict(self.ml_features)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'intent_id': self.intent_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction.value if isinstance(self.direction, IntentDirection) else self.direction,
            'strategy_id': self.strategy_id,
            'strategy_version': self.strategy_version,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'signal_strength': self.signal_strength,
            'market_regime': self.market_regime,
            'directional_bias': self.directional_bias,
            'volatility_regime': self.volatility_regime,
            'suggested_entry': self.suggested_entry,
            'suggested_stop': self.suggested_stop,
            'suggested_target': self.suggested_target,
            'risk_reward_ratio': self.risk_reward_ratio,
            'ml_probability': self.ml_probability,
            'ml_confidence': self.ml_confidence,
            'ml_model_id': self.ml_model_id,
            'ml_features': dict(self.ml_features),
            'risk_state': self.risk_state,
            'portfolio_heat': self.portfolio_heat,
            'priority': self.priority.value if isinstance(self.priority, IntentPriority) else self.priority,
            'ttl_seconds': self.ttl_seconds,
            'tags': list(self.tags),
            'checksum': self.checksum,
            'is_expired': self.is_expired(),
            'integrity_valid': self.verify_integrity(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeIntent":
        """Create TradeIntent from dictionary."""
        # Parse enums
        direction = IntentDirection(data['direction'])
        priority = IntentPriority(data.get('priority', 'normal'))
        
        # Parse timestamp
        timestamp = data['timestamp']
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Convert ml_features and tags
        ml_features = tuple(sorted(data.get('ml_features', {}).items()))
        tags = tuple(data.get('tags', []))
        
        return cls(
            intent_id=data['intent_id'],
            timestamp=timestamp,
            symbol=data['symbol'],
            direction=direction,
            strategy_id=data['strategy_id'],
            strategy_version=data.get('strategy_version', '1.0'),
            confidence=data['confidence'],
            reasoning=data['reasoning'],
            signal_strength=data.get('signal_strength', 0.0),
            market_regime=data.get('market_regime', 'unknown'),
            directional_bias=data.get('directional_bias', 'neutral'),
            volatility_regime=data.get('volatility_regime', 'normal'),
            suggested_entry=data.get('suggested_entry'),
            suggested_stop=data.get('suggested_stop'),
            suggested_target=data.get('suggested_target'),
            risk_reward_ratio=data.get('risk_reward_ratio'),
            ml_probability=data.get('ml_probability'),
            ml_confidence=data.get('ml_confidence'),
            ml_model_id=data.get('ml_model_id'),
            ml_features=ml_features,
            risk_state=data.get('risk_state', 'unknown'),
            portfolio_heat=data.get('portfolio_heat', 0.0),
            priority=priority,
            ttl_seconds=data.get('ttl_seconds', 300),
            tags=tags,
            checksum=data.get('checksum', ''),
        )
    
    def __str__(self) -> str:
        return (
            f"TradeIntent({self.intent_id[:8]}... | {self.symbol} {self.direction.value.upper()} | "
            f"strategy={self.strategy_id} | confidence={self.confidence:.2f})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()


class TradeIntentValidator:
    """
    Validates TradeIntent objects before processing.
    
    Ensures intents are:
    - Well-formed with required fields
    - Within acceptable bounds
    - Not expired
    - Integrity verified
    """
    
    MIN_CONFIDENCE = 0.1
    MAX_SYMBOL_LENGTH = 20
    
    @classmethod
    def validate(cls, intent: TradeIntent) -> tuple[bool, str]:
        """
        Validate a TradeIntent.
        
        Returns:
            (is_valid, reason)
        """
        # Check required fields
        if not intent.intent_id:
            return False, "Missing intent_id"
        
        if not intent.symbol:
            return False, "Missing symbol"
        
        if len(intent.symbol) > cls.MAX_SYMBOL_LENGTH:
            return False, f"Symbol too long (max {cls.MAX_SYMBOL_LENGTH})"
        
        if not intent.strategy_id:
            return False, "Missing strategy_id"
        
        if not intent.reasoning:
            return False, "Missing reasoning"
        
        # Check confidence bounds
        if intent.confidence < cls.MIN_CONFIDENCE:
            return False, f"Confidence too low (min {cls.MIN_CONFIDENCE})"
        
        if intent.confidence > 1.0:
            return False, "Confidence exceeds 1.0"
        
        # Check expiration
        if intent.is_expired():
            return False, f"Intent expired (TTL: {intent.ttl_seconds}s)"
        
        # Verify integrity
        if intent.checksum and not intent.verify_integrity():
            return False, "Integrity check failed"
        
        # Check direction
        if intent.direction not in IntentDirection:
            return False, f"Invalid direction: {intent.direction}"
        
        return True, "Valid"
    
    @classmethod
    def validate_batch(cls, intents: List[TradeIntent]) -> tuple[List[TradeIntent], List[tuple[TradeIntent, str]]]:
        """
        Validate a batch of intents.
        
        Returns:
            (valid_intents, rejected_intents_with_reasons)
        """
        valid = []
        rejected = []
        
        for intent in intents:
            is_valid, reason = cls.validate(intent)
            if is_valid:
                valid.append(intent)
            else:
                rejected.append((intent, reason))
                logger.warning(f"Intent rejected: {intent.intent_id} - {reason}")
        
        return valid, rejected


class TradeIntentRegistry:
    """
    Registry for tracking TradeIntent lifecycle.
    
    Provides:
    - Intent status tracking
    - Statistics collection
    - History lookup
    """
    
    def __init__(self, max_history: int = 10000):
        self._intents: Dict[str, TradeIntent] = {}
        self._status: Dict[str, IntentStatus] = {}
        self._history: List[str] = []
        self._max_history = max_history
        self._stats = {
            'total_created': 0,
            'total_approved': 0,
            'total_rejected': 0,
            'total_expired': 0,
            'by_strategy': {},
            'by_direction': {'long': 0, 'short': 0},
        }
    
    def register(self, intent: TradeIntent) -> None:
        """Register a new intent."""
        self._intents[intent.intent_id] = intent
        self._status[intent.intent_id] = IntentStatus.CREATED
        self._history.append(intent.intent_id)
        
        # Update stats
        self._stats['total_created'] += 1
        self._stats['by_direction'][intent.direction.value] += 1
        
        if intent.strategy_id not in self._stats['by_strategy']:
            self._stats['by_strategy'][intent.strategy_id] = {'created': 0, 'approved': 0, 'rejected': 0}
        self._stats['by_strategy'][intent.strategy_id]['created'] += 1
        
        # Trim history if needed
        if len(self._history) > self._max_history:
            old_id = self._history.pop(0)
            if old_id in self._intents:
                del self._intents[old_id]
            if old_id in self._status:
                del self._status[old_id]
    
    def update_status(self, intent_id: str, status: IntentStatus) -> None:
        """Update intent status."""
        if intent_id in self._status:
            old_status = self._status[intent_id]
            self._status[intent_id] = status
            
            # Update stats
            intent = self._intents.get(intent_id)
            if intent:
                if status == IntentStatus.APPROVED:
                    self._stats['total_approved'] += 1
                    self._stats['by_strategy'][intent.strategy_id]['approved'] += 1
                elif status == IntentStatus.REJECTED:
                    self._stats['total_rejected'] += 1
                    self._stats['by_strategy'][intent.strategy_id]['rejected'] += 1
                elif status == IntentStatus.EXPIRED:
                    self._stats['total_expired'] += 1
    
    def get_intent(self, intent_id: str) -> Optional[TradeIntent]:
        """Get intent by ID."""
        return self._intents.get(intent_id)
    
    def get_status(self, intent_id: str) -> Optional[IntentStatus]:
        """Get intent status."""
        return self._status.get(intent_id)
    
    def get_pending_intents(self) -> List[TradeIntent]:
        """Get all intents in processing state."""
        return [
            self._intents[id] 
            for id, status in self._status.items() 
            if status in (IntentStatus.CREATED, IntentStatus.SUBMITTED, IntentStatus.PROCESSING)
            and id in self._intents
        ]
    
    def get_stats(self) -> Dict:
        """Get registry statistics."""
        return {
            **self._stats,
            'current_registered': len(self._intents),
            'approval_rate': (
                self._stats['total_approved'] / self._stats['total_created'] 
                if self._stats['total_created'] > 0 else 0
            ),
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired intents. Returns count removed."""
        expired_ids = [
            id for id, intent in self._intents.items()
            if intent.is_expired() and self._status.get(id) not in (
                IntentStatus.APPROVED, IntentStatus.EXECUTED
            )
        ]
        
        for id in expired_ids:
            self.update_status(id, IntentStatus.EXPIRED)
            del self._intents[id]
        
        return len(expired_ids)


# Singleton registry
_intent_registry: Optional[TradeIntentRegistry] = None


def get_intent_registry() -> TradeIntentRegistry:
    """Get the global TradeIntentRegistry instance."""
    global _intent_registry
    if _intent_registry is None:
        _intent_registry = TradeIntentRegistry()
    return _intent_registry
