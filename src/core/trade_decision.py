"""
Trade Decision - Complete Decision Contract

A TradeDecision is the output of the execution pipeline.
It captures:
- The original intent
- All pipeline stage results
- Final approval/rejection status
- Execution parameters if approved

This is the SINGLE SOURCE OF TRUTH for why any trade happened (or didn't).

v11.0 - Production-Grade Platform Upgrade
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
import json

from .trade_intent import TradeIntent, IntentDirection

logger = logging.getLogger(__name__)


class DecisionStatus(str, Enum):
    """Final decision status."""
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


class RiskState(str, Enum):
    """Risk state at decision time."""
    SAFE = "safe"
    ELEVATED = "elevated"
    RESTRICTED = "restricted"
    HALTED = "halted"


class RejectionStage(str, Enum):
    """Stage at which rejection occurred."""
    INCIDENT_GATE = "incident_gate"
    MARKET_CONTEXT = "market_context"
    CONTEXT_STATE = "context_state"  # Alias for context_state_machine
    CONTEXT_STATE_MACHINE = "context_state_machine"
    BIAS_ENGINE = "bias_engine"
    BIAS_FILTER = "bias_filter"  # Alias for bias_pre_filter
    BIAS_PRE_FILTER = "bias_pre_filter"
    SCORING = "scoring"  # Combined scoring stage
    PROPOSAL_SCORING = "proposal_scoring"
    PROPOSAL_SELECTION = "proposal_selection"
    PERMISSION = "permission"  # Alias for trade_permission
    TRADE_PERMISSION = "trade_permission"
    CAPITAL = "capital"  # Alias for capital_governor
    CAPITAL_GOVERNOR = "capital_governor"
    EXECUTION = "execution"  # Alias for execution_router
    EXECUTION_ROUTER = "execution_router"
    VALIDATION = "validation"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass
class StageResultRecord:
    """Record of a single pipeline stage execution."""
    stage: str
    passed: bool
    reason: str
    duration_ms: float
    timestamp: datetime
    data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'stage': self.stage,
            'passed': self.passed,
            'reason': self.reason,
            'duration_ms': self.duration_ms,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
        }


@dataclass
class ExecutionParams:
    """Final execution parameters for approved trades."""
    position_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: float = 1.0
    order_type: str = "limit"
    time_in_force: str = "GTC"
    reduce_only: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'leverage': self.leverage,
            'order_type': self.order_type,
            'time_in_force': self.time_in_force,
            'reduce_only': self.reduce_only,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ExecutionParams":
        return cls(**data)


@dataclass
class TradeDecision:
    """
    Complete Trade Decision - The output of the execution pipeline.
    
    This is the formal contract that documents every trading decision.
    Every TradeIntent results in exactly one TradeDecision.
    
    Key Properties:
    - Complete: Contains full context from all pipeline stages
    - Explainable: Clear rejection reasons if not approved
    - Auditable: Full trace of decision-making process
    - Immutable: Cannot be modified after creation
    
    Example (approved):
        decision = TradeDecision(
            intent=intent,
            status=DecisionStatus.APPROVED,
            execution_params=ExecutionParams(
                position_size=0.1,
                entry_price=45000.0,
                stop_loss=44000.0,
                take_profit=47000.0
            )
        )
    
    Example (rejected):
        decision = TradeDecision(
            intent=intent,
            status=DecisionStatus.REJECTED,
            rejection_stage=RejectionStage.TRADE_PERMISSION,
            rejection_reason="Daily drawdown limit reached"
        )
    """
    
    # === Core Identity ===
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # === Source Intent ===
    intent: Optional[TradeIntent] = None
    intent_id: str = ""  # Backup reference if intent not stored
    
    # === Symbol & Direction (denormalized for quick access) ===
    symbol: str = ""
    direction: str = ""
    
    # === Pipeline Context ===
    market_regime: str = "unknown"
    directional_bias: str = "neutral"
    bias_confidence: float = 0.0
    volatility_regime: str = "normal"
    
    # === Strategy Analysis ===
    strategy_contributors: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    
    # === ML Analysis ===
    ml_probability: Optional[float] = None
    ml_confidence: Optional[float] = None
    ml_model_id: Optional[str] = None
    ml_influence: float = 0.0  # 0.0 - 1.0, how much ML affected decision
    
    # === Risk Assessment ===
    risk_state: RiskState = RiskState.SAFE
    portfolio_heat: float = 0.0
    drawdown_current: float = 0.0
    drawdown_limit: float = 5.0
    
    # === Decision Result ===
    status: DecisionStatus = DecisionStatus.REJECTED
    rejection_stage: Optional[RejectionStage] = None
    rejection_reason: Optional[str] = None
    
    # === Execution Parameters (only if approved) ===
    execution_params: Optional[ExecutionParams] = None
    
    # === Pipeline Trace ===
    stage_results: List[StageResultRecord] = field(default_factory=list)
    total_pipeline_ms: float = 0.0
    
    # === Execution Result (filled after execution) ===
    executed: bool = False
    execution_timestamp: Optional[datetime] = None
    fill_price: Optional[float] = None
    fill_size: Optional[float] = None
    slippage_bps: Optional[float] = None
    order_id: Optional[str] = None
    
    # === Metadata ===
    engine_version: str = "11.0"
    mode: str = "paper"  # 'paper' or 'live'
    
    @classmethod
    def create(
        cls,
        intent: Optional["TradeIntent"] = None,
        symbol: str = "",
        strategy_id: str = "",
        direction: str = "",
        **kwargs
    ) -> "TradeDecision":
        """
        Factory method to create a TradeDecision.
        
        Args:
            intent: Optional TradeIntent (will auto-populate fields)
            symbol: Trading pair
            strategy_id: Strategy that generated the intent
            direction: 'long' or 'short'
            **kwargs: Additional TradeDecision fields
            
        Returns:
            New TradeDecision instance
        """
        decision = cls(
            intent=intent,
            symbol=symbol or (intent.symbol if intent else ""),
            direction=direction or (intent.direction.value if intent else ""),
            **kwargs
        )
        
        if intent:
            decision.strategy_contributors = [strategy_id or intent.strategy_id]
        elif strategy_id:
            decision.strategy_contributors = [strategy_id]
            
        return decision
    
    def __post_init__(self):
        """Initialize from intent if provided."""
        if self.intent:
            self.intent_id = self.intent.intent_id
            self.symbol = self.intent.symbol
            self.direction = self.intent.direction.value if isinstance(self.intent.direction, IntentDirection) else self.intent.direction
            self.market_regime = self.intent.market_regime
            self.directional_bias = self.intent.directional_bias
            self.volatility_regime = self.intent.volatility_regime
            self.confidence_score = self.intent.confidence
            self.strategy_contributors = [self.intent.strategy_id]
            self.ml_probability = self.intent.ml_probability
            self.ml_confidence = self.intent.ml_confidence
            self.ml_model_id = self.intent.ml_model_id
    
    @property
    def is_approved(self) -> bool:
        """Check if decision was approved."""
        return self.status == DecisionStatus.APPROVED
    
    @property
    def is_rejected(self) -> bool:
        """Check if decision was rejected."""
        return self.status == DecisionStatus.REJECTED
    
    @property
    def is_executed(self) -> bool:
        """Check if decision was executed."""
        return self.executed
    
    def add_stage_result(
        self,
        stage_name: str,
        passed: bool,
        reason: str,
        duration_ms: float = 0.0,
        data: Dict = None
    ) -> None:
        """Add a stage result to the trace."""
        self.stage_results.append(StageResultRecord(
            stage=stage_name,
            passed=passed,
            reason=reason,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            data=data or {}
        ))
        self.total_pipeline_ms += duration_ms
    
    def reject(self, stage: RejectionStage, reason: str) -> None:
        """Mark decision as rejected."""
        self.status = DecisionStatus.REJECTED
        self.rejection_stage = stage
        self.rejection_reason = reason
        logger.info(f"Decision {self.decision_id[:8]} rejected at {stage.value}: {reason}")
    
    def approve(
        self, 
        execution_params: "ExecutionParams" = None,
        final_confidence: float = None,
        **kwargs
    ) -> None:
        """Mark decision as approved."""
        self.status = DecisionStatus.APPROVED
        
        # Handle execution params - can be passed directly or via kwargs
        if execution_params:
            self.execution_params = execution_params
        elif kwargs:
            # Build ExecutionParams from kwargs
            self.execution_params = ExecutionParams(
                position_size=kwargs.get('final_size', kwargs.get('position_size', 0)),
                entry_price=kwargs.get('final_entry', kwargs.get('entry_price', 0)),
                stop_loss=kwargs.get('final_stop', kwargs.get('stop_loss', 0)) or 0,
                take_profit=kwargs.get('final_target', kwargs.get('take_profit', 0)) or 0,
                order_type=kwargs.get('order_type', 'market'),
            )
        
        if final_confidence is not None:
            self.confidence_score = final_confidence
            
        self.rejection_stage = None
        self.rejection_reason = None
        
        size_str = self.execution_params.position_size if self.execution_params else 'N/A'
        price_str = self.execution_params.entry_price if self.execution_params else 'N/A'
        logger.info(f"Decision {self.decision_id[:8]} approved: {size_str} @ {price_str}")
    
    def record_execution(
        self,
        order_id: str,
        fill_price: float,
        fill_size: float,
        expected_price: float
    ) -> None:
        """Record execution result."""
        self.executed = True
        self.execution_timestamp = datetime.now()
        self.order_id = order_id
        self.fill_price = fill_price
        self.fill_size = fill_size
        
        # Calculate slippage in basis points
        if expected_price > 0:
            slippage = ((fill_price - expected_price) / expected_price) * 10000
            # Negative slippage is good for buys, bad for sells (and vice versa)
            if self.direction == "short":
                slippage = -slippage
            self.slippage_bps = slippage
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        if self.is_approved:
            return (
                f"APPROVED: {self.symbol} {self.direction.upper()} | "
                f"Size: {self.execution_params.position_size if self.execution_params else 'N/A'} | "
                f"Confidence: {self.confidence_score:.2f}"
            )
        else:
            return (
                f"REJECTED: {self.symbol} {self.direction.upper()} | "
                f"Stage: {self.rejection_stage.value if self.rejection_stage else 'N/A'} | "
                f"Reason: {self.rejection_reason}"
            )
    
    def get_stage_trace(self) -> List[Dict]:
        """Get the full stage trace for debugging."""
        return [sr.to_dict() for sr in self.stage_results]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'decision_id': self.decision_id,
            'timestamp': self.timestamp.isoformat(),
            'intent_id': self.intent_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'market_regime': self.market_regime,
            'directional_bias': self.directional_bias,
            'bias_confidence': self.bias_confidence,
            'volatility_regime': self.volatility_regime,
            'strategy_contributors': self.strategy_contributors,
            'confidence_score': self.confidence_score,
            'ml_probability': self.ml_probability,
            'ml_confidence': self.ml_confidence,
            'ml_model_id': self.ml_model_id,
            'ml_influence': self.ml_influence,
            'risk_state': self.risk_state.value if isinstance(self.risk_state, RiskState) else self.risk_state,
            'portfolio_heat': self.portfolio_heat,
            'drawdown_current': self.drawdown_current,
            'drawdown_limit': self.drawdown_limit,
            'status': self.status.value if isinstance(self.status, DecisionStatus) else self.status,
            'rejection_stage': self.rejection_stage.value if self.rejection_stage else None,
            'rejection_reason': self.rejection_reason,
            'execution_params': self.execution_params.to_dict() if self.execution_params else None,
            'stage_results': [sr.to_dict() for sr in self.stage_results],
            'total_pipeline_ms': self.total_pipeline_ms,
            'executed': self.executed,
            'execution_timestamp': self.execution_timestamp.isoformat() if self.execution_timestamp else None,
            'fill_price': self.fill_price,
            'fill_size': self.fill_size,
            'slippage_bps': self.slippage_bps,
            'order_id': self.order_id,
            'engine_version': self.engine_version,
            'mode': self.mode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeDecision":
        """Create TradeDecision from dictionary."""
        # Parse enums
        status = DecisionStatus(data['status'])
        risk_state = RiskState(data.get('risk_state', 'safe'))
        rejection_stage = RejectionStage(data['rejection_stage']) if data.get('rejection_stage') else None
        
        # Parse timestamp
        timestamp = data['timestamp']
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        execution_timestamp = data.get('execution_timestamp')
        if execution_timestamp and isinstance(execution_timestamp, str):
            execution_timestamp = datetime.fromisoformat(execution_timestamp)
        
        # Parse execution params
        execution_params = None
        if data.get('execution_params'):
            execution_params = ExecutionParams.from_dict(data['execution_params'])
        
        # Parse stage results
        stage_results = []
        for sr_data in data.get('stage_results', []):
            sr_timestamp = sr_data['timestamp']
            if isinstance(sr_timestamp, str):
                sr_timestamp = datetime.fromisoformat(sr_timestamp)
            stage_results.append(StageResultRecord(
                stage=sr_data['stage'],
                passed=sr_data['passed'],
                reason=sr_data['reason'],
                duration_ms=sr_data['duration_ms'],
                timestamp=sr_timestamp,
                data=sr_data.get('data', {})
            ))
        
        return cls(
            decision_id=data['decision_id'],
            timestamp=timestamp,
            intent_id=data.get('intent_id', ''),
            symbol=data['symbol'],
            direction=data['direction'],
            market_regime=data.get('market_regime', 'unknown'),
            directional_bias=data.get('directional_bias', 'neutral'),
            bias_confidence=data.get('bias_confidence', 0.0),
            volatility_regime=data.get('volatility_regime', 'normal'),
            strategy_contributors=data.get('strategy_contributors', []),
            confidence_score=data.get('confidence_score', 0.0),
            ml_probability=data.get('ml_probability'),
            ml_confidence=data.get('ml_confidence'),
            ml_model_id=data.get('ml_model_id'),
            ml_influence=data.get('ml_influence', 0.0),
            risk_state=risk_state,
            portfolio_heat=data.get('portfolio_heat', 0.0),
            drawdown_current=data.get('drawdown_current', 0.0),
            drawdown_limit=data.get('drawdown_limit', 5.0),
            status=status,
            rejection_stage=rejection_stage,
            rejection_reason=data.get('rejection_reason'),
            execution_params=execution_params,
            stage_results=stage_results,
            total_pipeline_ms=data.get('total_pipeline_ms', 0.0),
            executed=data.get('executed', False),
            execution_timestamp=execution_timestamp,
            fill_price=data.get('fill_price'),
            fill_size=data.get('fill_size'),
            slippage_bps=data.get('slippage_bps'),
            order_id=data.get('order_id'),
            engine_version=data.get('engine_version', '11.0'),
            mode=data.get('mode', 'paper'),
        )
    
    def __str__(self) -> str:
        status_icon = "✓" if self.is_approved else "✗"
        return (
            f"TradeDecision({status_icon} {self.decision_id[:8]}... | "
            f"{self.symbol} {self.direction.upper()} | "
            f"{self.status.value})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()


class DecisionStore:
    """
    In-memory store for trade decisions with persistence support.
    
    Provides:
    - Decision storage and retrieval
    - Statistics computation
    - Export to JSONL for audit
    """
    
    def __init__(self, max_decisions: int = 50000, persist_path: str = "logs/decisions"):
        self._decisions: Dict[str, TradeDecision] = {}
        self._by_symbol: Dict[str, List[str]] = {}
        self._by_status: Dict[DecisionStatus, List[str]] = {s: [] for s in DecisionStatus}
        self._history: List[str] = []
        self._max_decisions = max_decisions
        self._persist_path = persist_path
        
        self._stats = {
            'total_decisions': 0,
            'total_approved': 0,
            'total_rejected': 0,
            'total_executed': 0,
            'by_rejection_stage': {},
            'by_strategy': {},
            'avg_pipeline_ms': 0.0,
            'avg_slippage_bps': 0.0,
        }
    
    def store(self, decision: TradeDecision) -> None:
        """Store a decision."""
        self._decisions[decision.decision_id] = decision
        self._history.append(decision.decision_id)
        
        # Index by symbol
        if decision.symbol not in self._by_symbol:
            self._by_symbol[decision.symbol] = []
        self._by_symbol[decision.symbol].append(decision.decision_id)
        
        # Index by status
        self._by_status[decision.status].append(decision.decision_id)
        
        # Update stats
        self._stats['total_decisions'] += 1
        if decision.is_approved:
            self._stats['total_approved'] += 1
        else:
            self._stats['total_rejected'] += 1
            if decision.rejection_stage:
                stage_key = decision.rejection_stage.value
                self._stats['by_rejection_stage'][stage_key] = \
                    self._stats['by_rejection_stage'].get(stage_key, 0) + 1
        
        # Track by strategy
        for strategy_id in decision.strategy_contributors:
            if strategy_id not in self._stats['by_strategy']:
                self._stats['by_strategy'][strategy_id] = {'approved': 0, 'rejected': 0, 'executed': 0}
            if decision.is_approved:
                self._stats['by_strategy'][strategy_id]['approved'] += 1
            else:
                self._stats['by_strategy'][strategy_id]['rejected'] += 1
        
        # Update average pipeline time
        n = self._stats['total_decisions']
        old_avg = self._stats['avg_pipeline_ms']
        self._stats['avg_pipeline_ms'] = old_avg + (decision.total_pipeline_ms - old_avg) / n
        
        # Trim if needed
        self._trim()
    
    def update_execution(self, decision_id: str, order_id: str, fill_price: float, fill_size: float, expected_price: float) -> None:
        """Update decision with execution result."""
        decision = self._decisions.get(decision_id)
        if decision:
            decision.record_execution(order_id, fill_price, fill_size, expected_price)
            self._stats['total_executed'] += 1
            
            # Update slippage average
            if decision.slippage_bps is not None:
                n = self._stats['total_executed']
                old_avg = self._stats['avg_slippage_bps']
                self._stats['avg_slippage_bps'] = old_avg + (decision.slippage_bps - old_avg) / n
            
            # Update strategy stats
            for strategy_id in decision.strategy_contributors:
                if strategy_id in self._stats['by_strategy']:
                    self._stats['by_strategy'][strategy_id]['executed'] += 1
    
    def get(self, decision_id: str) -> Optional[TradeDecision]:
        """Get decision by ID."""
        return self._decisions.get(decision_id)
    
    def get_by_symbol(self, symbol: str, limit: int = 100) -> List[TradeDecision]:
        """Get recent decisions for a symbol."""
        ids = self._by_symbol.get(symbol, [])[-limit:]
        return [self._decisions[id] for id in ids if id in self._decisions]
    
    def get_recent(self, limit: int = 100, status: DecisionStatus = None) -> List[TradeDecision]:
        """Get recent decisions."""
        if status:
            ids = self._by_status[status][-limit:]
        else:
            ids = self._history[-limit:]
        return [self._decisions[id] for id in ids if id in self._decisions]
    
    def get_stats(self) -> Dict:
        """Get decision statistics."""
        return {
            **self._stats,
            'current_stored': len(self._decisions),
            'approval_rate': (
                self._stats['total_approved'] / self._stats['total_decisions']
                if self._stats['total_decisions'] > 0 else 0
            ),
            'execution_rate': (
                self._stats['total_executed'] / self._stats['total_approved']
                if self._stats['total_approved'] > 0 else 0
            ),
        }
    
    def _trim(self) -> None:
        """Remove old decisions if over limit."""
        while len(self._history) > self._max_decisions:
            old_id = self._history.pop(0)
            if old_id in self._decisions:
                del self._decisions[old_id]
    
    def export_jsonl(self, filepath: str = None) -> str:
        """Export all decisions to JSONL file."""
        import os
        from pathlib import Path
        
        if filepath is None:
            Path(self._persist_path).mkdir(parents=True, exist_ok=True)
            filepath = os.path.join(
                self._persist_path,
                f"decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            )
        
        with open(filepath, 'w') as f:
            for decision in self._decisions.values():
                f.write(json.dumps(decision.to_dict()) + '\n')
        
        logger.info(f"Exported {len(self._decisions)} decisions to {filepath}")
        return filepath


# Singleton store
_decision_store: Optional[DecisionStore] = None


def get_decision_store() -> DecisionStore:
    """Get the global DecisionStore instance."""
    global _decision_store
    if _decision_store is None:
        _decision_store = DecisionStore()
    return _decision_store
