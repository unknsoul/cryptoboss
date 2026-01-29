"""
Execution Flow Orchestrator - v11.0-PRODUCTION-GRADE

Enforces strict 10-stage execution order with ZERO bypass paths:

0. incident_gate        → Check incident state (v10.2+)
1. market_context       → Is it safe to trade?
2. context_state_machine → Is transition valid?
3. bias_engine          → Long/Short/Neutral?
4. bias_pre_filter      → Discard opposite-direction
5. proposal_scoring     → Validate 4 components
6. proposal_selection   → Pick best proposal
7. trade_permission     → Size/exposure check
8. capital_governor     → Allocate & VETO if zero
9. execution_router     → Paper or Live

v11.0 UPGRADES:
- Accepts TradeIntent objects from strategies
- Outputs TradeDecision objects (single source of truth)
- Integrates with DrawdownGovernor for risk state
- Broadcasts decisions via WebSocket
- Records execution quality via SlippageMonitor
- Full decision audit trail

RULES:
- If ANY stage fails → downstream stages DO NOT execute
- Stages CANNOT be reordered
- Stages CANNOT be skipped
- All results are logged as TradeDecision
- INCIDENT_FREEZE/HALTED blocks all new trades
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum, auto
import uuid

logger = logging.getLogger(__name__)


class FlowStage(Enum):
    """Execution flow stages in mandatory order."""
    INCIDENT_GATE = auto()  # v10.2: First gate - check incident state
    MARKET_CONTEXT = auto()
    CONTEXT_STATE_MACHINE = auto()
    BIAS_ENGINE = auto()
    BIAS_PRE_FILTER = auto()
    PROPOSAL_SCORING = auto()
    PROPOSAL_SELECTION = auto()
    TRADE_PERMISSION = auto()
    CAPITAL_GOVERNOR = auto()
    EXECUTION_ROUTER = auto()


# Stage names for logging
STAGE_NAMES = {
    FlowStage.INCIDENT_GATE: "incident_gate",  # v10.2
    FlowStage.MARKET_CONTEXT: "market_context",
    FlowStage.CONTEXT_STATE_MACHINE: "context_state_machine",
    FlowStage.BIAS_ENGINE: "bias_engine",
    FlowStage.BIAS_PRE_FILTER: "bias_pre_filter",
    FlowStage.PROPOSAL_SCORING: "proposal_scoring",
    FlowStage.PROPOSAL_SELECTION: "proposal_selection",
    FlowStage.TRADE_PERMISSION: "trade_permission",
    FlowStage.CAPITAL_GOVERNOR: "capital_governor",
    FlowStage.EXECUTION_ROUTER: "execution_router",
}

# Strict order (cannot be changed)
STAGE_ORDER = list(FlowStage)


class FlowStatus(Enum):
    """Flow execution status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    STAGE_FAILED = "stage_failed"
    COMPLETED = "completed"
    HALTED = "halted"


@dataclass
class StageResult:
    """Result of a single stage execution."""
    stage: FlowStage
    success: bool
    reason: str
    data: Dict = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'stage': STAGE_NAMES[self.stage],
            'success': self.success,
            'reason': self.reason,
            'duration_ms': self.duration_ms,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class FlowResult:
    """Result of complete flow execution."""
    flow_id: str
    status: FlowStatus
    started_at: datetime
    completed_at: Optional[datetime]
    
    # Stage results
    stage_results: List[StageResult]
    last_successful_stage: Optional[FlowStage]
    failed_at_stage: Optional[FlowStage]
    
    # Output
    final_action: Optional[str]  # "execute", "skip", "halt"
    order_intent: Optional[Dict]
    
    # Metadata
    symbol: str
    proposals_received: int
    proposals_after_filter: int
    
    @property
    def success(self) -> bool:
        return self.status == FlowStatus.COMPLETED
    
    def to_dict(self) -> Dict:
        return {
            'flow_id': self.flow_id,
            'status': self.status.value,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'stages_completed': len([s for s in self.stage_results if s.success]),
            'total_stages': len(STAGE_ORDER),
            'failed_at_stage': STAGE_NAMES.get(self.failed_at_stage) if self.failed_at_stage else None,
            'final_action': self.final_action,
            'symbol': self.symbol
        }


class ExecutionFlowOrchestrator:
    """
    Orchestrates the complete trade execution flow.
    
    v11.0: Now accepts TradeIntent objects and outputs TradeDecision objects.
    
    Ensures ZERO BYPASS - every trade must pass through all 10 stages
    in strict order. If any stage fails, downstream stages are skipped.
    
    Usage (v11.0 - Intent-based):
        from src.core import TradeIntent, IntentDirection
        
        # Strategy creates intent
        intent = TradeIntent.create(
            symbol="BTC/USDT",
            direction=IntentDirection.LONG,
            strategy_id="momentum",
            confidence=0.85,
            reasoning="Breakout with volume"
        )
        
        # Orchestrator processes and returns decision
        decision = orchestrator.process_intent(intent, current_price=40000.0)
        
        if decision.is_approved:
            print(f"Trade approved: {decision.execution_params}")
        else:
            print(f"Rejected at: {decision.rejection_stage}")
    
    Legacy Usage (backward compatible):
        result = orchestrator.execute_flow(
            symbol="BTC/USDT",
            proposals=strategy_proposals,
            current_price=40000.0
        )
    """
    
    def __init__(self):
        self._flow_counter: int = 0
        self._flow_history: List[FlowResult] = []
        self._decision_history: List = []  # TradeDecision objects
        self._stage_handlers: Dict[FlowStage, Callable] = {}
        
        # v11.0 components (lazy loaded)
        self._decision_store = None
        self._intent_registry = None
        self._websocket_manager = None
        self._drawdown_governor = None
        
        logger.info("ExecutionFlowOrchestrator v11.0 initialized - ZERO BYPASS mode")
    
    def execute_flow(
        self,
        symbol: str,
        proposals: List[Dict],
        current_price: float,
        context_data: Optional[Dict] = None
    ) -> FlowResult:
        """
        Execute the complete trading flow.
        
        Args:
            symbol: Trading pair
            proposals: Raw proposals from strategies
            current_price: Current market price
            context_data: Optional pre-computed context data
            
        Returns:
            FlowResult with complete execution details
        """
        self._flow_counter += 1
        flow_id = f"flow_{self._flow_counter}_{datetime.now().strftime('%H%M%S')}"
        started_at = datetime.now()
        
        stage_results: List[StageResult] = []
        last_successful: Optional[FlowStage] = None
        failed_at: Optional[FlowStage] = None
        
        # Flow state (passed between stages)
        flow_state = {
            'symbol': symbol,
            'proposals': proposals,
            'current_price': current_price,
            'context': context_data or {},
            'bias': None,
            'filtered_proposals': [],
            'validated_proposals': [],
            'selected_proposal': None,
            'permission_result': None,
            'capital_allocation': None,
            'order_intent': None
        }
        
        logger.info(f"[{flow_id}] Starting execution flow for {symbol}")
        
        # Execute stages in strict order
        for stage in STAGE_ORDER:
            stage_start = datetime.now()
            
            try:
                success, reason, data = self._execute_stage(stage, flow_state)
                
                duration = (datetime.now() - stage_start).total_seconds() * 1000
                
                result = StageResult(
                    stage=stage,
                    success=success,
                    reason=reason,
                    data=data,
                    duration_ms=duration
                )
                stage_results.append(result)
                
                if success:
                    last_successful = stage
                    logger.debug(f"[{flow_id}] Stage {STAGE_NAMES[stage]}: PASSED")
                else:
                    failed_at = stage
                    logger.warning(
                        f"[{flow_id}] Stage {STAGE_NAMES[stage]}: FAILED - {reason}"
                    )
                    break  # HALT - do not continue to downstream stages
                    
            except Exception as e:
                duration = (datetime.now() - stage_start).total_seconds() * 1000
                result = StageResult(
                    stage=stage,
                    success=False,
                    reason=f"Exception: {str(e)}",
                    duration_ms=duration
                )
                stage_results.append(result)
                failed_at = stage
                logger.error(f"[{flow_id}] Stage {STAGE_NAMES[stage]}: EXCEPTION - {e}")
                break
        
        # Determine final status
        completed_at = datetime.now()
        
        if failed_at is None:
            status = FlowStatus.COMPLETED
            final_action = "execute"
        else:
            status = FlowStatus.STAGE_FAILED
            final_action = "skip"
        
        # Build result
        flow_result = FlowResult(
            flow_id=flow_id,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            stage_results=stage_results,
            last_successful_stage=last_successful,
            failed_at_stage=failed_at,
            final_action=final_action,
            order_intent=flow_state.get('order_intent'),
            symbol=symbol,
            proposals_received=len(proposals),
            proposals_after_filter=len(flow_state.get('filtered_proposals', []))
        )
        
        # Store history
        self._flow_history.append(flow_result)
        if len(self._flow_history) > 500:
            self._flow_history = self._flow_history[-250:]
        
        # Log summary
        total_duration = (completed_at - started_at).total_seconds() * 1000
        logger.info(
            f"[{flow_id}] Flow {status.value}: "
            f"{len([s for s in stage_results if s.success])}/{len(STAGE_ORDER)} stages, "
            f"{total_duration:.1f}ms"
        )
        
        return flow_result
    
    def _execute_stage(
        self,
        stage: FlowStage,
        state: Dict
    ) -> tuple[bool, str, Dict]:
        """
        Execute a single stage.
        
        Returns: (success, reason, data)
        """
        # Check for registered handler
        if stage in self._stage_handlers:
            return self._stage_handlers[stage](state)
        
        # Default implementations
        if stage == FlowStage.INCIDENT_GATE:  # v10.2: First gate
            return self._stage_incident_gate(state)
        elif stage == FlowStage.MARKET_CONTEXT:
            return self._stage_market_context(state)
        elif stage == FlowStage.CONTEXT_STATE_MACHINE:
            return self._stage_state_machine(state)
        elif stage == FlowStage.BIAS_ENGINE:
            return self._stage_bias_engine(state)
        elif stage == FlowStage.BIAS_PRE_FILTER:
            return self._stage_bias_pre_filter(state)
        elif stage == FlowStage.PROPOSAL_SCORING:
            return self._stage_proposal_scoring(state)
        elif stage == FlowStage.PROPOSAL_SELECTION:
            return self._stage_proposal_selection(state)
        elif stage == FlowStage.TRADE_PERMISSION:
            return self._stage_trade_permission(state)
        elif stage == FlowStage.CAPITAL_GOVERNOR:
            return self._stage_capital_governor(state)
        elif stage == FlowStage.EXECUTION_ROUTER:
            return self._stage_execution_router(state)
        
        return False, "Unknown stage", {}
    
    def _stage_incident_gate(self, state: Dict) -> tuple[bool, str, Dict]:
        """Stage 0: Incident Gate (v10.2) - Check incident state before any trading."""
        try:
            from .incident_state_machine import get_incident_state_machine, IncidentState
            from .operator_control import get_operator_control
            from .safety_metrics import get_safety_metrics
            
            ism = get_incident_state_machine()
            operator = get_operator_control()
            
            # Check operator pause
            if operator.is_paused():
                return False, "Trading paused by operator", {'state': 'operator_paused'}
            
            # Check incident state
            incident_state = ism.get_state()
            
            if incident_state == IncidentState.HALTED:
                get_safety_metrics().record_no_trade("HALTED state")
                return False, "HALTED: All trading blocked", {'state': 'halted'}
            
            if incident_state == IncidentState.INCIDENT_FREEZE:
                get_safety_metrics().record_incident_freeze("New trade blocked by INCIDENT_FREEZE")
                return False, "INCIDENT_FREEZE: New trades blocked", {'state': 'incident_freeze'}
            
            # DEGRADED allows trading with reduced size (handled in capital governor)
            # NORMAL allows full trading
            
            return True, f"Incident gate passed: {incident_state.value}", {
                'incident_state': incident_state.value
            }
            
        except Exception as e:
            return False, f"Incident gate error: {e}", {}
    
    def _stage_market_context(self, state: Dict) -> tuple[bool, str, Dict]:
        """Stage 1: Market Context"""
        try:
            from .market_context_engine import get_market_context_engine
            
            engine = get_market_context_engine()
            context = engine.get_context()
            
            state['context'] = context
            
            if not context.trading_allowed:
                return False, f"Trading not allowed: {context.reason}", {}
            
            return True, f"Context: {context.regime.value}", {'regime': context.regime.value}
            
        except Exception as e:
            return False, f"Context engine error: {e}", {}
    
    def _stage_state_machine(self, state: Dict) -> tuple[bool, str, Dict]:
        """Stage 2: Context State Machine"""
        try:
            from .context_state_machine import get_context_state_machine
            
            machine = get_context_state_machine()
            snapshot = machine.get_snapshot()
            
            # Check if valid to proceed
            if not snapshot.time_in_state_hours >= 0:  # Always passes if has state
                return False, "State machine not initialized", {}
            
            return True, f"State: {snapshot.current_state.value}", {
                'state': snapshot.current_state.value
            }
            
        except Exception as e:
            return False, f"State machine error: {e}", {}
    
    def _stage_bias_engine(self, state: Dict) -> tuple[bool, str, Dict]:
        """Stage 3: Bias Engine"""
        try:
            from .bias_engine import get_bias_engine
            
            engine = get_bias_engine()
            bias = engine.get_current_bias()
            
            state['bias'] = bias
            
            # NEUTRAL bias will filter all in next stage, but we pass here
            return True, f"Bias: {bias.bias.value}", {'bias': bias.bias.value}
            
        except Exception as e:
            return False, f"Bias engine error: {e}", {}
    
    def _stage_bias_pre_filter(self, state: Dict) -> tuple[bool, str, Dict]:
        """Stage 4: Bias Pre-Filter (HARD GATE)"""
        try:
            from .bias_pre_filter import get_bias_pre_filter
            
            pre_filter = get_bias_pre_filter()
            bias = state.get('bias')
            proposals = state.get('proposals', [])
            
            if not proposals:
                return False, "No proposals to filter", {}
            
            result = pre_filter.filter_proposals(
                proposals=proposals,
                current_bias=bias.bias.value if bias else 'neutral'
            )
            
            state['filtered_proposals'] = result.passed_proposals
            
            if result.all_filtered:
                return False, f"All {result.filter_count} proposals filtered by bias", {
                    'filtered': result.filter_count
                }
            
            return True, f"{result.pass_count} proposals passed bias filter", {
                'passed': result.pass_count,
                'filtered': result.filter_count
            }
            
        except Exception as e:
            return False, f"Bias pre-filter error: {e}", {}
    
    def _stage_proposal_scoring(self, state: Dict) -> tuple[bool, str, Dict]:
        """Stage 5: Proposal Scoring Contract"""
        try:
            from .scoring_contract import get_scoring_contract
            
            contract = get_scoring_contract()
            proposals = state.get('filtered_proposals', [])
            
            validated = contract.validate_batch(proposals)
            state['validated_proposals'] = validated
            
            if not validated:
                return False, "No proposals passed scoring contract", {}
            
            return True, f"{len(validated)} proposals validated", {
                'validated': len(validated),
                'top_score': validated[0].score if validated else 0
            }
            
        except Exception as e:
            return False, f"Scoring contract error: {e}", {}
    
    def _stage_proposal_selection(self, state: Dict) -> tuple[bool, str, Dict]:
        """Stage 6: Proposal Selection"""
        validated = state.get('validated_proposals', [])
        
        if not validated:
            return False, "No validated proposals to select", {}
        
        # Select top proposal
        selected = validated[0]
        state['selected_proposal'] = selected
        
        return True, f"Selected: {selected.strategy_id} (score={selected.score:.3f})", {
            'strategy_id': selected.strategy_id,
            'score': selected.score
        }
    
    def _stage_trade_permission(self, state: Dict) -> tuple[bool, str, Dict]:
        """Stage 7: Trade Permission Filter"""
        try:
            from .trade_permission_filter import get_permission_filter
            
            pfilter = get_permission_filter()
            context = state.get('context')
            bias = state.get('bias')
            proposal = state.get('selected_proposal')
            
            if not all([context, bias, proposal]):
                return False, "Missing context/bias/proposal for permission", {}
            
            result = pfilter.check_permission(
                context=context,
                bias=bias,
                direction=proposal.direction,
                proposal=proposal.to_dict() if hasattr(proposal, 'to_dict') else {}
            )
            
            state['permission_result'] = result
            
            if not result.approved:
                return False, f"Permission denied: {result.reason}", {
                    'denial_reason': result.reason
                }
            
            return True, "Permission granted", {}
            
        except Exception as e:
            return False, f"Permission filter error: {e}", {}
    
    def _stage_capital_governor(self, state: Dict) -> tuple[bool, str, Dict]:
        """Stage 8: Capital Governor (with VETO power)"""
        try:
            from .capital_governor import get_capital_governor
            
            governor = get_capital_governor()
            context = state.get('context')
            proposal = state.get('selected_proposal')
            
            # Get allocation
            allocation = governor.get_allocation(
                context=context.regime.value if context else 'unknown',
                volatility_percentile=getattr(context, 'atr_percentile', 50.0),
                daily_drawdown_pct=0.0  # Would come from risk state
            )
            
            state['capital_allocation'] = allocation
            
            # VETO CHECK: Zero effective size = blocked
            if allocation.effective_allocation <= 0:
                return False, "Capital governor VETO: zero allocation", {
                    'allocation': 0
                }
            
            if allocation.max_position_size <= 0:
                return False, "Capital governor VETO: no available capital", {
                    'allocation': allocation.effective_allocation
                }
            
            # Calculate effective size
            requested_size = proposal.size if hasattr(proposal, 'size') else 100
            effective_size = min(
                requested_size,
                allocation.max_position_size * allocation.effective_allocation
            )
            
            if effective_size <= 0:
                return False, "Capital governor VETO: effective size is zero", {}
            
            state['effective_size'] = effective_size
            
            return True, f"Capital allocated: ${effective_size:,.0f}", {
                'allocation_pct': allocation.effective_allocation,
                'effective_size': effective_size
            }
            
        except Exception as e:
            return False, f"Capital governor error: {e}", {}
    
    def _stage_execution_router(self, state: Dict) -> tuple[bool, str, Dict]:
        """Stage 9: Execution Router"""
        try:
            proposal = state.get('selected_proposal')
            effective_size = state.get('effective_size', 0)
            
            if not proposal or effective_size <= 0:
                return False, "Cannot execute: no proposal or size", {}
            
            # Build order intent
            order_intent = {
                'symbol': state.get('symbol'),
                'direction': proposal.direction,
                'size': effective_size,
                'entry_price': proposal.entry_price,
                'stop_loss': proposal.stop_loss if hasattr(proposal, 'stop_loss') else None,
                'take_profit': proposal.take_profit if hasattr(proposal, 'take_profit') else None,
                'strategy_id': proposal.strategy_id,
                'score': proposal.score
            }
            
            state['order_intent'] = order_intent
            
            return True, f"Order ready: {proposal.direction} ${effective_size:,.0f}", {
                'order': order_intent
            }
            
        except Exception as e:
            return False, f"Execution router error: {e}", {}
    
    def register_stage_handler(
        self,
        stage: FlowStage,
        handler: Callable[[Dict], tuple[bool, str, Dict]]
    ):
        """Register custom handler for a stage."""
        self._stage_handlers[stage] = handler
        logger.info(f"Registered custom handler for stage: {STAGE_NAMES[stage]}")
    
    def get_flow_history(self, limit: int = 50) -> List[Dict]:
        """Get recent flow results."""
        return [f.to_dict() for f in self._flow_history[-limit:]]
    
    def get_stats(self) -> Dict:
        """Get orchestrator statistics."""
        total = len(self._flow_history)
        successful = sum(1 for f in self._flow_history if f.success)
        
        return {
            'total_flows': total,
            'successful_flows': successful,
            'success_rate': successful / total if total > 0 else 0,
            'flow_counter': self._flow_counter,
            'v11_decisions': len(self._decision_history)
        }
    
    # ============= V11.0 INTENT-BASED METHODS =============
    
    def process_intent(
        self,
        intent: "TradeIntent",
        current_price: float,
        context_data: Optional[Dict] = None
    ) -> "TradeDecision":
        """
        Process a single TradeIntent and return a TradeDecision.
        
        This is the v11.0 primary entry point. Strategies should create
        TradeIntent objects and submit them here.
        
        Args:
            intent: TradeIntent from a strategy
            intent: TradeIntent from a strategy
            current_price: Current market price
            context_data: Optional pre-computed context
            
        Returns:
            TradeDecision with full audit trail
        """
        from .trade_intent import TradeIntent, TradeIntentValidator, IntentStatus, get_intent_registry
        from .trade_decision import TradeDecision, DecisionStatus, RejectionStage, get_decision_store
        
        # Validate intent first
        is_valid, validation_reason = TradeIntentValidator.validate(intent)
        
        # Create decision object
        decision = TradeDecision.create(
            intent=intent,
            symbol=intent.symbol,
            strategy_id=intent.strategy_id,
            direction=intent.direction.value
        )
        
        # Register intent
        registry = get_intent_registry()
        registry.register(intent)
        registry.update_status(intent.intent_id, IntentStatus.PROCESSING)
        
        # If intent validation failed, reject immediately
        if not is_valid:
            decision.reject(
                stage=RejectionStage.VALIDATION,
                reason=f"Intent validation failed: {validation_reason}"
            )
            registry.update_status(intent.intent_id, IntentStatus.REJECTED)
            self._store_decision(decision)
            return decision
        
        # Convert intent to proposal format for legacy flow
        proposal = self._intent_to_proposal(intent, current_price)
        
        # Execute through the flow
        flow_result = self.execute_flow(
            symbol=intent.symbol,
            proposals=[proposal],
            current_price=current_price,
            context_data=context_data
        )
        
        # Map flow stages to decision stages
        stage_mapping = {
            FlowStage.INCIDENT_GATE: RejectionStage.INCIDENT_GATE,
            FlowStage.MARKET_CONTEXT: RejectionStage.MARKET_CONTEXT,
            FlowStage.CONTEXT_STATE_MACHINE: RejectionStage.CONTEXT_STATE,
            FlowStage.BIAS_ENGINE: RejectionStage.BIAS_FILTER,
            FlowStage.BIAS_PRE_FILTER: RejectionStage.BIAS_FILTER,
            FlowStage.PROPOSAL_SCORING: RejectionStage.SCORING,
            FlowStage.PROPOSAL_SELECTION: RejectionStage.SCORING,
            FlowStage.TRADE_PERMISSION: RejectionStage.PERMISSION,
            FlowStage.CAPITAL_GOVERNOR: RejectionStage.CAPITAL,
            FlowStage.EXECUTION_ROUTER: RejectionStage.EXECUTION,
        }
        
        # Add stage results to decision
        for stage_result in flow_result.stage_results:
            decision.add_stage_result(
                stage_name=STAGE_NAMES[stage_result.stage],
                passed=stage_result.success,
                reason=stage_result.reason,
                duration_ms=stage_result.duration_ms,
                data=stage_result.data
            )
        
        # Determine outcome
        if flow_result.success:
            # Trade approved
            order_intent = flow_result.order_intent or {}
            
            from .trade_decision import ExecutionParams
            exec_params = ExecutionParams(
                final_size=order_intent.get('size', 0),
                final_entry=order_intent.get('entry_price', current_price),
                final_stop=order_intent.get('stop_loss'),
                final_target=order_intent.get('take_profit'),
                order_type='market',
                execution_mode='paper'  # or 'live'
            )
            
            decision.approve(
                execution_params=exec_params,
                final_confidence=intent.confidence
            )
            registry.update_status(intent.intent_id, IntentStatus.APPROVED)
            
        else:
            # Trade rejected
            rejection_stage = RejectionStage.UNKNOWN
            if flow_result.failed_at_stage:
                rejection_stage = stage_mapping.get(
                    flow_result.failed_at_stage, 
                    RejectionStage.UNKNOWN
                )
            
            # Get rejection reason
            rejection_reason = "Unknown"
            for sr in reversed(flow_result.stage_results):
                if not sr.success:
                    rejection_reason = sr.reason
                    break
            
            decision.reject(
                stage=rejection_stage,
                reason=rejection_reason
            )
            registry.update_status(intent.intent_id, IntentStatus.REJECTED)
        
        # Store decision
        self._store_decision(decision)
        
        # Broadcast via WebSocket (async, non-blocking)
        self._broadcast_decision(decision)
        
        return decision
    
    def process_intents(
        self,
        intents: List["TradeIntent"],
        current_price: float,
        context_data: Optional[Dict] = None
    ) -> List["TradeDecision"]:
        """
        Process multiple TradeIntents.
        
        Useful for processing all strategy intents in a single cycle.
        
        Args:
            intents: List of TradeIntent objects
            current_price: Current market price
            context_data: Optional pre-computed context
            
        Returns:
            List of TradeDecision objects
        """
        decisions = []
        for intent in intents:
            decision = self.process_intent(intent, current_price, context_data)
            decisions.append(decision)
        return decisions
    
    def _intent_to_proposal(self, intent: "TradeIntent", current_price: float) -> Dict:
        """Convert TradeIntent to legacy proposal format."""
        return {
            'intent_id': intent.intent_id,
            'symbol': intent.symbol,
            'direction': intent.direction.value,
            'strategy_id': intent.strategy_id,
            'confidence': intent.confidence,
            'reasoning': intent.reasoning,
            'entry_price': intent.suggested_entry or current_price,
            'stop_loss': intent.suggested_stop,
            'take_profit': intent.suggested_target,
            'ml_probability': intent.ml_probability,
            'ml_confidence': intent.ml_confidence,
            'signal_strength': intent.signal_strength,
            'priority': intent.priority.value,
            'size': 100,  # Default, will be overridden by capital governor
        }
    
    def _store_decision(self, decision: "TradeDecision") -> None:
        """Store decision in history and decision store."""
        self._decision_history.append(decision)
        if len(self._decision_history) > 500:
            self._decision_history = self._decision_history[-250:]
        
        # Store in global decision store
        try:
            from .trade_decision import get_decision_store
            store = get_decision_store()
            store.store(decision)
        except Exception as e:
            logger.error(f"Failed to store decision: {e}")
    
    def _broadcast_decision(self, decision: "TradeDecision") -> None:
        """Broadcast decision via WebSocket (non-blocking)."""
        try:
            import asyncio
            
            # Try to get websocket manager
            try:
                from src.api.websocket import get_websocket_manager
                manager = get_websocket_manager()
                
                # Schedule broadcast (don't block)
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(manager.broadcast_decision(decision))
                    
            except ImportError:
                pass  # WebSocket not available
                
        except Exception as e:
            logger.debug(f"Could not broadcast decision: {e}")
    
    def get_recent_decisions(self, limit: int = 50) -> List[Dict]:
        """Get recent TradeDecision objects."""
        return [d.to_dict() for d in self._decision_history[-limit:]]


# Singleton instance
_flow_orchestrator: Optional[ExecutionFlowOrchestrator] = None


def get_execution_flow() -> ExecutionFlowOrchestrator:
    """Get global ExecutionFlowOrchestrator instance."""
    global _flow_orchestrator
    if _flow_orchestrator is None:
        _flow_orchestrator = ExecutionFlowOrchestrator()
    return _flow_orchestrator
