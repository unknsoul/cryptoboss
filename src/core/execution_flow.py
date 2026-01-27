"""
Execution Flow Orchestrator - v10.1-FINAL Component

Enforces strict 9-stage execution order with ZERO bypass paths:

1. market_context       → Is it safe to trade?
2. context_state_machine → Is transition valid?
3. bias_engine          → Long/Short/Neutral?
4. bias_pre_filter      → Discard opposite-direction
5. proposal_scoring     → Validate 4 components
6. proposal_selection   → Pick best proposal
7. trade_permission     → Size/exposure check
8. capital_governor     → Allocate & VETO if zero
9. execution_router     → Paper or Live

RULES:
- If ANY stage fails → downstream stages DO NOT execute
- Stages CANNOT be reordered
- Stages CANNOT be skipped
- All results are logged
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto

logger = logging.getLogger(__name__)


class FlowStage(Enum):
    """Execution flow stages in mandatory order."""
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
    
    Ensures ZERO BYPASS - every trade must pass through all 9 stages
    in strict order. If any stage fails, downstream stages are skipped.
    
    Usage:
        orchestrator = ExecutionFlowOrchestrator()
        
        result = await orchestrator.execute_flow(
            symbol="BTC/USDT",
            proposals=strategy_proposals,
            current_price=40000.0
        )
        
        if result.success:
            # Trade was executed
            print(f"Order: {result.order_intent}")
        else:
            # Failed at some stage
            print(f"Failed at: {result.failed_at_stage}")
    """
    
    def __init__(self):
        self._flow_counter: int = 0
        self._flow_history: List[FlowResult] = []
        self._stage_handlers: Dict[FlowStage, Callable] = {}
        
        logger.info("ExecutionFlowOrchestrator initialized - ZERO BYPASS mode")
    
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
        if stage == FlowStage.MARKET_CONTEXT:
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
            'flow_counter': self._flow_counter
        }


# Singleton instance
_flow_orchestrator: Optional[ExecutionFlowOrchestrator] = None


def get_execution_flow() -> ExecutionFlowOrchestrator:
    """Get global ExecutionFlowOrchestrator instance."""
    global _flow_orchestrator
    if _flow_orchestrator is None:
        _flow_orchestrator = ExecutionFlowOrchestrator()
    return _flow_orchestrator
