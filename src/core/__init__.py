"""
CryptoBoss Core Module

Clean, professional, institutional-grade architecture components.
Professional discretionary trader architecture implemented.
"""

# === Core Architecture ===
from .state_manager import StateManager, get_state_manager, StrategyState
from .execution_router import (
    ExecutionRouter, ExecutionMode, OrderIntent, OrderResult,
    OrderSide, OrderType, PaperBroker, LiveBroker
)
from .risk_guardian import RiskGuardian, RiskLimits, get_risk_guardian
from .event_bus import (
    EventBus, Event, EventType, get_event_bus,
    emit_price_tick, emit_order_filled, emit_signal, emit_risk_breach
)
from .engine import TradingEngine, create_engine, EngineStatus, EngineConfig

# === Professional Decision Architecture ===
from .market_context_engine import (
    MarketContextEngine, MarketContext, MarketRegime, LiquidityMetrics,
    get_market_context_engine
)
from .bias_engine import (
    BiasEngine, BiasState, TradeBias, get_bias_engine
)
from .trade_permission_filter import (
    TradePermissionFilter, PermissionResult, PermissionDenialReason,
    get_permission_filter
)
from .decision_logger import (
    DecisionLogger, DecisionLog, DecisionType, get_decision_logger
)

# === Phase 6: Live Readiness Hardening ===
from .context_state_machine import (
    ContextStateMachine, ContextState, ContextTransitionEvent,
    ContextStateSnapshot, get_context_state_machine, VALID_TRANSITIONS
)
from .risk_state_persistence import (
    RiskStatePersistence, PersistedRiskState, get_risk_persistence
)
from .trade_budget_manager import (
    TradeBudgetManager, BudgetLimits, BudgetStatus, BudgetType,
    get_budget_manager
)
from .proposal_scorer import (
    ProposalScorer, ScoredProposal, StrategyHealth, get_proposal_scorer
)
from .exchange_health_monitor import (
    ExchangeHealthMonitor, ExchangeHealthSnapshot, HealthLevel,
    EscalationStage, ESCALATION_ORDER, get_exchange_monitor
)
from .cold_start_controller import (
    ColdStartController, ColdStartPhase, ColdStartStatus,
    get_cold_start_controller, reset_cold_start_controller
)
from .replay_engine import (
    DeterministicReplayEngine, ReplaySession, ReplayEvent, ReplayDecision,
    ReplayMismatch, get_replay_engine
)

# === Phase 7: Production Finalization v10.0 ===
from .scoring_contract import (
    ScoringContract, ContractValidatedProposal, ScoreBreakdown,
    ScoreComponent, COMPONENT_WEIGHTS, get_scoring_contract
)
from .ml_containment import (
    MLContainmentManager, MLFeatureOutput, MLInfluenceLog,
    MLOutputType, MLContainmentError, get_ml_containment
)
from .capital_governor import (
    CapitalAllocationGovernor, AllocationSnapshot, AllocationContext,
    DEFAULT_CONTEXT_ALLOCATIONS, get_capital_governor
)

# === Phase 1: Core Stability ===
from .exchange_state import ExchangeStateManager, get_exchange_state, OpenOrder
from .portfolio_risk import PortfolioRiskModel, get_portfolio_risk, RiskMetrics
from .integration_hub import IntegrationHub, get_integration_hub, BaseIntegration
from .persistent_event_bus import PersistentEventBus, create_persistent_event_bus

# === Phase 2: Production Operations ===
from .data_validation import (
    OHLCVCandle, PriceTick, OrderRequest, OrderResponse, Signal,
    DataValidator, get_validator
)
from .graceful_shutdown import GracefulShutdown, get_shutdown_manager
from .observability import ObservabilityManager, get_observability, Metrics
from .secrets_manager import SecretsManager, get_secrets, require_secret
from .config_manager import ConfigManager, get_config

# === Phase 3: Advanced Features ===
from .tax_integration import IntegratedTaxTracker, get_tax_tracker

__all__ = [
    # Core Architecture
    "StateManager", "get_state_manager", "StrategyState",
    "ExecutionRouter", "ExecutionMode", "OrderIntent", "OrderResult",
    "OrderSide", "OrderType", "PaperBroker", "LiveBroker",
    "RiskGuardian", "RiskLimits", "get_risk_guardian",
    "EventBus", "Event", "EventType", "get_event_bus",
    "emit_price_tick", "emit_order_filled", "emit_signal", "emit_risk_breach",
    "TradingEngine", "create_engine", "EngineStatus", "EngineConfig",
    
    # Professional Decision Architecture
    "MarketContextEngine", "MarketContext", "MarketRegime", "LiquidityMetrics",
    "get_market_context_engine",
    "BiasEngine", "BiasState", "TradeBias", "get_bias_engine",
    "TradePermissionFilter", "PermissionResult", "PermissionDenialReason",
    "get_permission_filter",
    "DecisionLogger", "DecisionLog", "DecisionType", "get_decision_logger",
    
    # Phase 6: Live Readiness Hardening
    "ContextStateMachine", "ContextState", "ContextTransitionEvent",
    "ContextStateSnapshot", "get_context_state_machine", "VALID_TRANSITIONS",
    "RiskStatePersistence", "PersistedRiskState", "get_risk_persistence",
    "TradeBudgetManager", "BudgetLimits", "BudgetStatus", "BudgetType",
    "get_budget_manager",
    "ProposalScorer", "ScoredProposal", "StrategyHealth", "get_proposal_scorer",
    "ExchangeHealthMonitor", "ExchangeHealthSnapshot", "HealthLevel",
    "EscalationStage", "ESCALATION_ORDER", "get_exchange_monitor",
    "ColdStartController", "ColdStartPhase", "ColdStartStatus",
    "get_cold_start_controller", "reset_cold_start_controller",
    "DeterministicReplayEngine", "ReplaySession", "ReplayEvent", "ReplayDecision",
    "ReplayMismatch", "get_replay_engine",
    
    # Phase 7: Production Finalization v10.0
    "ScoringContract", "ContractValidatedProposal", "ScoreBreakdown",
    "ScoreComponent", "COMPONENT_WEIGHTS", "get_scoring_contract",
    "MLContainmentManager", "MLFeatureOutput", "MLInfluenceLog",
    "MLOutputType", "MLContainmentError", "get_ml_containment",
    "CapitalAllocationGovernor", "AllocationSnapshot", "AllocationContext",
    "DEFAULT_CONTEXT_ALLOCATIONS", "get_capital_governor",
    
    # Phase 1
    "ExchangeStateManager", "get_exchange_state", "OpenOrder",
    "PortfolioRiskModel", "get_portfolio_risk", "RiskMetrics",
    "IntegrationHub", "get_integration_hub", "BaseIntegration",
    "PersistentEventBus", "create_persistent_event_bus",
    
    # Phase 2
    "OHLCVCandle", "PriceTick", "OrderRequest", "OrderResponse", "Signal",
    "DataValidator", "get_validator",
    "GracefulShutdown", "get_shutdown_manager",
    "ObservabilityManager", "get_observability", "Metrics",
    "SecretsManager", "get_secrets", "require_secret",
    "ConfigManager", "get_config",
    
    # Phase 3
    "IntegratedTaxTracker", "get_tax_tracker",
]

