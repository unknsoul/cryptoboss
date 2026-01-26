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
