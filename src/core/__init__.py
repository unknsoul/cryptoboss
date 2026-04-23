"""
CryptoBoss v11.0 - Production-Grade Trading Platform

A context-first, risk-governed, event-driven autonomous trading engine.

Core Philosophy:
- No Direct Orders: Strategies produce TradeIntent, never direct orders
- Risk Sovereignty: Risk engine has veto over all trade intents
- ML Containment: ML is advisory only, cannot generate proposals
- Zero Bypass: Every trade passes through all 10 pipeline stages
- Full Auditability: Every decision logged with complete context
- Operator Control: Critical actions require operator acknowledgment

Version: 11.0-FINAL
"""

# ============================================================================
# v11.0 Trade Intent/Decision Pipeline
# ============================================================================

from .trade_intent import (
    TradeIntent,
    IntentDirection,
    IntentPriority,
    TradeIntentValidator,
)

from .trade_decision import (
    TradeDecision,
    DecisionStatus,
    RejectionStage,
    RiskState,
    ExecutionParams,
    StageResultRecord,
    DecisionStore,
    get_decision_store,
)

# ============================================================================
# Execution Flow Orchestrator
# ============================================================================

from .execution_flow import (
    ExecutionFlowOrchestrator,
    FlowStage,
    FlowResult,
    get_execution_flow,
)

# ============================================================================
# Risk Management
# ============================================================================

from .drawdown_governor import (
    DrawdownGovernor,
    DrawdownSeverity,
    get_drawdown_governor,
)

from .capital_governor import (
    CapitalAllocationGovernor,
    AllocationContext,
    AllocationSnapshot,
)

from .portfolio_risk import (
    PortfolioRiskModel,
    PortfolioPosition,
    RiskMetrics,
    get_portfolio_risk,
)

try:
    from .trade_permission_filter import (
        TradePermissionFilter,
        PermissionStatus,
    )
except ImportError:
    TradePermissionFilter = None
    PermissionStatus = None

# ============================================================================
# Execution Hardening
# ============================================================================

from .slippage_monitor import (
    SlippageMonitor,
    SlippageRecord,
    get_slippage_monitor,
)

from .exchange_recovery import (
    ExchangeRecoveryHandler,
    RecoveryResult,
    ErrorCategory,
    get_recovery_handler,
)

from .execution_router import (
    ExecutionRouter,
    ExecutionMode,
    OrderIntent,
    OrderResult,
)

# ============================================================================
# Market Context & Analysis
# ============================================================================

from .market_context_engine import (
    MarketContextEngine,
)

try:
    from .context_state_machine import (
        ContextStateMachine,
        ContextState,
        get_context_state_machine,
    )
    # Alias for backward compatibility
    MarketState = ContextState
except ImportError:
    ContextStateMachine = None
    ContextState = None
    MarketState = None
    get_context_state_machine = None

from .bias_engine import (
    BiasEngine,
)

from .scoring_contract import (
    ScoringContract,
)

try:
    from .trade_management_engine import (
        ManagedTrade,
        PositionSide,
        TPLevel,
        TradeManagementDecision,
        TradeManagementEngine,
    )
except ImportError:
    ManagedTrade = None
    PositionSide = None
    TPLevel = None
    TradeManagementDecision = None
    TradeManagementEngine = None

# ============================================================================
# Operator Controls (v10.2+)
# ============================================================================

try:
    from .operator_control import (
        OperatorControlLayer,
        OperatorAction,
        get_operator_control,
    )
except ImportError:
    OperatorControlLayer = None
    OperatorAction = None
    get_operator_control = None

try:
    from .incident_state_machine import (
        IncidentStateMachine,
        IncidentState,
        get_incident_state_machine,
    )
except ImportError:
    IncidentStateMachine = None
    IncidentState = None
    get_incident_state_machine = None

try:
    from .safety_metrics import (
        SafetyMetrics,
        get_safety_metrics,
    )
except ImportError:
    SafetyMetrics = None
    get_safety_metrics = None

try:
    from .config_guard import (
        ConfigGuard,
    )
except ImportError:
    ConfigGuard = None

try:
    from .drift_detection import (
        DriftDetector,
    )
except ImportError:
    DriftDetector = None

# ============================================================================
# v10.3-OPERATIONAL-GRADE Modules
# ============================================================================

try:
    from .operator_discipline import (
        OperatorDiscipline,
        OperatorIdentity,
        ActionReason,
        ActionReasonCode,
        ActionType,
        InterventionAuditLog,
        get_operator_discipline,
    )
except ImportError:
    OperatorDiscipline = None
    OperatorIdentity = None
    ActionReason = None
    ActionReasonCode = None
    ActionType = None
    InterventionAuditLog = None
    get_operator_discipline = None

try:
    from .drift_guard import (
        DecisionDriftGuard,
        ConfigChecksum,
        DriftType,
        DriftSeverity,
        DriftEvent,
        get_drift_guard,
    )
except ImportError:
    DecisionDriftGuard = None
    ConfigChecksum = None
    DriftType = None
    DriftSeverity = None
    DriftEvent = None
    get_drift_guard = None

try:
    from .config_seal import (
        LiveConfigGuard,
        SealedConfig,
        SealStatus,
        get_config_guard,
    )
except ImportError:
    LiveConfigGuard = None
    SealedConfig = None
    SealStatus = None
    get_config_guard = None

# ============================================================================
# v10.4-TRUST-GRADE Modules
# ============================================================================

try:
    from .environment_guard import (
        EnvironmentGuard,
        EnvironmentMode,
        EnvironmentSignature,
        get_environment_guard,
    )
except ImportError:
    EnvironmentGuard = None
    EnvironmentMode = None
    EnvironmentSignature = None
    get_environment_guard = None

try:
    from .data_authenticity import (
        DataAuthenticityGuard,
        AuthenticData,
        DataSource,
        AuthenticityStatus,
        get_auth_guard,
    )
except ImportError:
    DataAuthenticityGuard = None
    AuthenticData = None
    DataSource = None
    AuthenticityStatus = None
    get_auth_guard = None

try:
    from .decision_narrative import (
        NarrativeEngine,
        DecisionNarrative,
        NarrativeType,
        get_narrative_engine,
    )
except ImportError:
    NarrativeEngine = None
    DecisionNarrative = None
    NarrativeType = None
    get_narrative_engine = None

try:
    from .cold_start import (
        ColdStartManager,
        StartupState,
        StartupStep,
        get_cold_start_manager,
    )
except ImportError:
    ColdStartManager = None
    StartupState = None
    StartupStep = None
    get_cold_start_manager = None

# ============================================================================
# Database Layer (v11.0)
# ============================================================================

from .database import (
    DatabaseManager,
    DecisionRepository,
    EventRepository,
    StateRepository,
    get_database,
    get_decision_repository,
    get_event_repository,
    get_state_repository,
)

# ============================================================================
# WebSocket (v11.0)
# ============================================================================

try:
    from .websocket import (
        WebSocketManager,
        get_websocket_manager,
    )
except ImportError:
    WebSocketManager = None
    get_websocket_manager = None

# ============================================================================
# Exchange Integration
# ============================================================================

from .exchange_state import (
    ExchangeStateManager,
)

from .exchange_health_monitor import (
    ExchangeHealthMonitor,
)

# ============================================================================
# Engine & Configuration
# ============================================================================

from .engine import (
    create_engine,
    TradingEngine,
)

from .config_manager import (
    get_config,
    ConfigManager,
)

from .secrets_manager import (
    get_secrets,
    SecretsManager,
)

from .observability import (
    get_observability,
    ObservabilityManager,
)

from .graceful_shutdown import (
    get_shutdown_manager,
    GracefulShutdown,
)

from .integration_hub import (
    get_integration_hub,
    IntegrationHub,
)

# ============================================================================
# Exports
# ============================================================================

__version__ = "11.0-FINAL"

__all__ = [
    # v11.0 Trade Intent/Decision
    'TradeIntent',
    'IntentDirection',
    'IntentPriority',
    'TradeIntentValidator',
    'TradeDecision',
    'DecisionStatus',
    'RejectionStage',
    'RiskState',
    'ExecutionParams',
    'StageResultRecord',
    'DecisionStore',
    'get_decision_store',
    
    # Execution Flow
    'ExecutionFlowOrchestrator',
    'FlowStage',
    'FlowResult',
    'get_execution_flow',
    
    # Risk Management
    'DrawdownGovernor',
    'DrawdownSeverity',
    'get_drawdown_governor',
    'CapitalAllocationGovernor',
    'AllocationContext',
    'AllocationSnapshot',
    'PortfolioRiskModel',
    'PortfolioPosition',
    'RiskMetrics',
    'get_portfolio_risk',
    'TradePermissionFilter',
    'PermissionStatus',
    
    # Execution Hardening
    'SlippageMonitor',
    'SlippageRecord',
    'get_slippage_monitor',
    'ExchangeRecoveryHandler',
    'RecoveryResult',
    'ErrorCategory',
    'get_recovery_handler',
    'ExecutionRouter',
    'ExecutionMode',
    'OrderIntent',
    'OrderResult',
    
    # Market Context
    'MarketContextEngine',
    'ContextStateMachine',
    'MarketState',
    'BiasEngine',
    'ScoringContract',
    'TradeManagementEngine',
    'ManagedTrade',
    'PositionSide',
    'TPLevel',
    'TradeManagementDecision',
    
    # Operator Controls
    'OperatorControlLayer',
    'OperatorAction',
    'get_operator_control',
    'IncidentStateMachine',
    'IncidentState',
    'get_incident_state_machine',
    'SafetyMetrics',
    'get_safety_metrics',
    'ConfigGuard',
    'DriftDetector',
    
    # Database
    'DatabaseManager',
    'DecisionRepository',
    'EventRepository',
    'StateRepository',
    'get_database',
    'get_decision_repository',
    'get_event_repository',
    'get_state_repository',
    
    # WebSocket
    'WebSocketManager',
    'get_websocket_manager',
    
    # Exchange
    'ExchangeStateManager',
    'ExchangeHealthMonitor',
    
    # Engine & Configuration
    'create_engine',
    'TradingEngine',
    'get_config',
    'ConfigManager',
    'get_secrets',
    'SecretsManager',
    'get_observability',
    'ObservabilityManager',
    'get_shutdown_manager',
    'GracefulShutdown',
    'get_integration_hub',
    'IntegrationHub',
]
