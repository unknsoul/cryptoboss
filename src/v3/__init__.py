"""CryptoBoss v3.0 Intraday Scalper modular microservices package."""

from .config import V3SystemConfig, build_default_v3_config
from .orchestrator import IntradayScalperV3System
from .data_engine import DataEngine
from .market_structure_engine import MarketStructureEngine
from .smart_money_engine import SmartMoneyEngine
from .signal_engine import SignalEngine, SignalOutput
from .risk_engine import RiskEngine, RiskDecision
from .execution_engine import ExecutionEngine, ExecutionReport
from .strategy_builder_service import StrategyBuilderService
from .strategy_builder_engine import StrategyBuilderEngine, StrategyValidationEngine
from .backtesting_engine import BacktestingEngine
from .performance_tracker import PerformanceTracker
from .ai_optimizer import AIOptimizer
from .quantified_smc import detect_bos, detect_fvg, detect_order_block, detect_swing_points_fractal

__all__ = [
    "V3SystemConfig",
    "build_default_v3_config",
    "IntradayScalperV3System",
    "DataEngine",
    "MarketStructureEngine",
    "SmartMoneyEngine",
    "SignalEngine",
    "SignalOutput",
    "RiskEngine",
    "RiskDecision",
    "ExecutionEngine",
    "ExecutionReport",
    "StrategyBuilderService",
    "StrategyBuilderEngine",
    "StrategyValidationEngine",
    "BacktestingEngine",
    "PerformanceTracker",
    "AIOptimizer",
    "detect_bos",
    "detect_fvg",
    "detect_order_block",
    "detect_swing_points_fractal",
]
