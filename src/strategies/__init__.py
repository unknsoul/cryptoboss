"""
Strategies Module - v12.0 Professional SMC Scalper

All trading strategies with TradeIntent integration for the
production pipeline.

Exports:
- BaseStrategy, StrategyConfig, SignalResult
- LegacyStrategyWrapper for backward compatibility
- StrategyIntentAdapter for signal conversion
- DCAStrategy, GridTradingStrategy, MarketMakingStrategy
- RegimeDetector, StrategySelector, MarketRegime
- IntradayScalper, StrategyBuilder, StrategyTester
"""

# v11.0 Base classes
from .base_strategy import (
    BaseStrategy,
    StrategyConfig,
    SignalResult,
    LegacyStrategyWrapper,
)

# v11.0 Intent adapter
from .intent_adapter import (
    StrategyIntentAdapter,
    StrategySignal,
    StrategySignalType,
    create_intent,
    submit_intent,
)

# Trading strategies
from .dca_strategy import DCAStrategy, DCADeal
from .grid_strategy import GridTradingStrategy, GridConfig, GridLevel
from .market_making import MarketMakingStrategy
from .intraday_scalper import IntradayScalper
from .strategy_builder import StrategyBuilder
from .strategy_tester import StrategyTester
from .session_manager_pro import SessionManagerPro, SessionWindow
from .pro_strategy_builder import ProStrategyBuilder, ProStrategy, INDICATOR_LIBRARY
from .range_scalp import RangeScalpStrategy
from .smc_scalper import SMCScalperStrategy
from .smc_trend_follow import SMCTrendFollowStrategy

# Regime detection and selection
from .regime_selection import RegimeDetector, StrategySelector, MarketRegime

__all__ = [
    # v11.0 Base classes
    'BaseStrategy',
    'StrategyConfig',
    'SignalResult',
    'LegacyStrategyWrapper',
    
    # v11.0 Intent adapter
    'StrategyIntentAdapter',
    'StrategySignal',
    'StrategySignalType',
    'create_intent',
    'submit_intent',
    
    # Strategies
    'DCAStrategy',
    'DCADeal',
    'GridTradingStrategy',
    'GridConfig',
    'GridLevel',
    'MarketMakingStrategy',
    'IntradayScalper',
    'StrategyBuilder',
    'StrategyTester',
    'SMCTrendFollowStrategy',
    'SMCScalperStrategy',
    'RangeScalpStrategy',
    'ProStrategyBuilder',
    'ProStrategy',
    'INDICATOR_LIBRARY',
    'SessionManagerPro',
    'SessionWindow',
    
    # Regime
    'RegimeDetector',
    'StrategySelector',
    'MarketRegime',
]

__version__ = "12.0"
