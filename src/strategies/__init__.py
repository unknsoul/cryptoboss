"""
Strategies Module - v11.0 Production-Grade

All trading strategies with TradeIntent integration for the
production pipeline.

Exports:
- BaseStrategy, StrategyConfig, SignalResult
- LegacyStrategyWrapper for backward compatibility
- StrategyIntentAdapter for signal conversion
- DCAStrategy, GridTradingStrategy, MarketMakingStrategy
- RegimeDetector, StrategySelector, MarketRegime
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
    
    # Regime
    'RegimeDetector',
    'StrategySelector',
    'MarketRegime',
]

__version__ = "11.0"
