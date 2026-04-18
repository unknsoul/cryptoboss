"""Configuration contracts for CryptoBoss v3.0 intraday scalper system."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DataEngineConfig:
    sources: List[str] = field(default_factory=lambda: ["mt5", "binance", "forex_feed"])
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m"])
    candles_required: int = 10000
    features: List[str] = field(
        default_factory=lambda: [
            "ohlc",
            "tick_volume",
            "spread",
            "volatility",
            "momentum",
            "liquidity_zones",
        ]
    )


@dataclass
class MarketStructureConfig:
    bos_threshold: float = 0.0005
    volume_spike_mult: float = 1.2
    trend_method: str = "multi_timeframe"


@dataclass
class SmartMoneyConfig:
    impulse_strength_threshold: float = 0.5
    mitigation_pending_required: bool = True
    liquidity_zones: List[str] = field(default_factory=lambda: ["equal highs", "equal lows", "previous highs/lows"])


@dataclass
class SignalEngineConfig:
    require_trend_alignment: bool = True
    require_structure_confirmation: bool = True
    require_price_in_ob_or_fvg: bool = True
    require_liquidity_sweep: bool = True
    confirmations: List[str] = field(default_factory=lambda: ["engulfing_candle", "momentum_spike"])


@dataclass
class RiskEngineConfig:
    risk_per_trade: float = 1.0
    rr_ratio: float = 2.0
    stop_loss_policy: str = "below_ob_or_structure"
    take_profit_policy: str = "next_liquidity_zone"
    max_trades_per_day: int = 10
    max_drawdown: float = 5.0


@dataclass
class ExecutionEngineConfig:
    platform: str = "mt5"
    order_types: List[str] = field(default_factory=lambda: ["market", "limit"])
    slippage_control: bool = True
    spread_filter: bool = True
    execution_speed: str = "high_frequency"
    max_spread_pct: float = 0.15
    max_slippage_pct: float = 0.10


@dataclass
class StrategyBuilderConfig:
    strategy_type: str = "rule_based + ai_assisted"
    output: str = "custom_strategy.json"
    features: List[str] = field(
        default_factory=lambda: [
            "drag_and_drop_logic",
            "condition_chaining",
            "multi_timeframe_rules",
        ]
    )


@dataclass
class BacktestingConfig:
    data_range: str = "last_2_years"
    metrics: List[str] = field(default_factory=lambda: ["win_rate", "profit_factor", "drawdown", "sharpe_ratio"])
    simulate_spread: bool = True
    simulate_slippage: bool = True
    simulate_commission: bool = True


@dataclass
class PerformanceTrackerConfig:
    logs: List[str] = field(default_factory=lambda: ["trade_entry", "trade_exit", "reason_for_trade", "strategy_used"])
    dashboard: bool = True
    real_time_stats: bool = True


@dataclass
class AIOptimizerConfig:
    models: List[str] = field(default_factory=lambda: ["xgboost", "lightgbm", "neural_network"])
    purposes: List[str] = field(
        default_factory=lambda: [
            "optimize_parameters",
            "filter_false_signals",
            "predict_trade_success_probability",
        ]
    )


@dataclass
class V3SystemConfig:
    version: str = "3.0_intraday_scalper"
    architecture: str = "modular_microservices"
    modules: List[str] = field(
        default_factory=lambda: [
            "data_engine",
            "market_structure_engine",
            "smart_money_engine",
            "signal_engine",
            "risk_engine",
            "execution_engine",
            "strategy_builder",
            "backtesting_engine",
            "performance_tracker",
            "ai_optimizer",
        ]
    )
    data_engine: DataEngineConfig = field(default_factory=DataEngineConfig)
    market_structure_engine: MarketStructureConfig = field(default_factory=MarketStructureConfig)
    smart_money_engine: SmartMoneyConfig = field(default_factory=SmartMoneyConfig)
    signal_engine: SignalEngineConfig = field(default_factory=SignalEngineConfig)
    risk_engine: RiskEngineConfig = field(default_factory=RiskEngineConfig)
    execution_engine: ExecutionEngineConfig = field(default_factory=ExecutionEngineConfig)
    strategy_builder: StrategyBuilderConfig = field(default_factory=StrategyBuilderConfig)
    backtesting_engine: BacktestingConfig = field(default_factory=BacktestingConfig)
    performance_tracker: PerformanceTrackerConfig = field(default_factory=PerformanceTrackerConfig)
    ai_optimizer: AIOptimizerConfig = field(default_factory=AIOptimizerConfig)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "architecture": self.architecture,
            "modules": self.modules,
            "data_engine": self.data_engine.__dict__,
            "market_structure_engine": self.market_structure_engine.__dict__,
            "smart_money_engine": self.smart_money_engine.__dict__,
            "signal_engine": self.signal_engine.__dict__,
            "risk_engine": self.risk_engine.__dict__,
            "execution_engine": self.execution_engine.__dict__,
            "strategy_builder": self.strategy_builder.__dict__,
            "backtesting_engine": self.backtesting_engine.__dict__,
            "performance_tracker": self.performance_tracker.__dict__,
            "ai_optimizer": self.ai_optimizer.__dict__,
        }


def build_default_v3_config() -> V3SystemConfig:
    return V3SystemConfig()
