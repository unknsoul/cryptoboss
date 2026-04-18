"""Visual/programmatic strategy builder for CryptoBoss v12.0."""

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class ConditionOperator(Enum):
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "=="
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"
    IN_ZONE = "in_zone"


class LogicGate(Enum):
    AND = "AND"
    OR = "OR"


class IndicatorType(Enum):
    EMA = "ema"
    SMA = "sma"
    SUPERTREND = "supertrend"
    VWAP = "vwap"

    RSI = "rsi"
    STOCH_RSI = "stoch_rsi"
    MACD = "macd"
    CCI = "cci"
    WILLIAMS_R = "williams_r"

    ATR = "atr"
    BOLLINGER = "bollinger_bands"
    KELTNER = "keltner_channel"

    VOLUME_MA = "volume_ma"
    OBV = "obv"
    CVD = "cvd"

    ORDER_BLOCK = "order_block"
    FVG = "fvg"
    BOS = "bos"
    CHOCH = "choch"
    LIQUIDITY = "liquidity"

    CANDLE_PATTERN = "candle_pattern"
    PIVOT_POINTS = "pivot_points"
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass
class Condition:
    condition_id: str
    indicator: IndicatorType
    params: Dict[str, Any]
    operator: ConditionOperator
    threshold: Any
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.condition_id,
            "indicator": self.indicator.value,
            "params": self.params,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "description": self.description,
        }


@dataclass
class ConditionGroup:
    group_id: str
    gate: LogicGate
    conditions: List[Condition]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.group_id,
            "gate": self.gate.value,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "description": self.description,
        }


@dataclass
class ExitConfig:
    stop_loss_type: str
    stop_loss_value: float
    take_profit_type: str
    take_profit_value: float
    tp2_rr: Optional[float] = None
    tp3_rr: Optional[float] = None
    trail_after_tp1: bool = False
    trail_atr_mult: float = 1.0
    max_hold_bars: Optional[int] = None
    break_even_after_rr: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskConfig:
    max_risk_pct: float = 0.01
    max_daily_loss_pct: float = 0.03
    max_positions: int = 3
    position_sizing: str = "fixed_risk"
    kelly_fraction: float = 0.25
    correlation_limit: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FilterConfig:
    session_filter: Optional[List[str]] = None
    kill_zone_only: bool = False
    min_atr_percentile: Optional[float] = None
    max_atr_percentile: Optional[float] = None
    trend_filter: Optional[str] = None
    min_volume_ratio: Optional[float] = None
    avoid_news: bool = False
    avoid_weekend: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BuiltStrategy:
    strategy_id: str
    name: str
    description: str
    version: str
    symbol: str
    timeframe: str
    entry_long: Optional[ConditionGroup]
    entry_short: Optional[ConditionGroup]
    exit_config: ExitConfig
    risk_config: RiskConfig
    filter_config: FilterConfig
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    author: str = "CryptoBoss"
    performance: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "strategy_id": self.strategy_id,
                "name": self.name,
                "description": self.description,
                "version": self.version,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "entry_long": self.entry_long.to_dict() if self.entry_long else None,
                "entry_short": self.entry_short.to_dict() if self.entry_short else None,
                "exit_config": self.exit_config.to_dict(),
                "risk_config": self.risk_config.to_dict(),
                "filter_config": self.filter_config.to_dict(),
                "tags": self.tags,
                "created_at": self.created_at,
                "author": self.author,
                "performance": self.performance,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "BuiltStrategy":
        parsed = json.loads(json_str)
        entry_long = _group_from_dict(parsed["entry_long"]) if parsed.get("entry_long") else None
        entry_short = _group_from_dict(parsed["entry_short"]) if parsed.get("entry_short") else None

        return cls(
            strategy_id=parsed["strategy_id"],
            name=parsed["name"],
            description=parsed["description"],
            version=parsed["version"],
            symbol=parsed["symbol"],
            timeframe=parsed["timeframe"],
            entry_long=entry_long,
            entry_short=entry_short,
            exit_config=ExitConfig(**parsed["exit_config"]),
            risk_config=RiskConfig(**parsed["risk_config"]),
            filter_config=FilterConfig(**parsed["filter_config"]),
            tags=parsed.get("tags", []),
            created_at=parsed.get("created_at", ""),
            author=parsed.get("author", "CryptoBoss"),
            performance=parsed.get("performance", {}),
        )


def _group_from_dict(data: Dict[str, Any]) -> ConditionGroup:
    conditions = [
        Condition(
            condition_id=item["id"],
            indicator=IndicatorType(item["indicator"]),
            params=item["params"],
            operator=ConditionOperator(item["operator"]),
            threshold=item["threshold"],
            description=item.get("description", ""),
        )
        for item in data.get("conditions", [])
    ]

    return ConditionGroup(
        group_id=data["id"],
        gate=LogicGate(data["gate"]),
        conditions=conditions,
        description=data.get("description", ""),
    )


class StrategyBuilder:
    """Creates custom trading strategies from indicator/SMC blocks."""

    PRESETS: Dict[str, Dict[str, Any]] = {
        "smc_scalper": {
            "name": "SMC Intraday Scalper",
            "description": "Order Block + FVG + BOS confluence scalper",
            "tags": ["smc", "scalper", "intraday"],
            "entry_indicators": [
                {
                    "indicator": "order_block",
                    "params": {"type": "bullish"},
                    "operator": "in_zone",
                    "threshold": "price",
                },
                {
                    "indicator": "fvg",
                    "params": {"type": "bullish"},
                    "operator": "in_zone",
                    "threshold": "price",
                },
                {
                    "indicator": "bos",
                    "params": {"direction": "bullish"},
                    "operator": "==",
                    "threshold": True,
                },
            ],
            "exit": {
                "stop_loss_type": "ob_bottom",
                "stop_loss_value": 0.8,
                "take_profit_type": "rr",
                "take_profit_value": 1.5,
            },
        },
        "ema_trend_scalper": {
            "name": "EMA Trend Scalper",
            "description": "9/21 EMA crossover with RSI confirmation",
            "tags": ["ema", "scalper", "trend"],
            "entry_indicators": [
                {
                    "indicator": "ema",
                    "params": {"fast": 9, "slow": 21},
                    "operator": "cross_above",
                    "threshold": 0,
                },
                {
                    "indicator": "rsi",
                    "params": {"period": 14},
                    "operator": ">",
                    "threshold": 50,
                },
                {
                    "indicator": "volume_ma",
                    "params": {"period": 20},
                    "operator": ">",
                    "threshold": 1.2,
                },
            ],
            "exit": {
                "stop_loss_type": "atr",
                "stop_loss_value": 1.5,
                "take_profit_type": "rr",
                "take_profit_value": 2.0,
            },
        },
        "vwap_bounce": {
            "name": "VWAP Bounce Scalper",
            "description": "Price bounces off VWAP with volume surge",
            "tags": ["vwap", "scalper", "mean_reversion"],
            "entry_indicators": [
                {
                    "indicator": "vwap",
                    "params": {},
                    "operator": "in_zone",
                    "threshold": {"band": 0.002},
                },
                {
                    "indicator": "rsi",
                    "params": {"period": 7},
                    "operator": "<",
                    "threshold": 35,
                },
                {
                    "indicator": "volume_ma",
                    "params": {"period": 10},
                    "operator": ">",
                    "threshold": 1.5,
                },
            ],
            "exit": {
                "stop_loss_type": "atr",
                "stop_loss_value": 1.0,
                "take_profit_type": "rr",
                "take_profit_value": 1.5,
            },
        },
        "choch_reversal": {
            "name": "CHoCH Reversal",
            "description": "Change of Character reversal at key levels",
            "tags": ["smc", "reversal", "swing"],
            "entry_indicators": [
                {
                    "indicator": "choch",
                    "params": {},
                    "operator": "==",
                    "threshold": True,
                },
                {
                    "indicator": "order_block",
                    "params": {},
                    "operator": "in_zone",
                    "threshold": "price",
                },
                {
                    "indicator": "rsi",
                    "params": {"period": 14},
                    "operator": "<",
                    "threshold": 40,
                },
            ],
            "exit": {
                "stop_loss_type": "swing_low",
                "stop_loss_value": 1.0,
                "take_profit_type": "rr",
                "take_profit_value": 3.0,
            },
        },
    }

    def __init__(self) -> None:
        self._strategies: Dict[str, BuiltStrategy] = {}

    def new_strategy(
        self,
        name: str,
        symbol: str = "BTC/USDT",
        timeframe: str = "5m",
        description: str = "",
    ) -> str:
        strategy_id = str(uuid.uuid4())[:8].upper()
        strategy = BuiltStrategy(
            strategy_id=strategy_id,
            name=name,
            description=description,
            version="1.0",
            symbol=symbol,
            timeframe=timeframe,
            entry_long=ConditionGroup(
                group_id=str(uuid.uuid4())[:8],
                gate=LogicGate.AND,
                conditions=[],
                description="Long entry conditions",
            ),
            entry_short=ConditionGroup(
                group_id=str(uuid.uuid4())[:8],
                gate=LogicGate.AND,
                conditions=[],
                description="Short entry conditions",
            ),
            exit_config=ExitConfig(
                stop_loss_type="atr",
                stop_loss_value=1.5,
                take_profit_type="rr",
                take_profit_value=2.0,
            ),
            risk_config=RiskConfig(),
            filter_config=FilterConfig(),
            created_at=pd.Timestamp.utcnow().isoformat(),
        )
        self._strategies[strategy_id] = strategy
        return strategy_id

    def add_entry_condition(
        self,
        strategy_id: str,
        direction: str,
        indicator: str,
        params: Dict[str, Any],
        operator: str,
        threshold: Any,
        description: str = "",
    ) -> str:
        strategy = self._strategies[strategy_id]
        group = strategy.entry_long if direction == "long" else strategy.entry_short

        condition_id = str(uuid.uuid4())[:8]
        condition = Condition(
            condition_id=condition_id,
            indicator=IndicatorType(indicator),
            params=params,
            operator=ConditionOperator(operator),
            threshold=threshold,
            description=description,
        )

        if group is None:
            raise ValueError(f"Entry group for direction '{direction}' is not configured")

        group.conditions.append(condition)
        return condition_id

    def set_exit_config(self, strategy_id: str, **kwargs: Any) -> None:
        strategy = self._strategies[strategy_id]
        for key, value in kwargs.items():
            setattr(strategy.exit_config, key, value)

    def set_risk_config(self, strategy_id: str, **kwargs: Any) -> None:
        strategy = self._strategies[strategy_id]
        for key, value in kwargs.items():
            setattr(strategy.risk_config, key, value)

    def set_filter_config(self, strategy_id: str, **kwargs: Any) -> None:
        strategy = self._strategies[strategy_id]
        for key, value in kwargs.items():
            setattr(strategy.filter_config, key, value)

    def load_preset(self, preset_name: str, symbol: str = "BTC/USDT", timeframe: str = "5m") -> str:
        if preset_name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.PRESETS)}")

        preset = self.PRESETS[preset_name]
        strategy_id = self.new_strategy(
            preset["name"],
            symbol=symbol,
            timeframe=timeframe,
            description=preset["description"],
        )

        for indicator in preset["entry_indicators"]:
            self.add_entry_condition(
                strategy_id,
                "long",
                indicator=indicator["indicator"],
                params=indicator.get("params", {}),
                operator=indicator["operator"],
                threshold=indicator.get("threshold"),
            )

        self.set_exit_config(strategy_id, **preset["exit"])
        self._strategies[strategy_id].tags = preset.get("tags", [])
        return strategy_id

    def get_strategy(self, strategy_id: str) -> BuiltStrategy:
        return self._strategies[strategy_id]

    def export_json(self, strategy_id: str) -> str:
        return self._strategies[strategy_id].to_json()

    def import_json(self, json_str: str) -> str:
        strategy = BuiltStrategy.from_json(json_str)
        self._strategies[strategy.strategy_id] = strategy
        return strategy.strategy_id

    def list_strategies(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": strategy.strategy_id,
                "name": strategy.name,
                "symbol": strategy.symbol,
                "timeframe": strategy.timeframe,
                "tags": strategy.tags,
                "performance": strategy.performance,
            }
            for strategy in self._strategies.values()
        ]
