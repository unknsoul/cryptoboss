"""Professional strategy builder with indicator DSL, scoring, and canvas export."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


INDICATOR_LIBRARY: Dict[str, Dict[str, Any]] = {
    "ema": {
        "category": "trend",
        "name": "EMA",
        "description": "Exponential Moving Average",
        "params": {"period": {"type": "int", "default": 20, "min": 2, "max": 500}},
        "outputs": ["ema"],
        "operators": [">", "<", "cross_above", "cross_below"],
    },
    "sma": {
        "category": "trend",
        "name": "SMA",
        "description": "Simple Moving Average",
        "params": {"period": {"type": "int", "default": 50, "min": 2, "max": 500}},
        "outputs": ["sma"],
        "operators": [">", "<", "cross_above", "cross_below"],
    },
    "ema_ribbon": {
        "category": "trend",
        "name": "EMA Ribbon",
        "description": "8/13/21/34/55/89 EMA stack",
        "params": {"fast": {"type": "int", "default": 8}, "slow": {"type": "int", "default": 89}},
        "outputs": ["ribbon_bullish", "ribbon_bearish", "ribbon_spread"],
        "operators": ["==", "cross_above", "cross_below"],
    },
    "supertrend": {
        "category": "trend",
        "name": "Supertrend",
        "description": "ATR-based dynamic trend line",
        "params": {"period": {"type": "int", "default": 10}, "multiplier": {"type": "float", "default": 3.0}},
        "outputs": ["direction", "support", "resistance"],
        "operators": ["=="],
    },
    "vwap": {
        "category": "trend",
        "name": "VWAP",
        "description": "Volume weighted average price",
        "params": {"band_mult": {"type": "float", "default": 1.0}},
        "outputs": ["vwap", "upper_band", "lower_band"],
        "operators": [">", "<", "in_zone"],
    },
    "hma": {
        "category": "trend",
        "name": "HMA",
        "description": "Hull moving average",
        "params": {"period": {"type": "int", "default": 20}},
        "outputs": ["hma"],
        "operators": [">", "<", "cross_above", "cross_below"],
    },
    "rsi": {
        "category": "momentum",
        "name": "RSI",
        "description": "Relative strength index",
        "params": {"period": {"type": "int", "default": 14, "min": 2, "max": 100}},
        "outputs": ["rsi"],
        "operators": [">", "<", ">=", "<=", "cross_above", "cross_below"],
    },
    "stoch_rsi": {
        "category": "momentum",
        "name": "Stoch RSI",
        "description": "Stochastic RSI",
        "params": {
            "rsi_period": {"type": "int", "default": 14},
            "stoch_period": {"type": "int", "default": 14},
            "k": {"type": "int", "default": 3},
            "d": {"type": "int", "default": 3},
        },
        "outputs": ["k", "d"],
        "operators": [">", "<", "cross_above", "cross_below"],
    },
    "macd": {
        "category": "momentum",
        "name": "MACD",
        "description": "Moving average convergence divergence",
        "params": {
            "fast": {"type": "int", "default": 12},
            "slow": {"type": "int", "default": 26},
            "signal": {"type": "int", "default": 9},
        },
        "outputs": ["macd", "signal", "histogram"],
        "operators": [">", "<", "cross_above", "cross_below"],
    },
    "cci": {
        "category": "momentum",
        "name": "CCI",
        "description": "Commodity channel index",
        "params": {"period": {"type": "int", "default": 20}},
        "outputs": ["cci"],
        "operators": [">", "<", "cross_above", "cross_below"],
    },
    "williams_r": {
        "category": "momentum",
        "name": "Williams %R",
        "description": "Overbought and oversold oscillator",
        "params": {"period": {"type": "int", "default": 14}},
        "outputs": ["williams_r"],
        "operators": [">", "<"],
    },
    "mfi": {
        "category": "momentum",
        "name": "MFI",
        "description": "Money flow index",
        "params": {"period": {"type": "int", "default": 14}},
        "outputs": ["mfi"],
        "operators": [">", "<"],
    },
    "atr": {
        "category": "volatility",
        "name": "ATR",
        "description": "Average true range",
        "params": {"period": {"type": "int", "default": 14}},
        "outputs": ["atr", "atr_pct"],
        "operators": [">", "<", "percentile_above", "percentile_below"],
    },
    "bollinger": {
        "category": "volatility",
        "name": "Bollinger Bands",
        "description": "2-sigma price envelope",
        "params": {"period": {"type": "int", "default": 20}, "std": {"type": "float", "default": 2.0}},
        "outputs": ["upper", "mid", "lower", "bandwidth", "percent_b"],
        "operators": [">", "<", "in_zone", "cross_above", "cross_below"],
    },
    "keltner": {
        "category": "volatility",
        "name": "Keltner Channel",
        "description": "ATR-based envelope",
        "params": {"period": {"type": "int", "default": 20}, "atr_mult": {"type": "float", "default": 2.0}},
        "outputs": ["upper", "mid", "lower"],
        "operators": [">", "<", "in_zone"],
    },
    "squeeze": {
        "category": "volatility",
        "name": "TTM Squeeze",
        "description": "Low-vol squeeze detection",
        "params": {"bb_period": {"type": "int", "default": 20}, "kc_period": {"type": "int", "default": 20}},
        "outputs": ["squeeze_on", "momentum"],
        "operators": ["==", ">", "<"],
    },
    "volume_ma": {
        "category": "volume",
        "name": "Volume MA",
        "description": "Volume relative to moving average",
        "params": {"period": {"type": "int", "default": 20}},
        "outputs": ["volume_ratio"],
        "operators": [">", "<"],
    },
    "obv": {
        "category": "volume",
        "name": "OBV",
        "description": "On-balance volume",
        "params": {},
        "outputs": ["obv", "obv_slope"],
        "operators": [">", "<", "cross_above", "cross_below"],
    },
    "cvd": {
        "category": "volume",
        "name": "CVD",
        "description": "Cumulative volume delta",
        "params": {"period": {"type": "int", "default": 20}},
        "outputs": ["cvd", "cvd_slope"],
        "operators": [">", "<"],
    },
    "vwap_deviation": {
        "category": "volume",
        "name": "VWAP Deviation",
        "description": "Distance from VWAP in std deviations",
        "params": {"std_period": {"type": "int", "default": 20}},
        "outputs": ["deviation"],
        "operators": [">", "<"],
    },
    "order_block": {
        "category": "smc",
        "name": "Order Block",
        "description": "Institutional order block zone",
        "params": {"impulse_mult": {"type": "float", "default": 2.0}, "lookback": {"type": "int", "default": 10}},
        "outputs": ["ob_signal", "ob_top", "ob_bottom", "ob_strength"],
        "operators": ["in_zone", "==", "!="],
    },
    "fvg": {
        "category": "smc",
        "name": "FVG",
        "description": "Fair value gap",
        "params": {"min_gap": {"type": "float", "default": 0.00005}},
        "outputs": ["fvg_signal", "fvg_top", "fvg_bottom", "fvg_size"],
        "operators": ["in_zone", "==", "!="],
    },
    "bos": {
        "category": "smc",
        "name": "BOS",
        "description": "Break of structure",
        "params": {"lookback": {"type": "int", "default": 20}},
        "outputs": ["bos_bullish", "bos_bearish", "bos_level"],
        "operators": ["==", "!="],
    },
    "choch": {
        "category": "smc",
        "name": "CHoCH",
        "description": "Change of character",
        "params": {"swing_lookback": {"type": "int", "default": 5}},
        "outputs": ["choch_bullish", "choch_bearish"],
        "operators": ["==", "!="],
    },
    "liquidity": {
        "category": "smc",
        "name": "Liquidity",
        "description": "Liquidity sweep / stop hunt detector",
        "params": {"threshold": {"type": "float", "default": 0.001}},
        "outputs": ["sweep_detected", "sweep_direction", "level"],
        "operators": ["==", "!="],
    },
    "pivot_points": {
        "category": "price_action",
        "name": "Pivot Points",
        "description": "Classic pivot levels",
        "params": {"period": {"type": "str", "default": "D", "choices": ["D", "W", "M"]}},
        "outputs": ["pivot", "r1", "r2", "r3", "s1", "s2", "s3"],
        "operators": [">", "<", "in_zone"],
    },
    "candle_pattern": {
        "category": "price_action",
        "name": "Candle Pattern",
        "description": "Named candlestick patterns",
        "params": {
            "pattern": {
                "type": "str",
                "default": "engulfing",
                "choices": ["engulfing", "doji", "hammer", "pin_bar", "morning_star", "evening_star"],
            }
        },
        "outputs": ["pattern_found", "direction"],
        "operators": ["==", "!="],
    },
    "market_session": {
        "category": "filter",
        "name": "Market Session",
        "description": "Session gate filter",
        "params": {"sessions": {"type": "list", "default": ["London", "NY"]}},
        "outputs": ["in_session", "session_name"],
        "operators": ["==", "!="],
    },
    "kill_zone": {
        "category": "filter",
        "name": "Kill Zone",
        "description": "High-probability intraday windows",
        "params": {"zones": {"type": "list", "default": ["London Open", "NY Open"]}},
        "outputs": ["in_kill_zone", "zone_name"],
        "operators": ["==", "!="],
    },
}


class EntryType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class ExitMode(Enum):
    FIXED_RR = "fixed_rr"
    TRAILING_ATR = "trailing_atr"
    INDICATOR = "indicator"
    TIME_BASED = "time_based"
    PARTIAL = "partial"


class FilterGate(Enum):
    AND = "AND"
    OR = "OR"


@dataclass
class CanvasPosition:
    x: float = 0.0
    y: float = 0.0


@dataclass
class ConditionBlock:
    block_id: str
    indicator: str
    params: Dict[str, Any]
    operator: str
    threshold: Any
    direction: str
    output_key: str = ""
    description: str = ""
    canvas_pos: CanvasPosition = field(default_factory=CanvasPosition)
    color: str = "#5B8DEF"
    enabled: bool = True

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.indicator not in INDICATOR_LIBRARY:
            errors.append(f"Unknown indicator: {self.indicator}")
            return errors

        lib = INDICATOR_LIBRARY[self.indicator]
        if self.operator not in lib.get("operators", []):
            errors.append(f"{self.indicator} does not support operator '{self.operator}'")

        for param_name, schema in lib.get("params", {}).items():
            if param_name not in self.params:
                continue
            value = self.params[param_name]
            if schema.get("type") == "int" and not isinstance(value, int):
                errors.append(f"{self.indicator}.{param_name} must be int")
            if schema.get("type") == "float" and not isinstance(value, (int, float)):
                errors.append(f"{self.indicator}.{param_name} must be float")
            if schema.get("type") == "str" and not isinstance(value, str):
                errors.append(f"{self.indicator}.{param_name} must be str")
            choices = schema.get("choices")
            if choices and value not in choices:
                errors.append(f"{self.indicator}.{param_name} must be one of {choices}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["canvas_pos"] = {"x": self.canvas_pos.x, "y": self.canvas_pos.y}
        return out


@dataclass
class ExitConfig:
    mode: ExitMode = ExitMode.FIXED_RR
    stop_loss_type: str = "atr"
    stop_loss_value: float = 1.5
    take_profit_1_rr: float = 1.5
    take_profit_2_rr: Optional[float] = 2.5
    take_profit_3_rr: Optional[float] = 4.0
    partial_exit_pct_1: float = 0.5
    partial_exit_pct_2: float = 0.3
    break_even_at_rr: float = 1.0
    trail_atr_mult: float = 1.0
    max_hold_bars: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["mode"] = self.mode.value
        return out


@dataclass
class RiskConfig:
    max_risk_pct: float = 0.01
    max_daily_loss_pct: float = 0.03
    max_positions: int = 3
    position_sizing: str = "fixed_risk"
    kelly_fraction: float = 0.25
    session_weight_enabled: bool = True
    correlation_limit: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MTFConfig:
    enabled: bool = False
    bias_timeframe: str = "1h"
    entry_timeframe: str = "5m"
    require_htf_agreement: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProStrategy:
    strategy_id: str
    name: str
    description: str
    symbol: str
    entry_timeframe: str
    entry_type: EntryType
    long_conditions: List[ConditionBlock]
    short_conditions: List[ConditionBlock]
    exit_conditions: List[ConditionBlock]
    filter_conditions: List[ConditionBlock]
    condition_gate: FilterGate
    exit_config: ExitConfig
    risk_config: RiskConfig
    mtf_config: MTFConfig
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    author: str = "CryptoBoss"
    version: str = "1.0"
    ai_score: Optional[float] = None
    backtest_summary: Optional[Dict[str, Any]] = None
    canvas_nodes: List[Dict[str, Any]] = field(default_factory=list)
    canvas_edges: List[Dict[str, Any]] = field(default_factory=list)

    def validate(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        all_blocks = self.long_conditions + self.short_conditions + self.exit_conditions + self.filter_conditions
        for block in all_blocks:
            errors.extend(block.validate())

        if not self.long_conditions and not self.short_conditions:
            errors.append("Strategy must have at least one long or short condition")

        return len(errors) == 0, errors

    def to_json(self, indent: int = 2) -> str:
        payload = {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "symbol": self.symbol,
            "entry_timeframe": self.entry_timeframe,
            "entry_type": self.entry_type.value,
            "long_conditions": [b.to_dict() for b in self.long_conditions],
            "short_conditions": [b.to_dict() for b in self.short_conditions],
            "exit_conditions": [b.to_dict() for b in self.exit_conditions],
            "filter_conditions": [b.to_dict() for b in self.filter_conditions],
            "condition_gate": self.condition_gate.value,
            "exit_config": self.exit_config.to_dict(),
            "risk_config": self.risk_config.to_dict(),
            "mtf_config": self.mtf_config.to_dict(),
            "tags": self.tags,
            "created_at": self.created_at,
            "author": self.author,
            "version": self.version,
            "ai_score": self.ai_score,
            "backtest_summary": self.backtest_summary,
            "canvas_nodes": self.canvas_nodes,
            "canvas_edges": self.canvas_edges,
        }
        return json.dumps(payload, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ProStrategy":
        data = json.loads(json_str)

        def _blocks(raw: List[Dict[str, Any]]) -> List[ConditionBlock]:
            blocks: List[ConditionBlock] = []
            for block in raw:
                pos = block.get("canvas_pos", {})
                blocks.append(
                    ConditionBlock(
                        block_id=block["block_id"],
                        indicator=block["indicator"],
                        params=block.get("params", {}),
                        operator=block["operator"],
                        threshold=block.get("threshold"),
                        direction=block.get("direction", "long"),
                        output_key=block.get("output_key", ""),
                        description=block.get("description", ""),
                        canvas_pos=CanvasPosition(x=pos.get("x", 0.0), y=pos.get("y", 0.0)),
                        color=block.get("color", "#5B8DEF"),
                        enabled=block.get("enabled", True),
                    )
                )
            return blocks

        return cls(
            strategy_id=data["strategy_id"],
            name=data["name"],
            description=data.get("description", ""),
            symbol=data.get("symbol", "BTC/USDT"),
            entry_timeframe=data.get("entry_timeframe", "5m"),
            entry_type=EntryType(data.get("entry_type", "market")),
            long_conditions=_blocks(data.get("long_conditions", [])),
            short_conditions=_blocks(data.get("short_conditions", [])),
            exit_conditions=_blocks(data.get("exit_conditions", [])),
            filter_conditions=_blocks(data.get("filter_conditions", [])),
            condition_gate=FilterGate(data.get("condition_gate", "AND")),
            exit_config=ExitConfig(**data.get("exit_config", {})),
            risk_config=RiskConfig(**data.get("risk_config", {})),
            mtf_config=MTFConfig(**data.get("mtf_config", {})),
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
            author=data.get("author", "CryptoBoss"),
            version=data.get("version", "1.0"),
            ai_score=data.get("ai_score"),
            backtest_summary=data.get("backtest_summary"),
            canvas_nodes=data.get("canvas_nodes", []),
            canvas_edges=data.get("canvas_edges", []),
        )


class StrategyScorer:
    """Heuristic quality scorer (0-100) before expensive backtest runs."""

    def score(self, strategy: ProStrategy) -> Tuple[float, Dict[str, Any]]:
        breakdown: Dict[str, Any] = {}
        score = 0.0

        all_blocks = [b for b in strategy.long_conditions + strategy.short_conditions if b.enabled]
        condition_score = min(len(all_blocks) * 8, 25)
        breakdown["conditions"] = {"score": condition_score, "count": len(all_blocks)}
        score += condition_score

        categories = {
            INDICATOR_LIBRARY.get(block.indicator, {}).get("category", "")
            for block in all_blocks
            if block.indicator in INDICATOR_LIBRARY
        }
        diversity_score = min(len(categories) * 5, 20)
        breakdown["diversity"] = {"score": diversity_score, "categories": sorted([c for c in categories if c])}
        score += diversity_score

        smc_blocks = [
            block
            for block in all_blocks
            if INDICATOR_LIBRARY.get(block.indicator, {}).get("category") == "smc"
        ]
        smc_score = min(len(smc_blocks) * 5, 15)
        breakdown["smc"] = {"score": smc_score, "count": len(smc_blocks)}
        score += smc_score

        exit_score = 15 if strategy.exit_config.stop_loss_value > 0 else 0
        breakdown["exit"] = {"score": exit_score}
        score += exit_score

        filters = [b for b in strategy.filter_conditions if b.enabled]
        filter_score = min(len(filters) * 5, 10)
        breakdown["filters"] = {"score": filter_score, "count": len(filters)}
        score += filter_score

        if len(all_blocks) > 8:
            overfit_penalty = min((len(all_blocks) - 8) * 2, 15)
            breakdown["overfit_penalty"] = -overfit_penalty
            score -= overfit_penalty
        else:
            breakdown["overfit_penalty"] = 0

        if strategy.mtf_config.enabled:
            breakdown["mtf_bonus"] = 10
            score += 10
        else:
            breakdown["mtf_bonus"] = 0

        rr = strategy.exit_config.take_profit_1_rr / max(strategy.exit_config.stop_loss_value, 0.1)
        if rr >= 1.5:
            breakdown["rr_bonus"] = 5
            score += 5
        else:
            breakdown["rr_warning"] = "RR < 1.5. Consider wider TP or tighter SL"

        final_score = round(max(0.0, min(score, 100.0)), 1)
        return final_score, breakdown

    @staticmethod
    def get_recommendation(score: float) -> str:
        if score >= 80:
            return "Excellent. Run backtest and optimize."
        if score >= 65:
            return "Good. Ready for backtest."
        if score >= 50:
            return "Fair. Add stronger confluence or filters."
        if score >= 40:
            return "Weak. Improve SMC and exits before testing."
        return "Poor. Redesign strategy before backtesting."


class ProStrategyBuilder:
    """CRUD + validation + scoring + canvas for professional strategy design."""

    PRESETS: Dict[str, Dict[str, Any]] = {
        "smc_scalper_pro": {
            "name": "SMC Pro Scalper",
            "description": "OB + FVG + BOS confluence with kill zone filter",
            "tags": ["smc", "scalper", "professional"],
            "symbol": "BTC/USDT",
            "timeframe": "5m",
            "long_conditions": [
                {"indicator": "bos", "operator": "==", "threshold": "bullish", "output_key": "bos_bullish", "params": {"lookback": 20}},
                {"indicator": "order_block", "operator": "in_zone", "threshold": "price", "output_key": "ob_signal", "params": {"impulse_mult": 2.0}},
                {"indicator": "fvg", "operator": "in_zone", "threshold": "price", "output_key": "fvg_signal", "params": {"min_gap": 0.00005}},
                {"indicator": "kill_zone", "operator": "==", "threshold": True, "output_key": "in_kill_zone", "params": {"zones": ["London Open", "NY Open"]}},
            ],
            "short_conditions": [
                {"indicator": "bos", "operator": "==", "threshold": "bearish", "output_key": "bos_bearish", "params": {"lookback": 20}},
                {"indicator": "order_block", "operator": "in_zone", "threshold": "price", "output_key": "ob_signal", "params": {"impulse_mult": 2.0}},
                {"indicator": "fvg", "operator": "in_zone", "threshold": "price", "output_key": "fvg_signal", "params": {"min_gap": 0.00005}},
            ],
            "filter_conditions": [
                {"indicator": "volume_ma", "operator": ">", "threshold": 1.2, "output_key": "volume_ratio", "params": {"period": 20}},
            ],
            "exit": {"stop_loss_type": "ob_bottom", "stop_loss_value": 0.8, "take_profit_1_rr": 1.5, "take_profit_2_rr": 2.5, "take_profit_3_rr": 4.0},
        },
        "ema_ribbon_momentum": {
            "name": "EMA Ribbon Momentum",
            "description": "EMA alignment + RSI + MACD + volume",
            "tags": ["ema", "momentum", "trend"],
            "long_conditions": [
                {"indicator": "ema", "operator": "cross_above", "threshold": "sma", "output_key": "ema", "params": {"period": 9}},
                {"indicator": "rsi", "operator": ">", "threshold": 50, "output_key": "rsi", "params": {"period": 14}},
                {"indicator": "macd", "operator": "cross_above", "threshold": 0, "output_key": "histogram", "params": {"fast": 12, "slow": 26, "signal": 9}},
                {"indicator": "volume_ma", "operator": ">", "threshold": 1.3, "output_key": "volume_ratio", "params": {"period": 20}},
            ],
            "short_conditions": [
                {"indicator": "ema", "operator": "cross_below", "threshold": "sma", "output_key": "ema", "params": {"period": 9}},
                {"indicator": "rsi", "operator": "<", "threshold": 50, "output_key": "rsi", "params": {"period": 14}},
                {"indicator": "macd", "operator": "cross_below", "threshold": 0, "output_key": "histogram", "params": {"fast": 12, "slow": 26, "signal": 9}},
            ],
            "exit": {"stop_loss_type": "atr", "stop_loss_value": 1.5, "take_profit_1_rr": 2.0, "take_profit_2_rr": 3.0},
        },
        "vwap_mean_reversion": {
            "name": "VWAP Mean Reversion",
            "description": "VWAP extremes + stoch RSI + candle pattern",
            "tags": ["vwap", "mean_reversion", "scalper"],
            "long_conditions": [
                {"indicator": "vwap", "operator": "<", "threshold": "price", "output_key": "lower_band", "params": {"band_mult": 1.5}},
                {"indicator": "stoch_rsi", "operator": "<", "threshold": 20, "output_key": "k", "params": {"rsi_period": 14, "stoch_period": 14, "k": 3, "d": 3}},
                {"indicator": "candle_pattern", "operator": "==", "threshold": "bullish", "output_key": "direction", "params": {"pattern": "hammer"}},
            ],
            "short_conditions": [
                {"indicator": "vwap", "operator": ">", "threshold": "price", "output_key": "upper_band", "params": {"band_mult": 1.5}},
                {"indicator": "stoch_rsi", "operator": ">", "threshold": 80, "output_key": "k", "params": {"rsi_period": 14, "stoch_period": 14, "k": 3, "d": 3}},
                {"indicator": "candle_pattern", "operator": "==", "threshold": "bearish", "output_key": "direction", "params": {"pattern": "pin_bar"}},
            ],
            "exit": {"stop_loss_type": "atr", "stop_loss_value": 1.0, "take_profit_1_rr": 1.5},
        },
        "choch_reversal_pro": {
            "name": "CHoCH Reversal Pro",
            "description": "CHoCH + OB + liquidity sweep",
            "tags": ["smc", "reversal", "professional"],
            "long_conditions": [
                {"indicator": "choch", "operator": "==", "threshold": "bullish", "output_key": "choch_bullish", "params": {"swing_lookback": 5}},
                {"indicator": "order_block", "operator": "in_zone", "threshold": "price", "output_key": "ob_signal", "params": {}},
                {"indicator": "liquidity", "operator": "==", "threshold": True, "output_key": "sweep_detected", "params": {}},
                {"indicator": "rsi", "operator": "<", "threshold": 40, "output_key": "rsi", "params": {"period": 14}},
            ],
            "short_conditions": [
                {"indicator": "choch", "operator": "==", "threshold": "bearish", "output_key": "choch_bearish", "params": {"swing_lookback": 5}},
                {"indicator": "order_block", "operator": "in_zone", "threshold": "price", "output_key": "ob_signal", "params": {}},
                {"indicator": "liquidity", "operator": "==", "threshold": True, "output_key": "sweep_detected", "params": {}},
                {"indicator": "rsi", "operator": ">", "threshold": 60, "output_key": "rsi", "params": {"period": 14}},
            ],
            "exit": {"stop_loss_type": "swing_low", "stop_loss_value": 1.0, "take_profit_1_rr": 2.0, "take_profit_2_rr": 3.5, "take_profit_3_rr": 6.0},
        },
        "squeeze_breakout": {
            "name": "TTM Squeeze Breakout",
            "description": "Squeeze release + momentum + BOS",
            "tags": ["volatility", "breakout", "momentum"],
            "long_conditions": [
                {"indicator": "squeeze", "operator": "==", "threshold": False, "output_key": "squeeze_on", "params": {}},
                {"indicator": "squeeze", "operator": ">", "threshold": 0, "output_key": "momentum", "params": {}},
                {"indicator": "bos", "operator": "==", "threshold": "bullish", "output_key": "bos_bullish", "params": {}},
                {"indicator": "volume_ma", "operator": ">", "threshold": 1.5, "output_key": "volume_ratio", "params": {"period": 10}},
            ],
            "short_conditions": [
                {"indicator": "squeeze", "operator": "==", "threshold": False, "output_key": "squeeze_on", "params": {}},
                {"indicator": "squeeze", "operator": "<", "threshold": 0, "output_key": "momentum", "params": {}},
                {"indicator": "bos", "operator": "==", "threshold": "bearish", "output_key": "bos_bearish", "params": {}},
            ],
            "exit": {"stop_loss_type": "atr", "stop_loss_value": 2.0, "take_profit_1_rr": 2.0, "take_profit_2_rr": 4.0},
        },
    }

    def __init__(self):
        self._strategies: Dict[str, ProStrategy] = {}
        self.scorer = StrategyScorer()

    def new(self, name: str, symbol: str = "BTC/USDT", timeframe: str = "5m", description: str = "") -> str:
        strategy_id = str(uuid.uuid4())[:8].upper()
        strategy = ProStrategy(
            strategy_id=strategy_id,
            name=name,
            description=description,
            symbol=symbol,
            entry_timeframe=timeframe,
            entry_type=EntryType.MARKET,
            long_conditions=[],
            short_conditions=[],
            exit_conditions=[],
            filter_conditions=[],
            condition_gate=FilterGate.AND,
            exit_config=ExitConfig(),
            risk_config=RiskConfig(),
            mtf_config=MTFConfig(),
            created_at=pd.Timestamp.utcnow().isoformat(),
        )
        self._strategies[strategy_id] = strategy
        return strategy_id

    def add_condition(
        self,
        strategy_id: str,
        direction: str,
        indicator: str,
        operator: str,
        threshold: Any,
        params: Optional[Dict[str, Any]] = None,
        output_key: str = "",
        description: str = "",
        canvas_x: float = 0.0,
        canvas_y: float = 0.0,
        color: str = "#5B8DEF",
    ) -> str:
        strategy = self._strategies[strategy_id]
        block_id = str(uuid.uuid4())[:6]
        block = ConditionBlock(
            block_id=block_id,
            indicator=indicator,
            params=params or self._default_params(indicator),
            operator=operator,
            threshold=threshold,
            direction=direction,
            output_key=output_key or self._default_output(indicator),
            description=description,
            canvas_pos=CanvasPosition(x=canvas_x, y=canvas_y),
            color=color,
        )

        if direction == "long":
            strategy.long_conditions.append(block)
        elif direction == "short":
            strategy.short_conditions.append(block)
        elif direction == "exit":
            strategy.exit_conditions.append(block)
        elif direction == "filter":
            strategy.filter_conditions.append(block)
        else:
            raise ValueError("direction must be long, short, exit, or filter")

        return block_id

    def remove_condition(self, strategy_id: str, block_id: str) -> bool:
        strategy = self._strategies[strategy_id]
        for collection in [
            strategy.long_conditions,
            strategy.short_conditions,
            strategy.exit_conditions,
            strategy.filter_conditions,
        ]:
            for i, block in enumerate(collection):
                if block.block_id == block_id:
                    collection.pop(i)
                    return True
        return False

    def toggle_condition(self, strategy_id: str, block_id: str) -> bool:
        strategy = self._strategies[strategy_id]
        for collection in [
            strategy.long_conditions,
            strategy.short_conditions,
            strategy.exit_conditions,
            strategy.filter_conditions,
        ]:
            for block in collection:
                if block.block_id == block_id:
                    block.enabled = not block.enabled
                    return block.enabled
        return False

    def update_exit_config(self, strategy_id: str, **kwargs: Any) -> None:
        strategy = self._strategies[strategy_id]
        for key, value in kwargs.items():
            if key == "mode" and isinstance(value, str):
                strategy.exit_config.mode = ExitMode(value)
            elif hasattr(strategy.exit_config, key):
                setattr(strategy.exit_config, key, value)

    def update_risk_config(self, strategy_id: str, **kwargs: Any) -> None:
        strategy = self._strategies[strategy_id]
        for key, value in kwargs.items():
            if hasattr(strategy.risk_config, key):
                setattr(strategy.risk_config, key, value)

    def enable_mtf(self, strategy_id: str, bias_tf: str = "1h", entry_tf: str = "5m") -> None:
        strategy = self._strategies[strategy_id]
        strategy.mtf_config.enabled = True
        strategy.mtf_config.bias_timeframe = bias_tf
        strategy.mtf_config.entry_timeframe = entry_tf

    def set_condition_gate(self, strategy_id: str, gate: str) -> None:
        self._strategies[strategy_id].condition_gate = FilterGate(gate.upper())

    def score(self, strategy_id: str) -> Tuple[float, Dict[str, Any], str]:
        strategy = self._strategies[strategy_id]
        score, breakdown = self.scorer.score(strategy)
        recommendation = self.scorer.get_recommendation(score)
        strategy.ai_score = score
        return score, breakdown, recommendation

    def validate(self, strategy_id: str) -> Tuple[bool, List[str]]:
        return self._strategies[strategy_id].validate()

    def load_preset(self, preset_name: str, symbol: str = "BTC/USDT", timeframe: str = "5m") -> str:
        if preset_name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.PRESETS.keys())}")

        preset = self.PRESETS[preset_name]
        strategy_id = self.new(
            name=preset["name"],
            symbol=symbol,
            timeframe=preset.get("timeframe", timeframe),
            description=preset.get("description", ""),
        )
        strategy = self._strategies[strategy_id]
        strategy.tags = preset.get("tags", [])

        for cond in preset.get("long_conditions", []):
            self.add_condition(
                strategy_id,
                "long",
                cond["indicator"],
                cond["operator"],
                cond["threshold"],
                params=cond.get("params", {}),
                output_key=cond.get("output_key", ""),
            )

        for cond in preset.get("short_conditions", []):
            self.add_condition(
                strategy_id,
                "short",
                cond["indicator"],
                cond["operator"],
                cond["threshold"],
                params=cond.get("params", {}),
                output_key=cond.get("output_key", ""),
            )

        for cond in preset.get("filter_conditions", []):
            self.add_condition(
                strategy_id,
                "filter",
                cond["indicator"],
                cond["operator"],
                cond["threshold"],
                params=cond.get("params", {}),
                output_key=cond.get("output_key", ""),
            )

        if preset.get("exit"):
            self.update_exit_config(strategy_id, **preset["exit"])

        return strategy_id

    def get(self, strategy_id: str) -> ProStrategy:
        return self._strategies[strategy_id]

    def list(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": strategy.strategy_id,
                "name": strategy.name,
                "symbol": strategy.symbol,
                "timeframe": strategy.entry_timeframe,
                "tags": strategy.tags,
                "ai_score": strategy.ai_score,
                "long_conditions": len(strategy.long_conditions),
                "short_conditions": len(strategy.short_conditions),
            }
            for strategy in self._strategies.values()
        ]

    def export_json(self, strategy_id: str) -> str:
        return self._strategies[strategy_id].to_json()

    def import_json(self, json_str: str) -> str:
        strategy = ProStrategy.from_json(json_str)
        self._strategies[strategy.strategy_id] = strategy
        return strategy.strategy_id

    def get_indicator_library(self) -> Dict[str, Dict[str, Any]]:
        return INDICATOR_LIBRARY

    def get_canvas_data(self, strategy_id: str) -> Dict[str, Any]:
        strategy = self._strategies[strategy_id]
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        colors = {
            "long": "#22C55E",
            "short": "#EF4444",
            "exit": "#F59E0B",
            "filter": "#6366F1",
        }

        column_x = {
            "filter": 0,
            "long": 260,
            "short": 520,
            "exit": 780,
        }

        for direction, blocks in [
            ("filter", strategy.filter_conditions),
            ("long", strategy.long_conditions),
            ("short", strategy.short_conditions),
            ("exit", strategy.exit_conditions),
        ]:
            for idx, block in enumerate(blocks):
                lib = INDICATOR_LIBRARY.get(block.indicator, {})
                x = block.canvas_pos.x if block.canvas_pos.x else column_x[direction]
                y = block.canvas_pos.y if block.canvas_pos.y else (idx * 120)
                nodes.append(
                    {
                        "id": block.block_id,
                        "type": "conditionBlock",
                        "position": {"x": x, "y": y},
                        "data": {
                            "label": lib.get("name", block.indicator),
                            "indicator": block.indicator,
                            "operator": block.operator,
                            "threshold": block.threshold,
                            "params": block.params,
                            "direction": direction,
                            "enabled": block.enabled,
                            "description": block.description,
                            "category": lib.get("category", ""),
                        },
                        "style": {
                            "background": colors.get(direction, "#888"),
                            "border": "1px solid #FFFFFF",
                        },
                    }
                )

        for direction, blocks in [("long", strategy.long_conditions), ("short", strategy.short_conditions)]:
            if len(blocks) <= 1:
                continue
            gate_id = f"gate_{direction}"
            nodes.append(
                {
                    "id": gate_id,
                    "type": "gateNode",
                    "position": {"x": column_x[direction] + 180, "y": 120},
                    "data": {"gate": strategy.condition_gate.value, "direction": direction},
                }
            )
            for block in blocks:
                edges.append(
                    {
                        "id": f"{block.block_id}_{gate_id}",
                        "source": block.block_id,
                        "target": gate_id,
                    }
                )

        return {
            "strategy_id": strategy.strategy_id,
            "name": strategy.name,
            "nodes": nodes,
            "edges": edges,
        }

    @staticmethod
    def _default_params(indicator: str) -> Dict[str, Any]:
        lib = INDICATOR_LIBRARY.get(indicator, {})
        return {name: schema["default"] for name, schema in lib.get("params", {}).items() if "default" in schema}

    @staticmethod
    def _default_output(indicator: str) -> str:
        outputs = INDICATOR_LIBRARY.get(indicator, {}).get("outputs", [])
        return outputs[0] if outputs else ""
