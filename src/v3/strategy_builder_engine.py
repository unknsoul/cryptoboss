"""Backend strategy builder engine with rule composition and robustness validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RuleStrategy:
    name: str
    rules: List[str]
    weights: Dict[str, float]
    buy_threshold: float
    sell_threshold: float
    force_trade_if_no_signal: bool
    metadata: Dict[str, Any]


class StrategyBuilderEngine:
    """Builds and evaluates weighted rule-composer strategies."""

    def __init__(self):
        self._strategies: Dict[str, RuleStrategy] = {}

    def build(
        self,
        name: str,
        rules: List[str],
        weights: Optional[Dict[str, float]] = None,
        buy_threshold: float = 6.0,
        sell_threshold: float = 6.0,
        force_trade_if_no_signal: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RuleStrategy:
        normalized_rules = [str(rule) for rule in rules]
        default_weights = {rule: 1.0 for rule in normalized_rules}
        default_weights.update(weights or {})

        strategy = RuleStrategy(
            name=name,
            rules=normalized_rules,
            weights=default_weights,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            force_trade_if_no_signal=force_trade_if_no_signal,
            metadata=metadata or {},
        )
        self._strategies[name] = strategy
        return strategy

    def get(self, name: str) -> RuleStrategy:
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' does not exist")
        return self._strategies[name]

    def evaluate_dataframe(self, frame: pd.DataFrame, strategy: RuleStrategy) -> pd.DataFrame:
        df = frame.copy()
        buy_scores: List[float] = []
        sell_scores: List[float] = []
        signals: List[str] = []

        for _, row in df.iterrows():
            buy_score = 0.0
            sell_score = 0.0

            for rule in strategy.rules:
                weight = float(strategy.weights.get(rule, 1.0))
                value = float(row.get(rule, 0.0))

                if value > 0:
                    buy_score += weight * value
                elif value < 0:
                    sell_score += weight * abs(value)

            if buy_score >= strategy.buy_threshold and buy_score > sell_score:
                signal = "BUY"
            elif sell_score >= strategy.sell_threshold and sell_score > buy_score:
                signal = "SELL"
            elif strategy.force_trade_if_no_signal:
                if buy_score > sell_score and buy_score > 0:
                    signal = "BUY"
                elif sell_score > buy_score and sell_score > 0:
                    signal = "SELL"
                else:
                    signal = "HOLD"
            else:
                signal = "HOLD"

            buy_scores.append(buy_score)
            sell_scores.append(sell_score)
            signals.append(signal)

        df["buy_score"] = buy_scores
        df["sell_score"] = sell_scores
        df["strategy_score"] = np.array(buy_scores) - np.array(sell_scores)
        df["signal"] = signals
        return df

    def compile_signal_function(self, strategy: RuleStrategy) -> Callable[[pd.DataFrame], Dict[str, Any]]:
        def _signal_fn(df_slice: pd.DataFrame) -> Dict[str, Any]:
            if df_slice.empty:
                return {"action": "HOLD"}

            scored = self.evaluate_dataframe(df_slice.tail(1), strategy)
            last = scored.iloc[-1]
            action = str(last["signal"])
            price = float(df_slice["close"].iloc[-1])

            if action == "BUY":
                return {
                    "action": "BUY",
                    "entry": price,
                    "sl": price * 0.997,
                    "tp1": price * 1.004,
                    "tp2": price * 1.008,
                    "score": float(last["buy_score"]),
                }

            if action == "SELL":
                return {
                    "action": "SELL",
                    "entry": price,
                    "sl": price * 1.003,
                    "tp1": price * 0.996,
                    "tp2": price * 0.992,
                    "score": float(last["sell_score"]),
                }

            return {"action": "HOLD"}

        return _signal_fn


class StrategyValidationEngine:
    """Validates strategies using walk-forward and out-of-sample analysis."""

    def __init__(self, backtesting_engine):
        self.backtesting_engine = backtesting_engine

    def walk_forward_validation(
        self,
        df: pd.DataFrame,
        signal_fn_factory: Callable[[Dict[str, Any]], Callable[[pd.DataFrame], Dict[str, Any]]],
        param_grid: Dict[str, List[Any]],
        n_splits: int = 5,
        strategy_name: str = "RuleComposer",
        strategy_id: str = "RULE",
    ) -> Dict[str, Any]:
        wf = self.backtesting_engine.run_walk_forward(
            df,
            signal_fn_factory,
            param_grid,
            n_splits=n_splits,
            strategy_name=strategy_name,
            strategy_id=strategy_id,
        )
        wf["robustness_score"] = self.robustness_score(wf)
        return wf

    def out_of_sample_test(
        self,
        df: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame], Dict[str, Any]],
        split: float = 0.7,
        strategy_name: str = "RuleComposer",
        strategy_id: str = "RULE",
    ) -> Dict[str, Any]:
        if len(df) < 100:
            return {"error": "Not enough bars for out-of-sample testing"}

        split_index = int(len(df) * split)
        in_sample = df.iloc[:split_index]
        out_of_sample = df.iloc[split_index:]

        in_result = self.backtesting_engine.run(
            in_sample,
            signal_fn,
            strategy_name=f"{strategy_name}_in_sample",
            strategy_id=f"{strategy_id}_IS",
        )
        out_result = self.backtesting_engine.run(
            out_of_sample,
            signal_fn,
            strategy_name=f"{strategy_name}_out_of_sample",
            strategy_id=f"{strategy_id}_OOS",
        )

        return {
            "split": split,
            "in_sample": in_result["summary"],
            "out_of_sample": out_result["summary"],
            "degradation": {
                "win_rate_drop": float(in_result["summary"]["win_rate"] - out_result["summary"]["win_rate"]),
                "profit_factor_drop": float(
                    in_result["summary"]["profit_factor"] - out_result["summary"]["profit_factor"]
                ),
            },
        }

    @staticmethod
    def robustness_score(walk_forward_result: Dict[str, Any]) -> float:
        folds = walk_forward_result.get("folds", [])
        if not folds:
            return 0.0

        oos_pf = np.array([float(fold.get("oos_profit_factor", 0.0)) for fold in folds], dtype=float)
        oos_sharpe = np.array([float(fold.get("oos_sharpe", 0.0)) for fold in folds], dtype=float)

        pf_mean = float(np.mean(oos_pf))
        sharpe_mean = float(np.mean(oos_sharpe))
        pf_std = float(np.std(oos_pf))

        pf_score = max(min((pf_mean - 1.0) / 1.5, 1.0), 0.0)
        sharpe_score = max(min(sharpe_mean / 2.0, 1.0), 0.0)
        consistency_score = max(min(1.0 - min(pf_std, 1.0), 1.0), 0.0)

        robustness = 0.4 * pf_score + 0.3 * sharpe_score + 0.3 * consistency_score
        return round(float(robustness), 4)
