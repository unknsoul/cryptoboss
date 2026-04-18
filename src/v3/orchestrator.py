"""Orchestrator for CryptoBoss v3 intraday scalper modular microservices."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from src.smc.multi_timeframe_analyzer import MultiTimeframeAnalyzer

from .ai_optimizer import AIOptimizer
from .backtesting_engine import BacktestingEngine
from .config import V3SystemConfig, build_default_v3_config
from .data_engine import DataEngine
from .execution_engine import ExecutionEngine
from .market_structure_engine import MarketStructureEngine
from .models import ExecutionReport
from .performance_tracker import PerformanceTracker
from .risk_engine import RiskEngine
from .signal_engine import SignalEngine
from .smart_money_engine import SmartMoneyEngine
from .strategy_builder_service import StrategyBuilderService


class IntradayScalperV3System:
    """End-to-end modular system implementing the v3 intraday scalper architecture."""

    def __init__(self, config: Optional[V3SystemConfig] = None):
        self.config = config or build_default_v3_config()

        self.data_engine = DataEngine(self.config.data_engine)
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.market_structure_engine = MarketStructureEngine(self.config.market_structure_engine)
        self.smart_money_engine = SmartMoneyEngine(self.config.smart_money_engine)
        self.signal_engine = SignalEngine(self.config.signal_engine)
        self.risk_engine = RiskEngine(self.config.risk_engine)
        self.execution_engine = ExecutionEngine(self.config.execution_engine)
        self.strategy_builder = StrategyBuilderService(self.config.strategy_builder)
        self.backtesting_engine = BacktestingEngine(self.config.backtesting_engine)
        self.performance_tracker = PerformanceTracker(self.config.performance_tracker)
        self.ai_optimizer = AIOptimizer(self.config.ai_optimizer)

        self.last_cycle: Dict[str, Any] = {}

    def run_cycle(
        self,
        symbol: str,
        frames_by_timeframe: Dict[str, pd.DataFrame],
        account_state: Dict[str, float],
        strategy_used: str = "v3_intraday_scalper",
        order_type: str = "market",
    ) -> Dict[str, Any]:
        prepared = self.data_engine.prepare_multi_timeframe(frames_by_timeframe)
        mtf_result = self.mtf_analyzer.analyze(prepared)

        market_structure = self.market_structure_engine.analyze(
            prepared["15m"],
            prepared["5m"],
        )
        market_structure["mtf_alignment"] = mtf_result

        smart_money_all = self.smart_money_engine.analyze_multi_timeframe(prepared)
        smart_money_primary = smart_money_all.get("5m") or next(iter(smart_money_all.values()))

        feature_row = self._build_signal_feature_row(
            market_structure,
            smart_money_primary,
            prepared["1m"],
            mtf_result,
        )
        ai_probability = self.ai_optimizer.predict_trade_success_probability(feature_row)

        signal = self.signal_engine.evaluate(
            market_structure=market_structure,
            smart_money=smart_money_primary,
            ltf_frame=prepared["1m"],
            win_probability=ai_probability,
        )

        risk_decision = self.risk_engine.evaluate(
            signal=signal,
            account_state=account_state,
            smart_money=smart_money_primary,
            timestamp=datetime.utcnow(),
        )

        execution_report = ExecutionReport(
            accepted=False,
            status="skipped",
            action=signal.action,
            symbol=symbol,
            reason="Execution skipped",
        )
        trade_id = None

        if signal.action in ("BUY", "SELL") and risk_decision.approved:
            snapshot = self._build_market_snapshot(prepared["1m"])
            execution_report = self.execution_engine.execute(
                symbol=symbol,
                signal=signal,
                risk_decision=risk_decision,
                market_snapshot=snapshot,
                order_type=order_type,
            )

            if execution_report.accepted:
                self.risk_engine.register_trade()
                trade_id = self.performance_tracker.log_trade_entry(
                    symbol=symbol,
                    strategy_used=strategy_used,
                    action=signal.action,
                    entry_price=float(execution_report.filled_price or signal.entry_price or snapshot["last_price"]),
                    stop_loss=float(risk_decision.stop_loss or 0.0),
                    take_profit=float(risk_decision.take_profit or 0.0),
                    reason_for_trade=signal.reason,
                    metadata={
                        "confidence": signal.confidence,
                        "risk_metadata": risk_decision.metadata,
                    },
                )

        cycle_result = {
            "version": self.config.version,
            "architecture": self.config.architecture,
            "symbol": symbol,
            "modules": self.config.modules,
            "market_structure": market_structure,
            "smart_money": smart_money_primary,
            "signal": asdict(signal),
            "risk": asdict(risk_decision),
            "execution": asdict(execution_report),
            "trade_id": trade_id,
            "ai_probability": ai_probability,
            "performance": self.performance_tracker.stats(),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.last_cycle = cycle_result
        return cycle_result

    def close_trade(self, trade_id: str, exit_price: float, reason_for_trade: str) -> bool:
        return self.performance_tracker.log_trade_exit(
            trade_id=trade_id,
            exit_price=exit_price,
            reason_for_trade=reason_for_trade,
        )

    def create_strategy(
        self,
        name: str,
        indicators: list,
        smc_conditions: list,
        risk_rules: Dict[str, Any],
        symbol: str = "BTC/USDT",
        timeframe: str = "5m",
        multi_timeframe_rules: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        artifact = self.strategy_builder.create_custom_strategy(
            name=name,
            indicators=indicators,
            smc_conditions=smc_conditions,
            risk_rules=risk_rules,
            symbol=symbol,
            timeframe=timeframe,
            multi_timeframe_rules=multi_timeframe_rules,
        )
        return asdict(artifact)

    def backtest(
        self,
        df: pd.DataFrame,
        signal_fn,
        strategy_name: str,
        strategy_id: str,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
    ) -> Dict[str, Any]:
        return self.backtesting_engine.run(
            df=df,
            signal_fn=signal_fn,
            strategy_name=strategy_name,
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
        )

    def fit_ai_optimizer(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        return self.ai_optimizer.fit(features, labels)

    def optimize_strategy_parameters(self, evaluator, parameter_space: Dict[str, list], n_trials: int = 30) -> Dict[str, Any]:
        return self.ai_optimizer.optimize_parameters(
            evaluator=evaluator,
            parameter_space=parameter_space,
            n_trials=n_trials,
        )

    def create_rule_strategy(
        self,
        name: str,
        rules: list,
        weights: Dict[str, float],
        buy_threshold: float = 6.0,
        sell_threshold: float = 6.0,
        force_trade_if_no_signal: bool = True,
    ) -> Dict[str, Any]:
        return self.strategy_builder.create_rule_strategy(
            name=name,
            rules=rules,
            weights=weights,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            force_trade_if_no_signal=force_trade_if_no_signal,
        )

    def validate_rule_strategy(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        param_grid: Optional[Dict[str, list]] = None,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        return self.strategy_builder.validate_rule_strategy(
            strategy_name=strategy_name,
            df=df,
            backtesting_engine=self.backtesting_engine,
            param_grid=param_grid,
            n_splits=n_splits,
        )

    @staticmethod
    def _build_market_snapshot(frame: pd.DataFrame) -> Dict[str, float]:
        latest = frame.iloc[-1]
        close = float(latest["close"])

        spread_pct = float(latest.get("spread", ((float(latest["high"]) - float(latest["low"])) / close) * 100.0))
        volatility = float(latest.get("volatility", 0.0))
        expected_slippage_pct = max(volatility * 20.0, 0.01)

        return {
            "last_price": close,
            "spread_pct": spread_pct,
            "expected_slippage_pct": expected_slippage_pct,
        }

    @staticmethod
    def _build_signal_feature_row(
        market_structure: Dict[str, Any],
        smart_money: Dict[str, Any],
        ltf_frame: pd.DataFrame,
        mtf_result: Optional[Dict[str, Any]] = None,
    ) -> list:
        latest = ltf_frame.iloc[-1]
        mtf_score = float((mtf_result or {}).get("alignment_score", 0.0))
        return [
            float(bool(market_structure.get("bos", {}).get("confirmed", False))),
            float(bool(market_structure.get("choch", {}).get("confirmed", False))),
            float(smart_money.get("ob_signal", 0)),
            float(smart_money.get("fvg_signal", 0)),
            float(bool(smart_money.get("liquidity_sweep", False))),
            float(latest.get("volatility", 0.0)),
            float(latest.get("momentum", 0.0)),
            mtf_score,
        ]
