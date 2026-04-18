"""Backtesting engine service for v3 intraday scalper architecture."""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.strategies.strategy_tester import StrategyTester

from .config import BacktestingConfig
from .models import BacktestSummaryV3


class BacktestingEngine:
    """Professional backtesting service with spread/slippage/commission simulation."""

    def __init__(self, config: Optional[BacktestingConfig] = None, initial_capital: float = 10000.0):
        self.config = config or BacktestingConfig()
        self.initial_capital = initial_capital

        fee = 0.0004 if self.config.simulate_commission else 0.0
        slippage = 0.0001 if self.config.simulate_slippage else 0.0

        self.tester = StrategyTester(
            initial_capital=initial_capital,
            maker_fee=fee / 2,
            taker_fee=fee,
            slippage_pct=slippage,
            partial_exits=True,
        )
        self._last_result = None

    def run(
        self,
        df: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame], Dict],
        strategy_name: str = "v3_intraday_scalper",
        strategy_id: str = "V3",
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
    ) -> Dict[str, object]:
        data = self._prepare_data(df)

        result = self.tester.run(
            data,
            signal_fn,
            strategy_name=strategy_name,
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
        )
        self._last_result = result

        summary = BacktestSummaryV3(
            win_rate=float(result.win_rate),
            profit_factor=float(result.profit_factor),
            drawdown=float(abs(result.max_drawdown_pct)),
            sharpe_ratio=float(result.sharpe_ratio),
            total_trades=int(result.total_trades),
            net_pnl=float(result.net_profit),
            metadata={
                "metrics_requested": self.config.metrics,
                "simulation": {
                    "spread": self.config.simulate_spread,
                    "slippage": self.config.simulate_slippage,
                    "commission": self.config.simulate_commission,
                },
            },
        )

        trade_logs = [
            {
                "trade_id": trade.trade_id,
                "direction": trade.direction,
                "entry_time": str(trade.entry_time),
                "exit_time": str(trade.exit_time) if trade.exit_time is not None else None,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "net_pnl": trade.net_pnl,
                "reason": trade.exit_reason,
            }
            for trade in result.trades
        ]

        return {
            "summary": asdict(summary),
            "metrics": result.to_summary_dict(),
            "trade_logs": trade_logs,
            "equity_curve": [float(value) for value in result.equity_curve.tail(1000).tolist()],
            "drawdown_curve": [float(value) for value in result.drawdown_series.tail(1000).tolist()],
        }

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        signal_fn_factory: Callable[[Dict], Callable],
        param_grid: Dict[str, List],
        n_splits: int = 5,
        strategy_name: str = "v3_intraday_scalper",
        strategy_id: str = "V3",
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
    ) -> Dict[str, object]:
        data = self._prepare_data(df)
        return self.tester.run_walk_forward(
            data,
            signal_fn_factory,
            param_grid,
            n_splits=n_splits,
            strategy_name=strategy_name,
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
        )

    def run_monte_carlo(self, n_simulations: int = 500) -> Dict[str, object]:
        if self._last_result is None:
            return {}
        return self.tester.run_monte_carlo(self._last_result, n_simulations=n_simulations)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError("No data provided for backtesting")

        data = df.copy()
        data.columns = [str(column).lower() for column in data.columns]

        for required in ("open", "high", "low", "close"):
            if required not in data.columns:
                raise ValueError(f"Missing required column for backtest: {required}")

        if not isinstance(data.index, pd.DatetimeIndex):
            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
                data = data.set_index("timestamp")
            else:
                raise ValueError("Backtest data must include DatetimeIndex or timestamp column")

        if self.config.data_range == "last_2_years":
            cutoff = data.index.max() - timedelta(days=730)
            data = data.loc[data.index >= cutoff]

        return data.sort_index()
