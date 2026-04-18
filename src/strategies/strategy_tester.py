"""Professional strategy backtester for CryptoBoss v12.0."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    trade_id: str
    direction: str
    symbol: str
    timeframe: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    size: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    take_profit_3: Optional[float]
    exit_reason: str
    gross_pnl: float
    fees: float
    net_pnl: float
    pnl_pct: float
    risk_reward_achieved: float
    mae: float = 0.0
    mfe: float = 0.0
    bars_held: int = 0
    setup_type: str = ""
    components: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    strategy_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_capital: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    expectancy: float
    avg_rr: float

    max_drawdown_pct: float
    max_drawdown_usd: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    recovery_factor: float

    avg_win_pct: float
    avg_loss_pct: float
    avg_trade_duration: str
    longest_losing_streak: int
    longest_winning_streak: int
    max_consecutive_losses: int

    trades: List[TradeRecord]
    equity_curve: pd.Series
    drawdown_series: pd.Series
    monthly_returns: pd.Series

    walk_forward_results: Optional[Dict] = None
    monte_carlo_results: Optional[Dict] = None

    def to_summary_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "period": f"{self.start_date.date()} to {self.end_date.date()}",
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "net_profit_usd": round(self.net_profit, 2),
            "net_profit_pct": round(self.net_profit / self.initial_capital * 100.0, 2),
            "total_trades": self.total_trades,
            "win_rate_pct": round(self.win_rate * 100.0, 2),
            "profit_factor": round(self.profit_factor, 3),
            "expectancy": round(self.expectancy, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100.0, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "avg_rr": round(self.avg_rr, 2),
            "avg_win_pct": round(self.avg_win_pct, 2),
            "avg_loss_pct": round(self.avg_loss_pct, 2),
            "max_consecutive_losses": self.max_consecutive_losses,
            "longest_winning_streak": self.longest_winning_streak,
        }


class StrategyTester:
    """Event-driven backtester with fees, slippage, partial exits, and analytics."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        slippage_pct: float = 0.0001,
        partial_exits: bool = True,
        partial_exit_pcts: Optional[List[float]] = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_pct = slippage_pct
        self.partial_exits = partial_exits
        self.partial_exit_pcts = partial_exit_pcts or [0.5, 0.3, 0.2]

    def run(
        self,
        df: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame], Dict],
        strategy_name: str = "Unnamed",
        strategy_id: str = "TEST",
        symbol: str = "BTC/USDT",
        timeframe: str = "5m",
    ) -> BacktestResult:
        capital = self.initial_capital
        trades: List[TradeRecord] = []
        equity_curve = [capital]
        open_positions: List[Dict] = []
        trade_counter = 0

        warmup = 50

        for i in range(warmup, len(df)):
            bar = df.iloc[i]
            slice_df = df.iloc[: i + 1]

            closed_now, capital = self._update_positions(open_positions, bar, capital)
            trades.extend(closed_now)

            try:
                signal = signal_fn(slice_df)
            except Exception:
                signal = {"action": "HOLD"}

            if signal.get("action") in ("BUY", "SELL") and len(open_positions) < 5:
                direction = "long" if signal["action"] == "BUY" else "short"
                entry = float(signal.get("entry", float(bar["close"])))
                sl = float(signal.get("sl", 0.0))
                tp1 = float(signal.get("tp1", 0.0))
                tp2 = signal.get("tp2")
                tp3 = signal.get("tp3")

                if sl != 0.0 and abs(entry - sl) > 1e-9:
                    size = float(signal.get("size", capital * 0.01 / abs(entry - sl)))
                else:
                    size = float(signal.get("size", 0.01))

                slippage = entry * self.slippage_pct
                entry = entry + slippage if direction == "long" else entry - slippage

                entry_fee = entry * size * self.taker_fee
                capital -= entry_fee

                trade_counter += 1
                open_positions.append(
                    {
                        "trade_id": f"T{trade_counter:04d}",
                        "direction": direction,
                        "entry_time": bar.name,
                        "entry_price": entry,
                        "size": size,
                        "remaining_size": size,
                        "sl": sl,
                        "tp1": tp1,
                        "tp1_hit": False,
                        "tp2": tp2,
                        "tp2_hit": False,
                        "tp3": tp3,
                        "tp3_hit": False,
                        "mae": 0.0,
                        "mfe": 0.0,
                        "bars": 0,
                        "setup_type": signal.get("setup_type", ""),
                        "components": signal.get("components", {}),
                        "entry_fee": entry_fee,
                        "symbol": symbol,
                        "timeframe": timeframe,
                    }
                )

            equity_curve.append(capital + self._unrealized_pnl(open_positions, bar))

        last_bar = df.iloc[-1]
        for position in open_positions[:]:
            net_pnl = self._close_position(position, float(last_bar["close"]), "end_of_test")
            capital += net_pnl
            trades.append(
                self._make_trade_record(
                    position,
                    exit_time=last_bar.name,
                    exit_price=float(last_bar["close"]),
                    exit_reason="end_of_test",
                    net_pnl=net_pnl,
                )
            )
        open_positions.clear()
        equity_curve.append(capital)

        equity = pd.Series(equity_curve, name="equity")
        return self._compute_metrics(trades, equity, strategy_id, strategy_name, symbol, timeframe, df)

    def _update_positions(self, open_positions: List[Dict], bar: pd.Series, capital: float) -> Tuple[List[TradeRecord], float]:
        closed: List[TradeRecord] = []
        to_remove: List[Dict] = []

        for position in open_positions:
            position["bars"] += 1
            direction = position["direction"]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])

            if direction == "long":
                position["mae"] = min(position["mae"], low - position["entry_price"])
                position["mfe"] = max(position["mfe"], high - position["entry_price"])
            else:
                position["mae"] = min(position["mae"], position["entry_price"] - high)
                position["mfe"] = max(position["mfe"], position["entry_price"] - low)

            exit_price = None
            exit_reason = None

            if self.partial_exits:
                if not position["tp1_hit"] and position.get("tp1"):
                    tp1_hit = (direction == "long" and high >= position["tp1"]) or (
                        direction == "short" and low <= position["tp1"]
                    )
                    if tp1_hit:
                        position["tp1_hit"] = True
                        pct = self.partial_exit_pcts[0] if self.partial_exit_pcts else 0.5
                        exit_size = position["size"] * pct
                        position["remaining_size"] -= exit_size
                        capital += self._partial_pnl(position, float(position["tp1"]), exit_size, direction)
                        position["sl"] = position["entry_price"]

                if not position["tp2_hit"] and position.get("tp2"):
                    tp2_hit = (direction == "long" and high >= position["tp2"]) or (
                        direction == "short" and low <= position["tp2"]
                    )
                    if tp2_hit:
                        position["tp2_hit"] = True
                        pct = self.partial_exit_pcts[1] if len(self.partial_exit_pcts) > 1 else 0.0
                        exit_size = position["size"] * pct
                        position["remaining_size"] -= exit_size
                        capital += self._partial_pnl(position, float(position["tp2"]), exit_size, direction)

            if direction == "long":
                if position.get("sl") and low <= position["sl"]:
                    exit_price, exit_reason = float(position["sl"]), "sl"
                elif position.get("tp3") and high >= float(position["tp3"]):
                    exit_price, exit_reason = float(position["tp3"]), "tp3"
                elif not position.get("tp3") and position.get("tp2") and high >= float(position["tp2"]) and position["tp2_hit"]:
                    exit_price, exit_reason = float(position["tp2"]), "tp2"
                elif position.get("tp1") and high >= float(position["tp1"]) and not position.get("tp2"):
                    exit_price, exit_reason = float(position["tp1"]), "tp1"
            else:
                if position.get("sl") and high >= position["sl"]:
                    exit_price, exit_reason = float(position["sl"]), "sl"
                elif position.get("tp3") and low <= float(position["tp3"]):
                    exit_price, exit_reason = float(position["tp3"]), "tp3"
                elif position.get("tp1") and low <= float(position["tp1"]) and not position.get("tp2"):
                    exit_price, exit_reason = float(position["tp1"]), "tp1"

            if exit_price is not None:
                net_pnl = self._close_position(position, exit_price, exit_reason)
                capital += net_pnl
                closed.append(
                    self._make_trade_record(
                        position,
                        exit_time=bar.name,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        net_pnl=net_pnl,
                    )
                )
                to_remove.append(position)

        for position in to_remove:
            open_positions.remove(position)

        return closed, capital

    def _partial_pnl(self, position: Dict, price: float, size: float, direction: str) -> float:
        if size <= 0:
            return 0.0

        if direction == "long":
            gross = (price - position["entry_price"]) * size
        else:
            gross = (position["entry_price"] - price) * size

        fee = price * size * self.taker_fee
        return gross - fee

    def _close_position(self, position: Dict, exit_price: float, reason: str) -> float:
        _ = reason
        size = position.get("remaining_size", position["size"])
        if position["direction"] == "long":
            gross = (exit_price - position["entry_price"]) * size
        else:
            gross = (position["entry_price"] - exit_price) * size

        fee = exit_price * size * self.taker_fee
        return gross - fee

    def _make_trade_record(
        self,
        position: Dict,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        net_pnl: float,
    ) -> TradeRecord:
        entry_price = position["entry_price"]
        size = position["size"]
        direction = position["direction"]

        stop = float(position.get("sl", 0.0) or 0.0)
        risk_per_unit = abs(entry_price - stop) if stop else 0.0
        risk = risk_per_unit * size if risk_per_unit > 0 else 1.0
        rr_achieved = net_pnl / risk if risk else 0.0

        return TradeRecord(
            trade_id=position["trade_id"],
            direction=direction,
            symbol=position.get("symbol", "BTC/USDT"),
            timeframe=position.get("timeframe", "5m"),
            entry_time=position["entry_time"],
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            size=size,
            stop_loss=stop,
            take_profit_1=float(position.get("tp1", 0.0) or 0.0),
            take_profit_2=position.get("tp2"),
            take_profit_3=position.get("tp3"),
            exit_reason=exit_reason,
            gross_pnl=net_pnl,
            fees=entry_price * size * self.taker_fee,
            net_pnl=net_pnl,
            pnl_pct=(net_pnl / (entry_price * size) * 100.0) if size else 0.0,
            risk_reward_achieved=rr_achieved,
            mae=position.get("mae", 0.0),
            mfe=position.get("mfe", 0.0),
            bars_held=position.get("bars", 0),
            setup_type=position.get("setup_type", ""),
            components=position.get("components", {}),
        )

    @staticmethod
    def _unrealized_pnl(open_positions: List[Dict], bar: pd.Series) -> float:
        total = 0.0
        close = float(bar["close"])
        for position in open_positions:
            remaining = position.get("remaining_size", position["size"])
            if position["direction"] == "long":
                total += (close - position["entry_price"]) * remaining
            else:
                total += (position["entry_price"] - close) * remaining
        return total

    def _compute_metrics(
        self,
        trades: List[TradeRecord],
        equity: pd.Series,
        strategy_id: str,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> BacktestResult:
        if not trades:
            return self._empty_result(strategy_id, strategy_name, symbol, timeframe, df, equity)

        returns = equity.pct_change().dropna()
        drawdown = equity / equity.cummax() - 1.0
        max_drawdown_pct = float(drawdown.min())
        max_drawdown_usd = float((equity.cummax() - equity).max())

        wins = [trade for trade in trades if trade.net_pnl > 0]
        losses = [trade for trade in trades if trade.net_pnl <= 0]

        gross_profit = sum(trade.net_pnl for trade in wins)
        gross_loss = abs(sum(trade.net_pnl for trade in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win_pct = float(np.mean([trade.pnl_pct for trade in wins])) if wins else 0.0
        avg_loss_pct = float(np.mean([trade.pnl_pct for trade in losses])) if losses else 0.0
        win_rate = len(wins) / len(trades)
        expectancy = (win_rate * avg_win_pct) - ((1.0 - win_rate) * abs(avg_loss_pct))

        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(365.0 * 24.0))

        downside = returns[returns < 0]
        sortino = 0.0
        if len(downside) > 1 and downside.std() > 0:
            sortino = float((returns.mean() / downside.std()) * np.sqrt(365.0 * 24.0))

        annual_return = 0.0
        if len(equity) > 1 and equity.iloc[0] > 0:
            annual_return = float((equity.iloc[-1] / equity.iloc[0]) ** (365.0 / max(len(equity), 1)) - 1.0)

        calmar = annual_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0.0
        recovery_factor = (equity.iloc[-1] - equity.iloc[0]) / max_drawdown_usd if max_drawdown_usd > 0 else 0.0
        avg_rr = float(np.mean([trade.risk_reward_achieved for trade in trades])) if trades else 0.0

        max_win_streak = 0
        max_loss_streak = 0
        current_win = 0
        current_loss = 0
        for trade in trades:
            if trade.net_pnl > 0:
                current_win += 1
                current_loss = 0
            else:
                current_loss += 1
                current_win = 0
            max_win_streak = max(max_win_streak, current_win)
            max_loss_streak = max(max_loss_streak, current_loss)

        durations = []
        for trade in trades:
            if trade.exit_time is None:
                continue
            duration_hours = (pd.Timestamp(trade.exit_time) - pd.Timestamp(trade.entry_time)).total_seconds() / 3600.0
            durations.append(duration_hours)
        avg_duration = f"{float(np.mean(durations)):.1f}h" if durations else "N/A"

        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            eq_index = pd.date_range(start=df.index[0], periods=len(equity), freq="T")
            eq_series = pd.Series(equity.values, index=eq_index)
            monthly_returns = eq_series.resample("ME").last().pct_change().dropna()
        else:
            monthly_returns = pd.Series([], dtype=float)

        return BacktestResult(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=df.index[0] if len(df) > 0 else pd.Timestamp.utcnow(),
            end_date=df.index[-1] if len(df) > 0 else pd.Timestamp.utcnow(),
            initial_capital=self.initial_capital,
            final_capital=float(equity.iloc[-1]),
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=float(equity.iloc[-1]) - self.initial_capital,
            profit_factor=float(profit_factor),
            expectancy=float(expectancy),
            avg_rr=avg_rr,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_usd=max_drawdown_usd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=float(calmar),
            recovery_factor=float(recovery_factor),
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            avg_trade_duration=avg_duration,
            longest_losing_streak=max_loss_streak,
            longest_winning_streak=max_win_streak,
            max_consecutive_losses=max_loss_streak,
            trades=trades,
            equity_curve=equity,
            drawdown_series=drawdown,
            monthly_returns=monthly_returns,
        )

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        signal_fn_factory: Callable[[Dict], Callable],
        param_grid: Dict[str, List],
        n_splits: int = 5,
        **kwargs,
    ) -> Dict:
        if n_splits < 2:
            return {"n_splits": n_splits, "folds": []}

        split_size = max(len(df) // n_splits, 1)
        folds = []

        for fold in range(n_splits - 1):
            train_end = (fold + 1) * split_size
            test_end = min((fold + 2) * split_size, len(df))

            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:test_end]

            if len(train_df) < 60 or len(test_df) < 20:
                continue

            best_params = None
            best_pf = -np.inf

            import itertools

            keys, values = zip(*param_grid.items()) if param_grid else ([], [])
            combos = itertools.product(*values) if values else [()]

            for combo in combos:
                params = dict(zip(keys, combo))
                try:
                    signal_fn = signal_fn_factory(params)
                    result = self.run(train_df, signal_fn, **kwargs)
                    if result.profit_factor > best_pf:
                        best_pf = result.profit_factor
                        best_params = params
                except Exception:
                    continue

            if best_params is None:
                continue

            oos_fn = signal_fn_factory(best_params)
            oos_result = self.run(test_df, oos_fn, **kwargs)
            folds.append(
                {
                    "fold": fold,
                    "best_params": best_params,
                    "train_pf": best_pf,
                    "oos_profit_factor": oos_result.profit_factor,
                    "oos_win_rate": oos_result.win_rate,
                    "oos_net_profit": oos_result.net_profit,
                    "oos_sharpe": oos_result.sharpe_ratio,
                }
            )

        return {
            "n_splits": n_splits,
            "folds": folds,
            "avg_oos_profit_factor": float(np.mean([item["oos_profit_factor"] for item in folds])) if folds else 0.0,
            "avg_oos_win_rate": float(np.mean([item["oos_win_rate"] for item in folds])) if folds else 0.0,
            "avg_oos_sharpe": float(np.mean([item["oos_sharpe"] for item in folds])) if folds else 0.0,
        }

    def run_monte_carlo(
        self,
        result: BacktestResult,
        n_simulations: int = 1000,
        confidence_levels: Optional[List[float]] = None,
    ) -> Dict:
        confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
        if not result.trades:
            return {}

        trade_returns = np.array([trade.net_pnl for trade in result.trades], dtype=float)
        simulated_max_dd = []
        simulated_final_capital = []
        simulated_worst_streak = []

        for _ in range(n_simulations):
            shuffled = np.random.permutation(trade_returns)
            equity = self.initial_capital + np.cumsum(shuffled)
            equity = np.insert(equity, 0, self.initial_capital)
            equity_series = pd.Series(equity)

            drawdown = float((equity_series / equity_series.cummax() - 1.0).min())
            simulated_max_dd.append(drawdown)
            simulated_final_capital.append(float(equity[-1]))

            streak = 0
            max_streak = 0
            for value in shuffled:
                if value < 0:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            simulated_worst_streak.append(max_streak)

        output: Dict[str, float] = {}
        for level in confidence_levels:
            percentile = (1.0 - level) * 100.0
            output[f"max_dd_{int(level * 100)}pct"] = float(np.percentile(simulated_max_dd, percentile))
            output[f"final_capital_{int(level * 100)}pct"] = float(np.percentile(simulated_final_capital, percentile))

        output["median_final_capital"] = float(np.median(simulated_final_capital))
        output["mean_max_dd"] = float(np.mean(simulated_max_dd))
        output["worst_streak_95pct"] = int(np.percentile(simulated_worst_streak, 95))
        output["probability_of_profit"] = float(np.mean(np.array(simulated_final_capital) > self.initial_capital))
        output["n_simulations"] = int(n_simulations)

        return output

    def _empty_result(
        self,
        strategy_id: str,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        equity: pd.Series,
    ) -> BacktestResult:
        start = df.index[0] if len(df) > 0 else pd.Timestamp.utcnow()
        end = df.index[-1] if len(df) > 0 else pd.Timestamp.utcnow()

        return BacktestResult(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start,
            end_date=end,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            net_profit=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_rr=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_usd=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            recovery_factor=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            avg_trade_duration="N/A",
            longest_losing_streak=0,
            longest_winning_streak=0,
            max_consecutive_losses=0,
            trades=[],
            equity_curve=equity,
            drawdown_series=equity * 0,
            monthly_returns=pd.Series([], dtype=float),
        )
