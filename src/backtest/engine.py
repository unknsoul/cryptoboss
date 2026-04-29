"""
Simple Backtest Engine - Works with new architecture

A lightweight backtester that uses the new src/ structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from src.risk.controls import RiskConfig, RiskController
from src.risk.position_sizing import fixed_fractional_position_size

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    side: str = "long"
    size: float = 1.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission_cost: float = 0.0
    risk_amount: float = 0.0


@dataclass
class BacktestResult:
    """Backtest results container."""
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    num_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    equity_curve: List[float]
    trades: List[Trade]
    halted: bool = False
    halt_reason: str = ""
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_spread_cost: float = 0.0


@dataclass
class BacktestExecutionConfig:
    """Execution realism inputs."""
    commission_pct: float = 0.001
    spread_bps: float = 2.0
    slippage_bps: float = 5.0
    slippage_volatility_multiplier: float = 1.0


class SimpleBacktest:
    """
    Simple backtester for strategy validation.
    
    Usage:
        bt = SimpleBacktest(capital=10000)
        
        # Run with a strategy
        result = bt.run(df, strategy)
        
        # Get metrics
        print(f"Return: {result.total_return_pct:.2f}%")
        print(f"Sharpe: {result.sharpe_ratio:.2f}")
    """
    
    def __init__(
        self,
        capital: float = 10000.0,
        fee_rate: float = 0.001,
        slippage_bps: float = 5.0
    ):
        self.initial_capital = capital
        self.capital = capital
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.position = 0.0
        self.position_value = 0.0
    
    def run(self, df: pd.DataFrame, strategy) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            strategy: Strategy object with generate_signal(df, i, price) method
        """
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.position = 0.0
        self.position_value = 0.0
        
        current_trade: Optional[Trade] = None
        
        for i in range(len(df)):
            price = df['close'].iloc[i]
            timestamp = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
            
            # Update equity
            if self.position > 0:
                current_equity = self.capital + (self.position * price)
            else:
                current_equity = self.capital
            self.equity_curve.append(current_equity)
            
            # Get signal from strategy
            signal = strategy.generate_signal(df, i, price)
            action = signal.get('action', 'HOLD')
            
            # Process signal
            if action == 'BUY' and self.position == 0:
                # Open position
                size = signal.get('size', self.capital * 0.95 / price)
                cost = size * price * (1 + self.fee_rate + self.slippage_bps / 10000)
                
                if cost <= self.capital:
                    self.position = size
                    self.capital -= cost
                    current_trade = Trade(
                        entry_time=timestamp,
                        entry_price=price,
                        side='long',
                        size=size
                    )
            
            elif action == 'SELL' and self.position > 0:
                # Close position
                proceeds = self.position * price * (1 - self.fee_rate - self.slippage_bps / 10000)
                
                if current_trade:
                    current_trade.exit_time = timestamp
                    current_trade.exit_price = price
                    current_trade.pnl = proceeds - (current_trade.size * current_trade.entry_price)
                    current_trade.pnl_pct = (price - current_trade.entry_price) / current_trade.entry_price * 100
                    current_trade.exit_reason = signal.get('reason', 'SIGNAL')
                    self.trades.append(current_trade)
                
                self.capital += proceeds
                self.position = 0
                current_trade = None
        
        # Close any open position at end
        if self.position > 0 and current_trade:
            price = df['close'].iloc[-1]
            proceeds = self.position * price * (1 - self.fee_rate)
            current_trade.exit_time = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
            current_trade.exit_price = price
            current_trade.pnl = proceeds - (current_trade.size * current_trade.entry_price)
            current_trade.pnl_pct = (price - current_trade.entry_price) / current_trade.entry_price * 100
            current_trade.exit_reason = 'END_OF_DATA'
            self.trades.append(current_trade)
            self.capital += proceeds
            self.position = 0
        
        return self._calculate_results()
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest metrics."""
        final_capital = self.capital
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Trade stats
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)
        max_drawdown_pct = max_drawdown * 100
        
        # Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            num_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            equity_curve=self.equity_curve,
            trades=self.trades
        )


class SlippageModel:
    """Advanced slippage modeling."""
    
    @staticmethod
    def adaptive_slippage(
        price: float,
        size: float,
        side: str,
        volatility: float,
        liquidity_factor: float = 1.0
    ) -> float:
        """
        Calculate adaptive slippage based on volatility and size.
        
        Args:
            price: Current asset price
            size: Trade size
            side: 'BUY' or 'SELL'
            volatility: Current volatility (e.g., ATR/Price)
            liquidity_factor: Factor representing market liquidity
            
        Returns:
            Estimated slippage per unit in quote currency
        """
        # Base slippage relative to volatility
        base_slippage = price * volatility * 0.05
        
        # Impact model (square root law approximation)
        impact = 0.1 * np.sqrt(size) * base_slippage / liquidity_factor
        
        return impact


class RealBacktestEngine(SimpleBacktest):
    """
    Production-grade backtest engine with advanced features.
    Extends SimpleBacktest with realistic market constraints.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,
        slippage_model: Optional[SlippageModel] = None,
        execution_config: Optional[BacktestExecutionConfig] = None,
        risk_config: Optional[RiskConfig] = None,
    ):
        super().__init__(capital=initial_capital, fee_rate=fee_rate)
        self.slippage_model = slippage_model or SlippageModel()
        self.equity = initial_capital  # Alias for self.capital to match test expectation
        self.execution_config = execution_config or BacktestExecutionConfig(commission_pct=fee_rate)
        self.risk_controller = RiskController(initial_equity=initial_capital, config=risk_config or RiskConfig())
        self.halted = False
        self.halt_reason = ""
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.total_spread_cost = 0.0

    def run(self, df: pd.DataFrame, strategy) -> BacktestResult:
        """Run realistic backtest with spread, slippage, fees, and hard risk gates."""
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.position = 0.0
        self.position_value = 0.0
        self.halted = False
        self.halt_reason = ""
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.total_spread_cost = 0.0
        self.risk_controller = RiskController(initial_equity=self.initial_capital, config=self.risk_controller.config)

        current_trade: Optional[Trade] = None

        for i in range(len(df)):
            price = float(df["close"].iloc[i])
            timestamp = self._bar_timestamp(df, i)
            bar_volatility = self._bar_volatility(df, i)

            mark_equity = self.capital + (self.position * price if self.position > 0 else 0.0)
            self.equity = mark_equity
            self.equity_curve.append(mark_equity)

            signal = strategy.generate_signal(df, i, price)
            action = signal.get("action", "HOLD").upper()

            if self.halted:
                continue

            if action == "BUY" and self.position == 0:
                stop_price = float(signal.get("stop_loss", price * 0.99))
                risk_pct = self.risk_controller.dynamic_risk_pct(mark_equity)
                if risk_pct <= 0:
                    self.halted = True
                    self.halt_reason = "dynamic risk reduced to zero"
                    continue

                suggested_size = signal.get("size")
                if suggested_size is None:
                    suggested_size = fixed_fractional_position_size(
                        equity=mark_equity,
                        entry_price=price,
                        stop_price=stop_price,
                        risk_pct=risk_pct,
                    )
                size = float(max(suggested_size, 0.0))
                risk_amount = size * abs(price - stop_price)
                risk_check = self.risk_controller.validate_trade(
                    equity=mark_equity,
                    proposed_risk_amount=risk_amount,
                    timestamp=timestamp,
                )
                if not risk_check.allowed:
                    self.halted = True
                    self.halt_reason = risk_check.reason or "risk check rejected trade"
                    continue
                if risk_check.adjusted_size is not None:
                    size *= float(risk_check.adjusted_size)
                    risk_amount = size * abs(price - stop_price)

                estimated_unit_cost = price * (1 + self.execution_config.commission_pct)
                affordable_size = self.capital / max(estimated_unit_cost, 1e-9)
                size = min(size, max(0.0, affordable_size))
                risk_amount = size * abs(price - stop_price)

                fill_price, spread_cost, slippage_cost, fee_cost = self._apply_execution_costs(
                    price=price,
                    size=size,
                    side="BUY",
                    volatility=bar_volatility,
                )
                total_cost = (size * fill_price) + fee_cost
                if total_cost > (self.capital * 1.000000001) and fill_price > 0:
                    size = self.capital / (fill_price * (1 + self.execution_config.commission_pct))
                    size = max(0.0, size)
                    risk_amount = size * abs(price - stop_price)
                    fill_price, spread_cost, slippage_cost, fee_cost = self._apply_execution_costs(
                        price=price,
                        size=size,
                        side="BUY",
                        volatility=bar_volatility,
                    )
                    total_cost = (size * fill_price) + fee_cost

                if size <= 0 or total_cost > (self.capital * 1.000000001):
                    continue

                self.position = size
                self.capital -= total_cost
                self.total_fees += fee_cost
                self.total_spread_cost += spread_cost
                self.total_slippage += slippage_cost

                current_trade = Trade(
                    entry_time=timestamp,
                    entry_price=fill_price,
                    side="long",
                    size=size,
                    spread_cost=spread_cost,
                    slippage_cost=slippage_cost,
                    commission_cost=fee_cost,
                    risk_amount=risk_amount,
                )

            elif action == "SELL" and self.position > 0 and current_trade is not None:
                fill_price, spread_cost, slippage_cost, fee_cost = self._apply_execution_costs(
                    price=price,
                    size=self.position,
                    side="SELL",
                    volatility=bar_volatility,
                )
                proceeds = (self.position * fill_price) - fee_cost

                current_trade.exit_time = timestamp
                current_trade.exit_price = fill_price
                current_trade.spread_cost += spread_cost
                current_trade.slippage_cost += slippage_cost
                current_trade.commission_cost += fee_cost
                gross_entry = current_trade.size * current_trade.entry_price
                current_trade.pnl = proceeds - gross_entry
                current_trade.pnl_pct = ((fill_price - current_trade.entry_price) / current_trade.entry_price) * 100
                current_trade.exit_reason = signal.get("reason", "SIGNAL")
                self.trades.append(current_trade)

                self.capital += proceeds
                self.total_fees += fee_cost
                self.total_spread_cost += spread_cost
                self.total_slippage += slippage_cost
                self.position = 0.0
                current_trade = None

                self.risk_controller.update_after_trade(
                    realized_pnl=self.trades[-1].pnl,
                    equity=self.capital,
                    timestamp=timestamp,
                )
                if not self.risk_controller.trading_enabled:
                    self.halted = True
                    self.halt_reason = self.risk_controller.halt_reason or "risk halt"

        if self.position > 0 and current_trade is not None:
            price = float(df["close"].iloc[-1])
            timestamp = self._bar_timestamp(df, len(df) - 1)
            fill_price, spread_cost, slippage_cost, fee_cost = self._apply_execution_costs(
                price=price,
                size=self.position,
                side="SELL",
                volatility=self._bar_volatility(df, len(df) - 1),
            )
            proceeds = (self.position * fill_price) - fee_cost

            current_trade.exit_time = timestamp
            current_trade.exit_price = fill_price
            current_trade.spread_cost += spread_cost
            current_trade.slippage_cost += slippage_cost
            current_trade.commission_cost += fee_cost
            gross_entry = current_trade.size * current_trade.entry_price
            current_trade.pnl = proceeds - gross_entry
            current_trade.pnl_pct = ((fill_price - current_trade.entry_price) / current_trade.entry_price) * 100
            current_trade.exit_reason = "END_OF_DATA"
            self.trades.append(current_trade)
            self.capital += proceeds
            self.total_fees += fee_cost
            self.total_spread_cost += spread_cost
            self.total_slippage += slippage_cost
            self.position = 0.0

            self.risk_controller.update_after_trade(
                realized_pnl=current_trade.pnl,
                equity=self.capital,
                timestamp=timestamp,
            )
            if not self.risk_controller.trading_enabled:
                self.halted = True
                self.halt_reason = self.risk_controller.halt_reason or "risk halt"

        return self._calculate_results()

    def _calculate_results(self) -> BacktestResult:
        base = super()._calculate_results()
        base.halted = self.halted
        base.halt_reason = self.halt_reason
        base.total_fees = self.total_fees
        base.total_slippage = self.total_slippage
        base.total_spread_cost = self.total_spread_cost
        return base

    def _apply_execution_costs(
        self,
        *,
        price: float,
        size: float,
        side: str,
        volatility: float,
    ) -> tuple[float, float, float, float]:
        spread_half = price * (self.execution_config.spread_bps / 10000.0) * 0.5
        base_slippage = price * (self.execution_config.slippage_bps / 10000.0)
        vol_slippage = base_slippage * max(volatility, 0.0) * self.execution_config.slippage_volatility_multiplier
        slippage = base_slippage + vol_slippage

        if side.upper() == "BUY":
            fill_price = price + spread_half + slippage
        else:
            fill_price = price - spread_half - slippage

        spread_cost = spread_half * size
        slippage_cost = slippage * size
        fee_cost = fill_price * size * self.execution_config.commission_pct
        return fill_price, spread_cost, slippage_cost, fee_cost

    @staticmethod
    def _bar_timestamp(df: pd.DataFrame, index: int) -> datetime:
        if "timestamp" in df.columns:
            return pd.Timestamp(df["timestamp"].iloc[index]).to_pydatetime()
        if isinstance(df.index, pd.DatetimeIndex):
            return pd.Timestamp(df.index[index]).to_pydatetime()
        return datetime.now()

    @staticmethod
    def _bar_volatility(df: pd.DataFrame, index: int, lookback: int = 20) -> float:
        start = max(0, index - lookback + 1)
        window = df["close"].iloc[start : index + 1].astype(float)
        if len(window) < 2:
            return 0.0
        returns = window.pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        vol = float(returns.std())
        return float(np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0))


