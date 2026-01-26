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
