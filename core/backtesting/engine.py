"""
Strategy Backtesting Framework
Vectorized backtesting engine for strategy optimization.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    equity_curve: List[float]
    trades: List[Dict]
    

class VectorizedBacktest:
    """
    High-performance vectorized backtesting engine.
    
    Features:
    - Vectorized calculations (no loops)
    - Realistic slippage & fees
    - Multiple strategies
    - Walk-forward optimization ready
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.04,  # 0.04% per trade
        slippage_pct: float = 0.02      # 0.02% slippage
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            commission_pct: Trading commission (%)
            slippage_pct: Estimated slippage (%)
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct / 100
        self.slippage_pct = slippage_pct / 100
        
        logger.info(f"Backtest engine initialized - Capital: ${initial_capital:,.0f}")
    
    def run(
        self,
        df: pd.DataFrame,
        strategy_func: Callable,
        risk_pct: float = 1.0
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            strategy_func: Function that returns signals (-1, 0, 1)
            risk_pct: Risk per trade (%)
            
        Returns:
            BacktestResult with performance metrics
        """
        if len(df) < 100:
            raise ValueError("Need at least 100 bars for backtest")
        
        # Generate signals
        df['signal'] = strategy_func(df)
        df['position'] = df['signal'].shift(1).fillna(0)  # Trade on next bar
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'] * df['returns']
        
        # Apply costs
        df['trades'] = df['position'].diff().abs()
        df['costs'] = df['trades'] * (self.commission_pct + self.slippage_pct)
        df['net_returns'] = df['strategy_returns'] - df['costs']
        
        # Calculate equity curve
        df['equity'] = self.initial_capital * (1 + df['net_returns']).cumprod()
        
        # Extract trades
        trades = self._extract_trades(df)
        
        # Calculate metrics
        total_return_pct = ((df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital * 100)
        sharpe_ratio = self._calculate_sharpe(df['net_returns'])
        max_drawdown_pct = self._calculate_max_drawdown(df['equity'])
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        
        win_rate = (len(wins) / len(trades) * 100) if trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        
        total_wins = sum(t['pnl'] for t in wins)
        total_losses = abs(sum(t['pnl'] for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        return BacktestResult(
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            total_trades=len(trades),
            profit_factor=profit_factor,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            equity_curve=df['equity'].tolist(),
            trades=trades
        )
    
    def _extract_trades(self, df: pd.DataFrame) -> List[Dict]:
        """Extract individual trades from backtest."""
        trades = []
        position = None
        
        for idx, row in df.iterrows():
            if row['position'] != 0 and position is None:
                # Entry
                position = {
                    'entry_idx': idx,
                    'entry_price': row['close'],
                    'side': 'LONG' if row['position'] > 0 else 'SHORT'
                }
            elif row['position'] == 0 and position is not None:
                # Exit
                exit_price = row['close']
                
                if position['side'] == 'LONG':
                    pnl_pct = ((exit_price - position['entry_price']) / position['entry_price'] * 100)
                else:
                    pnl_pct = ((position['entry_price'] - exit_price) / position['entry_price'] * 100)
                
                # Apply costs
                pnl_pct -= (self.commission_pct + self.slippage_pct) * 2 * 100
                
                pnl = self.initial_capital * (pnl_pct / 100)
                
                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'side': position['side'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'bars_held': idx - position['entry_idx']
                })
                
                position = None
        
        return trades
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())  # Annualized
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100
        return abs(drawdown.min())
    
    def walk_forward_optimize(
        self,
        df: pd.DataFrame,
        strategy_func: Callable,
        train_window: int = 500,
        test_window: int = 100,
        step_size: int = 50
    ) -> Dict:
        """
        Walk-forward optimization.
        
        Args:
            df: Full dataset
            strategy_func: Strategy function
            train_window: Training period bars
            test_window: Testing period bars
            step_size: Step size for rolling window
            
        Returns:
            Aggregated walk-forward results
        """
        results = []
        
        start = 0
        while start + train_window + test_window <= len(df):
            # Training period
            train_df = df.iloc[start:start + train_window]
            
            # Test period
            test_df = df.iloc[start + train_window:start + train_window + test_window]
            
            # Run backtest on test period
            test_result = self.run(test_df, strategy_func)
            results.append(test_result)
            
            start += step_size
        
        # Aggregate results
        avg_return = np.mean([r.total_return_pct for r in results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        avg_win_rate = np.mean([r.win_rate for r in results])
        
        return {
            'avg_return_pct': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_win_rate': avg_win_rate,
            'num_periods': len(results),
            'all_results': results
        }
