"""
Performance Analytics Suite
Advanced performance attribution and analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class PerformanceAnalytics:
    """
    Institutional-grade performance analytics:
    - Detailed performance metrics
    - Risk-adjusted returns
    - Factor decomposition
    - Trade quality scoring
    """
    
    @staticmethod
    def calculate_all_metrics(equity_curve, trades, initial_capital, benchmark_returns=None):
        """
        Calculate comprehensive performance metrics
        
        Args:
            equity_curve: Array of equity values
            trades: List of trade dictionaries
            initial_capital: Starting capital
            benchmark_returns: Optional benchmark returns for alpha/beta
        
        Returns:
            metrics: Dict of all calculated metrics
        """
        
        equity = np.array(equity_curve)
        
        # Basic returns
        total_return = (equity[-1] - initial_capital) / initial_capital
        
        # Calculate returns series
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        # Annualization factor (assuming hourly data)
        ann_factor = np.sqrt(252 * 24)
        
        # Risk-adjusted ratios
        sharpe = PerformanceAnalytics.sharpe_ratio(returns, ann_factor)
        sortino = PerformanceAnalytics.sortino_ratio(returns, ann_factor)
        calmar = PerformanceAnalytics.calmar_ratio(equity, ann_factor)
        omega = PerformanceAnalytics.omega_ratio(returns)
        
        # Drawdown metrics
        max_dd, avg_dd, dd_duration = PerformanceAnalytics.drawdown_metrics(equity)
        
        # Trade metrics
        if trades:
            trade_metrics = PerformanceAnalytics.trade_metrics(trades)
        else:
            trade_metrics = {}
        
        # Alpha/Beta if benchmark provided
        if benchmark_returns is not None:
            alpha, beta = PerformanceAnalytics.alpha_beta(returns, benchmark_returns)
        else:
            alpha, beta = None, None
        
        return {
            # Returns
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (1 / (len(equity) / (252 * 24))) - 1,
            
            # Risk metrics
            'volatility': np.std(returns) * ann_factor if len(returns) > 0 else 0,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'omega_ratio': omega,
            
            # Drawdown
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'max_drawdown_duration': dd_duration,
            
            # Trade stats
            **trade_metrics,
            
            # Alpha/Beta
            'alpha': alpha,
            'beta': beta
        }
    
    @staticmethod
    def sharpe_ratio(returns, ann_factor):
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return (np.mean(returns) / np.std(returns)) * ann_factor
    
    @staticmethod
    def sortino_ratio(returns, ann_factor):
        """Calculate Sortino ratio (only penalize downside)"""
        if len(returns) == 0:
            return 0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0
        
        return (np.mean(returns) / np.std(downside_returns)) * ann_factor
    
    @staticmethod
    def calmar_ratio(equity, ann_factor):
        """Calculate Calmar ratio (return / max drawdown)"""
        
        # Annualized return
        total_return = (equity[-1] - equity[0]) / equity[0]
        years = len(equity) / (252 * 24)
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Max drawdown
        max_dd = 0
        peak = equity[0]
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        return ann_return / max_dd if max_dd > 0 else 0
    
    @staticmethod
    def omega_ratio(returns, threshold=0):
        """Calculate Omega ratio"""
        if len(returns) == 0:
            return 0
        
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        
        total_gains = gains.sum() if len(gains) > 0 else 0
        total_losses = losses.sum() if len(losses) > 0 else 0
        
        return total_gains / total_losses if total_losses > 0 else 0
    
    @staticmethod
    def drawdown_metrics(equity):
        """Calculate drawdown metrics"""
        
        max_dd = 0
        peak = equity[0]
        drawdowns = []
        current_dd = 0
        dd_duration = 0
        current_dd_duration = 0
        
        for value in equity:
            if value > peak:
                peak = value
                if current_dd > 0:
                    drawdowns.append(current_dd)
                current_dd = 0
                if current_dd_duration > dd_duration:
                    dd_duration = current_dd_duration
                current_dd_duration = 0
            else:
                current_dd = (peak - value) / peak if peak > 0 else 0
                current_dd_duration += 1
            
            if current_dd > max_dd:
                max_dd = current_dd
        
        avg_dd = np.mean(drawdowns) if drawdowns else 0
        
        return max_dd, avg_dd, dd_duration
    
    @staticmethod
    def trade_metrics(trades):
        """Calculate trade-specific metrics"""
        
        pnls = [t['pnl'] for t in trades]
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Profit factor
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Largest win/loss
        largest_win = max(pnls) if pnls else 0
        largest_loss = min(pnls) if pnls else 0
        
        # Average duration
        avg_duration = np.mean([t['duration'] for t in trades]) if trades else 0
        
        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        last_was_win = None
        
        for trade in trades:
            is_win = trade['pnl'] > 0
            
            if last_was_win is None or last_was_win == is_win:
                current_streak += 1
            else:
                if last_was_win:
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                else:
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)
                current_streak = 1
            
            last_was_win = is_win
        
        # Check final streak
        if last_was_win:
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
        elif last_was_win is not None:
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        
        return {
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_duration': avg_duration,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    @staticmethod
    def alpha_beta(strategy_returns, benchmark_returns):
        """Calculate alpha and beta vs benchmark"""
        
        if len(strategy_returns) != len(benchmark_returns):
            # Align lengths
            min_len = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]
        
        if len(strategy_returns) < 2:
            return 0, 0
        
        # Beta = Cov(strategy, benchmark) / Var(benchmark)
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        variance = np.var(benchmark_returns)
        
        beta = covariance / variance if variance > 0 else 0
        
        # Alpha = Strategy Return - (Risk-Free Rate + Beta * (Benchmark Return - Risk-Free Rate))
        # Assuming risk-free rate = 0 for simplicity
        alpha = np.mean(strategy_returns) - beta * np.mean(benchmark_returns)
        
        return alpha, beta
