"""
Testing & QA - Enterprise Features #290, #295, #300, #305
Strategy Backtester, Walk-Forward Analysis, Statistical Testing, Performance Attribution.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import random
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Feature #290: Strategy Backtester
    
    Backtests trading strategies on historical data.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        commission_pct: float = 0.04,
        slippage_pct: float = 0.05
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            commission_pct: Trading commission
            slippage_pct: Simulated slippage
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        
        self.results: List[Dict] = []
        
        logger.info(f"Strategy Backtester initialized - ${initial_capital:,.0f}")
    
    def run(
        self,
        candles: List[Dict],
        strategy_fn: Callable[[List[Dict], int], Optional[Dict]],
        position_size_pct: float = 10.0
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            candles: Historical OHLCV data
            strategy_fn: Function(candles, index) -> signal or None
            position_size_pct: Position size as % of equity
            
        Returns:
            Backtest results
        """
        equity = self.initial_capital
        position = None
        trades = []
        equity_curve = [equity]
        
        for i in range(50, len(candles)):  # Start at 50 for indicators
            candle = candles[i]
            price = candle['close']
            
            # Check for exit
            if position:
                # Check SL/TP
                if position['side'] == 'LONG':
                    if candle['low'] <= position['sl']:
                        exit_price = position['sl']
                        reason = 'stop_loss'
                    elif candle['high'] >= position['tp']:
                        exit_price = position['tp']
                        reason = 'take_profit'
                    else:
                        exit_price = None
                else:
                    if candle['high'] >= position['sl']:
                        exit_price = position['sl']
                        reason = 'stop_loss'
                    elif candle['low'] <= position['tp']:
                        exit_price = position['tp']
                        reason = 'take_profit'
                    else:
                        exit_price = None
                
                if exit_price:
                    # Close position
                    if position['side'] == 'LONG':
                        pnl = (exit_price - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - exit_price) * position['size']
                    
                    pnl -= (position['entry'] + exit_price) * position['size'] * self.commission_pct / 100
                    
                    trades.append({
                        'entry': position['entry'],
                        'exit': exit_price,
                        'side': position['side'],
                        'pnl': round(pnl, 2),
                        'reason': reason
                    })
                    
                    equity += pnl
                    position = None
            
            # Generate signal if no position
            if not position:
                signal = strategy_fn(candles[:i+1], i)
                
                if signal:
                    size_value = equity * position_size_pct / 100
                    size = size_value / price
                    
                    entry_price = price * (1 + self.slippage_pct / 100) if signal['side'] == 'LONG' else price * (1 - self.slippage_pct / 100)
                    
                    position = {
                        'entry': entry_price,
                        'side': signal['side'],
                        'size': size,
                        'sl': signal.get('sl', entry_price * 0.98 if signal['side'] == 'LONG' else entry_price * 1.02),
                        'tp': signal.get('tp', entry_price * 1.04 if signal['side'] == 'LONG' else entry_price * 0.96)
                    }
            
            equity_curve.append(equity)
        
        # Calculate metrics
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        total_return = (equity - self.initial_capital) / self.initial_capital * 100
        
        result = {
            'final_equity': round(equity, 2),
            'total_return_pct': round(total_return, 2),
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
            'avg_win': round(sum(t['pnl'] for t in wins) / len(wins), 2) if wins else 0,
            'avg_loss': round(sum(t['pnl'] for t in losses) / len(losses), 2) if losses else 0,
            'profit_factor': round(abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)), 2) if losses and sum(t['pnl'] for t in losses) != 0 else 0,
            'max_drawdown': self._calculate_max_dd(equity_curve),
            'trades': trades,
            'equity_curve': equity_curve[::max(1, len(equity_curve)//100)]  # Sample for chart
        }
        
        self.results.append(result)
        
        return result
    
    def _calculate_max_dd(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return round(max_dd, 2)


class WalkForwardAnalyzer:
    """
    Feature #295: Walk-Forward Analyzer
    
    Performs walk-forward optimization to test strategy robustness.
    """
    
    def __init__(
        self,
        train_pct: float = 70,
        test_pct: float = 30,
        n_folds: int = 5
    ):
        """
        Initialize walk-forward analyzer.
        
        Args:
            train_pct: Training window percentage
            test_pct: Testing window percentage
            n_folds: Number of fold iterations
        """
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.n_folds = n_folds
        
        logger.info(f"Walk-Forward Analyzer initialized - {n_folds} folds")
    
    def analyze(
        self,
        candles: List[Dict],
        optimize_fn: Callable[[List[Dict]], Dict],
        backtest_fn: Callable[[List[Dict], Dict], Dict]
    ) -> Dict:
        """
        Run walk-forward analysis.
        
        Args:
            candles: Full historical data
            optimize_fn: Function to optimize parameters on training data
            backtest_fn: Function to backtest with parameters on test data
            
        Returns:
            Walk-forward results
        """
        n = len(candles)
        fold_size = n // self.n_folds
        
        results = []
        in_sample_returns = []
        out_sample_returns = []
        
        for fold in range(self.n_folds - 1):
            train_start = fold * fold_size
            train_end = train_start + int(fold_size * self.train_pct / 100)
            test_start = train_end
            test_end = min(test_start + int(fold_size * self.test_pct / 100), n)
            
            train_data = candles[train_start:train_end]
            test_data = candles[test_start:test_end]
            
            if len(train_data) < 50 or len(test_data) < 10:
                continue
            
            # Optimize on training data
            params = optimize_fn(train_data)
            
            # Test on out-of-sample data
            train_result = backtest_fn(train_data, params)
            test_result = backtest_fn(test_data, params)
            
            in_sample_returns.append(train_result.get('total_return_pct', 0))
            out_sample_returns.append(test_result.get('total_return_pct', 0))
            
            results.append({
                'fold': fold + 1,
                'train_period': f"{train_start}-{train_end}",
                'test_period': f"{test_start}-{test_end}",
                'params': params,
                'in_sample_return': train_result.get('total_return_pct', 0),
                'out_sample_return': test_result.get('total_return_pct', 0),
                'overfitting_ratio': round(
                    train_result.get('total_return_pct', 1) / max(test_result.get('total_return_pct', 1), 0.01),
                    2
                ) if test_result.get('total_return_pct', 0) > 0 else 0
            })
        
        # Calculate summary statistics
        avg_is = sum(in_sample_returns) / len(in_sample_returns) if in_sample_returns else 0
        avg_oos = sum(out_sample_returns) / len(out_sample_returns) if out_sample_returns else 0
        
        return {
            'n_folds': len(results),
            'avg_in_sample_return': round(avg_is, 2),
            'avg_out_sample_return': round(avg_oos, 2),
            'robustness_ratio': round(avg_oos / avg_is, 2) if avg_is != 0 else 0,
            'is_robust': avg_oos > 0 and avg_oos / max(avg_is, 0.01) > 0.5,
            'fold_results': results
        }


class StatisticalSignificanceTester:
    """
    Feature #300: Statistical Significance Tester
    
    Tests if trading results are statistically significant.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize significance tester.
        
        Args:
            confidence_level: Required confidence (0.95 = 95%)
        """
        self.confidence_level = confidence_level
        
        logger.info(f"Significance Tester initialized - {confidence_level:.0%} confidence")
    
    def t_test(self, returns: List[float]) -> Dict:
        """
        Perform t-test for mean return > 0.
        
        Args:
            returns: List of trade returns
            
        Returns:
            Test results
        """
        n = len(returns)
        if n < 5:
            return {'significant': False, 'reason': 'Insufficient data'}
        
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
        std_error = (variance / n) ** 0.5
        
        if std_error == 0:
            return {'significant': False, 'reason': 'No variance in returns'}
        
        t_stat = mean / std_error
        
        # Critical values for common confidence levels
        critical_values = {30: 1.697, 50: 1.676, 100: 1.660, 200: 1.653}
        cv = critical_values.get(min(n, 200), 1.645)
        
        significant = t_stat > cv
        
        return {
            't_statistic': round(t_stat, 3),
            'critical_value': cv,
            'mean_return': round(mean, 4),
            'std_error': round(std_error, 4),
            'sample_size': n,
            'significant': significant,
            'confidence': self.confidence_level
        }
    
    def monte_carlo_test(
        self,
        trades: List[Dict],
        n_simulations: int = 1000
    ) -> Dict:
        """
        Monte Carlo permutation test for strategy significance.
        
        Args:
            trades: List of trades with 'pnl'
            n_simulations: Number of random permutations
            
        Returns:
            Test results
        """
        if len(trades) < 10:
            return {'significant': False, 'reason': 'Insufficient trades'}
        
        actual_return = sum(t['pnl'] for t in trades)
        
        # Generate random permutations
        better_count = 0
        for _ in range(n_simulations):
            shuffled = trades.copy()
            random.shuffle(shuffled)
            
            # Random entry/exit assignment
            random_return = sum(
                t['pnl'] * (1 if random.random() > 0.5 else -1)
                for t in shuffled
            )
            
            if random_return >= actual_return:
                better_count += 1
        
        p_value = better_count / n_simulations
        significant = p_value < (1 - self.confidence_level)
        
        return {
            'actual_return': round(actual_return, 2),
            'p_value': round(p_value, 4),
            'simulations': n_simulations,
            'significant': significant,
            'confidence': self.confidence_level
        }


class PerformanceAttributor:
    """
    Feature #305: Performance Attribution
    
    Breaks down performance by various factors.
    """
    
    def __init__(self):
        """Initialize performance attributor."""
        logger.info("Performance Attribution initialized")
    
    def attribute_by_time(self, trades: List[Dict]) -> Dict:
        """Attribute performance by time period."""
        by_hour = defaultdict(list)
        by_day = defaultdict(list)
        
        for trade in trades:
            ts = trade.get('timestamp', '')
            pnl = trade.get('pnl', 0)
            
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts)
                else:
                    dt = ts
                
                by_hour[dt.hour].append(pnl)
                by_day[dt.strftime('%A')].append(pnl)
            except:
                pass
        
        return {
            'by_hour': {
                k: {'count': len(v), 'total_pnl': round(sum(v), 2), 'avg_pnl': round(sum(v)/len(v), 2)}
                for k, v in sorted(by_hour.items())
            },
            'by_day': {
                k: {'count': len(v), 'total_pnl': round(sum(v), 2), 'avg_pnl': round(sum(v)/len(v), 2)}
                for k, v in by_day.items()
            }
        }
    
    def attribute_by_direction(self, trades: List[Dict]) -> Dict:
        """Attribute performance by trade direction."""
        longs = [t for t in trades if t.get('side') == 'LONG']
        shorts = [t for t in trades if t.get('side') == 'SHORT']
        
        return {
            'long': {
                'count': len(longs),
                'total_pnl': round(sum(t['pnl'] for t in longs), 2),
                'win_rate': round(len([t for t in longs if t['pnl'] > 0]) / len(longs) * 100, 1) if longs else 0
            },
            'short': {
                'count': len(shorts),
                'total_pnl': round(sum(t['pnl'] for t in shorts), 2),
                'win_rate': round(len([t for t in shorts if t['pnl'] > 0]) / len(shorts) * 100, 1) if shorts else 0
            }
        }
    
    def attribute_by_size(self, trades: List[Dict]) -> Dict:
        """Attribute performance by trade size."""
        if not trades:
            return {}
        
        sizes = [t.get('size', 0) for t in trades]
        avg_size = sum(sizes) / len(sizes)
        
        small = [t for t in trades if t.get('size', 0) < avg_size * 0.5]
        medium = [t for t in trades if avg_size * 0.5 <= t.get('size', 0) < avg_size * 1.5]
        large = [t for t in trades if t.get('size', 0) >= avg_size * 1.5]
        
        return {
            'small': {'count': len(small), 'total_pnl': round(sum(t['pnl'] for t in small), 2)},
            'medium': {'count': len(medium), 'total_pnl': round(sum(t['pnl'] for t in medium), 2)},
            'large': {'count': len(large), 'total_pnl': round(sum(t['pnl'] for t in large), 2)}
        }
    
    def full_attribution(self, trades: List[Dict]) -> Dict:
        """Get full performance attribution."""
        return {
            'by_time': self.attribute_by_time(trades),
            'by_direction': self.attribute_by_direction(trades),
            'by_size': self.attribute_by_size(trades)
        }


# Singletons
_backtester: Optional[StrategyBacktester] = None
_walk_forward: Optional[WalkForwardAnalyzer] = None
_sig_tester: Optional[StatisticalSignificanceTester] = None
_attributor: Optional[PerformanceAttributor] = None


def get_backtester() -> StrategyBacktester:
    global _backtester
    if _backtester is None:
        _backtester = StrategyBacktester()
    return _backtester


def get_walk_forward() -> WalkForwardAnalyzer:
    global _walk_forward
    if _walk_forward is None:
        _walk_forward = WalkForwardAnalyzer()
    return _walk_forward


def get_significance_tester() -> StatisticalSignificanceTester:
    global _sig_tester
    if _sig_tester is None:
        _sig_tester = StatisticalSignificanceTester()
    return _sig_tester


def get_attributor() -> PerformanceAttributor:
    global _attributor
    if _attributor is None:
        _attributor = PerformanceAttributor()
    return _attributor


if __name__ == '__main__':
    # Test significance
    sig = StatisticalSignificanceTester()
    returns = [random.gauss(0.01, 0.05) for _ in range(50)]
    result = sig.t_test(returns)
    print(f"T-test: {result}")
    
    # Test attribution
    attr = PerformanceAttributor()
    trades = [
        {'side': 'LONG', 'pnl': 50, 'size': 0.1, 'timestamp': '2024-01-15T10:30:00'},
        {'side': 'SHORT', 'pnl': -30, 'size': 0.05, 'timestamp': '2024-01-15T14:00:00'},
        {'side': 'LONG', 'pnl': 80, 'size': 0.15, 'timestamp': '2024-01-16T09:00:00'},
    ]
    print(f"Attribution: {attr.attribute_by_direction(trades)}")
