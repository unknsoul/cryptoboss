"""
Walk-Forward Analysis
Test strategy robustness with out-of-sample validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class WalkForwardAnalysis:
    """
    Implement walk-forward optimization and testing
    
    Splits data into train/test segments and evaluates
    performance degradation in out-of-sample periods
    """
    
    def __init__(self, train_ratio=0.7, num_windows=None):
        """
        Args:
            train_ratio: Fraction of data for training (default 70%)
            num_windows: Number of walk-forward windows (None = single split)
        """
        self.train_ratio = train_ratio
        self.num_windows = num_windows
    
    def single_split(self, data_length):
        """
        Single train/test split
        
        Returns:
            (train_start, train_end, test_start, test_end)
        """
        split_point = int(data_length * self.train_ratio)
        
        return (0, split_point, split_point, data_length)
    
    def rolling_windows(self, data_length):
        """
        Generate rolling walk-forward windows
        
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        if self.num_windows is None:
            return [self.single_split(data_length)]
        
        windows = []
        window_size = data_length // (self.num_windows + 1)
        train_size = int(window_size * self.train_ratio)
        test_size = window_size - train_size
        
        for i in range(self.num_windows):
            start = i * test_size
            train_end = start + train_size
            test_end = min(train_end + test_size, data_length)
            
            if test_end - train_end < test_size // 2:
                break  # Not enough data for test window
            
            windows.append((start, train_end, train_end, test_end))
        
        return windows
    
    def run_analysis(self, backtest_engine, strategy, highs, lows, closes, volumes=None):
        """
        Run walk-forward analysis
        
        Args:
            backtest_engine: Backtest instance (will be re-initialized)
            strategy: Trading strategy
            highs, lows, closes: Price data
            volumes: Optional volume data
        
        Returns:
            results: Dict with train/test metrics for each window
        """
        windows = self.rolling_windows(len(closes))
        results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"Window {i+1}/{len(windows)}: Train[{train_start}:{train_end}] Test[{test_start}:{test_end}]")
            
            # Run on training set
            backtest_engine.__init__(
                capital=backtest_engine.initial_capital,
                risk_per_trade=backtest_engine.risk_per_trade,
                fee=backtest_engine.fee,
                slippage=backtest_engine.slippage
            )
            
            train_equity = backtest_engine.run(
                highs[train_start:train_end],
                lows[train_start:train_end],
                closes[train_start:train_end],
                strategy,
                volumes=volumes[train_start:train_end] if volumes is not None else None
            )
            
            train_metrics = backtest_engine.get_metrics()
            
            # Run on test set
            backtest_engine.__init__(
                capital=backtest_engine.initial_capital,
                risk_per_trade=backtest_engine.risk_per_trade,
                fee=backtest_engine.fee,
                slippage=backtest_engine.slippage
            )
            
            test_equity = backtest_engine.run(
                highs[test_start:test_end],
                lows[test_start:test_end],
                closes[test_start:test_end],
                strategy,
                volumes=volumes[test_start:test_end] if volumes is not None else None
            )
            
            test_metrics = backtest_engine.get_metrics()
            
            # Calculate efficiency ratio
            train_return = train_metrics.get('total_return', 0)
            test_return = test_metrics.get('total_return', 0)
            
            if train_return > 0:
                efficiency = test_return / train_return
            else:
                efficiency = 0
            
            results.append({
                'window': i + 1,
                'train_return': train_return,
                'test_return': test_return,
                'train_sharpe': train_metrics.get('sharpe_ratio', 0),
                'test_sharpe': test_metrics.get('sharpe_ratio', 0),
                'train_max_dd': train_metrics.get('max_drawdown', 0),
                'test_max_dd': test_metrics.get('max_drawdown', 0),
                'train_trades': train_metrics.get('num_trades', 0),
                'test_trades': test_metrics.get('num_trades', 0),
                'efficiency_ratio': efficiency
            })
        
        return results
    
    def print_summary(self, results):
        """Print walk-forward results summary"""
        print("\n" + "=" * 80)
        print("WALK-FORWARD ANALYSIS RESULTS")
        print("=" * 80)
        
        for r in results:
            print(f"\nWindow {r['window']}:")
            print(f"  Train Return: {r['train_return']:>8.2%}  |  Test Return: {r['test_return']:>8.2%}")
            print(f"  Train Sharpe: {r['train_sharpe']:>8.2f}  |  Test Sharpe: {r['test_sharpe']:>8.2f}")
            print(f"  Train MaxDD:  {r['train_max_dd']:>8.2%}  |  Test MaxDD:  {r['test_max_dd']:>8.2%}")
            print(f"  Efficiency Ratio: {r['efficiency_ratio']:.2f}")
        
        # Overall statistics
        if results:
            avg_efficiency = np.mean([r['efficiency_ratio'] for r in results])
            avg_test_return = np.mean([r['test_return'] for r in results])
            avg_test_sharpe = np.mean([r['test_sharpe'] for r in results])
            
            print("\n" + "=" * 80)
            print("OVERALL STATISTICS")
            print("=" * 80)
            print(f"Average Test Return:     {avg_test_return:>8.2%}")
            print(f"Average Test Sharpe:     {avg_test_sharpe:>8.2f}")
            print(f"Average Efficiency:      {avg_efficiency:>8.2f}")
            
            if avg_efficiency >= 0.7:
                print("\n✅ Strategy shows good out-of-sample performance (efficiency >= 0.7)")
            elif avg_efficiency >= 0.5:
                print("\n⚠️  Strategy shows moderate robustness (0.5 <= efficiency < 0.7)")
            else:
                print("\n❌ Strategy may be overfit (efficiency < 0.5)")
