"""
Walk-Forward Validation Framework
Tests strategy performance on rolling windows to detect overfitting
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from core.backtest import ProfessionalBacktest
from core.monitoring.logger import get_logger


logger = get_logger()


class WalkForwardValidator:
    """
    Implements walk-forward analysis for strategy validation
    
    Process:
    1. Split data into windows (e.g., 6 months train, 1 month test)
    2. Train on in-sample period
    3. Test on out-of-sample period
    4. Roll forward and repeat
    5. Aggregate and compare results
    """
    
    def __init__(self, train_period_bars: int = 4320,  # 6 months of hourly data
                 test_period_bars: int = 720,           # 1 month of hourly data
                 step_size_bars: int = 720):            # Roll by 1 month
        """
        Args:
            train_period_bars: Number of bars for training
            test_period_bars: Number of bars for testing
            step_size_bars: Number of bars to roll forward
        """
        self.train_period = train_period_bars
        self.test_period = test_period_bars
        self.step_size = step_size_bars
        
        self.results: List[Dict[str, Any]] = []
        
    def validate(self, highs, lows, closes, strategy, volumes=None,
                 capital: float = 10000) -> Dict[str, Any]:
        """
        Run walk-forward validation
        
        Returns:
            Dictionary with aggregated results and window-by-window performance
        """
        total_bars = len(closes)
        logger.info(
            "Starting walk-forward validation",
            total_bars=total_bars,
            train_period=self.train_period,
            test_period=self.test_period
        )
        
        window_results = []
        current_position = 0
        window_num = 0
        
        while current_position + self.train_period + self.test_period <= total_bars:
            window_num += 1
            
            # Define train and test ranges
            train_start = current_position
            train_end = current_position + self.train_period
            test_start = train_end
            test_end = test_start + self.test_period
            
            logger.info(
                f"Processing window {window_num}",
                train_range=f"{train_start}-{train_end}",
                test_range=f"{test_start}-{test_end}"
            )
            
            # Train period (in-sample)
            train_highs = highs[train_start:train_end]
            train_lows = lows[train_start:train_end]
            train_closes = closes[train_start:train_end]
            train_volumes = volumes[train_start:train_end] if volumes is not None else None
            
            bt_train = ProfessionalBacktest(capital=capital)
            equity_train = bt_train.run(train_highs, train_lows, train_closes, 
                                       strategy, train_volumes)
            metrics_train = bt_train.get_metrics()
            
            # Test period (out-of-sample)
            test_highs = highs[test_start:test_end]
            test_lows = lows[test_start:test_end]
            test_closes = closes[test_start:test_end]
            test_volumes = volumes[test_start:test_end] if volumes is not None else None
            
            bt_test = ProfessionalBacktest(capital=capital)
            equity_test = bt_test.run(test_highs, test_lows, test_closes, 
                                     strategy, test_volumes)
            metrics_test = bt_test.get_metrics()
            
            # Calculate metrics
            window_result = {
                'window': window_num,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_return': metrics_train.get('total_return', 0),
                'test_return': metrics_test.get('total_return', 0),
                'train_sharpe': metrics_train.get('sharpe_ratio', 0),
                'test_sharpe': metrics_test.get('sharpe_ratio', 0),
                'train_max_dd': metrics_train.get('max_drawdown', 0),
                'test_max_dd': metrics_test.get('max_drawdown', 0),
                'train_win_rate': metrics_train.get('win_rate', 0),
                'test_win_rate': metrics_test.get('win_rate', 0),
                'train_num_trades': metrics_train.get('num_trades', 0),
                'test_num_trades': metrics_test.get('num_trades', 0)
            }
            
            # Calculate performance degradation
            if metrics_train.get('sharpe_ratio', 0) > 0:
                degradation = (
                    metrics_train['sharpe_ratio'] - metrics_test.get('sharpe_ratio', 0)
                ) / metrics_train['sharpe_ratio']
                window_result['sharpe_degradation'] = degradation
            else:
                window_result['sharpe_degradation'] = 0
            
            window_results.append(window_result)
            
            logger.info(
                f"Window {window_num} complete",
                train_sharpe=window_result['train_sharpe'],
                test_sharpe=window_result['test_sharpe'],
                degradation=window_result.get('sharpe_degradation', 0)
            )
            
            # Move to next window
            current_position += self.step_size
        
        # Aggregate results
        self.results = window_results
        return self._aggregate_results(window_results)
    
    def _aggregate_results(self, window_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across all windows"""
        if not window_results:
            return {}
        
        # Calculate averages
        avg_train_return = np.mean([w['train_return'] for w in window_results])
        avg_test_return = np.mean([w['test_return'] for w in window_results])
        avg_train_sharpe = np.mean([w['train_sharpe'] for w in window_results])
        avg_test_sharpe = np.mean([w['test_sharpe'] for w in window_results])
        avg_degradation = np.mean([w.get('sharpe_degradation', 0) for w in window_results])
        
        # Calculate consistency (std dev of returns across windows)
        test_returns = [w['test_return'] for w in window_results]
        consistency_score = 1 / (1 + np.std(test_returns))  # Higher is more consistent
        
        # Count profitable windows
        profitable_windows = sum(1 for w in window_results if w['test_return'] > 0)
        profitable_pct = profitable_windows / len(window_results)
        
        # Overfitting score (higher = more overfitting)
        overfitting_score = avg_degradation
        
        summary = {
            'total_windows': len(window_results),
            'avg_train_return': avg_train_return,
            'avg_test_return': avg_test_return,
            'avg_train_sharpe': avg_train_sharpe,
            'avg_test_sharpe': avg_test_sharpe,
            'avg_sharpe_degradation': avg_degradation,
            'consistency_score': consistency_score,
            'profitable_windows': profitable_windows,
            'profitable_window_pct': profitable_pct,
            'overfitting_score': overfitting_score,
            'overfitting_detected': overfitting_score > 0.30,  # 30% degradation threshold
            'windows': window_results
        }
        
        # Log summary
        logger.info(
            "Walk-forward validation complete",
            windows=len(window_results),
            avg_test_sharpe=avg_test_sharpe,
            overfitting_detected=summary['overfitting_detected']
        )
        
        if summary['overfitting_detected']:
            logger.warning(
                "⚠️ OVERFITTING DETECTED",
                sharpe_degradation=f"{overfitting_score:.1%}",
                message="Strategy may not generalize well to live trading"
            )
        
        return summary
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot walk-forward validation results"""
        if not self.results:
            print("No results to plot")
            return
        
        windows = [r['window'] for r in self.results]
        train_sharpe = [r['train_sharpe'] for r in self.results]
        test_sharpe = [r['test_sharpe'] for r in self.results]
        train_returns = [r['train_return'] for r in self.results]
        test_returns = [r['test_return'] for r in self.results]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Sharpe ratio comparison
        axes[0].plot(windows, train_sharpe, 'o-', label='In-Sample Sharpe', linewidth=2, markersize=6)
        axes[0].plot(windows, test_sharpe, 's-', label='Out-of-Sample Sharpe', linewidth=2, markersize=6)
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Window', fontsize=11)
        axes[0].set_ylabel('Sharpe Ratio', fontsize=11)
        axes[0].set_title('Walk-Forward Analysis: Sharpe Ratio', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Returns comparison
        axes[1].plot(windows, train_returns, 'o-', label='In-Sample Return', linewidth=2, markersize=6)
        axes[1].plot(windows, test_returns, 's-', label='Out-of-Sample Return', linewidth=2, markersize=6)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Window', fontsize=11)
        axes[1].set_ylabel('Return', fontsize=11)
        axes[1].set_title('Walk-Forward Analysis: Returns', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.savefig('walk_forward_analysis.png', dpi=150, bbox_inches='tight')
            print("✅ Plot saved to walk_forward_analysis.png")
        
        plt.close()


if __name__ == "__main__":
    # Test with dummy data
    print("Testing Walk-Forward Validator...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_bars = 10000
    closes = 45000 * (1 + np.random.randn(n_bars).cumsum() * 0.01)
    highs = closes * 1.002
    lows = closes * 0.998
    
    # Import a strategy
    from core.strategies.factory import StrategyFactory
    strategy = StrategyFactory.create("enhanced_momentum")
    
    # Run validation
    validator = WalkForwardValidator(
        train_period_bars=2000,
        test_period_bars=500,
        step_size_bars=500
    )
    
    results = validator.validate(highs, lows, closes, strategy)
    
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 70)
    print(f"Total Windows: {results['total_windows']}")
    print(f"Avg In-Sample Sharpe: {results['avg_train_sharpe']:.2f}")
    print(f"Avg Out-of-Sample Sharpe: {results['avg_test_sharpe']:.2f}")
    print(f"Sharpe Degradation: {results['avg_sharpe_degradation']:.1%}")
    print(f"Overfitting Detected: {results['overfitting_detected']}")
    print(f"Profitable Windows: {results['profitable_windows']}/{results['total_windows']} ({results['profitable_window_pct']:.1%})")
    
    # Plot results
    validator.plot_results()
    print("\n✅ Walk-forward validation test complete")
