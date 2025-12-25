"""
Advanced Performance Metrics Calculator
Institutional-grade risk-adjusted performance metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """
    Professional performance metrics calculator
    
    Features:
    - Sharpe Ratio (risk-adjusted returns)
    - Sortino Ratio (downside risk focus)
    - Calmar Ratio (return vs max drawdown)
    - Information Ratio (vs benchmark)
    - Maximum Adverse/Favorable Excursion
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2% = 0.02)
        """
        self.risk_free_rate = risk_free_rate
        self.returns_history = []
        self.equity_history = []
        
    def add_return(self, daily_return: float):
        """Add daily return to history"""
        self.returns_history.append(daily_return)
        
    def add_equity(self, equity: float):
        """Add equity snapshot to history"""
        self.equity_history.append({
            'timestamp': datetime.now(),
            'equity': equity
        })
    
    def calculate_sharpe_ratio(self, returns: Optional[List[float]] = None, 
                               window: int = 30) -> Dict:
        """
        Calculate Sharpe Ratio - risk-adjusted returns
        
        Formula: (E[R] - Rf) / σ[R] * √252
        
        Args:
            returns: List of daily returns (uses history if None)
            window: Rolling window in days (default 30)
            
        Returns:
            Dict with sharpe ratio and components
        """
        if returns is None:
            returns = self.returns_history
        
        if len(returns) < 2:
            return {
                'sharpe_ratio': 0.0,
                'status': 'insufficient_data',
                'data_points': len(returns)
            }
        
        # Use last N days if window specified
        if window and len(returns) > window:
            returns = returns[-window:]
        
        returns_array = np.array(returns)
        
        # Calculate mean excess return
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns_array - daily_rf
        
        # Calculate Sharpe
        mean_excess = np.mean(excess_returns)
        std_returns = np.std(excess_returns)
        
        if std_returns == 0:
            sharpe = 0.0
        else:
            sharpe = (mean_excess / std_returns) * np.sqrt(252)
        
        return {
            'sharpe_ratio': round(sharpe, 2),
            'annual_return': round(mean_excess * 252 * 100, 2),
            'annual_volatility': round(std_returns * np.sqrt(252) * 100, 2),
            'window_days': len(returns),
            'status': 'ok'
        }
    
    def calculate_sortino_ratio(self, returns: Optional[List[float]] = None,
                                 target_return: float = 0.0) -> Dict:
        """
        Calculate Sortino Ratio - penalizes only downside volatility
        
        Formula: (E[R] - Target) / σ[downside] * √252
        
        Args:
            returns: List of daily returns
            target_return: Target/required return (default 0%)
            
        Returns:
            Dict with sortino ratio and components
        """
        if returns is None:
            returns = self.returns_history
            
        if len(returns) < 2:
            return {'sortino_ratio': 0.0, 'status': 'insufficient_data'}
        
        returns_array = np.array(returns)
        
        # Calculate downside deviation (only negative returns)
        downside_returns = returns_array[returns_array < target_return]
        
        if len(downside_returns) == 0:
            return {'sortino_ratio': 999.99, 'status': 'no_downside'}
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            sortino = 0.0
        else:
            mean_return = np.mean(returns_array)
            sortino = ((mean_return - target_return) / downside_std) * np.sqrt(252)
        
        return {
            'sortino_ratio': round(sortino, 2),
            'downside_volatility': round(downside_std * np.sqrt(252) * 100, 2),
            'downside_periods': len(downside_returns),
            'status': 'ok'
        }
    
    def calculate_calmar_ratio(self, returns: Optional[List[float]] = None,
                               max_drawdown: Optional[float] = None) -> Dict:
        """
        Calculate Calmar Ratio - return vs maximum drawdown
        
        Formula: Annual Return / |Max Drawdown|
        
        Args:
            returns: List of daily returns
            max_drawdown: Maximum drawdown % (will calculate if None)
            
        Returns:
            Dict with calmar ratio
        """
        if returns is None:
            returns = self.returns_history
            
        if len(returns) < 30:
            return {'calmar_ratio': 0.0, 'status': 'insufficient_data'}
        
        returns_array = np.array(returns)
        annual_return = np.mean(returns_array) * 252
        
        # Calculate max drawdown if not provided
        if max_drawdown is None:
            cumulative = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
        else:
            max_drawdown = abs(max_drawdown)
        
        if max_drawdown == 0:
            calmar = 999.99
        else:
            calmar = annual_return / max_drawdown
        
        return {
            'calmar_ratio': round(calmar, 2),
            'annual_return_pct': round(annual_return * 100, 2),
            'max_drawdown_pct': round(max_drawdown * 100, 2),
            'status': 'ok'
        }
    
    def calculate_information_ratio(self, returns: Optional[List[float]] = None,
                                    benchmark_returns: Optional[List[float]] = None) -> Dict:
        """
        Calculate Information Ratio - excess return vs tracking error
        
        Formula: E[R - Rb] / σ[R - Rb] * √252
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns (defaults to 0% if None)
            
        Returns:
            Dict with information ratio
        """
        if returns is None:
            returns = self.returns_history
            
        if len(returns) < 2:
            return {'information_ratio': 0.0, 'status': 'insufficient_data'}
        
        returns_array = np.array(returns)
        
        if benchmark_returns is None:
            # Use risk-free rate as benchmark
            benchmark_returns = [self.risk_free_rate/252] * len(returns)
        
        benchmark_array = np.array(benchmark_returns)
        
        # Excess returns
        excess = returns_array - benchmark_array
        
        # Tracking error
        tracking_error = np.std(excess)
        
        if tracking_error == 0:
            info_ratio = 0.0
        else:
            info_ratio = (np.mean(excess) / tracking_error) * np.sqrt(252)
        
        return {
            'information_ratio': round(info_ratio, 2),
            'tracking_error': round(tracking_error * np.sqrt(252) * 100, 2),
            'excess_return': round(np.mean(excess) * 252 * 100, 2),
            'status': 'ok'
        }
    
    def calculate_all_metrics(self, returns: Optional[List[float]] = None) -> Dict:
        """
        Calculate all metrics at once
        
        Returns:
            Dict with all calculated metrics
        """
        if returns is None:
            returns = self.returns_history
        
        if len(returns) < 2:
            return {'status': 'insufficient_data', 'data_points': len(returns)}
        
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        calmar = self.calculate_calmar_ratio(returns)
        info = self.calculate_information_ratio(returns)
        
        return {
            'sharpe_ratio': sharpe['sharpe_ratio'],
            'sortino_ratio': sortino['sortino_ratio'],
            'calmar_ratio': calmar['calmar_ratio'],
            'information_ratio': info['information_ratio'],
            'annual_return_pct': sharpe.get('annual_return', 0),
            'annual_volatility_pct': sharpe.get('annual_volatility', 0),
            'downside_volatility_pct': sortino.get('downside_volatility', 0),
            'data_points': len(returns),
            'status': 'ok'
        }
    
    def get_rolling_sharpe(self, window: int = 30) -> List[Dict]:
        """
        Calculate rolling Sharpe ratio over time
        
        Args:
            window: Window size in days
            
        Returns:
            List of {timestamp, sharpe_ratio} dicts
        """
        if len(self.returns_history) < window:
            return []
        
        rolling_sharpe = []
        
        for i in range(window, len(self.returns_history) + 1):
            window_returns = self.returns_history[i-window:i]
            sharpe = self.calculate_sharpe_ratio(window_returns, window=None)
            
            rolling_sharpe.append({
                'index': i,
                'sharpe_ratio': sharpe['sharpe_ratio']
            })
        
        return rolling_sharpe


# Singleton instance
_metrics_calculator: Optional[AdvancedMetrics] = None


def get_metrics_calculator() -> AdvancedMetrics:
    """Get singleton metrics calculator"""
    global _metrics_calculator
    if _metrics_calculator is None:
        _metrics_calculator = AdvancedMetrics()
    return _metrics_calculator


if __name__ == '__main__':
    # Test the metrics calculator
    print("=" * 60)
    print("ADVANCED METRICS CALCULATOR - TEST")
    print("=" * 60)
    
    # Generate sample returns (trending upward with volatility)
    np.random.seed(42)
    n_days = 100
    returns = np.random.normal(0.001, 0.02, n_days)  # 0.1% daily return, 2% vol
    
    calc = AdvancedMetrics(risk_free_rate=0.02)
    for r in returns:
        calc.add_return(r)
    
    # Test all metrics
    print("\n📊 All Metrics:")
    all_metrics = calc.calculate_all_metrics()
    for key, value in all_metrics.items():
        print(f"  {key}: {value}")
    
    # Test individual metrics
    print("\n📈 Sharpe Ratio (30-day):")
    sharpe = calc.calculate_sharpe_ratio(window=30)
    for key, value in sharpe.items():
        print(f"  {key}: {value}")
    
    print("\n📉 Sortino Ratio:")
    sortino = calc.calculate_sortino_ratio()
    for key, value in sortino.items():
        print(f"  {key}: {value}")
    
    print("\n💹 Calmar Ratio:")
    calmar = calc.calculate_calmar_ratio()
    for key, value in calmar.items():
        print(f"  {key}: {value}")
    
    print("\n🎯 Information Ratio:")
    info = calc.calculate_information_ratio()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ All metrics calculated successfully!")
    print("=" * 60)
