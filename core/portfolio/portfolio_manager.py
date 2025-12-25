"""
Portfolio Management - Enterprise Features #161, #165, #170, #175
Multi-Asset Tracking, Correlation, Rebalancing, and Performance Ratios.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """
    Feature #161: Multi-Asset Portfolio Tracker
    
    Tracks positions and performance across multiple assets.
    """
    
    def __init__(self, initial_capital: float = 10000):
        """
        Initialize portfolio tracker.
        
        Args:
            initial_capital: Starting portfolio value
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        self.positions: Dict[str, Dict] = {}  # symbol -> position
        self.trades: List[Dict] = []
        self.equity_history: List[Dict] = []
        self.daily_returns: List[float] = []
        
        logger.info(f"Portfolio Tracker initialized - ${initial_capital:,.2f}")
    
    def add_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """Add a new position."""
        position = {
            'symbol': symbol,
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'current_price': entry_price,
            'entry_time': timestamp or datetime.now(),
            'unrealized_pnl': 0,
            'cost': size * entry_price
        }
        
        self.positions[symbol] = position
        self.cash -= position['cost']
        
        return position
    
    def update_price(self, symbol: str, price: float):
        """Update price for a position."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        pos['current_price'] = price
        
        if pos['side'] == 'LONG':
            pos['unrealized_pnl'] = (price - pos['entry_price']) * pos['size']
        else:
            pos['unrealized_pnl'] = (pos['entry_price'] - price) * pos['size']
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[Dict]:
        """Close a position."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        if pos['side'] == 'LONG':
            pnl = (exit_price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['size']
        
        trade = {
            'symbol': symbol,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'size': pos['size'],
            'pnl': round(pnl, 2),
            'return_pct': round((pnl / pos['cost']) * 100, 2),
            'hold_time': (datetime.now() - pos['entry_time']).total_seconds() / 3600,
            'timestamp': datetime.now().isoformat()
        }
        
        self.trades.append(trade)
        self.cash += pos['cost'] + pnl
        del self.positions[symbol]
        
        return trade
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        positions_value = sum(
            p['size'] * p['current_price'] + p['unrealized_pnl']
            for p in self.positions.values()
        )
        return self.cash + positions_value
    
    def record_equity(self):
        """Record current equity for history."""
        value = self.get_portfolio_value()
        self.equity_history.append({
            'timestamp': datetime.now().isoformat(),
            'value': round(value, 2)
        })
        
        # Calculate daily return
        if len(self.equity_history) >= 2:
            prev = self.equity_history[-2]['value']
            ret = (value - prev) / prev if prev > 0 else 0
            self.daily_returns.append(ret)
    
    def get_summary(self) -> Dict:
        """Get portfolio summary."""
        value = self.get_portfolio_value()
        total_pnl = value - self.initial_capital
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        
        return {
            'total_value': round(value, 2),
            'cash': round(self.cash, 2),
            'positions_count': len(self.positions),
            'total_pnl': round(total_pnl, 2),
            'return_pct': round((total_pnl / self.initial_capital) * 100, 2),
            'total_trades': len(self.trades),
            'win_rate': len(wins) / len(self.trades) if self.trades else 0,
            'positions': list(self.positions.keys())
        }


class CorrelationCalculator:
    """
    Feature #165: Correlation Matrix Calculator
    
    Calculates correlations between assets for diversification analysis.
    """
    
    def __init__(self, lookback: int = 30):
        """
        Initialize correlation calculator.
        
        Args:
            lookback: Number of periods for correlation
        """
        self.lookback = lookback
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"Correlation Calculator initialized - Lookback: {lookback}")
    
    def update_price(self, symbol: str, price: float):
        """Record price for correlation calculation."""
        self.price_history[symbol].append(price)
        # Keep only lookback periods
        self.price_history[symbol] = self.price_history[symbol][-self.lookback:]
    
    def calculate_returns(self, symbol: str) -> List[float]:
        """Calculate returns from prices."""
        prices = self.price_history.get(symbol, [])
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] != 0 else 0
            returns.append(ret)
        
        return returns
    
    def calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two assets."""
        returns1 = self.calculate_returns(symbol1)
        returns2 = self.calculate_returns(symbol2)
        
        if len(returns1) < 5 or len(returns2) < 5:
            return 0
        
        # Align lengths
        min_len = min(len(returns1), len(returns2))
        r1 = returns1[-min_len:]
        r2 = returns2[-min_len:]
        
        # Calculate correlation
        mean1 = sum(r1) / len(r1)
        mean2 = sum(r2) / len(r2)
        
        cov = sum((a - mean1) * (b - mean2) for a, b in zip(r1, r2)) / len(r1)
        std1 = (sum((x - mean1) ** 2 for x in r1) / len(r1)) ** 0.5
        std2 = (sum((x - mean2) ** 2 for x in r2) / len(r2)) ** 0.5
        
        if std1 == 0 or std2 == 0:
            return 0
        
        return cov / (std1 * std2)
    
    def get_correlation_matrix(self, symbols: List[str]) -> Dict:
        """Get full correlation matrix."""
        matrix = {}
        
        for s1 in symbols:
            matrix[s1] = {}
            for s2 in symbols:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                else:
                    matrix[s1][s2] = round(self.calculate_correlation(s1, s2), 3)
        
        return matrix
    
    def check_diversification(self, symbols: List[str], threshold: float = 0.7) -> Dict:
        """Check if portfolio is well-diversified."""
        high_correlations = []
        
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                corr = abs(self.calculate_correlation(s1, s2))
                if corr > threshold:
                    high_correlations.append({
                        'pair': (s1, s2),
                        'correlation': round(corr, 3)
                    })
        
        return {
            'is_diversified': len(high_correlations) == 0,
            'high_correlations': high_correlations,
            'recommendation': 'Well diversified' if not high_correlations else 'Consider reducing correlated positions'
        }


class RebalancingEngine:
    """
    Feature #170: Portfolio Rebalancing Engine
    
    Calculates rebalancing trades to maintain target allocations.
    """
    
    def __init__(self, tolerance_pct: float = 5.0):
        """
        Initialize rebalancing engine.
        
        Args:
            tolerance_pct: Tolerance before triggering rebalance
        """
        self.tolerance_pct = tolerance_pct
        self.target_allocations: Dict[str, float] = {}
        
        logger.info(f"Rebalancing Engine initialized - Tolerance: {tolerance_pct}%")
    
    def set_target_allocation(self, allocations: Dict[str, float]):
        """
        Set target allocations.
        
        Args:
            allocations: Symbol -> target percentage (0-100)
        """
        total = sum(allocations.values())
        if abs(total - 100) > 0.01:
            # Normalize
            allocations = {k: v / total * 100 for k, v in allocations.items()}
        
        self.target_allocations = allocations
    
    def calculate_rebalance(
        self,
        current_positions: Dict[str, float],  # symbol -> current value
        total_value: float
    ) -> Dict:
        """
        Calculate rebalancing trades needed.
        
        Args:
            current_positions: Current position values
            total_value: Total portfolio value
            
        Returns:
            Rebalancing instructions
        """
        if not self.target_allocations:
            return {'needs_rebalance': False, 'reason': 'No target allocations set'}
        
        trades = []
        needs_rebalance = False
        
        # Calculate current allocations
        current_allocations = {}
        for symbol, value in current_positions.items():
            current_allocations[symbol] = (value / total_value) * 100 if total_value > 0 else 0
        
        # Check each target
        for symbol, target_pct in self.target_allocations.items():
            current_pct = current_allocations.get(symbol, 0)
            diff = target_pct - current_pct
            
            if abs(diff) > self.tolerance_pct:
                needs_rebalance = True
                target_value = (target_pct / 100) * total_value
                current_value = current_positions.get(symbol, 0)
                trade_value = target_value - current_value
                
                trades.append({
                    'symbol': symbol,
                    'action': 'BUY' if trade_value > 0 else 'SELL',
                    'value': round(abs(trade_value), 2),
                    'current_pct': round(current_pct, 2),
                    'target_pct': round(target_pct, 2),
                    'diff_pct': round(diff, 2)
                })
        
        return {
            'needs_rebalance': needs_rebalance,
            'trades': trades,
            'total_trade_value': sum(t['value'] for t in trades)
        }


class PerformanceRatios:
    """
    Feature #175: Sharpe/Sortino/Calmar Ratios
    
    Calculates key risk-adjusted performance metrics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252
        
        logger.info(f"Performance Ratios initialized - RF: {risk_free_rate:.1%}")
    
    def calculate_sharpe(self, returns: List[float], annualize: bool = True) -> float:
        """
        Calculate Sharpe Ratio.
        
        Sharpe = (mean_return - rf) / std_return
        """
        if len(returns) < 10:
            return 0
        
        mean_return = sum(returns) / len(returns)
        excess_return = mean_return - self.daily_rf
        
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std = variance ** 0.5
        
        if std == 0:
            return 0
        
        sharpe = excess_return / std
        
        if annualize:
            sharpe *= (252 ** 0.5)
        
        return round(sharpe, 3)
    
    def calculate_sortino(self, returns: List[float], annualize: bool = True) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation).
        
        Sortino = (mean_return - rf) / downside_std
        """
        if len(returns) < 10:
            return 0
        
        mean_return = sum(returns) / len(returns)
        excess_return = mean_return - self.daily_rf
        
        # Downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float('inf')  # No negative returns
        
        downside_var = sum(r ** 2 for r in negative_returns) / len(returns)
        downside_std = downside_var ** 0.5
        
        if downside_std == 0:
            return 0
        
        sortino = excess_return / downside_std
        
        if annualize:
            sortino *= (252 ** 0.5)
        
        return round(sortino, 3)
    
    def calculate_calmar(self, returns: List[float], max_drawdown: float) -> float:
        """
        Calculate Calmar Ratio.
        
        Calmar = annualized_return / max_drawdown
        """
        if len(returns) < 10 or max_drawdown == 0:
            return 0
        
        total_return = 1
        for r in returns:
            total_return *= (1 + r)
        
        # Annualize
        periods = len(returns)
        annual_return = (total_return ** (252 / periods)) - 1
        
        calmar = annual_return / abs(max_drawdown)
        
        return round(calmar, 3)
    
    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(equity_curve) < 2:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return round(max_dd, 4)
    
    def get_all_ratios(
        self,
        returns: List[float],
        equity_curve: Optional[List[float]] = None
    ) -> Dict:
        """Calculate all performance ratios."""
        max_dd = self.calculate_max_drawdown(equity_curve) if equity_curve else 0.1
        
        return {
            'sharpe_ratio': self.calculate_sharpe(returns),
            'sortino_ratio': self.calculate_sortino(returns),
            'calmar_ratio': self.calculate_calmar(returns, max_dd),
            'max_drawdown': round(max_dd * 100, 2),
            'total_return': round(sum(returns) * 100, 2),
            'volatility': round((sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns)) ** 0.5 * (252**0.5) * 100, 2) if returns else 0,
            'sample_size': len(returns)
        }


# Singleton instances
_portfolio: Optional[PortfolioTracker] = None
_correlation: Optional[CorrelationCalculator] = None
_rebalancer: Optional[RebalancingEngine] = None
_ratios: Optional[PerformanceRatios] = None


def get_portfolio_tracker() -> PortfolioTracker:
    global _portfolio
    if _portfolio is None:
        _portfolio = PortfolioTracker()
    return _portfolio


def get_correlation_calculator() -> CorrelationCalculator:
    global _correlation
    if _correlation is None:
        _correlation = CorrelationCalculator()
    return _correlation


def get_rebalancer() -> RebalancingEngine:
    global _rebalancer
    if _rebalancer is None:
        _rebalancer = RebalancingEngine()
    return _rebalancer


def get_performance_ratios() -> PerformanceRatios:
    global _ratios
    if _ratios is None:
        _ratios = PerformanceRatios()
    return _ratios


if __name__ == '__main__':
    import random
    
    # Test portfolio tracker
    port = PortfolioTracker(10000)
    port.add_position('BTCUSDT', 'LONG', 0.1, 50000)
    port.update_price('BTCUSDT', 51000)
    print(f"Portfolio: {port.get_summary()}")
    
    # Test correlation
    corr = CorrelationCalculator()
    for i in range(30):
        corr.update_price('BTC', 50000 + random.uniform(-1000, 1000))
        corr.update_price('ETH', 3000 + random.uniform(-100, 100))
    print(f"Correlation BTC/ETH: {corr.calculate_correlation('BTC', 'ETH'):.3f}")
    
    # Test ratios
    ratios = PerformanceRatios()
    returns = [random.gauss(0.001, 0.02) for _ in range(100)]
    print(f"Performance: {ratios.get_all_ratios(returns)}")
