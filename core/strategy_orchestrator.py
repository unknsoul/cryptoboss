"""
Multi-Strategy Orchestrator
Manages multiple strategies with intelligent capital allocation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Advanced Portfolio Optimization Engine
    
    Implements:
    - Mean-Variance Optimization (Markowitz)
    - Kelly Criterion for position sizing
    - Risk Parity allocation
    - Minimum Variance Portfolio
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def optimize_mean_variance(self, 
                                returns: np.ndarray, 
                                target_return: Optional[float] = None,
                                max_weight: float = 0.4) -> np.ndarray:
        """
        Mean-Variance Optimization (Markowitz)
        
        Args:
            returns: 2D array of shape (n_periods, n_assets) containing historical returns
            target_return: Optional target return. If None, maximizes Sharpe ratio.
            max_weight: Maximum weight per asset (default 40%)
        
        Returns:
            Optimal weights array of shape (n_assets,)
        """
        n_assets = returns.shape[1]
        
        # Calculate expected returns and covariance matrix
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)
        
        # Handle single asset case
        if n_assets == 1:
            return np.array([1.0])
        
        # Objective: Minimize portfolio variance (for max Sharpe, we iterate)
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        def portfolio_return(weights):
            return np.dot(weights, mean_returns)
        
        def negative_sharpe(weights):
            port_ret = portfolio_return(weights)
            port_vol = np.sqrt(portfolio_variance(weights))
            return -(port_ret - self.risk_free_rate / 252) / (port_vol + 1e-8)
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # If target return specified, add constraint
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda w: portfolio_return(w) - target_return
            })
        
        # Bounds: 0 <= weight <= max_weight
        bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            # Normalize weights to sum to 1
            weights = result.x
            weights = np.maximum(weights, 0)  # Ensure non-negative
            weights = weights / weights.sum()
            return weights
        else:
            # Fallback to equal weights
            return initial_weights
    
    def optimize_kelly(self, 
                       win_probabilities: np.ndarray, 
                       win_loss_ratios: np.ndarray,
                       fraction: float = 0.25) -> np.ndarray:
        """
        Kelly Criterion for position sizing
        
        Args:
            win_probabilities: Array of win probabilities for each strategy
            win_loss_ratios: Array of (avg_win / avg_loss) for each strategy
            fraction: Kelly fraction (default 0.25 = quarter Kelly for safety)
        
        Returns:
            Optimal allocation weights
        """
        n_assets = len(win_probabilities)
        
        # Kelly formula: f* = (p * b - q) / b
        # where p = win prob, q = 1 - p, b = win/loss ratio
        kelly_fractions = []
        
        for p, b in zip(win_probabilities, win_loss_ratios):
            q = 1 - p
            if b > 0:
                kelly = (p * b - q) / b
            else:
                kelly = 0
            
            # Clamp between 0 and 1
            kelly = max(0, min(kelly, 1))
            
            # Apply fractional Kelly for safety
            kelly *= fraction
            
            kelly_fractions.append(kelly)
        
        kelly_fractions = np.array(kelly_fractions)
        
        # Normalize to sum to 1 (if any positive fractions)
        if kelly_fractions.sum() > 0:
            return kelly_fractions / kelly_fractions.sum()
        else:
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets
    
    def optimize_risk_parity(self, returns: np.ndarray) -> np.ndarray:
        """
        Risk Parity Allocation
        Each asset contributes equally to portfolio risk.
        
        Args:
            returns: 2D array of historical returns
        
        Returns:
            Risk parity weights
        """
        n_assets = returns.shape[1]
        cov_matrix = np.cov(returns.T)
        
        # Handle single asset
        if n_assets == 1:
            return np.array([1.0])
        
        # Inverse volatility as initial approximation
        vols = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1.0 / (vols + 1e-8)
        
        # Normalize
        weights = inv_vols / inv_vols.sum()
        return weights
    
    def optimize_minimum_variance(self, returns: np.ndarray) -> np.ndarray:
        """
        Minimum Variance Portfolio
        
        Args:
            returns: 2D array of historical returns
        
        Returns:
            Minimum variance weights
        """
        n_assets = returns.shape[1]
        cov_matrix = np.cov(returns.T)
        
        if n_assets == 1:
            return np.array([1.0])
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            weights = np.maximum(weights, 0)
            return weights / weights.sum()
        else:
            return initial_weights


class StrategyOrchestrator:
    """
    Professional multi-strategy manager:
    - Run multiple strategies simultaneously
    - Dynamic capital allocation
    - Conflict resolution
    - Performance tracking per strategy
    - Regime-aware strategy selection
    """
    
    def __init__(self, 
                 total_capital,
                 strategies: Dict[str, Any],
                 allocation_method='equal'):
        """
        Args:
            total_capital: Total capital to allocate
            strategies: Dict of {strategy_name: strategy_instance}
            allocation_method: 'equal', 'performance_weighted', 'dynamic',
                             'mean_variance', 'kelly', 'risk_parity', 'min_variance'
        """
        
        self.total_capital = total_capital
        self.strategies = strategies
        self.allocation_method = allocation_method
        
        # Portfolio optimizer for advanced methods
        self.optimizer = PortfolioOptimizer()
        
        # Historical returns for optimization (simulated for now)
        self.returns_history: List[Dict[str, float]] = []
        
        # Track performance per strategy
        self.strategy_performance = {name: {
            'total_return': 0,
            'sharpe': 0,
            'win_rate': 0.5,
            'avg_win': 1.0,
            'avg_loss': 1.0,
            'trades': [],
            'returns': [],
            'current_equity': total_capital / len(strategies)
        } for name in strategies.keys()}
        
        # Initial allocation
        self.allocations = self._calculate_allocations()
    
    def _calculate_allocations(self):
        """Calculate capital allocation for each strategy"""
        
        num_strategies = len(self.strategies)
        
        if self.allocation_method == 'equal':
            # Equal allocation
            allocation_per_strategy = self.total_capital / num_strategies
            return {name: allocation_per_strategy for name in self.strategies.keys()}
        
        elif self.allocation_method == 'performance_weighted':
            # Weight by recent performance (Sharpe ratio)
            sharpes = np.array([self.strategy_performance[name]['sharpe'] 
                               for name in self.strategies.keys()])
            
            # Avoid negative weights
            sharpes = np.maximum(sharpes, 0.1)
            
            # Normalize to sum to 1
            weights = sharpes / sharpes.sum()
            
            return {name: self.total_capital * weight 
                   for name, weight in zip(self.strategies.keys(), weights)}
        
        elif self.allocation_method == 'dynamic':
            # Allocate more to strategies with positive momentum
            returns = np.array([self.strategy_performance[name]['total_return'] 
                               for name in self.strategies.keys()])
            
            # Use exponential weighting (favor winners)
            weights = np.exp(returns)
            weights = weights / weights.sum()
            
            return {name: self.total_capital * weight 
                   for name, weight in zip(self.strategies.keys(), weights)}
        
        elif self.allocation_method == 'mean_variance':
            # Mean-Variance Optimization (Markowitz)
            returns_matrix = self._get_returns_matrix()
            if returns_matrix is not None and len(returns_matrix) > 10:
                weights = self.optimizer.optimize_mean_variance(returns_matrix)
                names = list(self.strategies.keys())
                return {name: self.total_capital * w for name, w in zip(names, weights)}
            else:
                # Not enough data, fallback to equal
                return {name: self.total_capital / num_strategies for name in self.strategies.keys()}
        
        elif self.allocation_method == 'kelly':
            # Kelly Criterion
            win_probs = np.array([self.strategy_performance[name].get('win_rate', 0.5) 
                                  for name in self.strategies.keys()])
            
            win_loss_ratios = []
            for name in self.strategies.keys():
                avg_win = self.strategy_performance[name].get('avg_win', 1.0)
                avg_loss = abs(self.strategy_performance[name].get('avg_loss', 1.0)) + 1e-8
                win_loss_ratios.append(avg_win / avg_loss)
            
            win_loss_ratios = np.array(win_loss_ratios)
            weights = self.optimizer.optimize_kelly(win_probs, win_loss_ratios)
            names = list(self.strategies.keys())
            return {name: self.total_capital * w for name, w in zip(names, weights)}
        
        elif self.allocation_method == 'risk_parity':
            # Risk Parity
            returns_matrix = self._get_returns_matrix()
            if returns_matrix is not None and len(returns_matrix) > 10:
                weights = self.optimizer.optimize_risk_parity(returns_matrix)
                names = list(self.strategies.keys())
                return {name: self.total_capital * w for name, w in zip(names, weights)}
            else:
                return {name: self.total_capital / num_strategies for name in self.strategies.keys()}
        
        elif self.allocation_method == 'min_variance':
            # Minimum Variance Portfolio
            returns_matrix = self._get_returns_matrix()
            if returns_matrix is not None and len(returns_matrix) > 10:
                weights = self.optimizer.optimize_minimum_variance(returns_matrix)
                names = list(self.strategies.keys())
                return {name: self.total_capital * w for name, w in zip(names, weights)}
            else:
                return {name: self.total_capital / num_strategies for name in self.strategies.keys()}
        
        else:
            # Default to equal
            allocation_per_strategy = self.total_capital / num_strategies
            return {name: allocation_per_strategy for name in self.strategies.keys()}
    
    def _get_returns_matrix(self) -> Optional[np.ndarray]:
        """
        Build returns matrix from strategy performance history.
        
        Returns:
            2D numpy array of shape (n_periods, n_strategies) or None if insufficient data
        """
        names = list(self.strategies.keys())
        
        # Check if we have returns history for all strategies
        min_returns = min(len(self.strategy_performance[name].get('returns', [])) 
                          for name in names)
        
        if min_returns < 10:
            return None
        
        # Build matrix
        returns_matrix = []
        for name in names:
            returns_matrix.append(self.strategy_performance[name]['returns'][-min_returns:])
        
        return np.array(returns_matrix).T  # Transpose to (n_periods, n_strategies)
    
    def get_signals(self, highs, lows, closes, volumes=None):
        """
        Get signals from all strategies
        
        Returns:
            List of (strategy_name, signal) tuples
        """
        
        signals = []
        
        for name, strategy in self.strategies.items():
            signal = strategy.signal(highs, lows, closes, volumes)
            
            if signal is not None:
                signals.append((name, signal))
        
        return signals
    
    def resolve_conflicts(self, signals: List[tuple]):
        """
        Resolve conflicting signals (e.g., one strategy long, another short)
        
        Args:
            signals: List of (strategy_name, signal) tuples
        
        Returns:
            filtered_signals: List of signals to execute
        """
        
        if len(signals) <= 1:
            return signals
        
        # Count long vs short signals
        long_count = sum(1 for _, sig in signals if sig['action'] == 'LONG')
        short_count = sum(1 for _, sig in signals if sig['action'] == 'SHORT')
        
        # If majority agrees, filter out minority
        if long_count > short_count:
            # Keep only long signals
            return [(name, sig) for name, sig in signals if sig['action'] == 'LONG']
        elif short_count > long_count:
            # Keep only short signals
            return [(name, sig) for name, sig in signals if sig['action'] == 'SHORT']
        else:
            # Tie - keep all (or implement tie-breaking logic)
            return signals
    
    def update_performance(self, strategy_name, metrics):
        """Update performance tracking for a strategy"""
        
        self.strategy_performance[strategy_name].update({
            'total_return': metrics.get('total_return', 0),
            'sharpe': metrics.get('sharpe_ratio', 0),
            'trades': metrics.get('trades', [])
        })
    
    def get_allocation(self, strategy_name):
        """Get current capital allocation for a strategy"""
        return self.allocations.get(strategy_name, 0)
    
    def rebalance(self):
        """Rebalance capital allocations based on performance"""
        self.allocations = self._calculate_allocations()
    
    def get_summary(self):
        """Get performance summary across all strategies"""
        
        total_return = sum(perf['total_return'] for perf in self.strategy_performance.values()) / len(self.strategies)
        avg_sharpe = np.mean([perf['sharpe'] for perf in self.strategy_performance.values()])
        
        return {
            'total_strategies': len(self.strategies),
            'avg_return': total_return,
            'avg_sharpe': avg_sharpe,
            'allocations': self.allocations,
            'performance_by_strategy': self.strategy_performance
        }
