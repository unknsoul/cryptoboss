"""
Modern Portfolio Theory - Portfolio Optimization Engine
Implements Markowitz mean-variance optimization, risk parity, and advanced allocation

Features:
- Efficient frontier calculation
- Sharpe ratio maximization
- Risk parity allocation
- Black-Litterman model
- Dynamic rebalancing
- Multi-asset optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy.optimize import minimize
from datetime import datetime, timedelta
import cvxpy as cp

from core.monitoring.logger import get_logger
from core.monitoring.metrics import get_metrics


logger = get_logger()
metrics = get_metrics()


class PortfolioOptimizer:
    """
    Modern Portfolio Theory optimizer
    
    Optimizes portfolio allocation to maximize risk-adjusted returns
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.optimal_weights = None
        self.efficient_frontier = None
    
    def calculate_portfolio_stats(self, 
                                  weights: np.ndarray,
                                  returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio statistics
        
        Args:
            weights: Asset weights (must sum to 1)
            returns: Historical returns dataframe
        
        Returns:
            Dict with return, volatility, sharpe
        """
        # Expected returns (annualized)
        mean_returns = returns.mean() * 252
        
        # Covariance matrix (annualized)
        cov_matrix = returns.cov() * 252
        
        # Portfolio return
        port_return = np.sum(mean_returns * weights)
        
        # Portfolio volatility
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Sharpe ratio
        sharpe = (port_return - self.risk_free_rate) / port_volatility
        
        return {
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': sharpe
        }
    
    def maximize_sharpe(self, 
                       returns: pd.DataFrame,
                       constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Find portfolio weights that maximize Sharpe ratio
        
        Args:
            returns: Historical returns dataframe
            constraints: Optional constraints dict:
                - min_weight: Minimum weight per asset (default 0)
                - max_weight: Maximum weight per asset (default 1)
                - target_return: Target return (if specified)
        
        Returns:
            Optimal weights array
        """
        n_assets = len(returns.columns)
        
        # Default constraints
        if constraints is None:
            constraints = {}
        
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        
        # Calculate statistics
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Objective: Minimize negative Sharpe (maximize Sharpe)
        def neg_sharpe(weights):
            port_return = np.sum(mean_returns * weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (port_return - self.risk_free_rate) / port_volatility
            return -sharpe
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Target return constraint (if specified)
        if 'target_return' in constraints:
            target = constraints['target_return']
            cons.append({
                'type': 'eq',
                'fun': lambda w: np.sum(mean_returns * w) - target
            })
        
        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Initial guess (equal weight)
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        if result.success:
            self.optimal_weights = result.x
            logger.info(
                "✅ Sharpe maximization successful",
                sharpe=-result.fun,
                weights=dict(zip(returns.columns, np.round(result.x, 4)))
            )
            metrics.increment("portfolio_optimization_success")
        else:
            logger.warning("⚠️ Optimization did not converge")
            self.optimal_weights = init_weights
            metrics.increment("portfolio_optimization_failed")
        
        return self.optimal_weights
    
    def efficient_frontier(self,
                          returns: pd.DataFrame,
                          n_points: int = 100) -> pd.DataFrame:
        """
        Calculate efficient frontier
        
        Args:
            returns: Historical returns
            n_points: Number of points on frontier
        
        Returns:
            DataFrame with frontier points
        """
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Range of target returns
        min_return = mean_returns.min()
        max_return = mean_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier = []
        
        for target in target_returns:
            # Minimize volatility for given return
            weights = self._minimize_volatility(returns, target)
            
            if weights is not None:
                stats = self.calculate_portfolio_stats(weights, returns)
                frontier.append({
                    'return': stats['return'],
                    'volatility': stats['volatility'],
                    'sharpe': stats['sharpe'],
                    'weights': weights
                })
        
        self.efficient_frontier = pd.DataFrame(frontier)
        
        logger.info(f"Calculated efficient frontier with {len(frontier)} points")
        
        return self.efficient_frontier
    
    def _minimize_volatility(self,
                            returns: pd.DataFrame,
                            target_return: float) -> Optional[np.ndarray]:
        """Minimize portfolio volatility for target return"""
        n_assets = len(returns.columns)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Objective: Minimize volatility
        def volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
            {'type': 'eq', 'fun': lambda w: np.sum(mean_returns * w) - target_return}  # Target return
        ]
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            volatility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return result.x if result.success else None
    
    def risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Risk parity allocation - equal risk contribution from each asset
        
        Args:
            returns: Historical returns
        
        Returns:
            Risk parity weights
        """
        n_assets = len(returns.columns)
        cov_matrix = returns.cov() * 252
        
        # Objective: Minimize difference in risk contributions
        def risk_contribution_variance(weights):
            # Portfolio volatility
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Marginal contribution to risk
            marginal_contrib = np.dot(cov_matrix, weights) / port_vol
            
            # Risk contribution
            risk_contrib = weights * marginal_contrib
            
            # Variance of risk contributions (we want them equal)
            target = port_vol / n_assets
            return np.sum((risk_contrib - target) ** 2)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            risk_contribution_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            weights = result.x
            logger.info(
                "✅ Risk parity allocation calculated",
                weights=dict(zip(returns.columns, np.round(weights, 4)))
            )
            return weights
        else:
            logger.warning("⚠️ Risk parity optimization failed")
            return init_weights
    
    def black_litterman(self,
                       returns: pd.DataFrame,
                       market_caps: Dict[str, float],
                       views: List[Dict[str, Any]],
                       tau: float = 0.05,
                       omega_scale: float = 1.0) -> np.ndarray:
        """
        Black-Litterman model - incorporate views into market equilibrium
        
        Args:
            returns: Historical returns
            market_caps: Market capitalization of each asset
            views: List of views, each dict with:
                - assets: List of asset names
                - weights: Weights for view (must sum to 0 for relative views)
                - return: Expected return for this view
                - confidence: Confidence level (0-1)
            tau: Uncertainty in equilibrium (typically 0.01-0.05)
            omega_scale: Scaling factor for view uncertainty
        
        Returns:
            Optimal weights incorporating views
        """
        # Market equilibrium weights (from market caps)
        total_cap = sum(market_caps.values())
        w_eq = np.array([market_caps.get(col, 0) / total_cap for col in returns.columns])
        
        # Covariance matrix
        cov_matrix = returns.cov() * 252
        
        # Implied equilibrium returns (reverse optimization)
        # π = λ * Σ * w_eq
        # where λ is risk aversion coefficient (approximate as 2.5)
        lam = 2.5
        pi = lam * np.dot(cov_matrix, w_eq)
        
        # Views matrix P and vector Q
        n_assets = len(returns.columns)
        n_views = len(views)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        Omega = np.zeros((n_views, n_views))
        
        for i, view in enumerate(views):
            # Build P matrix
            for asset, weight in zip(view['assets'], view['weights']):
                idx = returns.columns.get_loc(asset)
                P[i, idx] = weight
            
            # Expected return for this view
            Q[i] = view['return']
            
            # Uncertainty in view (Omega)
            # Higher confidence = lower uncertainty
            view_var = np.dot(P[i], np.dot(cov_matrix, P[i].T))
            Omega[i, i] = view_var * omega_scale / view['confidence']
        
        # Black-Litterman formula
        # Posterior returns = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)π + P'Ω^(-1)Q]
        
        tau_cov_inv = np.linalg.inv(tau * cov_matrix)
        omega_inv = np.linalg.inv(Omega)
        
        A = tau_cov_inv + np.dot(P.T, np.dot(omega_inv, P))
        b = np.dot(tau_cov_inv, pi) + np.dot(P.T, np.dot(omega_inv, Q))
        
        # Posterior expected returns
        posterior_returns = np.dot(np.linalg.inv(A), b)
        
        # Optimize with posterior returns
        def neg_utility(weights):
            port_return = np.sum(posterior_returns * weights)
            port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            # Mean-variance utility
            return -(port_return - lam/2 * port_var)
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = w_eq
        
        result = minimize(
            neg_utility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            logger.info(
                "✅ Black-Litterman optimization complete",
                weights=dict(zip(returns.columns, np.round(result.x, 4)))
            )
            return result.x
        else:
            logger.warning("⚠️ Black-Litterman optimization failed")
            return w_eq


class DynamicRebalancer:
    """
    Manages portfolio rebalancing with transaction cost awareness
    """
    
    def __init__(self,
                 threshold: float = 0.05,
                 transaction_cost: float = 0.001):
        """
        Args:
            threshold: Rebalance when drift exceeds this (5%)
            transaction_cost: Trading cost as fraction (0.1%)
        """
        self.threshold = threshold
        self.transaction_cost = transaction_cost
        self.last_rebalance = None
    
    def should_rebalance(self,
                        current_weights: Dict[str, float],
                        target_weights: Dict[str, float]) -> bool:
        """
        Check if portfolio should be rebalanced
        
        Returns:
            True if rebalancing is needed
        """
        # Calculate drift
        total_drift = 0
        for asset in target_weights:
            current = current_weights.get(asset, 0)
            target = target_weights[asset]
            drift = abs(current - target)
            total_drift += drift
        
        if total_drift > self.threshold:
            logger.info(f"Rebalancing triggered: drift = {total_drift:.2%}")
            return True
        
        return False
    
    def calculate_trades(self,
                        current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        portfolio_value: float) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance
        
        Returns:
            Dict of asset -> trade amount (positive = buy, negative = sell)
        """
        trades = {}
        
        for asset in target_weights:
            current = current_weights.get(asset, 0)
            target = target_weights[asset]
            
            # Trade amount
            trade_pct = target - current
            trade_amount = trade_pct * portfolio_value
            
            if abs(trade_amount) > 10:  # Minimum $10 trade
                trades[asset] = trade_amount
        
        logger.info(f"Rebalancing trades calculated: {len(trades)} assets")
        
        return trades


if __name__ == "__main__":
    print("=" * 70)
    print("📊 PORTFOLIO OPTIMIZATION TEST")
    print("=" * 70)
    
    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    returns = pd.DataFrame({
        'BTC': np.random.normal(0.001, 0.03, 252),
        'ETH': np.random.normal(0.0008, 0.035, 252),
        'BNB': np.random.normal(0.0006, 0.025, 252),
        'SOL': np.random.normal(0.0012, 0.04, 252)
    }, index=dates)
    
    optimizer = PortfolioOptimizer()
    
    # Test 1: Maximize Sharpe
    print("\n1. Maximum Sharpe Ratio Portfolio:")
    weights = optimizer.maximize_sharpe(returns)
    stats = optimizer.calculate_portfolio_stats(weights, returns)
    
    print(f"   Expected Return: {stats['return']:.2%}")
    print(f"   Volatility: {stats['volatility']:.2%}")
    print(f"   Sharpe Ratio: {stats['sharpe']:.2f}")
    print(f"\n   Weights:")
    for asset, weight in zip(returns.columns, weights):
        print(f"      {asset}: {weight:.1%}")
    
    # Test 2: Risk Parity
    print("\n2. Risk Parity Portfolio:")
    rp_weights = optimizer.risk_parity(returns)
    rp_stats = optimizer.calculate_portfolio_stats(rp_weights, returns)
    
    print(f"   Expected Return: {rp_stats['return']:.2%}")
    print(f"   Volatility: {rp_stats['volatility']:.2%}")
    print(f"   Sharpe Ratio: {rp_stats['sharpe']:.2f}")
    print(f"\n   Weights:")
    for asset, weight in zip(returns.columns, rp_weights):
        print(f"      {asset}: {weight:.1%}")
    
    # Test 3: Rebalancing
    print("\n3. Rebalancing Test:")
    rebalancer = DynamicRebalancer(threshold=0.05)
    
    current = {'BTC': 0.35, 'ETH': 0.30, 'BNB': 0.20, 'SOL': 0.15}
    target = dict(zip(returns.columns, weights))
    
    if rebalancer.should_rebalance(current, target):
        trades = rebalancer.calculate_trades(current, target, 10000)
        print("   Trades needed:")
        for asset, amount in trades.items():
            print(f"      {asset}: ${amount:,.2f}")
    
    print("\n✅ Portfolio optimization test complete")
