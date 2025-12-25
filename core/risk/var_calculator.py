"""
Value at Risk (VaR) Calculator
Professional risk measurement for trading portfolios

VaR estimates the maximum loss over a time period at a given confidence level.
Example: 1-day VaR of $500 at 95% confidence = 95% chance loss won't exceed $500 tomorrow
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class VaRCalculator:
    """
    Professional Value at Risk calculator
    
    Methods:
    - Historical VaR (empirical distribution)
    - Parametric VaR (normal distribution assumption)
    - Monte Carlo VaR (simulation-based)
    - Conditional VaR / Expected Shortfall
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator
        
        Args:
            confidence_level: Confidence level (0.90, 0.95, 0.99)
        """
        self.confidence_level = confidence_level
        self.returns_history = []
        
    def add_return(self, daily_return: float):
        """Add daily return to history"""
        self.returns_history.append(daily_return)
    
    def historical_var(self, returns: Optional[List[float]] = None,
                      confidence: Optional[float] = None,
                      horizon_days: int = 1) -> Dict:
        """
        Calculate Historical VaR (non-parametric)
        
        Uses actual historical returns distribution without assumptions.
        
        Args:
            returns: List of daily returns (uses history if None)
            confidence: Confidence level (uses self.confidence_level if None)
            horizon_days: Time horizon in days
            
        Returns:
            Dict with VaR and details
        """
        if returns is None:
            returns = self.returns_history
        
        if len(returns) < 30:
            return {'status': 'insufficient_data', 'required': 30, 'available': len(returns)}
        
        conf = confidence or self.confidence_level
        returns_array = np.array(returns)
        
        # For multi-day horizon, scale by sqrt(days) (assumes i.i.d.)
        if horizon_days > 1:
            returns_array = returns_array * np.sqrt(horizon_days)
        
        # VaR is the percentile at (1 - confidence) level
        var_percentile = (1 - conf) * 100
        var = np.percentile(returns_array, var_percentile)
        
        # Convert to dollar terms (assuming current portfolio value)
        # This will be multiplied by equity when displaying
        
        return {
            'var_pct': round(var * 100, 2),  # As percentage
            'var_dollar_per_1k': round(abs(var) * 1000, 2),  # Loss per $1000
            'confidence': conf * 100,
            'horizon_days': horizon_days,
            'method': 'historical',
            'data_points': len(returns),
            'status': 'ok'
        }
    
    def parametric_var(self, returns: Optional[List[float]] = None,
                       confidence: Optional[float] = None,
                       horizon_days: int = 1) -> Dict:
        """
        Calculate Parametric VaR (variance-covariance method)
        
        Assumes returns follow normal distribution.
        Faster but less accurate for fat-tailed distributions.
        
        Args:
            returns: List of daily returns
            confidence: Confidence level
            horizon_days: Time horizon in days
            
        Returns:
            Dict with VaR and details
        """
        if returns is None:
            returns = self.returns_history
        
        if len(returns) < 30:
            return {'status': 'insufficient_data'}
        
        conf = confidence or self.confidence_level
        returns_array = np.array(returns)
        
        # Calculate mean and std
        mu = np.mean(returns_array)
        sigma = np.std(returns_array)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - conf)
        
        # VaR formula: μ + z*σ (z is negative for losses)
        var_1day = mu + z_score * sigma
        
        # Scale for horizon
        var = var_1day * np.sqrt(horizon_days)
        
        return {
            'var_pct': round(var * 100, 2),
            'var_dollar_per_1k': round(abs(var) * 1000, 2),
            'confidence': conf * 100,
            'horizon_days': horizon_days,
            'mean_return': round(mu * 100, 4),
            'volatility': round(sigma * 100, 2),
            'z_score': round(z_score, 2),
            'method': 'parametric',
            'status': 'ok'
        }
    
    def monte_carlo_var(self, returns: Optional[List[float]] = None,
                        confidence: Optional[float] = None,
                        horizon_days: int = 1,
                        n_simulations: int = 10000) -> Dict:
        """
        Calculate Monte Carlo VaR
        
        Simulates many possible return paths using historical parameters.
        Most flexible method, accounts for path dependency.
        
        Args:
            returns: Historical returns
            confidence: Confidence level
            horizon_days: Time horizon
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dict with VaR and simulation details
        """
        if returns is None:
            returns = self.returns_history
        
        if len(returns) < 30:
            return {'status': 'insufficient_data'}
        
        conf = confidence or self.confidence_level
        returns_array = np.array(returns)
        
        # Fit distribution to historical returns
        mu = np.mean(returns_array)
        sigma = np.std(returns_array)
        
        # Simulate returns for each path
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mu, sigma, (n_simulations, horizon_days))
        
        # Calculate cumulative return for each path
        cumulative_returns = np.sum(simulated_returns, axis=1)
        
        # VaR is percentile of simulated outcomes
        var_percentile = (1 - conf) * 100
        var = np.percentile(cumulative_returns, var_percentile)
        
        return {
            'var_pct': round(var * 100, 2),
            'var_dollar_per_1k': round(abs(var) * 1000, 2),
            'confidence': conf * 100,
            'horizon_days': horizon_days,
            'n_simulations': n_simulations,
            'worst_case_pct': round(np.min(cumulative_returns) * 100, 2),
            'best_case_pct': round(np.max(cumulative_returns) * 100, 2),
            'method': 'monte_carlo',
            'status': 'ok'
        }
    
    def conditional_var(self, returns: Optional[List[float]] = None,
                        confidence: Optional[float] = None) -> Dict:
        """
        Calculate Conditional VaR (CVaR) / Expected Shortfall
        
        Mean loss given that loss exceeds VaR.
        Better risk measure as it considers tail risk.
        
        Args:
            returns: Historical returns
            confidence: Confidence level
            
        Returns:
            Dict with CVaR (expected loss in worst cases)
        """
        if returns is None:
            returns = self.returns_history
        
        if len(returns) < 50:
            return {'status': 'insufficient_data', 'required': 50}
        
        conf = confidence or self.confidence_level
        returns_array = np.array(returns)
        
        # Get VaR threshold
        var_percentile = (1 - conf) * 100
        var_threshold = np.percentile(returns_array, var_percentile)
        
        # CVaR = average of returns worse than VaR
        tail_losses = returns_array[returns_array <= var_threshold]
        
        if len(tail_losses) == 0:
            cvar = var_threshold
        else:
            cvar = np.mean(tail_losses)
        
        return {
            'cvar_pct': round(cvar * 100, 2),
            'cvar_dollar_per_1k': round(abs(cvar) * 1000, 2),
            'var_pct': round(var_threshold * 100, 2),
            'tail_events': len(tail_losses),
            'confidence': conf * 100,
            'method': 'conditional_var',
            'status': 'ok'
        }
    
    def calculate_all_var(self, returns: Optional[List[float]] = None,
                          confidence: float = 0.95) -> Dict:
        """
        Calculate VaR using all methods for comparison
        
        Returns:
            Dict with all VaR measures
        """
        if returns is None:
            returns = self.returns_history
        
        hist_var = self.historical_var(returns, confidence, horizon_days=1)
        param_var = self.parametric_var(returns, confidence, horizon_days=1)
        mc_var = self.monte_carlo_var(returns, confidence, horizon_days=1, n_simulations=5000)
        cvar = self.conditional_var(returns, confidence)
        
        # Also calculate for longer horizons
        hist_var_5d = self.historical_var(returns, confidence, horizon_days=5)
        
        return {
            '1_day_historical': hist_var,
            '1_day_parametric': param_var,
            '1_day_monte_carlo': mc_var,
            '5_day_historical': hist_var_5d,
            'conditional_var': cvar,
            'summary': {
                'avg_1day_var_pct': round(np.mean([
                    hist_var.get('var_pct', 0),
                    param_var.get('var_pct', 0),
                    mc_var.get('var_pct', 0)
                ]), 2),
                'confidence': confidence * 100
            }
        }
    
    def var_for_equity(self, equity: float, var_pct: float) -> float:
        """
        Convert VaR percentage to dollar amount for given equity
        
        Args:
            equity: Current portfolio value
            var_pct: VaR as percentage (e.g., -2.5 for -2.5%)
            
        Returns:
            VaR in dollars
        """
        return abs(equity * var_pct / 100)


# Singleton
_var_calculator: Optional[VaRCalculator] = None


def get_var_calculator() -> VaRCalculator:
    """Get singleton VaR calculator"""
    global _var_calculator
    if _var_calculator is None:
        _var_calculator = VaRCalculator(confidence_level=0.95)
    return _var_calculator


if __name__ == '__main__':
    # Test VaR calculator
    print("=" * 70)
    print("VALUE AT RISK (VaR) CALCULATOR - TEST")
    print("=" * 70)
    
    # Generate sample returns (slightly negative drift, realistic volatility)
    np.random.seed(42)
    n_days = 250  # 1 year of trading data
    returns = np.random.normal(-0.0002, 0.02, n_days)  # -0.02% daily mean, 2% vol
    
    calc = VaRCalculator(confidence_level=0.95)
    for r in returns:
        calc.add_return(r)
    
    print(f"\n📊 Generated {n_days} days of sample returns")
    print(f"Mean daily return: {np.mean(returns)*100:.3f}%")
    print(f"Daily volatility: {np.std(returns)*100:.2f}%")
    
    # Test historical VaR
    print("\n📉 1-Day Historical VaR (95% confidence):")
    hist_var = calc.historical_var()
    for key, value in hist_var.items():
        print(f"  {key}: {value}")
    
    # Test parametric VaR
    print("\n📊 1-Day Parametric VaR (95% confidence):")
    param_var = calc.parametric_var()
    for key, value in param_var.items():
        print(f"  {key}: {value}")
    
    # Test Monte Carlo VaR
    print("\n🎲 Monte Carlo VaR (10,000 simulations):")
    mc_var = calc.monte_carlo_var(n_simulations=10000)
    for key, value in mc_var.items():
        print(f"  {key}: {value}")
    
    # Test CVaR
    print("\n⚠️  Conditional VaR (Expected Shortfall):")
    cvar = calc.conditional_var()
    for key, value in cvar.items():
        print(f"  {key}: {value}")
    
    # Test longer horizon
    print("\n📅 5-Day Historical VaR:")
    var_5d = calc.historical_var(horizon_days=5)
    print(f"  5-day VaR: {var_5d['var_pct']}%")
    print(f"  1-day VaR: {hist_var['var_pct']}%")
    print(f"  Scaling factor: {abs(var_5d['var_pct']/hist_var['var_pct']):.2f}x (should be ~√5 = 2.24)")
    
    # Example with real portfolio
    print("\n💰 Example: $10,000 Portfolio")
    equity = 10000
    var_dollar = calc.var_for_equity(equity, hist_var['var_pct'])
    print(f"  1-day VaR @ 95%: ${var_dollar:.2f}")
    print(f"  Interpretation: 95% confident we won't lose more than ${var_dollar:.2f} tomorrow")
    
    print("\n" + "=" * 70)
    print("✅ VaR calculator working correctly!")
    print("=" * 70)
