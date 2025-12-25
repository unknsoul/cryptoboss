"""
Institutional Risk Management - Advanced Risk Metrics
VaR, CVaR, Stress Testing, Monte Carlo Simulation

Features:
- Value at Risk (VaR) - 95%, 99%
- Conditional VaR (CVaR/Expected Shortfall)
- Stress testing scenarios
- Monte Carlo simulation
- Scenario analysis
- Risk factor decomposition
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from datetime import datetime, timedelta

from core.monitoring.logger import get_logger
from core.monitoring.metrics import get_metrics
from core.monitoring.alerting import get_alerts


logger = get_logger()
metrics = get_metrics()
alerts = get_alerts()


class InstitutionalRiskManager:
    """
    Advanced risk metrics for institutional-grade risk management
    """
    
    def __init__(self):
        self.var_history = []
        self.stress_test_results = []
    
    def calculate_var(self,
                     returns: np.ndarray,
                     confidence: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Array of returns
            confidence: Confidence level (0.95 = 95%)
            method: 'historical', 'parametric', or 'monte_carlo'
        
        Returns:
            VaR value (negative = loss)
        """
        if method == 'historical':
            # Historical VaR - use empirical distribution
            var = np.percentile(returns, (1 - confidence) * 100)
            
        elif method == 'parametric':
            # Parametric VaR - assume normal distribution
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence)
            var = mean + z_score * std
            
        elif method == 'monte_carlo':
            # Monte Carlo VaR - simulate future returns
            simulated = self._monte_carlo_simulation(returns, n_simulations=10000)
            var = np.percentile(simulated, (1 - confidence) * 100)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(
            f"VaR calculated ({method})",
            confidence=confidence,
            var=var,
            var_pct=f"{var:.2%}"
        )
        
        # Store for monitoring
        self.var_history.append({
            'timestamp': datetime.now(),
            'confidence': confidence,
            'method': method,
            'var': var
        })
        
        metrics.gauge("risk_var", var)
        
        # Alert if VaR exceeds threshold
        if var < -0.10:  # 10% loss
            alerts.send_alert(
                "high_var_detected",
                f"High VaR detected: {var:.2%}",
                {'var': var, 'confidence': confidence}
            )
        
        return var
    
    def calculate_cvar(self,
                      returns: np.ndarray,
                      confidence: float = 0.95,
                      method: str = 'historical') -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)
        Average loss beyond VaR
        
        Args:
            returns: Array of returns
            confidence: Confidence level
            method: Calculation method
        
        Returns:
            CVaR value (expected loss in worst cases)
        """
        # First calculate VaR
        var = self.calculate_var(returns, confidence, method)
        
        # CVaR = average of all returns worse than VaR
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) > 0:
            cvar = np.mean(tail_returns)
        else:
            cvar = var  # If no tail, CVaR = VaR
        
        logger.info(
            f"CVaR calculated ({method})",
            confidence=confidence,
            cvar=cvar,
            cvar_pct=f"{cvar:.2%}"
        )
        
        metrics.gauge("risk_cvar", cvar)
        
        return cvar
    
    def _monte_carlo_simulation(self,
                                returns: np.ndarray,
                                n_simulations: int = 10000,
                                horizon: int = 1) -> np.ndarray:
        """
        Monte Carlo simulation of future returns
        
        Args:
            returns: Historical returns
            n_simulations: Number of simulations
            horizon: Number of periods to simulate
        
        Returns:
            Array of simulated returns
        """
        # Estimate parameters from historical data
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Simulate
        simulated = np.random.normal(mean, std, (n_simulations, horizon))
        
        # Cumulative returns
        cumulative_returns = np.prod(1 + simulated, axis=1) - 1
        
        return cumulative_returns
    
    def stress_test(self,
                   portfolio_value: float,
                   positions: Dict[str, float],
                   scenarios: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Stress test portfolio under extreme scenarios
        
        Args:
            portfolio_value: Current portfolio value
            positions: Dict of asset -> position size
            scenarios: List of stress scenarios (if None, use defaults)
        
        Returns:
            DataFrame with scenario results
        """
        if scenarios is None:
            # Default stress scenarios
            scenarios = [
                {
                    'name': '2008 Financial Crisis',
                    'btc_change': -0.85,  # 85% crash
                    'eth_change': -0.90,
                    'market_volatility_mult': 5.0
                },
                {
                    'name': 'COVID-19 Crash (Mar 2020)',
                    'btc_change': -0.50,
                    'eth_change': -0.60,
                    'market_volatility_mult': 3.0
                },
                {
                    'name': 'Flash Crash',
                    'btc_change': -0.30,
                    'eth_change': -0.35,
                    'market_volatility_mult': 2.0
                },
                {
                    'name': 'Extended Bear Market',
                    'btc_change': -0.70,
                    'eth_change': -0.75,
                    'market_volatility_mult': 2.5
                },
                {
                    'name': 'Black Swan Event',
                    'btc_change': -0.95,
                    'eth_change': -0.98,
                    'market_volatility_mult': 10.0
                }
            ]
        
        results = []
        
        for scenario in scenarios:
            # Calculate portfolio impact
            new_value = portfolio_value
            
            for asset, position in positions.items():
                if asset.lower() in ['btc', 'bitcoin']:
                    change = scenario.get('btc_change', 0)
                elif asset.lower() in ['eth', 'ethereum']:
                    change = scenario.get('eth_change', 0)
                else:
                    # Default to BTC change
                    change = scenario.get('btc_change', 0)
                
                new_value += position * change
            
            loss = new_value - portfolio_value
            loss_pct = loss / portfolio_value
            
            results.append({
                'scenario': scenario['name'],
                'portfolio_value_before': portfolio_value,
                'portfolio_value_after': new_value,
                'loss': loss,
                'loss_pct': loss_pct,
                'volatility_multiplier': scenario.get('market_volatility_mult', 1.0)
            })
            
            logger.info(
                f"Stress test: {scenario['name']}",
                loss=loss,
                loss_pct=f"{loss_pct:.2%}"
            )
        
        df = pd.DataFrame(results)
        
        # Store results
        self.stress_test_results.append({
            'timestamp': datetime.now(),
            'results': df
        })
        
        # Alert on severe losses
        max_loss_pct = df['loss_pct'].min()  # Most negative
        if max_loss_pct < -0.50:
            alerts.send_alert(
                "high_stress_test_loss",
                f"Stress test shows potential {max_loss_pct:.1%} loss",
                {'worst_scenario': df.loc[df['loss_pct'].idxmin(), 'scenario']}
            )
        
        return df
    
    def calculate_maximum_drawdown(self, equity_curve: np.ndarray) -> Dict[str, Any]:
        """
        Calculate maximum drawdown
        
        Args:
            equity_curve: Array of portfolio values over time
        
        Returns:
            Dict with max_drawdown, start_idx, end_idx, duration
        """
        # Running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Drawdown at each point
        drawdown = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Find start of drawdown (last peak before max DD)
        start_idx = np.argmax(equity_curve[:max_dd_idx])
        
        # Find recovery point (if any)
        recovery_idx = None
        if max_dd_idx < len(equity_curve) - 1:
            # Look for recovery to previous peak
            peak_value = equity_curve[start_idx]
            future_values = equity_curve[max_dd_idx:]
            recovery_points = np.where(future_values >= peak_value)[0]
            
            if len(recovery_points) > 0:
                recovery_idx = max_dd_idx + recovery_points[0]
        
        duration_bars = max_dd_idx - start_idx
        recovery_duration = (recovery_idx - max_dd_idx) if recovery_idx else None
        
        result = {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd,
            'start_idx': start_idx,
            'bottom_idx': max_dd_idx,
            'recovery_idx': recovery_idx,
            'drawdown_duration': duration_bars,
            'recovery_duration': recovery_duration,
            'current_drawdown': drawdown[-1]
        }
        
        logger.info(
            "Maximum drawdown calculated",
            max_dd=f"{max_dd:.2%}",
            duration=duration_bars
        )
        
        metrics.gauge("risk_max_drawdown", max_dd)
        
        return result
    
    def calculate_risk_metrics(self,
                              returns: np.ndarray,
                              benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns
        
        Returns:
            Dict with all risk metrics
        """
        # Basic stats
        mean_return = np.mean(returns) * 252  # Annualized
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        sharpe = mean_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252)
        sortino = mean_return / downside_std if downside_std > 0 else 0
        
        # VaR and CVaR
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        metrics_dict = {
            'annual_return': mean_return,
            'annual_volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        # Information ratio (if benchmark provided)
        if benchmark_returns is not None:
            excess_returns = returns - benchmark_returns
            tracking_error = np.std(excess_returns) * np.sqrt(252)
            information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0
            
            metrics_dict['information_ratio'] = information_ratio
            metrics_dict['tracking_error'] = tracking_error
        
        return metrics_dict


if __name__ == "__main__":
    print("=" * 70)
    print("📉 ADVANCED RISK METRICS TEST")
    print("=" * 70)
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.0008, 0.02, 1000)  # Daily returns
    
    risk_mgr = InstitutionalRiskManager()
    
    # Test 1: VaR
    print("\n1. Value at Risk (VaR):")
    var_95 = risk_mgr.calculate_var(returns, 0.95, 'historical')
    var_99 = risk_mgr.calculate_var(returns, 0.99, 'historical')
    print(f"   95% VaR: {var_95:.2%}")
    print(f"   99% VaR: {var_99:.2%}")
    
    # Test 2: CVaR
    print("\n2. Conditional VaR (Expected Shortfall):")
    cvar_95 = risk_mgr.calculate_cvar(returns, 0.95)
    cvar_99 = risk_mgr.calculate_cvar(returns, 0.99)
    print(f"   95% CVaR: {cvar_95:.2%}")
    print(f"   99% CVaR: {cvar_99:.2%}")
    
    # Test 3: Stress Testing
    print("\n3. Stress Testing:")
    portfolio = {
        'BTC': 50000,
        'ETH': 30000
    }
    stress_results = risk_mgr.stress_test(80000, portfolio)
    print(stress_results[['scenario', 'loss_pct']].to_string(index=False))
    
    # Test 4: Maximum Drawdown
    print("\n4. Maximum Drawdown:")
    equity = np.random.randn(1000).cumsum() + 100
    dd_stats = risk_mgr.calculate_maximum_drawdown(equity)
    print(f"   Max Drawdown: {dd_stats['max_drawdown_pct']:.2%}")
    print(f"   Duration: {dd_stats['drawdown_duration']} bars")
    
    # Test 5: Comprehensive Metrics
    print("\n5. Comprehensive Risk Metrics:")
    metrics = risk_mgr.calculate_risk_metrics(returns)
    for name, value in metrics.items():
        if 'ratio' in name.lower() or 'return' in name.lower():
            print(f"   {name}: {value:.2f}")
        else:
            print(f"   {name}: {value:.2%}")
    
    print("\n✅ Risk metrics test complete")
