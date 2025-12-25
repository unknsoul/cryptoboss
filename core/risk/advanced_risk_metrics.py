"""
Advanced Risk Management - Enterprise Features #122, #126, #130, #135, #150
Kelly Criterion, VaR, Monte Carlo, and Position Management.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
import random
from datetime import datetime

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Feature #122: Kelly Criterion Module
    
    Calculates optimal position size using Kelly formula:
    f* = (bp - q) / b
    
    Where:
    - b = odds (win/loss ratio)
    - p = probability of winning
    - q = probability of losing (1-p)
    """
    
    def __init__(
        self,
        max_kelly_fraction: float = 0.5,     # Max 50% of full Kelly
        min_win_rate: float = 0.4,           # Min win rate to use Kelly
        lookback_trades: int = 30            # Trades for calculation
    ):
        """
        Initialize Kelly Criterion calculator.
        
        Args:
            max_kelly_fraction: Maximum fraction of Kelly to use (safety)
            min_win_rate: Minimum win rate required
            lookback_trades: Number of trades to analyze
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.min_win_rate = min_win_rate
        self.lookback_trades = lookback_trades
        
        logger.info(f"Kelly Criterion initialized - Max fraction: {max_kelly_fraction:.0%}")
    
    def calculate(
        self,
        trades: List[Dict],
        confidence: float = 1.0
    ) -> Dict:
        """
        Calculate Kelly-optimal position size.
        
        Args:
            trades: Historical trade data
            confidence: Signal confidence (0-1) to adjust Kelly
            
        Returns:
            Kelly sizing information
        """
        recent = trades[-self.lookback_trades:] if trades else []
        
        if len(recent) < 10:
            return {
                'kelly_fraction': 0.02,  # Default conservative
                'adjusted_fraction': 0.02,
                'sufficient_data': False,
                'reason': 'Insufficient trade history'
            }
        
        # Calculate win rate
        wins = [t for t in recent if t.get('pnl', 0) > 0]
        losses = [t for t in recent if t.get('pnl', 0) <= 0]
        
        win_rate = len(wins) / len(recent)
        
        if win_rate < self.min_win_rate:
            return {
                'kelly_fraction': 0.01,
                'adjusted_fraction': 0.01,
                'win_rate': win_rate,
                'sufficient_data': True,
                'reason': f'Win rate {win_rate:.1%} below minimum {self.min_win_rate:.1%}'
            }
        
        # Calculate average win/loss
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t['pnl'] for t in losses) / len(losses)) if losses else 1
        
        # Win/loss ratio (odds)
        odds = avg_win / avg_loss if avg_loss > 0 else 1
        
        # Kelly formula: f* = (bp - q) / b
        # Where b=odds, p=win_rate, q=1-p
        q = 1 - win_rate
        kelly = (odds * win_rate - q) / odds if odds > 0 else 0
        
        # Apply safety fraction
        safe_kelly = max(min(kelly * self.max_kelly_fraction, 0.25), 0.005)
        
        # Further adjust by confidence
        adjusted = safe_kelly * confidence
        
        return {
            'kelly_fraction': round(kelly, 4),
            'adjusted_fraction': round(adjusted, 4),
            'win_rate': round(win_rate, 3),
            'odds': round(odds, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'sufficient_data': True,
            'reason': 'OK'
        }


class ValueAtRisk:
    """
    Feature #126: Value-at-Risk Estimator
    
    Estimates maximum expected loss at specified confidence level.
    Uses historical simulation and parametric VaR.
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,      # 95% VaR
        lookback_days: int = 30
    ):
        """
        Initialize VaR estimator.
        
        Args:
            confidence_level: Confidence level (0.95 = 95%)
            lookback_days: Historical period for calculation
        """
        self.confidence_level = confidence_level
        self.lookback_days = lookback_days
        
        logger.info(f"VaR Estimator initialized - {confidence_level:.0%} confidence")
    
    def calculate_historical_var(
        self,
        returns: List[float],
        portfolio_value: float
    ) -> Dict:
        """
        Calculate historical VaR from return distribution.
        
        Args:
            returns: List of daily returns (as decimals)
            portfolio_value: Current portfolio value
            
        Returns:
            VaR estimates
        """
        if len(returns) < 10:
            return {'var': 0, 'var_pct': 0, 'sufficient_data': False}
        
        # Sort returns (worst to best)
        sorted_returns = sorted(returns)
        
        # Find percentile for VaR
        index = int((1 - self.confidence_level) * len(sorted_returns))
        var_return = sorted_returns[max(index, 0)]
        
        # VaR in dollar terms
        var_dollars = abs(var_return * portfolio_value)
        
        # Also calculate conditional VaR (expected shortfall)
        tail_returns = sorted_returns[:max(index, 1)]
        cvar_return = sum(tail_returns) / len(tail_returns) if tail_returns else var_return
        cvar_dollars = abs(cvar_return * portfolio_value)
        
        return {
            'var': round(var_dollars, 2),
            'var_pct': round(abs(var_return) * 100, 2),
            'cvar': round(cvar_dollars, 2),
            'cvar_pct': round(abs(cvar_return) * 100, 2),
            'confidence': self.confidence_level,
            'sample_size': len(returns),
            'sufficient_data': True
        }
    
    def calculate_parametric_var(
        self,
        returns: List[float],
        portfolio_value: float
    ) -> Dict:
        """
        Calculate parametric VaR assuming normal distribution.
        
        Args:
            returns: List of daily returns
            portfolio_value: Current portfolio value
            
        Returns:
            Parametric VaR estimate
        """
        if len(returns) < 10:
            return {'var': 0, 'sufficient_data': False}
        
        # Calculate mean and std
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = variance ** 0.5
        
        # Z-score for confidence level
        z_scores = {0.90: 1.28, 0.95: 1.645, 0.99: 2.33}
        z = z_scores.get(self.confidence_level, 1.645)
        
        # VaR = portfolio * z * sigma (assuming mean ~0 for short periods)
        var_return = z * std
        var_dollars = var_return * portfolio_value
        
        return {
            'parametric_var': round(var_dollars, 2),
            'parametric_var_pct': round(var_return * 100, 2),
            'mean_return': round(mean * 100, 4),
            'volatility': round(std * 100, 2),
            'z_score': z,
            'sufficient_data': True
        }


class MonteCarloSimulator:
    """
    Feature #130: Monte Carlo Simulation
    
    Simulates portfolio outcomes under various scenarios.
    """
    
    def __init__(
        self,
        num_simulations: int = 1000,
        simulation_days: int = 30
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            num_simulations: Number of simulation paths
            simulation_days: Days to simulate forward
        """
        self.num_simulations = num_simulations
        self.simulation_days = simulation_days
        
        logger.info(f"Monte Carlo initialized - {num_simulations} sims, {simulation_days} days")
    
    def simulate_portfolio(
        self,
        initial_value: float,
        daily_returns: List[float],
        trades_per_day: float = 3
    ) -> Dict:
        """
        Run Monte Carlo simulation of portfolio performance.
        
        Args:
            initial_value: Starting portfolio value
            daily_returns: Historical daily returns for sampling
            trades_per_day: Expected trades per day
            
        Returns:
            Simulation results with percentiles
        """
        if len(daily_returns) < 10:
            return {'sufficient_data': False}
        
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
        std_return = variance ** 0.5
        
        final_values = []
        paths = []
        
        for _ in range(self.num_simulations):
            value = initial_value
            path = [value]
            
            for _ in range(self.simulation_days):
                # Sample random return from distribution
                daily_return = random.gauss(mean_return, std_return)
                value *= (1 + daily_return)
                path.append(value)
            
            final_values.append(value)
            if len(paths) < 10:  # Store some sample paths
                paths.append(path)
        
        # Calculate statistics
        final_values.sort()
        
        return {
            'initial_value': initial_value,
            'mean_final': round(sum(final_values) / len(final_values), 2),
            'median_final': round(final_values[len(final_values) // 2], 2),
            'percentile_5': round(final_values[int(0.05 * len(final_values))], 2),
            'percentile_25': round(final_values[int(0.25 * len(final_values))], 2),
            'percentile_75': round(final_values[int(0.75 * len(final_values))], 2),
            'percentile_95': round(final_values[int(0.95 * len(final_values))], 2),
            'worst_case': round(min(final_values), 2),
            'best_case': round(max(final_values), 2),
            'probability_profit': sum(1 for v in final_values if v > initial_value) / len(final_values),
            'simulation_days': self.simulation_days,
            'num_simulations': self.num_simulations,
            'sufficient_data': True
        }


class DailyLossLimit:
    """
    Feature #135: Daily Loss Limit
    
    Enforces maximum daily loss limits across the portfolio.
    """
    
    def __init__(
        self,
        max_daily_loss_pct: float = 3.0,     # 3% daily loss limit
        warning_threshold: float = 0.7        # Warn at 70% of limit
    ):
        """
        Initialize daily loss limiter.
        
        Args:
            max_daily_loss_pct: Maximum loss as % of equity
            warning_threshold: Warning level as fraction of limit
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.warning_threshold = warning_threshold
        
        self.daily_start_equity: float = 0
        self.current_date: Optional[str] = None
        
        logger.info(f"Daily Loss Limit initialized - {max_daily_loss_pct}% max")
    
    def update(self, current_equity: float) -> Dict:
        """
        Update daily tracking and check limits.
        
        Args:
            current_equity: Current portfolio equity
            
        Returns:
            Status with trading permission
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Reset for new day
        if self.current_date != today:
            self.current_date = today
            self.daily_start_equity = current_equity
            return {
                'can_trade': True,
                'daily_pnl': 0,
                'daily_pnl_pct': 0,
                'remaining_risk': self.max_daily_loss_pct,
                'status': 'NEW_DAY'
            }
        
        # Calculate daily P&L
        daily_pnl = current_equity - self.daily_start_equity
        daily_pnl_pct = (daily_pnl / self.daily_start_equity) * 100 if self.daily_start_equity > 0 else 0
        
        # Check limits
        loss_pct = abs(min(daily_pnl_pct, 0))
        remaining = self.max_daily_loss_pct - loss_pct
        
        if loss_pct >= self.max_daily_loss_pct:
            return {
                'can_trade': False,
                'daily_pnl': round(daily_pnl, 2),
                'daily_pnl_pct': round(daily_pnl_pct, 2),
                'remaining_risk': 0,
                'status': 'LIMIT_REACHED'
            }
        
        warning = loss_pct >= self.max_daily_loss_pct * self.warning_threshold
        
        return {
            'can_trade': True,
            'daily_pnl': round(daily_pnl, 2),
            'daily_pnl_pct': round(daily_pnl_pct, 2),
            'remaining_risk': round(remaining, 2),
            'status': 'WARNING' if warning else 'OK'
        }


class PyramidingManager:
    """
    Feature #150: Pyramiding Support
    
    Manages adding to winning positions (pyramiding).
    """
    
    def __init__(
        self,
        max_pyramid_levels: int = 3,
        min_profit_for_add: float = 0.5,     # 0.5R profit before adding
        size_reduction: float = 0.5           # Each level is 50% of previous
    ):
        """
        Initialize pyramiding manager.
        
        Args:
            max_pyramid_levels: Maximum position additions
            min_profit_for_add: Minimum profit (in R) before adding
            size_reduction: Size multiplier for each level
        """
        self.max_pyramid_levels = max_pyramid_levels
        self.min_profit_for_add = min_profit_for_add
        self.size_reduction = size_reduction
        
        logger.info(f"Pyramiding Manager initialized - {max_pyramid_levels} levels max")
    
    def should_pyramid(
        self,
        position: Dict,
        current_price: float
    ) -> Dict:
        """
        Check if we should add to position.
        
        Args:
            position: Current position data
            current_price: Current market price
            
        Returns:
            Pyramiding recommendation
        """
        if not position:
            return {'should_add': False, 'reason': 'No position'}
        
        pyramid_level = position.get('pyramid_level', 0)
        
        if pyramid_level >= self.max_pyramid_levels:
            return {'should_add': False, 'reason': 'Max pyramid reached'}
        
        entry = position.get('entry_price', 0)
        initial_risk = position.get('initial_risk', entry * 0.01)
        is_long = position.get('side') == 'LONG'
        
        # Calculate profit in R
        if is_long:
            profit = current_price - entry
        else:
            profit = entry - current_price
        
        profit_r = profit / initial_risk if initial_risk > 0 else 0
        
        if profit_r < self.min_profit_for_add:
            return {
                'should_add': False,
                'reason': f'Profit {profit_r:.2f}R below threshold {self.min_profit_for_add}R'
            }
        
        # Calculate new size
        original_size = position.get('original_size', position.get('size', 0))
        add_size = original_size * (self.size_reduction ** pyramid_level)
        
        return {
            'should_add': True,
            'add_size': round(add_size, 6),
            'new_level': pyramid_level + 1,
            'profit_r': round(profit_r, 2),
            'reason': 'Profit target reached'
        }


# Singleton instances
_kelly: Optional[KellyCriterion] = None
_var: Optional[ValueAtRisk] = None
_monte_carlo: Optional[MonteCarloSimulator] = None
_daily_limit: Optional[DailyLossLimit] = None
_pyramiding: Optional[PyramidingManager] = None


def get_kelly() -> KellyCriterion:
    global _kelly
    if _kelly is None:
        _kelly = KellyCriterion()
    return _kelly


def get_var() -> ValueAtRisk:
    global _var
    if _var is None:
        _var = ValueAtRisk()
    return _var


def get_monte_carlo() -> MonteCarloSimulator:
    global _monte_carlo
    if _monte_carlo is None:
        _monte_carlo = MonteCarloSimulator()
    return _monte_carlo


def get_daily_limit() -> DailyLossLimit:
    global _daily_limit
    if _daily_limit is None:
        _daily_limit = DailyLossLimit()
    return _daily_limit


if __name__ == '__main__':
    # Test Kelly
    kelly = KellyCriterion()
    trades = [
        {'pnl': 50}, {'pnl': -30}, {'pnl': 40}, {'pnl': 60},
        {'pnl': -25}, {'pnl': 45}, {'pnl': 35}, {'pnl': -20},
        {'pnl': 55}, {'pnl': 30}, {'pnl': -15}, {'pnl': 40}
    ]
    result = kelly.calculate(trades)
    print(f"Kelly: {result}")
    
    # Test VaR
    var = ValueAtRisk()
    returns = [random.gauss(0.001, 0.02) for _ in range(30)]
    result = var.calculate_historical_var(returns, 10000)
    print(f"VaR: {result}")
    
    # Test Monte Carlo
    mc = MonteCarloSimulator(num_simulations=100)
    result = mc.simulate_portfolio(10000, returns)
    print(f"Monte Carlo: {result}")
