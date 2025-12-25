"""
Simulation & Testing - Enterprise Features #285, #288, #292, #296
Monte Carlo Path, Stress Testing, Scenarios, Historical Replay.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import random
import math

logger = logging.getLogger(__name__)


class MonteCarloPathSimulator:
    """
    Feature #285: Monte Carlo Path Simulator
    
    Simulates price paths using Monte Carlo methods.
    """
    
    def __init__(self, num_simulations: int = 1000):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            num_simulations: Number of paths to simulate
        """
        self.num_simulations = num_simulations
        self.results: List[Dict] = []
        
        logger.info(f"Monte Carlo Path Simulator initialized - {num_simulations} sims")
    
    def simulate_gbm(
        self,
        current_price: float,
        days: int,
        daily_return: float = 0.0,
        daily_volatility: float = 0.02
    ) -> Dict:
        """
        Simulate price paths using Geometric Brownian Motion.
        
        Args:
            current_price: Starting price
            days: Number of days to simulate
            daily_return: Expected daily return
            daily_volatility: Daily volatility
            
        Returns:
            Simulation results
        """
        paths = []
        final_prices = []
        
        for _ in range(self.num_simulations):
            path = [current_price]
            price = current_price
            
            for _ in range(days):
                drift = daily_return
                shock = daily_volatility * random.gauss(0, 1)
                price *= math.exp(drift + shock)
                path.append(price)
            
            paths.append(path)
            final_prices.append(price)
        
        # Calculate statistics
        final_prices.sort()
        n = len(final_prices)
        
        result = {
            'current_price': current_price,
            'days_simulated': days,
            'num_simulations': self.num_simulations,
            'mean_final': round(sum(final_prices) / n, 2),
            'median_final': round(final_prices[n // 2], 2),
            'percentile_5': round(final_prices[int(n * 0.05)], 2),
            'percentile_25': round(final_prices[int(n * 0.25)], 2),
            'percentile_75': round(final_prices[int(n * 0.75)], 2),
            'percentile_95': round(final_prices[int(n * 0.95)], 2),
            'max_price': round(max(final_prices), 2),
            'min_price': round(min(final_prices), 2),
            'prob_up': round(len([p for p in final_prices if p > current_price]) / n, 3)
        }
        
        self.results.append(result)
        return result
    
    def simulate_portfolio(
        self,
        positions: List[Dict],
        days: int,
        correlations: Optional[Dict] = None
    ) -> Dict:
        """Simulate portfolio value evolution."""
        portfolio_values = []
        
        for _ in range(self.num_simulations):
            final_value = 0
            
            for pos in positions:
                # Simulate each position
                current = pos['value']
                vol = pos.get('volatility', 0.02)
                ret = pos.get('expected_return', 0)
                
                for _ in range(days):
                    shock = vol * random.gauss(0, 1)
                    current *= math.exp(ret / 365 + shock)
                
                final_value += current
            
            portfolio_values.append(final_value)
        
        portfolio_values.sort()
        n = len(portfolio_values)
        initial = sum(p['value'] for p in positions)
        
        return {
            'initial_value': initial,
            'mean_final': round(sum(portfolio_values) / n, 2),
            'var_95': round(initial - portfolio_values[int(n * 0.05)], 2),
            'var_99': round(initial - portfolio_values[int(n * 0.01)], 2),
            'best_case': round(portfolio_values[-1], 2),
            'worst_case': round(portfolio_values[0], 2)
        }


class StressTestFramework:
    """
    Feature #288: Stress Test Framework
    
    Tests portfolio under extreme market conditions.
    """
    
    def __init__(self):
        """Initialize stress test framework."""
        self.scenarios: Dict[str, Dict] = {}
        self.results: List[Dict] = []
        
        self._register_default_scenarios()
        logger.info("Stress Test Framework initialized")
    
    def _register_default_scenarios(self):
        """Register default stress scenarios."""
        self.add_scenario('market_crash', {
            'price_change': -0.30,
            'volatility_spike': 3.0,
            'liquidity_drop': 0.5,
            'description': '30% market crash'
        })
        
        self.add_scenario('flash_crash', {
            'price_change': -0.15,
            'volatility_spike': 5.0,
            'recovery_minutes': 30,
            'description': '15% flash crash'
        })
        
        self.add_scenario('black_swan', {
            'price_change': -0.50,
            'volatility_spike': 10.0,
            'correlation_spike': 1.0,
            'description': '50% black swan event'
        })
    
    def add_scenario(self, name: str, params: Dict):
        """Add a stress scenario."""
        self.scenarios[name] = params
    
    def run_scenario(
        self,
        scenario_name: str,
        portfolio: Dict
    ) -> Dict:
        """
        Run a stress test scenario.
        
        Args:
            scenario_name: Name of scenario
            portfolio: Portfolio to stress test
            
        Returns:
            Stress test results
        """
        if scenario_name not in self.scenarios:
            return {'error': 'Scenario not found'}
        
        scenario = self.scenarios[scenario_name]
        initial_value = portfolio['total_value']
        
        # Apply stress
        price_impact = scenario.get('price_change', 0)
        stressed_value = initial_value * (1 + price_impact)
        
        loss = initial_value - stressed_value
        loss_pct = abs(price_impact) * 100
        
        # Check margin calls
        margin_ratio = portfolio.get('margin_ratio', 1.0)
        new_margin_ratio = margin_ratio * (1 + price_impact)
        margin_call = new_margin_ratio < 0.3
        
        result = {
            'scenario': scenario_name,
            'description': scenario.get('description', ''),
            'initial_value': initial_value,
            'stressed_value': round(stressed_value, 2),
            'loss': round(loss, 2),
            'loss_pct': round(loss_pct, 2),
            'margin_call': margin_call,
            'survivable': stressed_value > 0 and not margin_call
        }
        
        self.results.append(result)
        return result
    
    def run_all_scenarios(self, portfolio: Dict) -> List[Dict]:
        """Run all stress scenarios."""
        results = []
        for name in self.scenarios:
            results.append(self.run_scenario(name, portfolio))
        return results


class ScenarioGenerator:
    """
    Feature #292: Scenario Generator
    
    Generates custom market scenarios for testing.
    """
    
    def __init__(self):
        """Initialize scenario generator."""
        self.generated: List[Dict] = []
        
        logger.info("Scenario Generator initialized")
    
    def generate_trend_scenario(
        self,
        direction: str,
        magnitude_pct: float,
        duration_days: int
    ) -> Dict:
        """Generate a trending market scenario."""
        daily_move = magnitude_pct / duration_days
        
        scenario = {
            'type': 'trend',
            'direction': direction,
            'total_move_pct': magnitude_pct,
            'duration_days': duration_days,
            'daily_move_pct': round(daily_move, 3),
            'price_path': []
        }
        
        price = 100  # Base price
        for day in range(duration_days):
            noise = random.gauss(0, abs(daily_move) * 0.5)
            if direction == 'up':
                price *= (1 + daily_move / 100 + noise / 100)
            else:
                price *= (1 - daily_move / 100 + noise / 100)
            scenario['price_path'].append(round(price, 2))
        
        self.generated.append(scenario)
        return scenario
    
    def generate_volatility_scenario(
        self,
        base_volatility: float,
        spike_multiplier: float,
        spike_duration_days: int
    ) -> Dict:
        """Generate a volatility spike scenario."""
        scenario = {
            'type': 'volatility_spike',
            'base_volatility': base_volatility,
            'spike_volatility': base_volatility * spike_multiplier,
            'duration_days': spike_duration_days,
            'price_path': []
        }
        
        price = 100
        for day in range(spike_duration_days):
            move = random.gauss(0, base_volatility * spike_multiplier)
            price *= (1 + move / 100)
            scenario['price_path'].append(round(price, 2))
        
        self.generated.append(scenario)
        return scenario
    
    def generate_range_bound(
        self,
        low: float,
        high: float,
        duration_days: int
    ) -> Dict:
        """Generate a range-bound market scenario."""
        scenario = {
            'type': 'range_bound',
            'low': low,
            'high': high,
            'duration_days': duration_days,
            'price_path': []
        }
        
        mid = (low + high) / 2
        range_size = high - low
        price = mid
        
        for day in range(duration_days):
            # Mean revert within range
            reversion = (mid - price) / range_size * 0.1
            noise = random.gauss(0, range_size * 0.05)
            price = max(low, min(high, price + reversion + noise))
            scenario['price_path'].append(round(price, 2))
        
        self.generated.append(scenario)
        return scenario


class HistoricalReplayEngine:
    """
    Feature #296: Historical Replay Engine
    
    Replays historical market data for backtesting.
    """
    
    def __init__(self):
        """Initialize replay engine."""
        self.data: Dict[str, List[Dict]] = {}
        self.current_index: Dict[str, int] = {}
        self.replay_speed: float = 1.0
        
        logger.info("Historical Replay Engine initialized")
    
    def load_data(self, symbol: str, candles: List[Dict]):
        """Load historical data for replay."""
        self.data[symbol] = sorted(candles, key=lambda x: x.get('timestamp', ''))
        self.current_index[symbol] = 0
        logger.info(f"Loaded {len(candles)} candles for {symbol}")
    
    def set_speed(self, multiplier: float):
        """Set replay speed multiplier."""
        self.replay_speed = multiplier
    
    def get_next(self, symbol: str) -> Optional[Dict]:
        """Get next candle in replay."""
        if symbol not in self.data:
            return None
        
        idx = self.current_index[symbol]
        if idx >= len(self.data[symbol]):
            return None
        
        candle = self.data[symbol][idx]
        self.current_index[symbol] += 1
        
        return candle
    
    def peek(self, symbol: str, n: int = 1) -> List[Dict]:
        """Peek at upcoming candles without advancing."""
        if symbol not in self.data:
            return []
        
        idx = self.current_index[symbol]
        return self.data[symbol][idx:idx + n]
    
    def reset(self, symbol: Optional[str] = None):
        """Reset replay to beginning."""
        if symbol:
            self.current_index[symbol] = 0
        else:
            for s in self.current_index:
                self.current_index[s] = 0
    
    def seek(self, symbol: str, index: int):
        """Seek to specific position."""
        if symbol in self.data:
            self.current_index[symbol] = max(0, min(index, len(self.data[symbol]) - 1))
    
    def get_progress(self, symbol: str) -> Dict:
        """Get replay progress."""
        if symbol not in self.data:
            return {}
        
        total = len(self.data[symbol])
        current = self.current_index[symbol]
        
        return {
            'symbol': symbol,
            'current_index': current,
            'total_candles': total,
            'progress_pct': round(current / total * 100, 1) if total > 0 else 0,
            'remaining': total - current
        }


# Singletons
_monte_carlo: Optional[MonteCarloPathSimulator] = None
_stress_test: Optional[StressTestFramework] = None
_scenario_gen: Optional[ScenarioGenerator] = None
_replay: Optional[HistoricalReplayEngine] = None


def get_monte_carlo() -> MonteCarloPathSimulator:
    global _monte_carlo
    if _monte_carlo is None:
        _monte_carlo = MonteCarloPathSimulator()
    return _monte_carlo


def get_stress_tester() -> StressTestFramework:
    global _stress_test
    if _stress_test is None:
        _stress_test = StressTestFramework()
    return _stress_test


def get_scenario_generator() -> ScenarioGenerator:
    global _scenario_gen
    if _scenario_gen is None:
        _scenario_gen = ScenarioGenerator()
    return _scenario_gen


def get_replay_engine() -> HistoricalReplayEngine:
    global _replay
    if _replay is None:
        _replay = HistoricalReplayEngine()
    return _replay


if __name__ == '__main__':
    # Test Monte Carlo
    mc = MonteCarloPathSimulator(num_simulations=1000)
    result = mc.simulate_gbm(50000, 30, daily_return=0.001, daily_volatility=0.02)
    print(f"Monte Carlo: Mean={result['mean_final']}, P5={result['percentile_5']}, P95={result['percentile_95']}")
    
    # Test stress test
    stress = StressTestFramework()
    portfolio = {'total_value': 100000, 'margin_ratio': 0.5}
    result = stress.run_scenario('market_crash', portfolio)
    print(f"Stress test: {result}")
    
    # Test scenario generator
    gen = ScenarioGenerator()
    trend = gen.generate_trend_scenario('up', 20, 30)
    print(f"Trend scenario: {len(trend['price_path'])} days")
