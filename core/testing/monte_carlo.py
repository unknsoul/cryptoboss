"""
Monte Carlo Simulation
Test strategy robustness by randomizing trade sequences
"""

import numpy as np
from typing import List, Dict


class MonteCarloSimulation:
    """
    Perform Monte Carlo simulation on trading results
    
    Randomizes trade sequence to understand confidence intervals
    and probability of different outcomes
    """
    
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations
    
    def run_simulation(self, trades: List[Dict], initial_capital: float):
        """
        Run Monte Carlo simulation by randomizing trade order
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            initial_capital: Starting capital
        
        Returns:
            simulation_results: Dict with confidence intervals and statistics
        """
        if not trades:
            return {
                'final_equity_mean': initial_capital,
                'final_equity_std': 0,
                'confidence_intervals': {},
                'probability_of_profit': 0,
                'worst_case': initial_capital,
                'best_case': initial_capital
            }
        
        final_equities = []
        max_drawdowns = []
        
        trade_pnls = [t['pnl'] for t in trades]
        
        for _ in range(self.num_simulations):
            # Randomize trade order
            shuffled_pnls = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
            
            # Calculate equity curve
            equity = initial_capital
            peak = initial_capital
            max_dd = 0
            
            for pnl in shuffled_pnls:
                equity += pnl
                
                if equity > peak:
                    peak = equity
                
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            
            final_equities.append(equity)
            max_drawdowns.append(max_dd)
        
        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate statistics
        mean_equity = np.mean(final_equities)
        std_equity = np.std(final_equities)
        
        # Confidence intervals
        confidence_intervals = {
            '50%': (np.percentile(final_equities, 25), np.percentile(final_equities, 75)),
            '90%': (np.percentile(final_equities, 5), np.percentile(final_equities, 95)),
            '95%': (np.percentile(final_equities, 2.5), np.percentile(final_equities, 97.5))
        }
        
        # Probability of profit
        prob_profit = np.sum(final_equities > initial_capital) / len(final_equities)
        
        # Worst and best case
        worst_case = np.min(final_equities)
        best_case = np.max(final_equities)
        
        # Drawdown statistics
        mean_dd = np.mean(max_drawdowns)
        worst_dd = np.max(max_drawdowns)
        
        return {
            'final_equity_mean': mean_equity,
            'final_equity_std': std_equity,
            'confidence_intervals': confidence_intervals,
            'probability_of_profit': prob_profit,
            'worst_case': worst_case,
            'best_case': best_case,
            'mean_max_drawdown': mean_dd,
            'worst_max_drawdown': worst_dd,
            'all_final_equities': final_equities  # For plotting
        }
    
    def print_results(self, results: Dict, initial_capital: float):
        """Print Monte Carlo simulation results"""
        print("\n" + "=" * 80)
        print(f"MONTE CARLO SIMULATION RESULTS ({self.num_simulations} runs)")
        print("=" * 80)
        
        print(f"\nInitial Capital:         ${initial_capital:>12,.2f}")
        print(f"Mean Final Equity:       ${results['final_equity_mean']:>12,.2f}")
        print(f"Std Deviation:           ${results['final_equity_std']:>12,.2f}")
        
        print(f"\nProbability of Profit:   {results['probability_of_profit']:>12.1%}")
        
        print(f"\nWorst Case:              ${results['worst_case']:>12,.2f}")
        print(f"Best Case:               ${results['best_case']:>12,.2f}")
        
        print("\nConfidence Intervals (Final Equity):")
        for level, (lower, upper) in results['confidence_intervals'].items():
            print(f"  {level:>3s}: ${lower:>12,.2f} to ${upper:>12,.2f}")
        
        print(f"\nMean Max Drawdown:       {results['mean_max_drawdown']:>12.2%}")
        print(f"Worst Max Drawdown:      {results['worst_max_drawdown']:>12.2%}")
        
        # Risk assessment
        print("\n" + "=" * 80)
        print("RISK ASSESSMENT")
        print("=" * 80)
        
        if results['probability_of_profit'] >= 0.70:
            print("✅ High probability of profit (>= 70%)")
        elif results['probability_of_profit'] >= 0.55:
            print("⚠️  Moderate probability of profit (55-70%)")
        else:
            print("❌ Low probability of profit (< 55%)")
        
        worst_loss_pct = (results['worst_case'] - initial_capital) / initial_capital
        if worst_loss_pct < -0.30:
            print(f"❌ Worst case scenario shows severe loss: {worst_loss_pct:.1%}")
        elif worst_loss_pct < -0.15:
            print(f"⚠️  Worst case scenario shows moderate loss: {worst_loss_pct:.1%}")
        else:
            print(f"✅ Worst case scenario is acceptable: {worst_loss_pct:.1%}")
