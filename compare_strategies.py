"""
Strategy Comparison Tool
Compare all available trading strategies on the same dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tabulate import tabulate

from core.strategies.factory import StrategyFactory
from core.backtest import ProfessionalBacktest


def compare_strategies(data_file="data/btc_1h.csv", capital=10000, risk=0.02, fee=0.001):
    """
    Compare all available strategies on the same dataset
    """
    print("=" * 80)
    print("STRATEGY COMPARISON - PROFESSIONAL TRADING BOT")
    print("=" * 80)
    print()
    
    # Load data
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df.get("volume", pd.Series([None] * len(df))).values
    
    print(f"Data loaded: {len(closes)} candles")
    print()
    
    # Get all available strategies
    strategies_info = StrategyFactory.list_strategies()
    strategy_names = list(strategies_info.keys())
    
    print(f"Testing {len(strategy_names)} strategies:")
    for name, info in strategies_info.items():
        print(f"  - {info['name']}: {info['description']}")
    print()
    
    # Results storage
    results = {}
    equity_curves = {}
    
    # Test each strategy
    for strategy_name in strategy_names:
        print(f"\n{'='*80}")
        print(f"Testing: {strategies_info[strategy_name]['name']}")
        print(f"{'='*80}")
        
        try:
            # Create strategy
            strategy = StrategyFactory.create(strategy_name)
            
            # Create backtest engine
            bt = ProfessionalBacktest(capital=capital, base_risk=risk, fee=fee)
            
            # Run backtest
            print("Running backtest...")
            equity = bt.run(highs, lows, closes, strategy, volumes=volumes)
            
            # Get metrics
            metrics = bt.get_metrics()
            
            # Store results
            results[strategy_name] = {
                'name': strategies_info[strategy_name]['name'],
                'type': strategies_info[strategy_name]['type'],
                'metrics': metrics,
                'trades': bt.trades
            }
            equity_curves[strategy_name] = equity
            
            # Print quick summary
            print(f"✅ Complete - {metrics['num_trades']} trades, "
                  f"{metrics['total_return']:.2%} return, "
                  f"Sharpe: {metrics['sharpe_ratio']:.2f}")
            
        except Exception as e:
            print(f"❌ Error testing {strategy_name}: {str(e)}")
            continue
    
    # Generate comparison report
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()
    
    # Build comparison table
    comparison_data = []
    for strategy_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append([
            result['name'],
            result['type'],
            f"{metrics['total_return']:.2%}",
            f"{metrics['annual_return']:.2%}",
            f"{metrics['sharpe_ratio']:.2f}",
            f"{metrics['sortino_ratio']:.2f}",
            f"{metrics['max_drawdown']:.2%}",
            metrics['num_trades'],
            f"{metrics['win_rate']:.2%}",
            f"{metrics['profit_factor']:.2f}"
        ])
    
    headers = [
        "Strategy", "Type", "Total Return", "Annual Return", 
        "Sharpe", "Sortino", "Max DD", "Trades", "Win Rate", "Profit Factor"
    ]
    
    print(tabulate(comparison_data, headers=headers, tablefmt="grid"))
    print()
    
    # Find best performers
    print("\n🏆 BEST PERFORMERS:")
    print("-" * 80)
    
    # Best by return
    best_return = max(results.items(), key=lambda x: x[1]['metrics']['total_return'])
    print(f"Highest Return: {best_return[1]['name']} "
          f"({best_return[1]['metrics']['total_return']:.2%})")
    
    # Best by Sharpe
    best_sharpe = max(results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
    print(f"Best Sharpe Ratio: {best_sharpe[1]['name']} "
          f"({best_sharpe[1]['metrics']['sharpe_ratio']:.2f})")
    
    # Best win rate
    best_winrate = max(results.items(), key=lambda x: x[1]['metrics']['win_rate'])
    print(f"Highest Win Rate: {best_winrate[1]['name']} "
          f"({best_winrate[1]['metrics']['win_rate']:.2%})")
    
    # Lowest drawdown
    best_dd = min(results.items(), key=lambda x: abs(x[1]['metrics']['max_drawdown']))
    print(f"Lowest Drawdown: {best_dd[1]['name']} "
          f"({best_dd[1]['metrics']['max_drawdown']:.2%})")
    
    print()
    
    # Generate comparison charts
    print("Generating comparison charts...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Equity curves comparison
    ax1 = fig.add_subplot(gs[0, :])
    buy_hold = (closes / closes[0]) * capital
    ax1.plot(buy_hold, label="Buy & Hold", linestyle="--", alpha=0.7, linewidth=2, color='black')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    for idx, (strategy_name, equity) in enumerate(equity_curves.items()):
        ax1.plot(equity, label=results[strategy_name]['name'], 
                linewidth=2, color=colors[idx % len(colors)])
    
    ax1.set_ylabel("Equity ($)", fontsize=12)
    ax1.set_title("Strategy Equity Curves Comparison", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Returns comparison bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    returns = [result['metrics']['total_return'] * 100 for result in results.values()]
    names = [result['name'] for result in results.values()]
    bars = ax2.barh(names, returns, color=colors[:len(names)])
    ax2.set_xlabel("Total Return (%)", fontsize=11)
    ax2.set_title("Total Returns Comparison", fontsize=12, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)
    
    # 3. Sharpe ratio comparison
    ax3 = fig.add_subplot(gs[1, 1])
    sharpes = [result['metrics']['sharpe_ratio'] for result in results.values()]
    ax3.barh(names, sharpes, color=colors[:len(names)])
    ax3.set_xlabel("Sharpe Ratio", fontsize=11)
    ax3.set_title("Sharpe Ratio Comparison", fontsize=12, fontweight='bold')
    ax3.grid(True, axis='x', alpha=0.3)
    ax3.axvline(x=1.0, color='red', linestyle='--', label='Threshold')
    
    # 4. Win rate vs Profit factor scatter
    ax4 = fig.add_subplot(gs[2, 0])
    win_rates = [result['metrics']['win_rate'] * 100 for result in results.values()]
    profit_factors = [result['metrics']['profit_factor'] for result in results.values()]
    
    for idx, name in enumerate(names):
        ax4.scatter(win_rates[idx], profit_factors[idx], 
                   s=200, color=colors[idx % len(colors)], alpha=0.7, label=name)
    
    ax4.set_xlabel("Win Rate (%)", fontsize=11)
    ax4.set_ylabel("Profit Factor", fontsize=11)
    ax4.set_title("Win Rate vs Profit Factor", fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    # 5. Max drawdown comparison
    ax5 = fig.add_subplot(gs[2, 1])
    drawdowns = [abs(result['metrics']['max_drawdown']) * 100 for result in results.values()]
    ax5.barh(names, drawdowns, color='red', alpha=0.6)
    ax5.set_xlabel("Max Drawdown (%)", fontsize=11)
    ax5.set_title("Maximum Drawdown Comparison", fontsize=12, fontweight='bold')
    ax5.grid(True, axis='x', alpha=0.3)
    ax5.invert_xaxis()  # Lower is better
    
    plt.savefig('strategy_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Charts saved to: strategy_comparison.png")
    
    # Export results to CSV
    print("Exporting results to CSV...")
    df_results = pd.DataFrame(comparison_data, columns=headers)
    df_results.to_csv('strategy_comparison.csv', index=False)
    print(f"Results exported to: strategy_comparison.csv")
    
    plt.show()
    
    print("\n✅ Strategy comparison complete!")
    
    return results, equity_curves


def main():
    parser = argparse.ArgumentParser(description="Compare trading strategies")
    parser.add_argument('--data', default='data/btc_1h.csv', help='Data file path')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.02, help='Risk per trade')
    parser.add_argument('--fee', type=float, default=0.001, help='Trading fee')
    
    args = parser.parse_args()
    
    compare_strategies(
        data_file=args.data,
        capital=args.capital,
        risk=args.risk,
        fee=args.fee
    )


if __name__ == "__main__":
    main()
