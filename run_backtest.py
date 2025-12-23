"""
Professional Trading Bot - Enhanced Main Execution
Multiple strategies, advanced analytics, walk-forward validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add core to path
sys.path.insert(0, os.path.dirname(__file__))

from core.strategy import SimpleTrendStrategy
from core.backtest import EnhancedBacktest
from core.testing.walk_forward import WalkForwardAnalysis
from core.testing.monte_carlo import MonteCarloSimulation


def load_data(filepath="data/btc_1h.csv"):
    """Load BTC data"""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found")
        sys.exit(1)


def run_single_backtest(df, strategy, capital=10000):
    """Run a single backtest"""
    
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df.get("volume", pd.Series([0]*len(df))).values
    
    # Create backtest engine
    bt = EnhancedBacktest(
        capital=capital,
        risk_per_trade=0.02,          # 2% risk per trade
        fee=0.001,                     # 0.1% fee
        slippage=0.0005,               # 0.05% slippage
        max_drawdown_limit=0.25,       # Halt at 25% drawdown
        daily_loss_limit=0.05,         # Max 5% daily loss
        cooldown_after_losses=3        # Cooldown after 3 losses
    )
    
    # Run backtest
    print("Running backtest...")
    equity = bt.run(highs, lows, closes, strategy, volumes=volumes)
    metrics = bt.get_metrics()
    
    return bt, equity, metrics


def print_metrics(metrics, initial_capital, buy_hold_return):
    """Print comprehensive metrics"""
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    
    print(f"\n💰 RETURNS")
    print(f"  Initial Capital:       ${initial_capital:>12,.2f}")
    print(f"  Final Equity:          ${metrics.get('final_equity', 0):>12,.2f}")
    print(f"  Total Return:          {metrics.get('total_return', 0):>12.2%}")
    print(f"  Buy & Hold Return:     {buy_hold_return:>12.2%}")
    
    print(f"\n📊 RISK METRICS")
    print(f"  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):>12.2f}")
    print(f"  Sortino Ratio:         {metrics.get('sortino_ratio', 0):>12.2f}")
    print(f"  Max Drawdown:          {metrics.get('max_drawdown', 0):>12.2%}")
    
    print(f"\n📈 TRADING METRICS")
    print(f"  Number of Trades:      {metrics.get('num_trades', 0):>12}")
    print(f"  Win Rate:              {metrics.get('win_rate', 0):>12.2%}")
    print(f"  Average Win:           ${metrics.get('avg_win', 0):>12,.2f}")
    print(f"  Average Loss:          ${metrics.get('avg_loss', 0):>12,.2f}")
    print(f"  Expectancy:            ${metrics.get('expectancy', 0):>12,.2f}")
    print(f"  Profit Factor:         {metrics.get('profit_factor', 0):>12.2f}")
    print(f"  Avg Duration (hours):  {metrics.get('avg_duration_hours', 0):>12.1f}")
    
    # Exit reasons
    if metrics.get('exit_reasons'):
        print(f"\n📋 EXIT REASONS")
        for reason, count in metrics['exit_reasons'].items():
            print(f"  {reason:<20s}: {count:>4}")
    
    # Risk warnings
    if metrics.get('trading_halted'):
        print(f"\n⚠️  WARNING: Trading was halted due to risk limits")


def plot_results(equity, buy_hold, trades, filename='backtest_results.png'):
    """Create comprehensive visualization"""
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Equity curve
    ax1 = axes[0]
    ax1.plot(equity, label="Strategy", linewidth=2, color='#2E86AB')
    ax1.plot(buy_hold, label="Buy & Hold", linestyle="--", alpha=0.7, color='#A23B72')
    ax1.set_ylabel("Equity ($)", fontsize=11)
    ax1.set_title("Professional BTC Trading Bot - Equity Curve", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2 = axes[1]
    equity_array = np.array(equity)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak
    ax2.fill_between(range(len(drawdown)), 0, -drawdown * 100, color='#E63946', alpha=0.3)
    ax2.plot(-drawdown * 100, color='#E63946', linewidth=1)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_title("Drawdown Over Time", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Trade distribution
    ax3 = axes[2]
    if trades:
        pnls = [t['pnl'] for t in trades]
        colors = ['#06D6A0' if pnl > 0 else '#E63946' for pnl in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel("PnL ($)", fontsize=11)
        ax3.set_xlabel("Trade Number", fontsize=11)
        ax3.set_title("Trade PnL Distribution", fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Chart saved to: {filename}")


def main():
    print("=" * 80)
    print("PROFESSIONAL BTC TRADING BOT - ENHANCED BACKTEST")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading BTC data...")
    df = load_data("data/btc_1h.csv")
    print(f"✅ Loaded {len(df)} hourly candles")
    print(f"   Period: {df.iloc[0].get('timestamp', 'N/A')} to {df.iloc[-1].get('timestamp', 'N/A')}")
    
    # Initialize strategy
    print("\nInitializing Simple Trend Strategy...")
    strategy = SimpleTrendStrategy(
        ema_fast=50,
        ema_slow=200,
        donchian_period=20,
        atr_period=14,
        atr_multiplier=2.0
    )
    
    # Run backtest
    initial_capital = 10000
    bt, equity, metrics = run_single_backtest(df, strategy, capital=initial_capital)
    
    # Calculate buy & hold
    closes = df["close"].values
    buy_hold_return = (closes[-1] / closes[0]) - 1
    buy_hold = (closes / closes[0]) * initial_capital
    
    # Print metrics
    print_metrics(metrics, initial_capital, buy_hold_return)
    
    # Plot results
    plot_results(equity, buy_hold, bt.trades, 'backtest_results.png')
    
    # Walk-forward analysis
    print("\n" + "=" * 80)
    print("WALK-FORWARD ANALYSIS")
    print("=" * 80)
    
    wf = WalkForwardAnalysis(train_ratio=0.7, num_windows=None)
    wf_results = wf.run_analysis(
        bt, strategy,
        df["high"].values, df["low"].values, df["close"].values,
        volumes=df.get("volume", pd.Series([0]*len(df))).values
    )
    wf.print_summary(wf_results)
    
    # Monte Carlo simulation
    if bt.trades:
        print("\n" + "=" * 80)
        print("MONTE CARLO SIMULATION")
        print("=" * 80)
        
        mc = MonteCarloSimulation(num_simulations=1000)
        mc_results = mc.run_simulation(bt.trades, initial_capital)
        mc.print_results(mc_results, initial_capital)
    
    print("\n" + "=" * 80)
    print("✅ BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
