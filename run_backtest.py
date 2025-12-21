"""
Professional AI Trading Bot - Main Execution Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.strategies.factory import StrategyFactory
from core.backtest import ProfessionalBacktest


def main():
    print("=" * 70)
    print("PROFESSIONAL AI TRADING BOT - BACKTEST")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    try:
        df = pd.read_csv("data/btc_1h.csv")
    except FileNotFoundError:
        print("Error: data/btc_1h.csv not found.")
        return
    
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    
    print(f"Data loaded: {len(closes)} hourly candles")
    print()
    
    # Initialize professional strategy
    print("Initializing professional strategy (Professional Trend)...")
    
    # Use factory to create strategy
    strategy = StrategyFactory.create("professional_trend",
        donchian_period=20,
        ema_fast=12,
        ema_slow=50,
        atr_period=14,
        atr_multiplier=3.0,
        adx_period=14,
        adx_threshold=25,
        rsi_period=14
    )
    
    bt = ProfessionalBacktest(
        capital=10000,
        base_risk=0.02,  # 2% base risk, adjusted by volatility
        fee=0.001        # 0.1% fee
    )
    
    # Run backtest
    print("Running backtest...")
    equity = bt.run(highs, lows, closes, strategy)
    
    # Calculate buy and hold benchmark
    buy_hold = (closes / closes[0]) * bt.initial_capital
    
    # Get comprehensive metrics
    metrics = bt.get_metrics()
    
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    # Performance metrics
    print(f"\n📊 PERFORMANCE METRICS")
    print(f"{'Initial Capital:':<25} ${bt.initial_capital:>12,.2f}")
    print(f"{'Final Equity:':<25} ${metrics.get('final_equity', 0):>12,.2f}")
    if 'total_return' in metrics:
        print(f"{'Total Return:':<25} {metrics['total_return']:>12.2%}")
    if 'annual_return' in metrics:
        print(f"{'Annual Return:':<25} {metrics['annual_return']:>12.2%}")
    print(f"{'Buy & Hold Return:':<25} {(closes[-1]/closes[0] - 1):>12.2%}")
    
    # Risk metrics
    print(f"\n⚠️ RISK METRICS")
    if 'sharpe_ratio' in metrics:
        print(f"{'Sharpe Ratio:':<25} {metrics['sharpe_ratio']:>12.2f}")
    if 'max_drawdown' in metrics:
        print(f"{'Max Drawdown:':<25} {metrics['max_drawdown']:>12.2%}")
    
    # Trading metrics
    print(f"\n📈 TRADING METRICS")
    print(f"{'Number of Trades:':<25} {metrics.get('num_trades', 0):>12}")
    if 'win_rate' in metrics:
        print(f"{'Win Rate:':<25} {metrics['win_rate']:>12.2%}")
    if 'profit_factor' in metrics:
        print(f"{'Profit Factor:':<25} {metrics['profit_factor']:>12.2f}")
    
    print("\n" + "=" * 70)
    
    # Plot results
    print(f"\nGenerating charts...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity, label="Professional Strategy", linewidth=2, color='#2E86AB')
    ax1.plot(buy_hold, label="Buy & Hold", linestyle="--", alpha=0.7, color='#A23B72')
    ax1.set_ylabel("Equity ($)", fontsize=11)
    ax1.set_title("Professional AI Trading Bot - Equity Curve", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.savefig('professional_strategy_results.png', dpi=150, bbox_inches='tight')
    print(f"Chart saved to: professional_strategy_results.png")
    
    print(f"\n✅ Backtest complete!")


if __name__ == "__main__":
    main()
