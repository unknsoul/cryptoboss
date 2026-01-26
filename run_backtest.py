#!/usr/bin/env python3
"""
CryptoBoss Backtest Runner

Run backtests using the new architecture.

Usage:
    python run_backtest.py
    python run_backtest.py --strategy=dca --capital=10000
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest import SimpleBacktest
from src.strategies.dca_strategy import DCAStrategy


def load_data(filepath: str = "data/btc_1h.csv") -> pd.DataFrame:
    """Load historical data."""
    try:
        df = pd.read_csv(filepath)
        
        # Ensure proper column names
        df.columns = [c.lower() for c in df.columns]
        
        # Set datetime index if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        print(f"✅ Loaded {len(df)} rows from {filepath}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found")
        print("   Run: python download_data.py to download data")
        sys.exit(1)


def run_dca_backtest(df: pd.DataFrame, capital: float = 10000) -> dict:
    """Run DCA strategy backtest."""
    print("\n" + "=" * 60)
    print("  DCA STRATEGY BACKTEST")
    print("=" * 60)
    
    # Create strategy
    strategy = DCAStrategy(
        base_order_size=capital * 0.02,      # 2% base order
        safety_order_size=capital * 0.04,    # 4% safety orders
        max_safety_orders=5,
        price_step_pct=2.5,
        target_profit_pct=2.0,
        safety_order_volume_scale=1.5,
        stop_loss_pct=15.0
    )
    
    print(f"\nConfig:")
    print(f"  Base Order: ${strategy.base_order_size:.2f}")
    print(f"  Safety Order: ${strategy.safety_order_size:.2f}")
    print(f"  Max Safety Orders: {strategy.max_safety_orders}")
    print(f"  Price Deviation: {strategy.price_step_pct}%")
    print(f"  Target Profit: {strategy.target_profit_pct}%")
    print(f"  Max Capital Required: ${strategy.calculate_total_investment():,.2f}")
    
    # Run backtest
    bt = SimpleBacktest(capital=capital, fee_rate=0.001, slippage_bps=5)
    result = bt.run(df, strategy)
    
    # Print results
    print(f"\n📊 RESULTS")
    print(f"  Initial Capital:   ${result.initial_capital:,.2f}")
    print(f"  Final Capital:     ${result.final_capital:,.2f}")
    print(f"  Total Return:      ${result.total_return:,.2f} ({result.total_return_pct:+.2f}%)")
    print(f"\n📈 PERFORMANCE")
    print(f"  Total Trades:      {result.num_trades}")
    print(f"  Win Rate:          {result.win_rate:.1f}%")
    print(f"  Avg Win:           ${result.avg_win:,.2f}")
    print(f"  Avg Loss:          ${result.avg_loss:,.2f}")
    print(f"  Max Drawdown:      {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"  Profit Factor:     {result.profit_factor:.2f}")
    
    # Strategy-specific metrics
    metrics = strategy.get_metrics()
    print(f"\n🔄 DCA METRICS")
    print(f"  Total Deals:       {metrics.get('total_deals', 0)}")
    print(f"  Avg Safety Orders: {metrics.get('avg_safety_orders_used', 0):.1f}")
    
    return {
        'result': result,
        'strategy': strategy,
        'metrics': metrics
    }


def plot_results(result, df: pd.DataFrame, save_path: str = "backtest_results.png"):
    """Plot backtest results."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Equity curve
    ax1 = axes[0]
    ax1.plot(result.equity_curve, label='Strategy Equity', color='blue', linewidth=1.5)
    ax1.axhline(y=result.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_title('Equity Curve', fontsize=14)
    ax1.set_xlabel('Bar')
    ax1.set_ylabel('Equity ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Price with trades
    ax2 = axes[1]
    ax2.plot(df['close'].values, label='BTC Price', color='gray', alpha=0.7)
    
    # Mark trades
    for trade in result.trades:
        if hasattr(trade, 'entry_time') and trade.entry_price:
            color = 'green' if trade.pnl > 0 else 'red'
            # Just plot entry points for simplicity
            ax2.axhline(y=trade.entry_price, color=color, alpha=0.3, linewidth=0.5)
    
    ax2.set_title('Price Chart with Trades', fontsize=14)
    ax2.set_xlabel('Bar')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n📊 Chart saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="CryptoBoss Backtest Runner")
    parser.add_argument("--data", default="data/btc_1h.csv", help="Data file path")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital")
    parser.add_argument("--strategy", choices=["dca", "grid"], default="dca", help="Strategy")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.data)
    
    # Run backtest
    if args.strategy == "dca":
        output = run_dca_backtest(df, args.capital)
    else:
        print(f"Strategy {args.strategy} not yet implemented in backtest")
        return
    
    # Plot
    if not args.no_plot:
        try:
            plot_results(output['result'], df)
        except Exception as e:
            print(f"⚠️ Could not plot: {e}")
    
    print("\n" + "=" * 60)
    print("  BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
