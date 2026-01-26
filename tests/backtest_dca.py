"""
DCA Strategy Backtest
Test DCA strategy with different configurations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from src.strategies.dca_strategy import DCAStrategy
from src.backtest.engine import RealBacktestEngine


def load_config(preset: str = "balanced") -> dict:
    """Load DCA configuration from YAML."""
    config_path = Path(__file__).parent.parent / "configs" / "dca_config.yaml"
    
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    if preset not in configs:
        raise ValueError(f"Preset '{preset}' not found. Available: {list(configs.keys())}")
    
    return configs[preset]


def run_dca_backtest(
    data_path: str = "data/btc_1h.csv",
    preset: str = "balanced",
    start_capital: float = 10000.0
):
    """
    Run DCA strategy backtest.
    
    Args:
        data_path: Path to OHLCV data
        preset: Configuration preset to use
        start_capital: Starting capital
    """
    print("=" * 80)
    print(f"DCA STRATEGY BACKTEST - {preset.upper()} PRESET")
    print("=" * 80)
    print()
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    print(f"✅ Loaded {len(df)} bars")
    print(f"   Period: {df.index[0]} to {df.index[-1]}")
    print()
    
    # Load configuration
    config = load_config(preset)
    print(f"Configuration ({preset}):")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize DCA strategy
    dca = DCAStrategy(
        base_order_size=config['base_order_size'],
        safety_order_size=config['safety_order_size'],
        max_safety_orders=config['max_safety_orders'],
        price_step_pct=config['price_step_pct'],
        target_profit_pct=config['target_profit_pct'],
        safety_order_volume_scale=config['safety_order_volume_scale'],
        stop_loss_pct=config['stop_loss_pct'],
        cooldown_bars=config['cooldown_bars']
    )
    
    # Calculate capital requirements
    required_capital = dca.calculate_total_investment()
    print(f"💰 Capital Required per Deal: ${required_capital:,.2f}")
    print(f"💼 Starting Capital: ${start_capital:,.2f}")
    print()
    
    if required_capital > start_capital:
        print(f"⚠️  WARNING: Required capital (${required_capital:,.2f}) exceeds starting capital!")
        print(f"   Strategy may not execute full safety order sequence.\n")
    
    # Run backtest
    print("Running backtest...")
    equity_curve = [start_capital]
    current_capital = start_capital
    
    for i in range(200, len(df)):  # Skip warmup period
        current_price = df['close'].iloc[i]
        
        # Generate signal
        signal = dca.generate_signal(df, i, current_price)
        
        if signal['action'] == 'BUY':
            # Deduct buying cost
            order_value = signal['size'] * signal['price']
            current_capital -= order_value
            
        elif signal['action'] == 'SELL':
            # Add selling proceeds + P&L
            proceeds = signal['size'] * signal['price']
            current_capital += proceeds
        
        equity_curve.append(current_capital)
    
    equity_curve = equity_curve[:len(df)-199]  # Align with data
    
    # Get metrics
    metrics = dca.get_metrics()
    deal_history = dca.get_deal_history()
    
    # Print results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    
    print(f"\n📊 DEAL STATISTICS")
    print(f"  Total Deals: {metrics['total_deals']}")
    print(f"  Profitable Deals: {metrics['profitable_deals']}")
    print(f"  Losing Deals: {metrics['losing_deals']}")
    print(f"  Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"  Avg Safety Orders Used: {metrics['avg_safety_orders_used']:.2f}")
    
    print(f"\n💰 PROFITABILITY")
    final_equity = equity_curve[-1]
    total_return = ((final_equity - start_capital) / start_capital) * 100
    
    print(f"  Starting Capital: ${start_capital:,.2f}")
    print(f"  Final Equity: ${final_equity:,.2f}")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"  Total P&L: ${metrics['total_pnl']:+,.2f}")
    print(f"  Avg P&L per Deal: {metrics['avg_pnl_pct']:+.2f}%")
    print(f"  Avg Win: {metrics['avg_win_pct']:+.2f}%")
    print(f"  Avg Loss: {metrics['avg_loss_pct']:+.2f}%")
    
    # Calculate buy & hold
    closes = df["close"].iloc[200:].values # Ensure closes starts from the same point as backtest
    buy_hold_return = ((closes[-1] - closes[0]) / closes[0]) * 100
    
    print(f"\n📈 COMPARISON")
    print(f"  DCA Return: {total_return:+.2f}%")
    print(f"  Buy & Hold: {buy_hold_return:+.2f}%")
    print(f"  Alpha: {total_return - buy_hold_return:+.2f}%")
    
    # Deal history
    if deal_history:
        print(f"\n📋 DEAL HISTORY (Last 10 deals)")
        print("-" * 120)
        print(f"{'Deal ID':<12} {'Start':<20} {'End':<20} {'SO Used':<8} {'Invested':<12} {'P&L $':<12} {'P&L %':<10} {'Reason':<15}")
        print("-" * 120)
        
        for deal in deal_history[-10:]:
            print(f"{deal['deal_id']:<12} "
                  f"{str(deal['start_time'])[:19]:<20} "
                  f"{str(deal['end_time'])[:19]:<20} "
                  f"{deal['safety_orders_used']:<8} "
                  f"${deal['total_invested']:<11,.0f} "
                  f"${deal['realized_pnl']:<11,.2f} "
                  f"{deal['realized_pnl_pct']:>8.2f}% "
                  f"{deal['exit_reason']:<15}")
    
    # Plot results
    print(f"\n📊 Generating chart...")
    plot_results(df[200:], equity_curve, deal_history, preset)
    
    print("\n" + "=" * 80)
    print("✅ BACKTEST COMPLETE")
    print("=" * 80)
    
    return metrics, equity_curve, deal_history


def plot_results(df, equity_curve, deal_history, preset):
    """Plot backtest results."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Price chart with deals
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='BTC Price', color='#2E86AB', linewidth=1.5)
    
    # Mark deal entries and exits
    for deal in deal_history:
        # Entry
        ax1.scatter(deal['start_time'], deal['base_price'], 
                   color='green', marker='^', s=100, zorder=5, alpha=0.7)
        # Exit
        color = 'darkgreen' if deal['realized_pnl'] > 0 else 'darkred'
        ax1.scatter(deal['end_time'], deal['exit_price'],
                   color=color, marker='v', s=100, zorder=5, alpha=0.7)
    
    ax1.set_ylabel('Price (USD)', fontsize=11)
    ax1.set_title(f'DCA Strategy Backtest - {preset.upper()} Preset', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Equity curve
    ax2 = axes[1]
    ax2.plot(df.index, equity_curve, color='#06D6A0', linewidth=2)
    ax2.set_ylabel('Equity (USD)', fontsize=11)
    ax2.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=equity_curve[0], color='gray', linestyle='--', alpha=0.5)
    
    # Deal P&L distribution
    ax3 = axes[2]
    if deal_history:
        pnls = [d['realized_pnl'] for d in deal_history]
        colors = ['#06D6A0' if pnl > 0 else '#E63946' for pnl in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('Deal P&L (USD)', fontsize=11)
        ax3.set_xlabel('Deal Number', fontsize=11)
        ax3.set_title('Deal P&L Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'dca_backtest_{preset}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   Chart saved to: {filename}")


def compare_presets():
    """Compare all DCA presets on same data."""
    print("\n" + "=" * 80)
    print("COMPARING ALL DCA PRESETS")
    print("=" * 80)
    print()
    
    presets = ['conservative', 'balanced', 'aggressive', 'commas_style', 'scalper', 'hodl']
    results = {}
    
    for preset in presets:
        print(f"\n{'='*40}")
        print(f"Testing: {preset.upper()}")
        print(f"{'='*40}\n")
        
        try:
            metrics, equity, deals = run_dca_backtest(preset=preset, start_capital=50000.0)
            results[preset] = metrics
        except Exception as e:
            print(f"❌ Error testing {preset}: {e}")
            continue
    
    # Summary comparison
    print("\n\n" + "=" * 120)
    print("PRESET COMPARISON SUMMARY")
    print("=" * 120)
    print(f"{'Preset':<15} {'Deals':<8} {'Win%':<8} {'Avg SO':<8} {'Avg P&L%':<12} {'Total P&L':<12} {'Capital Req':<12}")
    print("-" * 120)
    
    for preset, metrics in results.items():
        print(f"{preset:<15} "
              f"{metrics['total_deals']:<8} "
              f"{metrics['win_rate_pct']:<7.1f}% "
              f"{metrics['avg_safety_orders_used']:<7.2f} "
              f"{metrics['avg_pnl_pct']:>10.2f}% "
              f"${metrics['total_pnl']:>10,.2f} "
              f"${metrics['max_capital_required']:>10,.0f}")
    
    print("=" * 120)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DCA Strategy Backtest')
    parser.add_argument('--preset', type=str, default='balanced',
                       choices=['conservative', 'balanced', 'aggressive', 
                               'commas_style', 'scalper', 'hodl'],
                       help='Configuration preset to use')
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Starting capital')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all presets')
    parser.add_argument('--data', type=str, default='data/btc_1h.csv',
                       help='Path to OHLCV data')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_presets()
    else:
        run_dca_backtest(
            data_path=args.data,
            preset=args.preset,
            start_capital=args.capital
        )
