"""
Strategy Comparison Script - Test Enhanced vs Original
Tier 1 improvements validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from core.strategy import SimpleTrendStrategy
from core.enhanced_trend_strategy import EnhancedTrendStrategy
from core.backtest import EnhancedBacktest
from core.testing.walk_forward import WalkForwardAnalysis


def load_data(filepath="data/btc_1h.csv"):
    """Load BTC data"""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found")
        sys.exit(1)


def run_strategy_backtest(df, strategy, strategy_name, capital=10000):
    """Run backtest for a strategy"""
    
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df.get("volume", pd.Series([0]*len(df))).values
    
    # Create backtest engine
    bt = EnhancedBacktest(
        capital=capital,
        risk_per_trade=0.02,
        fee=0.001,
        slippage=0.0005,
        max_drawdown_limit=0.25,
        daily_loss_limit=0.05,
        cooldown_after_losses=3,
        # Tier 1 improvements (always on for fair comparison)
        use_partial_profits=True,
        partial_profit_1r=0.25,
        partial_profit_2r=0.50,
        use_breakeven_stop=True,
        max_hold_hours=48
    )
    
    print(f"\n🔄 Running backtest for: {strategy_name}")
    equity = bt.run(highs, lows, closes, strategy, volumes=volumes)
    metrics = bt.get_metrics()
    
    return bt, equity, metrics


def print_comparison(original_metrics, enhanced_metrics):
    """Print side-by-side comparison"""
    
    print("\n" + "="*100)
    print("STRATEGY COMPARISON - ORIGINAL vs ENHANCED (Tier 1)")
    print("="*100)
    
    metrics_to_compare = [
        ('Win Rate', 'win_rate', '%'),
        ('Total Return', 'total_return', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Sortino Ratio', 'sortino_ratio', ''),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Number of Trades', 'num_trades', ''),
        ('Expectancy', 'expectancy', '$'),
        ('Profit Factor', 'profit_factor', ''),
        ('Avg Win', 'avg_win', '$'),
        ('Avg Loss', 'avg_loss', '$'),
    ]
    
    print(f"\n{'Metric':<20} {'Original':<20} {'Enhanced':<20} {'Change':<15}")
    print("-" * 100)
    
    for label, key, unit in metrics_to_compare:
        orig_val = original_metrics.get(key, 0)
        enh_val = enhanced_metrics.get(key, 0)
        
        if unit == '%':
            orig_str = f"{orig_val:.2%}"
            enh_str = f"{enh_val:.2%}"
            change = ((enh_val - orig_val) * 100)
            change_str = f"{change:+.2f} pp"
        elif unit == '$':
            orig_str = f"${orig_val:,.2f}"
            enh_str = f"${enh_val:,.2f}"
            if orig_val != 0:
                change_pct = ((enh_val - orig_val) / abs(orig_val)) * 100
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = "N/A"
        else:
            orig_str = f"{orig_val:.2f}"
            enh_str = f"{enh_val:.2f}"
            if orig_val != 0:
                change_pct = ((enh_val - orig_val) / abs(orig_val)) * 100
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = "N/A"
        
        # Color coding
        if key in ['win_rate', 'total_return', 'sharpe_ratio', 'sortino_ratio', 'profit_factor', 'expectancy']:
            improvement = enh_val > orig_val
        elif key == 'max_drawdown':
            improvement = enh_val < orig_val
        elif key == 'avg_loss':
            improvement = enh_val > orig_val  # Less negative is better
        else:
            improvement = None
        
        if improvement is True:
            indicator = "✅"
        elif improvement is False:
            indicator = "⚠️"
        else:
            indicator = "  "
        
        print(f"{indicator} {label:<20} {orig_str:<20} {enh_str:<20} {change_str:<15}")


def plot_comparison(original_equity, enhanced_equity, original_trades, enhanced_trades, filename='tier1_comparison.png'):
    """Create comparison visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Equity curves
    ax1 = axes[0, 0]
    ax1.plot(original_equity, label="Original Strategy", linewidth=2, color='#2E86AB', alpha=0.7)
    ax1.plot(enhanced_equity, label="Enhanced Strategy (Tier 1)", linewidth=2, color='#06D6A0')
    ax1.set_ylabel("Equity ($)", fontsize=11)
    ax1.set_title("Equity Curve Comparison", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Drawdown comparison
    ax2 = axes[0, 1]
    
    orig_equity_array = np.array(original_equity)
    orig_peak = np.maximum.accumulate(orig_equity_array)
    orig_drawdown = (orig_peak - orig_equity_array) / orig_peak
    
    enh_equity_array = np.array(enhanced_equity)
    enh_peak = np.maximum.accumulate(enh_equity_array)
    enh_drawdown = (enh_peak - enh_equity_array) / enh_peak
    
    ax2.plot(-orig_drawdown * 100, label="Original", color='#E63946', linewidth=1, alpha=0.7)
    ax2.plot(-enh_drawdown * 100, label="Enhanced", color='#06D6A0', linewidth=1)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_title("Drawdown Comparison", fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Trade PnL distribution - Original
    ax3 = axes[1, 0]
    if original_trades:
        pnls = [t['pnl'] for t in original_trades]
        colors = ['#06D6A0' if pnl > 0 else '#E63946' for pnl in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel("PnL ($)", fontsize=11)
        ax3.set_xlabel("Trade Number", fontsize=11)
        ax3.set_title("Original Strategy - Trade PnL", fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Trade PnL distribution - Enhanced
    ax4 = axes[1, 1]
    if enhanced_trades:
        pnls = [t['pnl'] for t in enhanced_trades]
        colors = ['#06D6A0' if pnl > 0 else '#E63946' for pnl in pnls]
        ax4.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_ylabel("PnL ($)", fontsize=11)
        ax4.set_xlabel("Trade Number", fontsize=11)
        ax4.set_title("Enhanced Strategy - Trade PnL (Tier 1)", fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison chart saved to: {filename}")


def main():
    print("="*100)
    print("TIER 1 IMPROVEMENTS - STRATEGY COMPARISON")
    print("="*100)
    
    # Load data
    print("\nLoading BTC data...")
    df = load_data("data/btc_1h.csv")
    print(f"✅ Loaded {len(df)} hourly candles")
    
    initial_capital = 10000
    
    # Test Original Strategy
    print("\n" + "="*100)
    print("TESTING ORIGINAL STRATEGY")
    print("="*100)
    
    original_strategy = SimpleTrendStrategy(
        ema_fast=50,
        ema_slow=200,
        donchian_period=20,
        atr_period=14,
        atr_multiplier=2.0
    )
    
    original_bt, original_equity, original_metrics = run_strategy_backtest(
        df, original_strategy, "Simple Trend (Original)", initial_capital
    )
    
    # Test Enhanced Strategy
    print("\n" + "="*100)
    print("TESTING ENHANCED STRATEGY (TIER 1)")
    print("="*100)
    
    enhanced_strategy = EnhancedTrendStrategy(
        ema_fast=50,
        ema_slow=200,
        donchian_period=20,
        atr_period=14,
        atr_multiplier=2.0,
        # Entry filters
        use_volume_filter=True,
        volume_threshold=0.8,
        use_volatility_filter=True,
        volatility_percentile_threshold=30,
        use_adx_filter=True,
        adx_threshold=25,
        use_time_filter=False  # Disabled for backtest (no timestamps)
    )
    
    enhanced_bt, enhanced_equity, enhanced_metrics = run_strategy_backtest(
        df, enhanced_strategy, "Enhanced Trend (Tier 1)", initial_capital
    )
    
    # Print comparison
    print_comparison(original_metrics, enhanced_metrics)
    
    # Exit reasons comparison
    print("\n" + "="*100)
    print("EXIT REASONS BREAKDOWN")
    print("="*100)
    
    print("\nOriginal Strategy:")
    if original_metrics.get('exit_reasons'):
        for reason, count in original_metrics['exit_reasons'].items():
            print(f"  {reason:<25}: {count:>4}")
    
    print("\nEnhanced Strategy (Tier 1):")
    if enhanced_metrics.get('exit_reasons'):
        for reason, count in enhanced_metrics['exit_reasons'].items():
            print(f"  {reason:<25}: {count:>4}")
    
    # Plot comparison
    plot_comparison(
        original_equity, enhanced_equity,
        original_bt.trades, enhanced_bt.trades,
        'tier1_comparison.png'
    )
    
    # Walk-forward analysis on enhanced strategy
    print("\n" + "="*100)
    print("WALK-FORWARD ANALYSIS (Enhanced Strategy)")
    print("="*100)
    
    wf = WalkForwardAnalysis(train_ratio=0.7, num_windows=None)
    wf_results = wf.run_analysis(
        enhanced_bt, enhanced_strategy,
        df["high"].values, df["low"].values, df["close"].values,
        volumes=df.get("volume", pd.Series([0]*len(df))).values
    )
    wf.print_summary(wf_results)
    
    print("\n" + "="*100)
    print("✅ COMPARISON COMPLETE")
    print("="*100)
    
    # Summary
    print("\n📊 TIER 1 IMPROVEMENT SUMMARY:")
    print(f"  Win Rate:     {original_metrics.get('win_rate', 0):.1%} → {enhanced_metrics.get('win_rate', 0):.1%}")
    print(f"  Expectancy:   ${original_metrics.get('expectancy', 0):.2f} → ${enhanced_metrics.get('expectancy', 0):.2f}")
    print(f"  Sharpe Ratio: {original_metrics.get('sharpe_ratio', 0):.2f} → {enhanced_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {original_metrics.get('max_drawdown', 0):.2%} → {enhanced_metrics.get('max_drawdown', 0):.2%}")


if __name__ == "__main__":
    main()
