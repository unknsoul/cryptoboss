"""
Test Real Backtest Engine
Demonstrates proper backtesting with realistic modeling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.backtest import RealBacktestEngine
from datetime import datetime, timedelta


def fetch_sample_data() -> pd.DataFrame:
    """
    Fetch sample BTC data for testing.
    In production, this would pull real historical data.
    """
    # Generate synthetic OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
    
    # Simulate random walk with trend
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, len(dates))
    prices = 40000 * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0.002, 0.003, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0.002, 0.003, len(dates)))),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, len(dates))
    }, index=dates)
    
    return df


def simple_momentum_strategy(df: pd.DataFrame, index: int) -> dict:
    """
    Simple momentum strategy for testing.
    
    Signal:
    - BUY: 20-period SMA crosses above 50-period SMA
    - SELL: Opposite
    """
    if index < 100:
        return {'signal': 0}
    
    # Calculate SMAs
    sma_20 = df['close'].iloc[index-20:index].mean()
    sma_50 = df['close'].iloc[index-50:index].mean()
    
    prev_sma_20 = df['close'].iloc[index-21:index-1].mean()
    prev_sma_50 = df['close'].iloc[index-51:index-1].mean()
    
    # Crossover logic
    if prev_sma_20 <= prev_sma_50 and sma_20 > sma_50:
        return {'signal': 1, 'reason': 'SMA_CROSS_UP', 'confidence': 0.7}
    elif prev_sma_20 >= prev_sma_50 and sma_20 < sma_50:
        return {'signal': -1, 'reason': 'SMA_CROSS_DOWN', 'confidence': 0.7}
    
    return {'signal': 0}


def main():
    """Run backtest demonstration."""
    print("=" * 70)
    print("REAL BACKTEST ENGINE - DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Fetch data
    print("📊 Fetching sample data...")
    df = fetch_sample_data()
    print(f"   Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    print()
    
    # Initialize backtest engine
    print("⚙️  Initializing backtest engine...")
    engine = RealBacktestEngine(
        initial_capital=10000.0,
        fee_taker_pct=0.04,  # 0.04%
        fee_maker_pct=0.02,  # 0.02%
        slippage_model="adaptive"
    )
    print("   Engine ready")
    print()
    
    # Run backtest
    print("🚀 Running backtest...")
    result = engine.run(
        data=df,
        strategy_func=simple_momentum_strategy,
        risk_per_trade_pct=1.0,  # 1% risk per trade
        stop_loss_atr_mult=2.0   # 2x ATR stop loss
    )
    print("   Backtest complete")
    print()
    
    # Display results
    print("=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print()
    
    print("📈 PERFORMANCE METRICS:")
    print(f"   Total Return:        {result.total_return_pct:+.2f}%")
    print(f"   Sharpe Ratio:        {result.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio:       {result.sortino_ratio:.2f}")
    print(f"   Max Drawdown:        -{result.max_drawdown_pct:.2f}%")
    print(f"   Annual Volatility:   {result.volatility_annual*100:.2f}%")
    print()
    
    print("📊 TRADE STATISTICS:")
    print(f"   Total Trades:        {result.total_trades}")
    print(f"   Win Rate:            {result.win_rate:.1f}%")
    print(f"   Profit Factor:       {result.profit_factor:.2f}")
    print(f"   Avg Win:             +{result.avg_win_pct:.2f}%")
    print(f"   Avg Loss:            {result.avg_loss_pct:.2f}%")
    print(f"   Avg Duration:        {result.avg_trade_duration_bars:.1f} bars")
    print()
    
    print("💰 CAPITAL:")
    print(f"   Initial:             ${result.initial_capital:,.2f}")
    print(f"   Final:               ${result.final_equity:,.2f}")
    print(f"   Profit/Loss:         ${result.final_equity - result.initial_capital:+,.2f}")
    print()
    
    # Show sample trades
    print("🔍 SAMPLE TRADES (First 5):")
    print("-" * 70)
    for i, trade in enumerate(result.trades[:5]):
        print(f"   Trade {i+1}: {trade.side}")
        print(f"      Entry:  ${trade.entry_price:.2f} ({trade.entry_time})")
        print(f"      Exit:   ${trade.exit_price:.2f} ({trade.exit_time})")
        print(f"      P&L:    ${trade.pnl:+.2f} ({trade.pnl_pct:+.2f}%)")
        print(f"      Fees:   ${trade.fees:.2f}")
        print(f"      Held:   {trade.bars_held} bars")
        print()
    
    # Export results
    results_dict = result.to_dict()
    print("=" * 70)
    print("✅ Backtest complete. Results exported.")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = main()
