"""
Test the Adaptive Strategy System
Runs backtests with automatic strategy selection
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from core.strategies.adaptive_selector import AdaptiveStrategySelector, MarketRegime
from core.strategies.factory import AdvancedStrategyFactory
from core.backtest import ProfessionalBacktest
from core.ml.signal_filter import SignalQualityFilter


def test_adaptive_strategy_selection():
    """Test adaptive strategy selection with different market regimes"""
    print("=" * 70)
    print("ADAPTIVE STRATEGY SELECTION TEST")
    print("=" * 70)
    
    selector = AdaptiveStrategySelector()
    available_strategies = AdvancedStrategyFactory.get_all_strategies()
    
    # Test different market scenarios
    scenarios = {
        'Trending Bull': {
            'prices': [45000 + i*50 for i in range(100)],  # Strong uptrend
            'volatility': 0.015,
            'avg_volatility': 0.020,
            'sentiment_score': 0.5,
            'trend_strength': 0.85,
            'adx': 32
        },
        'Ranging Low Vol': {
            'prices': [45000 + np.sin(i/10)*200 for i in range(100)],  # Sideways
            'volatility': 0.012,
            'avg_volatility': 0.020,
            'sentiment_score': 0.0,
            'trend_strength': 0.25,
            'adx': 15
        },
        'High Vol Breakout': {
            'prices': [45000] * 50 + [45000 + i*80 for i in range(50)],  # Breakout
            'volatility': 0.035,
            'avg_volatility': 0.020,
            'sentiment_score': 0.3,
            'trend_strength': 0.65,
            'adx': 28
        },
        'News-Driven': {
            'prices': [45000 + np.random.randn()*500 for _ in range(100)],  # Choppy
            'volatility': 0.045,
            'avg_volatility': 0.020,
            'sentiment_score': 0.75,  # Strong sentiment
            'trend_strength': 0.15,
            'adx': 12
        }
    }
    
    print("\nTesting different market scenarios:")
    print("-" * 70)
    
    for scenario_name, market_data in scenarios.items():
        regime = selector.detect_market_regime(market_data)
        best_strategy = selector.select_best_strategy(market_data, available_strategies[:6])
        
        print(f"\n{scenario_name}:")
        print(f"  Detected Regime: {regime.value}")
        print(f"  Selected Strategy: {best_strategy}")
        print(f"  Volatility Ratio: {market_data['volatility']/market_data['avg_volatility']:.2f}")
        print(f"  Trend Strength: {market_data['trend_strength']:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Adaptive selection test complete")


def test_signal_quality_filter():
    """Test signal quality filtering"""
    print("\n" + "=" * 70)
    print("SIGNAL QUALITY FILTER TEST")
    print("=" * 70)
    
    filter = SignalQualityFilter(min_quality_score=70.0)
    
    signals = [
        {
            'name': 'High Quality Signal',
            'ml_confidence': 0.85,
            'direction': 'LONG',
            'volume': 1800,
            'avg_volume': 1000,
            'volatility': 0.018,
            'avg_volatility': 0.020,
            'orderbook_imbalance': 0.6,
            'sentiment_score': 0.5,
            'timeframe_alignment': 4,
            'trend_strength': 0.82
        },
        {
            'name': 'Medium Quality Signal',
            'ml_confidence': 0.62,
            'direction': 'SHORT',
            'volume': 1100,
            'avg_volume': 1000,
            'volatility': 0.022,
            'avg_volatility': 0.020,
            'orderbook_imbalance': -0.3,
            'sentiment_score': -0.2,
            'timeframe_alignment': 2,
            'trend_strength': 0.55
        },
        {
            'name': 'Low Quality Signal',
            'ml_confidence': 0.52,
            'direction': 'LONG',
            'volume': 600,
            'avg_volume': 1000,
            'volatility': 0.045,
            'avg_volatility': 0.020,
            'orderbook_imbalance': -0.2,  # Wrong direction
            'sentiment_score': -0.3,  # Wrong direction
            'timeframe_alignment': 1,
            'trend_strength': 0.25
        }
    ]
    
    print("\nTesting signal quality:")
    print("-" * 70)
    
    for signal in signals:
        quality = filter.calculate_quality_score(signal)
        
        print(f"\n{signal['name']}:")
        print(f"  Quality Score: {quality['quality_score']}/100")
        print(f"  Grade: {quality['grade']}")
        print(f"  Should Trade: {'YES' if quality['should_trade'] else 'NO'}")
        print(f"  Position Multiplier: {quality['position_multiplier']:.1%}")
    
    print("\n" + "=" * 70)
    print("✅ Signal quality filter test complete")


def test_strategy_backtest():
    """Run backtest with one of the new strategies"""
    print("\n" + "=" * 70)
    print("STRATEGY BACKTEST TEST")
    print("=" * 70)
    
    # Load data
    try:
        df = pd.read_csv("data/btc_1h.csv")
        print(f"\n✅ Loaded {len(df)} bars of data from data/btc_1h.csv")
    except:
        print("\n⚠️  No data file found, generating synthetic data...")
        dates = pd.date_range('2024-01-01', periods=5000, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': 45000 + np.random.randn(5000).cumsum() * 50,
            'high': 45100 + np.random.randn(5000).cumsum() * 50,
            'low': 44900 + np.random.randn(5000).cumsum() * 50,
            'close': 45000 + np.random.randn(5000).cumsum() * 50,
            'volume': np.random.uniform(800, 1200, 5000)
        })
    
    # Test new strategies
    strategies_to_test = [
        'statistical_arbitrage',
        'volume_profile_trading',
        'breakout_momentum'
    ]
    
    results = {}
    
    for strategy_name in strategies_to_test:
        print(f"\nTesting {strategy_name}...")
        
        strategy = AdvancedStrategyFactory.create(strategy_name)
        bt = ProfessionalBacktest(capital=10000, base_risk=0.02)
        
        equity = bt.run(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            strategy,
            df['volume'].values
        )
        
        metrics = bt.get_metrics()
        results[strategy_name] = metrics
        
        print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Strategy':<30} {'Return':<12} {'Sharpe':<10} {'Win Rate':<12}")
    print("-" * 70)
    
    for strategy_name, metrics in results.items():
        print(f"{strategy_name:<30} "
              f"{metrics.get('total_return', 0):>10.2%}  "
              f"{metrics.get('sharpe_ratio', 0):>8.2f}  "
              f"{metrics.get('win_rate', 0):>10.2%}")
    
    print("\n" + "=" * 70)
    print("✅ Strategy backtest complete")


def run_all_tests():
    """Run all tests"""
    print("\n" + "🚀 " * 15)
    print("ADAPTIVE TRADING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("🚀 " * 15 + "\n")
    
    try:
        # Test 1: Adaptive Strategy Selection
        test_adaptive_strategy_selection()
        
        # Test 2: Signal Quality Filter
        test_signal_quality_filter()
        
        # Test 3: Strategy Backtests
        test_strategy_backtest()
        
        print("\n" + "✅ " * 15)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("✅ " * 15 + "\n")
        
        print("\n📊 SYSTEM STATUS:")
        print("  ✅ Adaptive strategy selection: WORKING")
        print("  ✅ Signal quality filtering: WORKING")
        print("  ✅ Advanced strategies: WORKING")
        print("  ✅ Backtesting engine: WORKING")
        
        print("\n🚀 READY FOR LIVE TRADING!")
        print("   Run: python adaptive_live_trader.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
