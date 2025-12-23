"""
Quick Accuracy Test Script
Tests the trading system accuracy on historical data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

# Load data
try:
    df = pd.read_csv('data/btc_1h.csv')
    print(f'[OK] Loaded {len(df)} candles from data/btc_1h.csv')
except:
    print('[!] No data file found. Downloading...')
    from core.data_manager import DataManager
    dm = DataManager()
    df = dm.download_binance_data('BTCUSDT', '1h', '2024-01-01')
    print(f'[OK] Downloaded {len(df)} candles')

# Run backtest
from core.strategy import SimpleTrendStrategy
from core.backtest import EnhancedBacktest

print('\n' + '='*60)
print('TESTING TRADING STRATEGY ACCURACY')
print('='*60)

strategy = SimpleTrendStrategy()
bt = EnhancedBacktest(capital=10000, risk_per_trade=0.02, fee=0.001, slippage=0.0005)

equity = bt.run(
    df['high'].values,
    df['low'].values,
    df['close'].values,
    strategy,
    volumes=df.get('volume', pd.Series([0]*len(df))).values
)
metrics = bt.get_metrics()

print('\nBACKTEST RESULTS')
print('='*60)
print(f"Total Return:    {metrics.get('total_return', 0)*100:.2f}%")
print(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}")
print(f"Win Rate:        {metrics.get('win_rate', 0)*100:.2f}%")
print(f"Max Drawdown:    {metrics.get('max_drawdown', 0)*100:.2f}%")
print(f"Total Trades:    {metrics.get('num_trades', 0)}")
print(f"Profit Factor:   {metrics.get('profit_factor', 0):.2f}")
print(f"Expectancy:      ${metrics.get('expectancy', 0):.2f}")
print('='*60)

# Test portfolio optimizer
print('\nTESTING NEW FEATURES')
print('='*60)

try:
    from core.strategy_orchestrator import PortfolioOptimizer
    optimizer = PortfolioOptimizer()
    fake_returns = np.random.randn(50, 3) * 0.02
    weights = optimizer.optimize_mean_variance(fake_returns)
    print(f"[OK] Portfolio Optimizer: Mean-Variance weights = {weights.round(3)}")
except Exception as e:
    print(f"[FAIL] Portfolio Optimizer: {e}")

try:
    from core.execution.smart_execution import SmartOrderRouter, ExecutionOrder, ExecutionAlgorithm
    router = SmartOrderRouter()
    order = ExecutionOrder(
        symbol="BTCUSDT",
        side="BUY",
        total_quantity=0.1,
        algorithm=ExecutionAlgorithm.TWAP,
        duration_seconds=30,
        limit_price=45000
    )
    result = router.execute_twap(order)
    print(f"[OK] Smart Execution: TWAP avg price = ${result['avg_fill_price']:.2f}")
except Exception as e:
    print(f"[FAIL] Smart Execution: {e}")

try:
    from core.analysis.sentiment_analyzer import SentimentAnalyzer
    analyzer = SentimentAnalyzer(use_mock_data=True)
    fg = analyzer.get_fear_greed_index()
    print(f"[OK] Sentiment Analyzer: Fear & Greed Index = {fg}")
except Exception as e:
    print(f"[FAIL] Sentiment Analyzer: {e}")

print('='*60)
print('[OK] ALL TESTS COMPLETE')
print('='*60)
