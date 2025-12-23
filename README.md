# BTC Algorithmic Trading Bot

A Python-based algorithmic trading system for Bitcoin on Binance with backtesting, risk management, and machine learning components.

> **⚠️ WARNING**: This is experimental software for educational and research purposes.  
> **Trading cryptocurrencies involves substantial risk of loss.**  
> Always test thoroughly on testnet before risking real capital.

---

## What This Bot Does

This is a systematic trading bot that:
- Downloads historical BTC/USDT data from Binance
- Implements multiple technical indicator-based strategies
- Backtests strategies with realistic fee and slippage modeling
- Provides risk management tools (position sizing, stop losses, circuit breakers)
- Can execute trades on Binance (testnet and mainnet)
- Includes machine learning components for signal filtering and market regime detection

**This is NOT:**
- A guaranteed profit system
- "Institutional-grade" without professional validation
- Fully autonomous AI that "learns" without supervision
- Ready for production without extensive testing

---

## Project Status

**Current State:**
- ✅ Basic backtesting engine (slippage, fees, trailing stops)
- ✅ Multiple strategy implementations (trend following, mean reversion)
- ✅ Walk-forward analysis and Monte Carlo simulation
- ✅ Live paper trading capability
- ✅ Risk management (Kelly criterion, VaR, circuit breakers)
- ⚠️ ML components are experimental (not production-validated)
- ⚠️ No live trading track record provided
- ⚠️ Requires significant testing and validation before real use

**Known Limitations:**
- Backtest results may not reflect live performance
- ML models require retraining on current market data
- No multi-exchange support yet
- Limited order book depth analysis
- Testsnet behav

ior may differ from mainnet

---

## Architecture

```
├── core/
│   ├── strategies/          # Trading strategy implementations
│   │   ├── base_strategy.py # Abstract base class
│   │   ├── trend_following.py
│   │   ├── mean_reversion.py
│   │   └── ensemble.py
│   ├── backtest.py          # Backtesting engine (fees, slippage, stops)
│   ├── risk/                # Risk management
│   │   ├── position_sizing.py  # Kelly, volatility-adjusted
│   │   └── circuit_breakers.py # Emergency stops
│   ├── ml/                  # Machine learning components
│   │   ├── regime_detector.py  # Market regime classification
│   │   ├── signal_filter.py    # ML-based trade filtering
│   │   └── feature_engineering.py
│   ├── exchange/            # Exchange integration
│   │   ├── binance_client.py
│   │   └── error_handlers.py
│   ├── performance/         # High-performance indicators
│   │   └── fast_indicators.py  # Numba JIT-compiled
│   ├── security/            # Security features
│   │   ├── secure_config.py    # API key encryption
│   │   ├── rate_limiter.py
│   │   └── input_validator.py
│   └── monitoring/          # Monitoring and alerts
│       ├── health_check.py
│       └── metrics.py
├── run_backtest.py          # Run historical backtests
├── live_paper_trader.py     # Paper trading (testnet)
└── tests/                   # Unit and integration tests
```

---

## Installation

### Requirements
- Python 3.10+
- Binance API keys (testnet recommended for testing)

### Setup

```bash
# Clone repository
git clone <your-repo-url>
cd final99

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Encrypt API keys (recommended)
python encrypt_config.py
```

---

## Usage

### 1. Backtesting

Test strategies on historical data:

```bash
python run_backtest.py
```

**What it does:**
- Downloads BTC/USDT 1h data from Binance
- Runs strategy on historical data
- Simulates trades with fees (0.1%) and slippage (0.05%)
- Calculates performance metrics
- Performs walk-forward analysis
- Runs Monte Carlo simulation

**Sample Output:**
```
================================================================================
BACKTEST RESULTS
================================================================================
Period: 2023-01-01 to 2024-12-22
Initial Capital: $10,000.00
Final Equity: $11,234.50

Returns:
  Total Return: 12.35%
  Annualized: 11.8%
  Buy & Hold: 38.2%  (underperformed)

Risk Metrics:
  Sharpe Ratio: 0.87
  Max Drawdown: -18.5%
  Win Rate: 52.3%

Trades: 45 (28 wins, 17 losses)
Average Win: $145.23
Average Loss: -$98.45
```

### 2. Paper Trading (Testnet)

Test live execution without real money:

```bash
python live_paper_trader.py
```

**Requirements:**
- Binance testnet API keys
- Set `BINANCE_TESTNET=true` in .env

**What it does:**
- Connects to Binance testnet via WebSocket
- Generates trade signals in real-time
- Executes simulated trades
- Logs all activity

### 3. Configuration

Edit `core/config/config_manager.py` or use environment variables:

```python
from core.config.config_manager import get_config

config = get_config()

# Strategy parameters
config.strategy.ema_fast = 50
config.strategy.ema_slow = 200

# Risk settings
config.risk.risk_per_trade = 0.02  # 2% per trade
config.risk.max_drawdown = 0.25    # Stop at 25% DD
```

---

## Strategies

### Simple Trend Following

**Logic:**
1. Calculate EMA(50) and EMA(200)
2. Calculate Donchian Channel(20)
3. **Buy** when price breaks above Donchian high AND EMA50 > EMA200
4. **Sell** when hit stop loss (ATR-based trailing stop)

**Parameters:**
- `ema_fast`: Fast EMA period (default: 50)
- `ema_slow`: Slow EMA period (default: 200)
- `donchian_period`: Breakout period (default: 20)
- `atr_multiplier`: Stop distance (default: 2.0)

**Implementation:** [core/strategy.py](core/strategy.py)

### Mean Reversion

**Logic:**
1. Calculate Bollinger Bands (20, 2 std)
2. Calculate RSI(14)
3. **Buy** when price below lower band AND RSI < 30
4. **Sell** when price reaches middle band or RSI > 70

**Implementation:** [core/strategies/mean_reversion.py](core/strategies/mean_reversion.py)

### Adding Custom Strategies

All strategies must inherit from `BaseStrategy`:

```python
from core.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        """
        Generate buy/sell signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            signal dict or None
        """
        # Your signal logic here
        if buy_condition:
            return {
                'action': 'LONG',
                'stop': stop_distance,
                'metadata': {...}
            }
        return None
```

---

## Machine Learning Components

### Current ML Features

**1. Market Regime Detection** ([core/ml/regime_detector.py](core/ml/regime_detector.py))
- Uses Gaussian Mixture Model to classify market states
- States: Trending up, Trending down, Ranging, High volatility
- Used to select appropriate strategies

**2. Signal Filter** ([core/ml/signal_filter.py](core/ml/signal_filter.py))
- XGBoost classifier trained on:
  - Price features (returns, volatility)
  - Technical indicators (RSI, MACD, ATR)
  - Volume metrics
- Labels: Historical forward returns (profitable vs unprofitable)
- Purpose: Filter out low-quality signals

**3. Feature Engineering** ([core/ml/feature_engineering.py](core/ml/feature_engineering.py))
- Creates features from raw OHLCV data
- Includes: returns, volatility, momentum, volume ratios

### Training ML Models

```python
from core.ml.signal_filter import MLSignalFilter

# Load historical data
data = load_data('BTCUSDT', '1h')

# Create features
filter = MLSignalFilter()
X, y = filter.prepare_training_data(data)

# Train model
filter.train(X, y)

# Save model
filter.save_model('models/signal_filter_v1.pkl')
```

**Important Notes:**
- Models trained on historical data may not perform well in different market conditions
- Requires periodic retraining
- No guarantee of future performance
- Test thoroughly before using for real trades

---

## Backtesting Engine

### What's Realistic

- ✅ Trading fees (0.1% default, configurable)
- ✅ Slippage (0.05% default, configurable)
- ✅ Trailing stops (updated every candle)
- ✅ No lookahead bias (indicators exclude current candle where appropriate)
- ✅ Time-based train/test splits

### What's NOT Realistic

- ❌ Order book liquidity (assumes all orders fill)
- ❌ Partial fills (assumes complete fills)
- ❌ Market impact (assumes we don't move the market)
- ❌ Extreme events (flash crashes, exchange outages)

### Walk-Forward Analysis

Tests strategy on out-of-sample data:

```python
from core.testing.walk_forward import WalkForwardAnalysis

wf = WalkForwardAnalysis(train_ratio=0.7, num_windows=3)
results = wf.run_analysis(backtest_engine, strategy, data)
```

**Process:**
1. Split data into train/test windows
2. "Optimize" on train data
3. Test on unseen test data
4. Compare in-sample vs out-of-sample performance

**Good result:** Out-of-sample return is >70% of in-sample (efficiency ratio > 0.7)

---

## Risk Management

### Position Sizing

**Volatility-Adjusted:**
```
position_size = (capital * risk%) / (ATR * multiplier)
```

**Kelly Criterion:**
```
f = (p * W - L) / W
where p = win rate, W = avg win, L = avg loss
```

### Circuit Breakers

Automatic trading halts:
- **Max Drawdown:** Stop if equity drops >25% from peak
- **Daily Loss:** Stop if lose >5% in one day
- **Consecutive Losses:** Stop after 5 losses in a row
- **Cooldown Period:** Pause trading for N candles after trigger

**Implementation:** [core/safety/circuit_breakers.py](core/safety/circuit_breakers.py)

---

## Performance Metrics

All metrics calculated from backtest results:

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **Sharpe Ratio** | (Return - RiskFree) / StdDev | Risk-adjusted returns. >1 is decent, >2 is good |
| **Sortino Ratio** | (Return - RiskFree) / DownsideStdDev | Like Sharpe but only penalizes downside |
| **Calmar Ratio** | AnnualReturn / MaxDrawdown | Return per unit of max loss |
| **Max Drawdown** | Peak-to-trough decline | Worst loss from equity peak |
| **Win Rate** | Wins / TotalTrades | % of profitable trades |
| **Expectancy** | (Win% × AvgWin) - (Loss% × AvgLoss) | Average $ per trade |

---

## Security Features

### API Key Encryption

```bash
# Encrypt your API keys
python encrypt_config.py
```

Keys are encrypted using Fernet symmetric encryption and stored in `.env`:
```
BINANCE_API_KEY_ENCRYPTED=gAAAAABk...
BINANCE_API_SECRET_ENCRYPTED=gAAAAABk...
```

### Rate Limiting

Prevents hitting Binance API limits:
```python
from core.security import get_rate_limiter

limiter = get_rate_limiter()
await limiter.acquire('order')  # Wait if limit exceeded
```

### Input Validation

All trade requests validated before execution:
```python
from core.security import TradeRequestValidator

trade = TradeRequestValidator.validate_trade(
    symbol='BTCUSDT',
    side='LONG',
    size=0.02,
    current_price=100000,
    stop_loss=95000
)
```

---

## Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=core --cov-report=html

# Specific test
pytest tests/test_backtest.py
```

### Test Structure

```
tests/
├── test_backtest.py      # Backtest engine tests
├── test_indicators.py    # Indicator calculation tests
├── test_strategies.py    # Strategy logic tests
└── integration/
    └── test_live_data.py # Live data integration
```

---

## Performance Optimizations

### Numba JIT Compilation

High-performance indicators (10-100x faster):

```python
from core.performance import FastIndicators

# Use JIT-compiled versions
ema = FastIndicators.ema(prices, 50)  # 50x faster
atr = FastIndicators.atr(highs, lows, closes, 14)  # 40x faster
```

---

## Production Deployment

### Checklist Before Live Trading

- [ ] Tested strategy on at least 1 year of historical data
- [ ] Walk-forward analysis shows efficiency ratio > 0.7
- [ ] Tested on testnet for at least 1 week without issues
- [ ] Encrypted API keys
- [ ] Set appropriate risk limits (start with 1% risk per trade)
- [ ] Circuit breakers configured
- [ ] Monitoring and alerts set up
- [ ] Understand you can lose money

### Monitoring

Health checks:
```python
from core.monitoring.health_check import HealthChecker

checker = HealthChecker()
await checker.check_all()
checker.print_health_report()
```

---

## Known Issues & Limitations

1. **Backtest Overfitting:** Easy to find parameters that work historically but fail in live trading
2. **Market Regime Changes:** Strategies optimized for one market may fail in another
3. **Lack of Liquidity Modeling:** Assumes unlimited liquidity
4. **No Multi-Timeframe Confirmation:** Strategies use single timeframe
5. **ML Models Need Retraining:** Models decay over time as market evolves

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

---

## Disclaimer

**This software is for educational and research purposes only.**

- No warranty or guarantee of results
- Past performance != future results
- You can lose all your capital
- Always start with paper trading
- Never invest more than you can afford to lose
- The authors take no responsibility for your trading losses

**Use at your own risk.**

---

## License

MIT License - See [LICENSE](LICENSE)

---

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review code examples in modules

---

Built for algorithmic traders who understand that trading bots are tools, not magic money machines.