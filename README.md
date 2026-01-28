# 🤖 CryptoBoss - Professional-Grade Crypto Trading Bot

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/architecture-institutional-purple.svg)]()

**Next-generation crypto trading system designed to think like a professional discretionary trader.**

---

## 🎯 Current Status: Architectural Redesign in Progress

> [!IMPORTANT]
> This project is currently undergoing a major architectural transformation from a signal-based system to a professional discretionary trading architecture. See [Implementation Plan](C:/Users/hardi/.gemini/antigravity/brain/9d4b36f6-903c-4aa7-819d-97cf87489b4b/implementation_plan.md) for details.

**Current Architecture Level**: 7.5/10 (Signal-based with good risk management)  
**Target Architecture Level**: 9.5/10 (Context-first discretionary trader model)

---

## 🏗️ Architecture Overview

### Current System Components

#### Core Decision Flow (Being Redesigned)
```
Current: Strategy Signal → Risk Check → Execution
Target:  Market Context → Bias → Permission → Execution → Management
```

#### Existing Components (Production-Ready)

| Component | Status | Description |
|-----------|--------|-------------|
| **TradingEngine** | ✅ Stable | Main orchestrator with lifecycle management |
| **RiskGuardian** | ✅ Stable | Multi-level risk protection (order/strategy/portfolio) |
| **ExecutionRouter** | ✅ Stable | Unified paper/live execution with realistic slippage |
| **StateManager** | ✅ Stable | Crash-proof SQLite persistence |
| **EventBus** | ✅ Stable | Event-driven communication system |
| **AdvancedRegimeDetector** | ✅ Stable | Market regime classification (ATR, ADX, Hurst) |

#### Components Under Development

| Component | Status | Description |
|-----------|--------|-------------|
| **MarketContextEngine** | 🚧 Planned | Multi-timeframe market context classifier |
| **BiasEngine** | 🚧 Planned | Higher-timeframe directional bias system |
| **TradePermissionFilter** | 🚧 Planned | Pre-execution permission gate |
| **TradeManagementEngine** | 🚧 Planned | Professional position management |
| **DecisionLogger** | 🚧 Planned | Complete decision audit trail |

---

## 📦 Current Features

### Trading Strategies
- **DCA Strategy**: Dollar-cost averaging with safety orders (3Commas-style)
- **Grid Strategy**: Arithmetic/geometric grid trading with dynamic spacing
- **Market Making**: Inventory-aware spread management

### Risk Management
- ✅ Per-order size and value limits
- ✅ Per-strategy allocation and drawdown limits
- ✅ Portfolio-level daily/weekly loss limits
- ✅ Position concentration limits
- ✅ Circuit breakers for consecutive errors
- ✅ Rate limiting (orders per minute)
- ✅ Emergency stop mechanism

### Execution Modes
- **Paper Trading**: Realistic simulation with slippage, fees, latency
- **Live Trading**: Direct exchange integration (Binance)
- **Backtest**: Historical data testing (being enhanced)

### Integrations & Monitoring
- 📊 **Observability**: Structured logging, Prometheus metrics
- 💬 **Notifications**: Telegram, Discord, Slack, Email
- 🔐 **Secrets Management**: Environment, Vault, AWS support
- 🚀 **CI/CD**: GitHub Actions pipeline
- 🎨 **Dashboard**: React-based frontend (in `frontend/`)

---

## 📁 Project Structure

```
d:/projects/final99/
├── src/
│   ├── core/                    # ⭐ Core architecture (16 modules)
│   │   ├── engine.py            # Main trading engine orchestrator
│   │   ├── risk_guardian.py     # Multi-level risk protection
│   │   ├── execution_router.py  # Unified execution abstraction
│   │   ├── state_manager.py     # Crash-proof persistence
│   │   ├── event_bus.py         # Event-driven communication
│   │   ├── config_manager.py    # Configuration management
│   │   ├── secrets_manager.py   # Secure secrets handling
│   │   ├── observability.py     # Logging and metrics
│   │   └── monitoring/          # Alert and metric subsystems
│   ├── strategies/              # Trading strategy implementations
│   │   ├── dca_strategy.py      # DCA with safety orders
│   │   ├── grid_strategy.py     # Grid trading
│   │   ├── market_making.py     # Market maker
│   │   └── regime_selection.py  # Regime-based strategy selection
│   ├── analysis/                # Market analysis (14 modules)
│   │   ├── regime_detector_advanced.py  # Advanced regime detection
│   │   ├── trade_quality_scorer.py      # Trade quality analysis
│   │   ├── multi_timeframe.py           # MTF analysis
│   │   ├── order_flow_analyzer.py       # Order flow metrics
│   │   └── sentiment_analyzer.py        # Sentiment aggregation
│   ├── ml/                      # Machine learning (14 modules)
│   ├── execution/               # Trade execution layer
│   ├── risk/                    # Risk calculation modules
│   ├── data/                    # Data pipeline
│   └── exchange/                # Exchange integrations
├── configs/                     # YAML configuration files
│   ├── dca_config.yaml
│   ├── grid_config.yaml
│   ├── market_making_config.yaml
│   └── onchain_config.yaml
├── tests/                       # Test suite
│   ├── test_production.py       # Production component tests
│   ├── test_dca_strategy.py
│   └── test_indicators.py
├── frontend/                    # React dashboard
├── data/                        # Historical market data
├── logs/                        # Application logs
├── dashboard/                   # Flask API for frontend
├── run_trading_bot.py          # Main entry point
├── run_backtest.py             # Backtesting script
├── setup_binance.py            # Exchange setup utility
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- Binance account (for live trading)

### Installation

```bash
# Clone repository
git clone https://github.com/unknsoul/cryptoboss.git
cd cryptoboss

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Run Paper Trading (Recommended)

```bash
# Default: Paper mode, $10,000 capital, DCA strategy on BTC/USDT
python run_trading_bot.py

# With custom parameters
python run_trading_bot.py --mode=paper --capital=5000 --symbols BTC/USDT ETH/USDT --strategy=dca
```

### Run Backtest

```bash
# Backtest DCA strategy
python run_backtest.py --strategy=dca --capital=10000 --start=2023-01-01 --end=2023-12-31

# With custom config
python run_backtest.py --config=configs/dca_config.yaml
```

### Run Live Trading (Advanced)

> [!CAUTION]
> Only use live trading after extensive paper trading validation (6+ months recommended)

```bash
# Requires --env=prod for safety
python run_trading_bot.py --mode=live --env=prod --capital=1000
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
# Exchange API (required for live mode)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_USER_ID=your_telegram_user_id
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# Environment
ENVIRONMENT=dev  # dev, staging, or prod
```

### Strategy Configuration

See `configs/` directory for strategy-specific YAML files:
- `dca_config.yaml`: DCA strategy parameters
- `grid_config.yaml`: Grid trading settings
- `market_making_config.yaml`: Market maker configuration

---

## 🔌 Binance Testnet Configuration

CryptoBoss supports both Binance Testnet and Live environments. **Always test on testnet first!**

### Getting Testnet API Keys

1. Go to [Binance Testnet](https://testnet.binance.vision/)
2. Log in with your GitHub account
3. Click "Generate HMAC_SHA256 Key"
4. Copy the API Key and Secret Key

### Environment Variables

```env
# Binance API Configuration
BINANCE_API_KEY=your_testnet_or_live_api_key
BINANCE_API_SECRET=your_testnet_or_live_api_secret

# Testnet Mode (set to 'true' for testnet, 'false' for live)
BINANCE_TESTNET_ENABLED=true
```

### Endpoints Used

| Environment | REST Endpoint | WebSocket Endpoint |
|-------------|---------------|-------------------|
| **Testnet** | `https://testnet.binance.vision/api` | `wss://testnet.binance.vision/ws` |
| **Live** | `https://api.binance.com/api` | `wss://stream.binance.com:9443/ws` |

### Testing Your Connection

```python
import asyncio
from src.exchange.binance_client import test_binance_connection

async def main():
    result = await test_binance_connection(testnet=True)
    if result["success"]:
        print(f"✅ Connected to {result['environment']}")
        print(f"   Balances: {result['balances']}")
    else:
        print(f"❌ Failed: {result['message']}")

asyncio.run(main())
```

### Expected Output (Successful)

```
BinanceClient initialized (TESTNET)
  REST endpoint: https://testnet.binance.vision
  WebSocket endpoint: wss://testnet.binance.vision/ws
Credentials validated (testnet). Found 2 assets.
✅ Connected to testnet
   Balances: {'BTC': 1.0, 'USDT': 10000.0}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Invalid signature" | Using testnet keys on live or vice versa | Ensure `BINANCE_TESTNET_ENABLED` matches your key type |
| "API key format invalid" | Malformed or incomplete key | Re-copy the full key from Binance |
| "Timestamp out of recv window" | Clock sync issue | System clock should be accurate (within 10 seconds) |

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suite
```bash
pytest tests/test_production.py -v
```

### Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 📊 Dashboard

The frontend dashboard (React + TypeScript) provides:
- Real-time portfolio overview
- Strategy performance metrics
- Trade history and analysis
- Risk monitoring
- Live price charts

```bash
cd frontend
npm install
npm run dev
```

Access at `http://localhost:3000`

---

## 🎯 Roadmap

### ✅ Completed (Current System)
- [x] Core trading engine architecture
- [x] Multi-level risk management
- [x] Paper/live execution abstraction
- [x] State persistence and recovery
- [x] Event-driven communication
- [x] DCA, Grid, Market Making strategies
- [x] Advanced regime detection
- [x] Telegram/Discord notifications

### 🚧 In Progress (Architectural Redesign)
- [ ] Market Context Engine (multi-timeframe)
- [ ] Bias Engine (higher-timeframe trend)
- [ ] Trade Permission Filter (pre-execution gate)
- [ ] Trade Management Engine (professional exits)
- [ ] Decision Logger (complete audit trail)
- [ ] Strategy adaptation to proposal-based pattern

### 📅 Planned (Future Enhancements)
- [ ] Advanced ML models for trade quality
- [ ] Multi-exchange support (FTX, Coinbase)
- [ ] Options and futures support
- [ ] Portfolio optimization
- [ ] Advanced backtesting engine
- [ ] Web-based configuration UI

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚠️ Risk Disclaimer

**IMPORTANT: READ BEFORE USE**

Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. This software is provided for **educational and research purposes only**.

- Past performance does not guarantee future results
- You may lose all invested capital
- The authors assume no liability for trading losses
- This is NOT financial advice

**Before live trading:**
1. ✅ Paper trade for minimum 6 months
2. ✅ Verify Sharpe Ratio > 1.5
3. ✅ Verify Win Rate > 45%
4. ✅ Verify Max Drawdown < 20%
5. ✅ Start with minimal capital you can afford to lose
6. ✅ Understand all code and risk parameters
7. ✅ Monitor daily and adjust as needed

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🔗 Links

- **Repository**: [https://github.com/unknsoul/cryptoboss](https://github.com/unknsoul/cryptoboss)
- **Issues**: [https://github.com/unknsoul/cryptoboss/issues](https://github.com/unknsoul/cryptoboss/issues)
- **Documentation**: Coming soon

---

## 🙏 Acknowledgments

Built with:
- [ccxt](https://github.com/ccxt/ccxt) - Cryptocurrency exchange library
- [pandas](https://pandas.pydata.org/) - Data analysis
- [numpy](https://numpy.org/) - Numerical computing
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [React](https://reactjs.org/) - Frontend UI

---

**Made with ❤️ for the crypto trading community**