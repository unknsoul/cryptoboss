# 🤖 CryptoBoss - Crypto Trading Scalper Bot

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-development-orange.svg)]()

**Intraday crypto scalping system with Smart Money Concepts + technical indicators.**

---

## 🎯 Current Status: In Development

> [!NOTE]
> This project is being built as a professional-grade crypto trading bot.
> Currently supports paper trading (simulated) and live/testnet trading via Binance.

---

## 🏗️ Architecture Overview

### Decision Flow
```
Market Data → Signal Engine (RSI/EMA/ATR + SMC) → Risk Guardian → Execution Router → Exchange
```

### Core Components

| Component | Status | Description |
|-----------|--------|-------------|
| **TradingEngine** | ✅ Working | Main orchestrator with lifecycle management |
| **RiskGuardian** | ✅ Working | Multi-level risk protection (order/strategy/portfolio) |
| **ExecutionRouter** | ✅ Working | Paper/Testnet/Live execution with slippage simulation |
| **StateManager** | ✅ Working | Crash-proof SQLite persistence with WAL mode |
| **EventBus** | ✅ Working | Event-driven communication with persistence |
| **SignalEngine** | ✅ Working | RSI + EMA + ATR + SMC composite scoring |
| **BotInstance** | ✅ Working | Per-account isolated bot with reconnection |
| **MockPriceGenerator** | ✅ Working | Realistic price simulation for paper trading |

---

## 📦 Features

### Trading Strategies
- **DCA Strategy**: Dollar-cost averaging with safety orders
- **SMC Scalper**: Smart Money Concepts based scalping
- **SMC Trend Follow**: Trend-following with structure analysis
- **Range Scalp**: Range-bound scalping

### Signal Engine
- RSI (14-period) oversold/overbought detection
- EMA crossover (9/21) trend direction
- ATR volatility filter (avoids low-vol environments)
- Smart Money: BOS, Order Blocks, FVG, Liquidity Sweeps

### Risk Management
- ✅ Per-trade risk limit (default 2% of portfolio)
- ✅ Max concurrent trades (default 5)
- ✅ Daily/weekly loss limits
- ✅ Per-strategy allocation limits
- ✅ Circuit breaker for consecutive errors
- ✅ Rate limiting (orders per minute)
- ✅ Emergency stop mechanism

### Execution Modes
- **Paper Trading**: Simulated execution with slippage (no API keys needed)
- **Testnet**: Real orders on Binance Testnet
- **Live Trading**: Real orders on Binance Mainnet

---

## 📁 Project Structure

```
cryptoboss/
├── src/
│   ├── core/                    # Core architecture
│   │   ├── engine.py            # Main trading engine
│   │   ├── risk_guardian.py     # Risk protection
│   │   ├── execution_router.py  # Order execution
│   │   ├── state_manager.py     # Crash-proof persistence
│   │   ├── event_bus.py         # Event-driven communication
│   │   ├── bot_instance.py      # Per-account bot isolation
│   │   ├── mock_price_generator.py  # Paper trading prices
│   │   └── auth/                # Account/key management
│   ├── strategies/              # Trading strategies
│   ├── v3/                      # Signal engine + SMC modules
│   ├── exchange/                # Binance client integration
│   └── risk/                    # Risk calculation modules
├── configs/                     # YAML configuration files
├── tests/                       # Test suite
├── run_trading_bot.py           # Main entry point
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
git clone https://github.com/unknsoul/cryptoboss.git
cd cryptoboss
pip install -r requirements.txt
```

### Run Paper Trading (No API keys needed)

```bash
# Default: Paper mode, $10,000 capital, DCA strategy on BTC/USDT
python run_trading_bot.py

# Custom parameters
python run_trading_bot.py --capital=5000 --symbols BTC/USDT ETH/USDT --strategy=smc_scalper
```

### Run Live Trading

> [!CAUTION]
> Only use live trading after extensive paper trading validation.

```bash
# 1. Set up API keys
cp .env.example .env
# Edit .env with your Binance API keys

# 2. Run with safety flag
python run_trading_bot.py --mode=live --env=prod --capital=1000
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
# Exchange API (required for live/testnet mode)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_USER_ID=your_telegram_user_id
```

---

## 🔌 Binance Testnet

| Environment | REST Endpoint | WebSocket Endpoint |
|-------------|---------------|-------------------|
| **Testnet** | `https://testnet.binance.vision/api` | `wss://testnet.binance.vision/ws` |
| **Live** | `https://api.binance.com/api` | `wss://stream.binance.com:9443/ws` |

### Getting Testnet Keys
1. Go to [Binance Testnet](https://testnet.binance.vision/)
2. Log in with GitHub
3. Generate HMAC_SHA256 Key

---

## 🧪 Testing

```bash
pytest tests/ -v
pytest tests/test_system_integration.py -v
```

---

## ⚠️ Risk Disclaimer

Trading cryptocurrencies involves substantial risk. This software is for **educational and research purposes only**.

- Past performance does not guarantee future results
- You may lose all invested capital
- This is NOT financial advice

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for the crypto trading community**