# Binance API Configuration Quick Start

## Setup Instructions

### 1. Create Configuration File

```bash
# Copy the example file
cp .env.example .env
```

### 2. Choose Environment

Edit `.env` and set:

```bash
# For testing (RECOMMENDED)
BINANCE_ENV=testnet

# For real trading (USE WITH CAUTION)
# BINANCE_ENV=mainnet
```

### 3. Get API Keys

#### Testnet (Safe for Testing)
1. Go to: https://testnet.binance.vision/
2. Login with GitHub
3. Generate API keys
4. Copy to `.env`:
   ```
   BINANCE_TESTNET_API_KEY=your_key_here
   BINANCE_TESTNET_SECRET_KEY=your_secret_here
   ```

#### Mainnet (Real Money)
1. Go to: https://www.binance.com/
2. Account → API Management
3. Create API key
4. **Enable only "Enable Reading"** for paper trading
5. Copy to `.env`:
   ```
   BINANCE_MAINNET_API_KEY=your_key_here
   BINANCE_MAINNET_SECRET_KEY=your_secret_here
   ```

### 4. Configure Trading Parameters

In `.env`:

```bash
# Trading Settings
INITIAL_CAPITAL=10000
RISK_PER_TRADE=0.02
SYMBOL=btcusdt
TIMEFRAMES=1h

# Safety Limits
MAX_DRAWDOWN_LIMIT=0.20
DAILY_LOSS_LIMIT=0.05
```

### 5. Run

```bash
python live_paper_trader.py
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BINANCE_ENV` | testnet or mainnet | testnet |
| `INITIAL_CAPITAL` | Starting capital | 10000 |
| `RISK_PER_TRADE` | Risk per trade (2% = 0.02) | 0.02 |
| `SYMBOL` | Trading pair | btcusdt |
| `TIMEFRAMES` | Comma-separated (e.g., 1h,4h) | 1h |
| `MAX_DRAWDOWN_LIMIT` | Max drawdown before halt | 0.20 |
| `DAILY_LOSS_LIMIT` | Max daily loss | 0.05 |

### Testing Timeframes

For faster testing, use shorter timeframes:

```bash
# Fast testing (1-minute candles)
TIMEFRAMES=1m

# Standard (1-hour candles)
TIMEFRAMES=1h

# Multi-timeframe
TIMEFRAMES=4h,1h,15m
```

## Safety Notes

⚠️ **Always test on testnet first!**
- Testnet uses fake money
- Same market data as mainnet
- Perfect for validation

🚨 **Mainnet warnings:**
- Real money at risk
- Start with small capital
- Monitor closely
- Use stop losses

## Example Console Output

```
============================================================
📡 BINANCE CONFIGURATION
============================================================
Environment:  TESTNET
WebSocket:    wss://testnet.binance.vision/ws
REST API:     https://testnet.binance.vision/api
API Key:      ABCD1234...XYZ9 ✅

⚠️  TESTNET MODE - Safe for testing
============================================================
```

## Troubleshooting

### "API keys not configured"
- WebSocket still works (no keys needed for price data)
- Only need keys for actual order execution
- For paper trading, keys are optional

### "Invalid BINANCE_ENV"
- Must be exactly `testnet` or `mainnet`
- Check for typos in `.env`

### Connection errors
- Check internet connection
- Verify Binance is accessible
- Try switching between testnet/mainnet
