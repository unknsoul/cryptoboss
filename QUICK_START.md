# Quick Start Guide 🚀

## Get Your Trading Bot Running in 10 Minutes

---

## Step 1: Install Dependencies (2 minutes)

```bash
# Make sure you have Python 3.9+ installed
python --version

# Install all requirements
pip install -r requirements.txt
```

**Expected output**: All packages install successfully without errors.

---

## Step 2: Configure Environment (3 minutes)

```bash
# Copy the example environment file
cp .env.example .env
```

Edit `.env` file (use any text editor):

```env
# IMPORTANT: Start with testnet!
USE_TESTNET=true

# Get your Binance testnet API keys from: https://testnet.binance.vision/
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_api_secret_here

# Optional: Google Gemini API for sentiment analysis
# Get free key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Slack alerts (leave empty if not using)
SLACK_WEBHOOK=

# Optional: Discord alerts (leave empty if not using)
DISCORD_WEBHOOK=
```

### 🔑 Getting API Keys

#### Binance Testnet (Safe for Testing)
1. Go to https://testnet.binance.vision/
2. Click "Generate HMAC_SHA256 Key"
3. Save your API Key and Secret
4. Transfer some test funds to your account

#### Binance Mainnet (For Live Trading - USE WITH CAUTION)
1. Go to https://www.binance.com/en/my/settings/api-management
2. Create new API key
3. Enable "Enable Spot & Margin Trading"
4. **IMPORTANT**: Disable "Enable Withdrawals" for safety
5. Set IP restrictions if possible

---

## Step 3: Test Configuration (1 minute)

```bash
python -c "from core.config import get_settings; settings = get_settings(); print('✅ Configuration loaded successfully')"
```

**Expected output**: 
```
✅ Configuration loaded successfully
```

---

## Step 4: Run Backtest (2 minutes)

```bash
python run_backtest.py
```

**Expected output**: 
- Backtest results with Sharpe ratio, win rate, etc.
- Chart saved to `professional_strategy_results.png`

---

## Step 5: Launch Dashboard (1 minute)

```bash
streamlit run dashboard/app.py
```

**Expected output**:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Open your browser and go to: **http://localhost:8501**

---

## Step 6: Test Components (1 minute)

### Test Logging
```python
python -c "from core.monitoring.logger import get_logger; logger = get_logger(); logger.info('Test message'); print('✅ Logging works')"
```

### Test Mock Exchange
```bash
python core/testing/mock_exchange.py
```

### Test Alerts
```bash  
python core/monitoring/alerting.py
```

---

## 🎯 What's Next?

### **For Paper Trading**
```python
from live_trader import LiveTrader

# Start with testnet
trader = LiveTrader(
    symbol="BTCUSDT",
    strategy_name="enhanced_momentum",
    capital=10000
)

trader.start()
```

### **For Validation**
```python
from tests.backtest.walk_forward import WalkForwardValidator

# Test your strategy
validator = WalkForwardValidator()
results = validator.validate(highs, lows, closes, strategy)
print(f"Overfitting Detected: {results['overfitting_detected']}")
```

---

## 🛡️ Safety Checklist

Before going live, make sure:

- [ ] Tested on testnet for at least 1 week
- [ ] Walk-forward validation shows no overfitting
- [ ] All alerts are configured and working
- [ ] Dashboard shows expected behavior
- [ ] Logs are being recorded properly
- [ ] Position reconciliation is enabled
- [ ] Circuit breakers are set appropriately
- [ ] Starting with small capital ($500-$1000)
- [ ] API keys have withdrawal disabled
- [ ] You understand the risks involved

---

## 📊 Monitor Your Bot

### Check Logs
```bash
# Main logs
tail -f logs/main.log

# Trade logs
tail -f logs/trades.log

# Error logs
tail -f logs/errors.log
```

### View Metrics
```python
from core.monitoring.metrics import get_metrics

metrics = get_metrics()
stats = metrics.get_trading_stats()
print(stats)
```

### Check Alerts
Alerts will be sent to configured channels (Email/Slack/Discord) when:
- Daily loss exceeds 3%
- Circuit breaker triggers
- Position stuck for > 5 minutes
- API issues detected

---

## 🆘 Troubleshooting

### "Module not found" Error
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### "Configuration Error"
```bash
# Check your .env file exists
ls -la .env

# Validate configuration
python core/config/settings.py
```

### "Exchange Connection Failed"
- Check your API keys are correct
- Verify you're using testnet keys with `USE_TESTNET=true`
- Check your internet connection
- Verify Binance API status: https://www.binance-status.com/

### Dashboard Won't Start
```bash
# Check streamlit is installed
pip install streamlit --upgrade

# Try running on different port
streamlit run dashboard/app.py --server.port=8502
```

---

## 📚 Learn More

- **Full Documentation**: See [README.md](README.md)
- **Upgrade Details**: See [UPGRADE_SUMMARY.md](UPGRADE_SUMMARY.md)
- **Strategy Guide**: See `core/strategies/` directory
- **Configuration Options**: See `core/config/settings.py`

---

## 🎉 Success!

If you've completed all steps, you now have:

✅ A working trading bot  
✅ Professional dashboard  
✅ Comprehensive monitoring  
✅ Advanced execution  
✅ Safety features enabled

**Happy Trading! 📈**

---

*Remember: Always start with paper trading and never risk more than you can afford to lose.*
