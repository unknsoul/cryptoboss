"""
Centralized Configuration for CryptoBoss Pro
Stores all hardcoded values, risk parameters, and feature toggles.
"""
import os
from pathlib import Path

# === PROJECT PATHS ===
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DB_PATH = ROOT_DIR / "trading_data.db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# === TRADING CONFIGURATION ===
INITIAL_CAPITAL = 10000.0
SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
MAX_POSITIONS = 3  # Max concurrent positions
LEVERAGE = 1  # Spot trading (1x)

# === RISK MANAGEMENT ===
RISK_PER_TRADE_PCT = 0.5        # 0.5% risk per trade (Professional Standard)
MIN_RISK_REWARD_RATIO = 2.5     # Minimum 2.5:1 reward to risk
MAX_DRAWDOWN_PCT = 10.0         # Max portfolio drawdown allowed
DAILY_LOSS_LIMIT_PCT = 5.0      # Max daily loss allowed
WEEKLY_LOSS_LIMIT_PCT = 10.0    # Max weekly loss allowed
MARKET_TIMEZONE = "America/New_York"  # Market timezone for loss tracking
MARKET_OPEN_TIME = "09:30"      # Market open time for daily reset

# === STRATEGY SETTINGS ===
MIN_CONFIDENCE_THRESHOLD = 0.50 # 50% min confidence for trade execution
STOP_LOSS_ATR_MULTIPLIER = 1.0  # 1.0x ATR for Stop Loss
TRAILING_ACTIVATION_RISK_MULT = 0.75 # Activate trailing stop at 0.75x Risk

# === FEATURE TOGGLES ===
ENABLE_ML = True
ENABLE_REGIME_DETECTION = True
ENABLE_SENTIMENT_ANALYSIS = True
ENABLE_DASHBOARD = True
ENABLE_DAILY_LOSS_LIMITS = True  # Enable daily/weekly loss tracking

# === SYSTEM SETTINGS ===
LOG_LEVEL = "INFO"
DB_URL = f"sqlite:///{DB_PATH}"

# === API CONFIGURATION ===
# Load from environment variables in production
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
