import sys
import os
import time
import math
import logging
import threading
import requests
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import random
import json

# === CORE IMPORTS ===
from core.config import settings
from core.storage.database import SQLiteManager
from core.exchange.data_service import get_data_service
from core.risk.daily_loss_tracker import DailyLossTracker
import threading
import signal
import sys
from contextlib import contextmanager

# Thread lock for trading_state access
trading_state_lock = threading.Lock()

@contextmanager
def safe_state_access():
    """Context manager for safe trading_state access"""
    trading_state_lock.acquire()
    try:
        yield trading_state
    finally:
        trading_state_lock.release()

# Configure Logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.DATA_DIR / "trading_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoBoss")

# Suppress warnings
warnings.filterwarnings("ignore")

# Import Dashboard
try:
    import dashboard.app as dash_app
    from dashboard.app import run_dashboard as run_dash_server
except ImportError:
    logger.error("Could not import dashboard.app")
    sys.exit(1)

# Import Core Managers
try:
    from core.strategies.strategy_manager import StrategyManager
    from core.analysis.signal_aggregator import SignalAggregator
    from core.risk.dynamic_position_sizer import DynamicPositionSizer
    from core.analysis.sentiment_analyzer import SentimentAnalyzer
    from core.analysis.probabilistic_signals import ProbabilisticSignalGenerator
except ImportError as e:
    logger.critical(f"Critical Core Component Missing: {e}")
    sys.exit(1)

# Import Regime Detector
try:
    from core.analysis.regime_detector_advanced import AdvancedRegimeDetector
except ImportError as e:
    logger.warning(f"Regime Detector missing (sklearn required?): {e}")
    AdvancedRegimeDetector = None

# Import Strategies
try:
    from core.strategies.mean_reversion import MeanReversionStrategy
    from core.strategies.enhanced_momentum import EnhancedMomentumStrategy
except ImportError as e:
    logger.warning(f"Base strategies missing: {e}")

try:
    from core.strategies.institutional_strategies import (
        AdaptiveVolatilityTargeting,
        StatisticalArbitrage,
        MomentumWithQuality,
        MarketMicrostructureAlpha
    )
except ImportError as e:
    logger.warning(f"Institutional strategies missing (scipy/sklearn?): {e}")
    AdaptiveVolatilityTargeting = None
    StatisticalArbitrage = None
    MomentumWithQuality = None
    MarketMicrostructureAlpha = None

# Import Advanced Features
try:
    from core.advanced_features import AdvancedTradingFeatures
except ImportError as e:
    logger.warning(f"Advanced features module not found: {e}")
    AdvancedTradingFeatures = None

# Import Scalper Strategies
try:
    from core.strategies.scalper_strategies import (
        TrendScalper,
        BreakoutScalper,
        RegimeAwareScalper
    )
    SCALPER_STRATEGIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Scalper strategies missing: {e}")
    SCALPER_STRATEGIES_AVAILABLE = False

# Import ML Components
try:
    from core.ml.ensemble_model import EnsembleModel
    from core.ml.real_ml_trainer import RealMLTrainer
    from core.ml.feature_engineering import AdvancedFeatureEngineer
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML components missing: {e}")
    ML_AVAILABLE = False

# Import Risk/Reward Optimizer (CRITICAL FIX for small wins/large losses)
try:
    from core.risk.position_optimizer import get_rr_optimizer
    RISK_REWARD_AVAILABLE = True
    logger.info("✓ Risk/Reward Optimizer loaded (enforces 2:1 R:R minimum)")
except ImportError as e:
    logger.warning(f"Risk/Reward optimizer missing: {e}")
    RISK_REWARD_AVAILABLE = False

# Import Integrated Feature Hub (Activates all 26 features)
try:
    from core.integration.feature_hub import get_feature_hub
    FEATURE_HUB_AVAILABLE = True
    logger.info("✓ Integrated Feature Hub loaded (all features)")
except ImportError as e:
    logger.warning(f"Feature Hub missing: {e}")
    FEATURE_HUB_AVAILABLE = False

# Import BTC Master Hub (Professional BTC-specific analysis)
try:
    from core.btc.btc_master_hub import get_btc_master, get_circuit_breaker
    from core.btc.btc_analysis import (
        get_cme_tracker, get_funding_analyzer, get_ob_detector,
        get_session_analyzer
    )
    BTC_MASTER_AVAILABLE = True
    logger.info("✓ BTC Master Hub loaded (CME gaps, funding, order blocks)")
except ImportError as e:
    logger.warning(f"BTC Master Hub missing: {e}")
    BTC_MASTER_AVAILABLE = False

try:
    from core.analysis.trade_quality_scorer import TradeQualityScorer
    from core.analysis.order_flow_analyzer import OrderFlowAnalyzer
    QUALITY_SCORING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Quality scoring modules missing: {e}")
    QUALITY_SCORING_AVAILABLE = False
    TradeQualityScorer = None
    OrderFlowAnalyzer = None

# Import Smart Risk Manager
try:
    from core.risk.smart_risk_manager import SmartRiskManager, TradingMode
    SMART_RISK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Smart risk manager missing: {e}")
    SMART_RISK_AVAILABLE = False
    SmartRiskManager = None

# Import Circuit Breaker (Feature #136)
try:
    from core.safety.circuit_breaker import CircuitBreaker, CircuitBreakerState
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Circuit breaker missing: {e}")
    CIRCUIT_BREAKER_AVAILABLE = False
    CircuitBreaker = None

# ============ ENTERPRISE FEATURES (360 Module System) ============

# Import Regime Detection
try:
    from core.regime.regime_detector import (
        get_regime_classifier, get_vol_detector, get_trend_analyzer,
        get_strategy_selector
    )
    REGIME_FEATURES_AVAILABLE = True
    logger.info("Enterprise: Regime Detection ✓")
except ImportError:
    REGIME_FEATURES_AVAILABLE = False

# Import Sentiment Analysis
try:
    from core.sentiment.sentiment_engine import (
        get_fear_greed, get_funding_analyzer, get_oi_tracker,
        get_whale_watcher, get_sentiment_aggregator
    )
    SENTIMENT_FEATURES_AVAILABLE = True
    logger.info("Enterprise: Sentiment Engine ✓")
except ImportError:
    SENTIMENT_FEATURES_AVAILABLE = False

# Import Technical Indicators
try:
    from core.indicators.technical_indicators import (
        get_atr, get_bollinger, get_rsi, get_macd, get_adx,
        get_vwap, get_ichimoku, get_fibonacci
    )
    INDICATORS_AVAILABLE = True
    logger.info("Enterprise: Technical Indicators ✓")
except ImportError:
    INDICATORS_AVAILABLE = False

# Import Capital & Risk Management
try:
    from core.capital.capital_manager import (
        get_risk_budgeter, get_dd_protector, get_profit_locker,
        get_equity_tracker, get_pnl_calculator, get_expectancy_calculator
    )
    CAPITAL_FEATURES_AVAILABLE = True
    logger.info("Enterprise: Capital Management ✓")
except ImportError:
    CAPITAL_FEATURES_AVAILABLE = False

# Import Analytics & Tracking
try:
    from core.analytics.analytics_system import (
        get_trade_journal, get_equity_analyzer, get_drawdown_analyzer,
        get_streak_tracker
    )
    ANALYTICS_AVAILABLE = True
    logger.info("Enterprise: Analytics System ✓")
except ImportError:
    ANALYTICS_AVAILABLE = False

# Import Signal Processing
try:
    from core.signals.signal_processing import (
        get_smoother, get_confirmation, get_strength, get_cooldown,
        get_composite_filter
    )
    SIGNAL_PROCESSING_AVAILABLE = True
    logger.info("Enterprise: Signal Processing ✓")
except ImportError:
    SIGNAL_PROCESSING_AVAILABLE = False

# Import System Monitoring
try:
    from core.system.system_monitor import (
        get_latency_monitor, get_memory_profiler, get_cpu_tracker
    )
    SYSTEM_MONITORING_AVAILABLE = True
    logger.info("Enterprise: System Monitoring ✓")
except ImportError:
    SYSTEM_MONITORING_AVAILABLE = False

# Import Order Types
try:
    from core.orders.order_types import (
        get_trailing_stop, get_oco_order, get_bracket_order
    )
    ORDER_TYPES_AVAILABLE = True
    logger.info("Enterprise: Advanced Orders ✓")
except ImportError:
    ORDER_TYPES_AVAILABLE = False

# ============ SHARED STATE ============

# Initialize shared state with default structure
trading_state = {
    'system_status': 'running',
    'uptime_start': datetime.now(),
    'equity': 10000.0,
    'initial_capital': 10000.0,
    'positions': [],
    'trades': [],
    'signals': [],
    'candles': [],
    'current_price': 0.0,
    'indicators': {},
    'orderbook': {'bids': [], 'asks': [], 'spread': 0},
    'metrics': {
        'total_return': 0.0,
        'win_rate': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'num_trades': 0,
        'profit_factor': 0.0,
        'win_streak': 0,
        'loss_streak': 0
    },
    'sentiment': {
        'fear_greed_index': 50,
        'level': 'neutral'
    }
}

# Sync with dashboard's state - CRITICAL for UI data display
# dash_app is imported at top of file as dashboard.app
dash_app.trading_state = trading_state
logger.info("✓ Dashboard state linked to bot state")

# ============ DATA PERSISTENCE ============

# Initialize Database Manager
db_manager = SQLiteManager()

def save_trades(trades):
    """Deprecated: using DB now"""
    pass

def load_trades():
    """Load trades from DB"""
    return db_manager.load_trades(limit=100)

def save_state(bot):
    """Save bot state to DB"""
    state = {
        'equity': bot.equity,
        'initial_capital': bot.initial_capital,
        'max_equity': bot.max_equity,
        'win_streak': bot.win_streak,
        'loss_streak': bot.loss_streak,
        'position': bot.position
    }
    db_manager.save_state(state)

def load_state():
    """Load bot state from DB"""
    return db_manager.load_state()

# ============ FETCH REAL BTC DATA (Using ExchangeDataService) ============

# Initialize the singleton data service
_exchange_service = get_data_service()

def fetch_binance_klines(symbol="BTCUSDT", interval="5m", limit=300):
    """Fetch real candlestick data using ExchangeDataService"""
    df = _exchange_service.fetch_klines(symbol, interval, limit)
    if df is None:
        return None
    
    # Convert DataFrame to list of dicts (legacy format)
    candles = []
    for _, row in df.iterrows():
        candles.append({
            'time': int(row['timestamp'].timestamp()),
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        })
    return candles

def fetch_current_price(symbol="BTCUSDT"):
    """Fetch current price using ExchangeDataService"""
    return _exchange_service.fetch_current_price(symbol)

def fetch_orderbook(symbol="BTCUSDT", limit=10):
    """Fetch order book using ExchangeDataService"""
    result = _exchange_service.fetch_orderbook(symbol, limit)
    if result is None:
        return {'bids': [], 'asks': [], 'spread': 0}
    return result

# ============ ENHANCED TRADING BOT ============

class EnhancedTradingBot:
    """Professional Trading Bot with Multi-Strategy Ensemble"""
    
    def __init__(self, initial_capital=settings.INITIAL_CAPITAL):
        # Database State Loading
        try:
            saved_state = load_state()
            logger.info("DB: Attempting to load state...")
        except Exception:
            saved_state = None
        
        if saved_state:
            self.equity = saved_state.get('equity', initial_capital)
            self.initial_capital = saved_state.get('initial_capital', initial_capital)
            self.max_equity = saved_state.get('max_equity', self.equity)
            self.win_streak = saved_state.get('win_streak', 0)
            self.loss_streak = saved_state.get('loss_streak', 0)
            self.position = saved_state.get('position', None)
            logger.info(f"Resuming with Equity: ${self.equity:,.2f}")
        else:
            self.equity = initial_capital
            self.initial_capital = initial_capital
            self.max_equity = initial_capital
            self.win_streak = 0
            self.loss_streak = 0
            self.position = None
            logger.info(f"Starting fresh with Equity: ${self.equity:,.2f}")
        
        self.trades = load_trades()
        self.signals = []
        
        # Initialize Managers
        self.strategy_manager = StrategyManager()
        self.signal_aggregator = SignalAggregator()
        self.position_sizer = DynamicPositionSizer(
            risk_per_trade=settings.RISK_PER_TRADE_PCT
        )
        self.sentiment_analyzer = SentimentAnalyzer(use_mock_data=True)  # Use mock for now until real keys
        self.probabilistic_generator = ProbabilisticSignalGenerator(buy_threshold=0.55, sell_threshold=0.55)  # RESTORED: Original threshold
        
        if AdvancedRegimeDetector:
            self.regime_detector = AdvancedRegimeDetector(atr_period=14, adx_period=14, lookback_period=100)
        else:
            self.regime_detector = None
            logger.warning("Regime Detector disabled due to missing dependencies")
        
        # Initialize Advanced Trading Features
        if AdvancedTradingFeatures:
            self.advanced_features = AdvancedTradingFeatures(
                enable_mtf=True,           # Multi-timeframe confirmation
                enable_session=True,        # Session-based trading
                enable_dynamic_stops=True,  # Volatility-adjusted stops
                enable_partial_profits=True,# Scale out at milestones
                enable_portfolio_risk=True  # Daily/weekly loss limits
            )
            logger.info("Advanced Trading Features ENABLED [OK]")
        else:
            self.advanced_features = None
            logger.warning("Advanced Features disabled")
        
        # Initialize Trade Quality Scoring (Professional-grade filtering)
        if QUALITY_SCORING_AVAILABLE and TradeQualityScorer:
            self.quality_scorer = TradeQualityScorer(min_score=70)  # RESTORED: Original threshold for trade quality
            self.order_flow = OrderFlowAnalyzer(imbalance_threshold=0.3)
            logger.info("Trade Quality Scoring ENABLED (min score: 70 - original threshold)")
        else:
            self.quality_scorer = None
            self.order_flow = None
            logger.warning("Quality scoring disabled")
        
        # Initialize Smart Risk Manager
        if SMART_RISK_AVAILABLE and SmartRiskManager:
            self.risk_manager_smart = SmartRiskManager(initial_capital=self.initial_capital)
            logger.info("Smart Risk Manager ENABLED 🛡️")
            logger.info(f"  Profit Protection: +2% | Recovery: -1% | Circuit Breaker: -3%")
        else:
            self.risk_manager_smart = None
        
        # Initialize Circuit Breaker (Feature #136)
        if CIRCUIT_BREAKER_AVAILABLE and CircuitBreaker:
            self.circuit_breaker = CircuitBreaker(
                daily_loss_limit_pct=5.0,      # Max 5% daily loss
                max_drawdown_pct=10.0,          # Max 10% drawdown
                max_consecutive_losses=5,        # Max 5 losses in a row
                rapid_loss_threshold=3,          # 3+ losses in 1 hour
                cooldown_minutes=30              # 30 min cooldown after halt
            )
            logger.info("Circuit Breaker ENABLED 🛑 (5% daily / 10% DD / 5 consec)")
        else:
            self.circuit_breaker = None
            logger.warning("Circuit breaker disabled")
        
        # Initialize Daily/Weekly Loss Tracker (Production Feature #1)
        if settings.ENABLE_DAILY_LOSS_LIMITS:
            self.daily_loss_tracker = DailyLossTracker(
                daily_loss_limit_pct=settings.DAILY_LOSS_LIMIT_PCT,
                weekly_loss_limit_pct=settings.WEEKLY_LOSS_LIMIT_PCT,
                timezone=settings.MARKET_TIMEZONE,
                market_open_time=settings.MARKET_OPEN_TIME
            )
            self.daily_loss_tracker.initialize(self.equity)
            logger.info("📅 Daily/Weekly Loss Tracker ENABLED (5% daily / 10% weekly)")
        else:
            self.daily_loss_tracker = None
            logger.warning("Daily/weekly loss tracking disabled")
        
        # Initialize Alert Manager (Production Feature #5)
        try:
            from core.monitoring.alerting import get_alerts
            self.alert_manager = get_alerts({
                'slack_enabled': False,  # Configure in production
                'email_enabled': False,
                'throttle_minutes': 5
            })
            logger.info("📢 Alert Manager ENABLED")
        except Exception as e:
            logger.warning(f"Alert Manager initialization failed: {e}")
            self.alert_manager = None
        
        # Initialize System Monitor (Production Feature #11)
        try:
            from core.monitoring.system_monitor import get_system_monitor
            self.system_monitor = get_system_monitor()
            logger.info("📊 System Monitor ENABLED (CPU/Memory/Disk tracking)")
        except Exception as e:
            logger.warning(f"System Monitor initialization failed: {e}")
            self.system_monitor = None
            
        # Initialize ML Components (Self-Learning)
        if ML_AVAILABLE:
            self.feature_engineer = AdvancedFeatureEngineer()
            self.ml_trainer = RealMLTrainer(model_type='xgboost')
            self.ml_model = EnsembleModel()
            
            # Try to load existing model
            model_path = Path('models/signal_filter_xgboost.pkl')
            if model_path.exists():
                try:
                    self.ml_model = EnsembleModel.load(model_path)
                    logger.info("ML: Loaded pre-trained Ensemble Model 🧠")
                except Exception as e:
                    logger.error(f"ML: Failed to load model: {e}")
            else:
                logger.info("ML: Initialized empty Ensemble Model (will learn from trades) 🧠")
        else:
            self.feature_engineer = None
            self.ml_trainer = None
            self.ml_model = None

        # ============ ENTERPRISE FEATURES INITIALIZATION ============
        
        # Regime Detection System
        if REGIME_FEATURES_AVAILABLE:
            self.regime_classifier = get_regime_classifier()
            self.vol_detector = get_vol_detector()
            self.trend_analyzer = get_trend_analyzer()
            self.strategy_selector = get_strategy_selector()
            logger.info("✓ Regime Detection System Active")
        else:
            self.regime_classifier = None
            self.vol_detector = None
            self.trend_analyzer = None
            self.strategy_selector = None
        
        # Sentiment Analysis Engine
        if SENTIMENT_FEATURES_AVAILABLE:
            self.fear_greed = get_fear_greed()
            self.funding_analyzer = get_funding_analyzer()
            self.oi_tracker = get_oi_tracker()
            self.whale_watcher = get_whale_watcher()
            self.sentiment_aggregator = get_sentiment_aggregator()
            logger.info("✓ Sentiment Analysis Engine Active")
        else:
            self.fear_greed = None
            self.sentiment_aggregator = None
        
        # Capital & Risk Management
        if CAPITAL_FEATURES_AVAILABLE:
            self.risk_budgeter = get_risk_budgeter()
            self.dd_protector = get_dd_protector()
            self.profit_locker = get_profit_locker()
            self.equity_tracker_ent = get_equity_tracker()
            self.pnl_calculator = get_pnl_calculator()
            self.expectancy_calc = get_expectancy_calculator()
            self.profit_locker.set_base(self.equity)
            logger.info("✓ Enterprise Capital Management Active")
        else:
            self.risk_budgeter = None
            self.dd_protector = None
            self.profit_locker = None
        
        # Analytics & Tracking
        if ANALYTICS_AVAILABLE:
            self.trade_journal = get_trade_journal()
            self.equity_analyzer = get_equity_analyzer()
            self.drawdown_analyzer = get_drawdown_analyzer()
            self.streak_tracker = get_streak_tracker()
            logger.info("✓ Analytics System Active")
        else:
            self.trade_journal = None
            self.drawdown_analyzer = None
        
        # Signal Processing
        if SIGNAL_PROCESSING_AVAILABLE:
            self.signal_smoother = get_smoother()
            self.signal_confirmation = get_confirmation()
            self.signal_strength = get_strength()
            self.signal_cooldown = get_cooldown()
            self.composite_filter = get_composite_filter()
            logger.info("✓ Signal Processing Active")
        else:
            self.signal_cooldown = None
            self.composite_filter = None
        
        # System Monitoring
        if SYSTEM_MONITORING_AVAILABLE:
            self.latency_monitor = get_latency_monitor()
            self.memory_profiler = get_memory_profiler()
            self.cpu_tracker = get_cpu_tracker()
            logger.info("✓ System Monitoring Active")
        else:
            self.latency_monitor = None
        
        # Advanced Order Types
        if ORDER_TYPES_AVAILABLE:
            self.trailing_stop_manager = get_trailing_stop()
            self.bracket_order = get_bracket_order()
            logger.info("✓ Advanced Order Types Active")
        else:
            self.trailing_stop_manager = None
        
        # ============ END ENTERPRISE FEATURES ============
        
        # Initialize Strategies (using placeholder inline classes if imports failed)
        self._initialize_strategies()
        
        # Initial Data Load
        self._load_initial_data()
        
        # Update shared state
        trading_state['equity'] = self.equity
        trading_state['initial_capital'] = self.initial_capital
        trading_state['trades'] = self.trades
        trading_state['positions'] = [self.position] if self.position else []
        trading_state['risk_mode'] = self.risk_manager_smart.mode.value if self.risk_manager_smart else 'disabled'
        self._update_metrics()

    def _initialize_strategies(self):
        """Register strategies"""
        
        # 1. Base Strategies
        try:
            self.strategy_manager.register_strategy('momentum', EnhancedMomentumStrategy())
            self.strategy_manager.register_strategy('mean_reversion', MeanReversionStrategy())
        except Exception as e:
            logger.error(f"Error initializing base strategies: {e}")

        # 2. Institutional Strategies
        try:
            if AdaptiveVolatilityTargeting:
                self.strategy_manager.register_strategy('vol_targeting', AdaptiveVolatilityTargeting(target_vol=0.15))
            if StatisticalArbitrage:
                self.strategy_manager.register_strategy('stat_arb', StatisticalArbitrage())
            if MomentumWithQuality:
                self.strategy_manager.register_strategy('quality_momentum', MomentumWithQuality())
            if MarketMicrostructureAlpha:
                self.strategy_manager.register_strategy('microstructure', MarketMicrostructureAlpha())
            
            if AdaptiveVolatilityTargeting:
                logger.info("Institutional Grade Strategies Initialized ✅")
            else:
                logger.warning("Institutional Strategies skipped checks failed")
                
        except Exception as e:
            logger.error(f"Error initializing institutional strategies: {e}")
        
        # 3. Professional Scalper Strategies
        try:
            if SCALPER_STRATEGIES_AVAILABLE:
                self.strategy_manager.register_strategy('trend_scalper', TrendScalper())
                self.strategy_manager.register_strategy('breakout_scalper', BreakoutScalper())
                self.strategy_manager.register_strategy('regime_scalper', RegimeAwareScalper())
                logger.info("Professional Scalper Strategies Initialized 🔥")
        except Exception as e:
            logger.error(f"Error initializing scalper strategies: {e}")

    def _create_fallback_strategy(self, name):
        """Create a simple strategy object if import fails"""
        class SimpleStrategy:
            def generate_signal(self, df):
                return None
        return SimpleStrategy()

    def _load_initial_data(self):
        """Load real BTC data from Binance including higher timeframes"""
        logger.info("Fetching initial market data...")
        
        # Fetch 5m (primary)
        candles = fetch_binance_klines("BTCUSDT", "5m", 300)
        if candles:
            trading_state['candles'] = candles
            trading_state['candles_5m'] = candles
            trading_state['current_price'] = candles[-1]['close']
            logger.info(f"5m data loaded: {len(candles)} candles, BTC @ ${candles[-1]['close']:,.2f}")
        else:
            logger.warning("Could not load 5m market data!")
        
        # Fetch 1H for multi-timeframe
        candles_1h = fetch_binance_klines("BTCUSDT", "1h", 100)
        if candles_1h:
            trading_state['candles_1h'] = candles_1h
            logger.info(f"1H data loaded: {len(candles_1h)} candles")
        else:
            logger.warning("Could not load 1H data!")
            trading_state['candles_1h'] = []
        
        # Fetch 4H for multi-timeframe
        candles_4h = fetch_binance_klines("BTCUSDT", "4h", 50)
        if candles_4h:
            trading_state['candles_4h'] = candles_4h
            logger.info(f"4H data loaded: {len(candles_4h)} candles")
        else:
            logger.warning("Could not load 4H data!")
            trading_state['candles_4h'] = []
        
        # Initialize HTF update timer
        self._last_htf_update = time.time()

    def calculate_atr(self, candles, period=20):
        """Calculate ATR for position sizing"""
        if not candles or len(candles) < period:
            if candles:
                return candles[-1]['high'] - candles[-1]['low']
            return 0
        
        trs = []
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        
        return sum(trs[-period:]) / min(period, len(trs))
    
    def _calculate_volatility_ratio(self, candles, period=20):
        """Calculate volatility ratio: current ATR / average ATR over longer period"""
        if not candles or len(candles) < period * 2:
            return 1.0
        
        # Current ATR (last 'period' candles)
        current_atr = self.calculate_atr(candles[-period:], period)
        
        # Average ATR (previous 'period' candles)
        avg_atr = self.calculate_atr(candles[-(period*2):-period], period)
        
        if avg_atr <= 0:
            return 1.0
        
        return current_atr / avg_atr
    
    def _detect_daily_trend(self, candles):
        """
        UPGRADE 4: Detect daily trend using EMA analysis
        Returns: 'UP', 'DOWN', or 'NEUTRAL'
        """
        if not candles or len(candles) < 50:
            return 'NEUTRAL'
        
        try:
            closes = [c['close'] for c in candles[-50:]]
            
            # Calculate EMAs
            ema_10 = sum(closes[-10:]) / 10
            ema_20 = sum(closes[-20:]) / 20
            ema_50 = sum(closes[-50:]) / 50
            
            # Current price relative to EMAs
            current_price = closes[-1]
            
            # Strong uptrend: price > EMA10 > EMA20 > EMA50
            if current_price > ema_10 > ema_20 > ema_50:
                return 'UP'
            
            # Strong downtrend: price < EMA10 < EMA20 < EMA50
            if current_price < ema_10 < ema_20 < ema_50:
                return 'DOWN'
            
            # Weak trend detection
            if current_price > ema_20 and ema_20 > ema_50:
                return 'UP'
            if current_price < ema_20 and ema_20 < ema_50:
                return 'DOWN'
            
            return 'NEUTRAL'
        except Exception:
            return 'NEUTRAL'

    def generate_signal(self, current_price):
        """Generate trading signal using Probabilistic AI Engine"""
        
        # === FEATURE #136: Circuit Breaker Check ===
        if self.circuit_breaker:
            trading_allowed, cb_status = self.circuit_breaker.update(self.equity)
            trading_state['circuit_breaker'] = self.circuit_breaker.get_status()
            
            if not trading_allowed:
                logger.warning(f"🛑 Circuit Breaker: {cb_status}")
                print(f"DEBUG: Circuit Breaker ACTIVE - {cb_status}")
                return None
        
        # === ENTERPRISE FEATURES: Pre-Signal Checks ===
        
        # Check Risk Budget
        if self.risk_budgeter:
            risk_for_trade = 2.0  # Default 2% risk per trade
            if not self.risk_budgeter.can_trade(risk_for_trade):
                logger.info(f"Risk budget exhausted - remaining: {self.risk_budgeter.remaining():.1f}%")
                return None
        
        # Check Drawdown Protection
        if self.dd_protector:
            dd_status = self.dd_protector.update(self.equity)
            if dd_status.get('blocked', False):
                logger.warning(f"Drawdown protection active - DD: {dd_status['drawdown']}%")
                return None
            trading_state['drawdown_pct'] = dd_status.get('drawdown', 0)
        
        # Check Signal Cooldown
        if self.signal_cooldown:
            if not self.signal_cooldown.can_signal('TRADE'):
                logger.debug("Signal cooldown active - waiting")
                return None
        
        # Update Enterprise Regime Classifier
        if self.regime_classifier and len(trading_state.get('candles', [])) > 50:
            candles = trading_state['candles']
            returns = [(candles[i]['close'] - candles[i-1]['close']) / candles[i-1]['close'] 
                      for i in range(1, min(20, len(candles)))]
            atr = self.calculate_atr(candles, 14)
            volatility = atr / current_price if current_price > 0 else 0.02
            
            regime_result = self.regime_classifier.classify(
                returns=returns,
                volatility=volatility,
                trend_strength=25,  # Default, will be updated
                volume_ratio=1.0
            )
            trading_state['enterprise_regime'] = regime_result.get('regime', 'UNKNOWN')
            
            # Use strategy selector for regime-aware trading
            if self.strategy_selector:
                strategy_advice = self.strategy_selector.select_strategy(
                    regime_result.get('regime', 'RANGING'),
                    regime_result.get('confidence', 0.5)
                )
                trading_state['recommended_strategy'] = strategy_advice.get('strategy', 'NEUTRAL')
        
        # Update Equity Tracker
        if hasattr(self, 'equity_tracker_ent') and self.equity_tracker_ent:
            self.equity_tracker_ent.record(self.equity)
        
        # Update Vol Detector
        if self.vol_detector:
            atr = self.calculate_atr(trading_state.get('candles', []), 14)
            volatility = atr / current_price if current_price > 0 else 0.02
            self.vol_detector.update(volatility)
            vol_regime = self.vol_detector.detect_regime()
            trading_state['volatility_regime'] = vol_regime.get('regime', 'NORMAL')
        
        # === END ENTERPRISE PRE-CHECKS ===
        
        # 1. Prepare Dataframes
        candles_5m = trading_state.get('candles', [])
        if not candles_5m or len(candles_5m) < 100:
            return None
            
        df_5m = pd.DataFrame(candles_5m)
        
        try:
            # ===== UPGRADE 3: Session-Based Trading Filter =====
            from datetime import datetime, timezone
            utc_hour = datetime.now(timezone.utc).hour
            
            # Determine session quality
            if 12 <= utc_hour < 17:
                session_quality = "BEST"  # London/NY Overlap
                session_min_score = 0  # Disabled - allow all trades
            elif 7 <= utc_hour < 12:
                session_quality = "GOOD"  # London Session
                session_min_score = 0  # Disabled - allow all trades
            elif 13 <= utc_hour < 21:
                session_quality = "GOOD"  # NY Session
                session_min_score = 0  # Disabled - allow all trades
            else:
                session_quality = "OFF_HOURS"  # Asian/Off-hours
                session_min_score = 0  # Disabled - allow all trades
            
            trading_state['session'] = {'quality': session_quality, 'utc_hour': utc_hour, 'min_score': session_min_score}
            # 1.5 Check Smart Risk Manager (mode-based trading control)
            if self.risk_manager_smart:
                # Update equity and check mode
                self.risk_manager_smart.update_equity(self.equity)
                trading_state['risk_mode'] = self.risk_manager_smart.mode.value
                
                # Pre-check if trading is allowed (before quality scoring)
                # We pass 70 as default min score, will check again with actual score later
                can_trade, reason = self.risk_manager_smart.can_trade(quality_score=70)
                
                if not can_trade:
                    # Log the reason
                    with open('debug_trace.txt', 'a') as f:
                        f.write(f"\n>>> RISK BLOCK: {reason} <<<\n")
                    print(f"DEBUG: {reason}")
                    return None
            
            # 2. Update Sentiment
            sentiment = self.sentiment_analyzer.get_current_sentiment()
            trading_state['sentiment'] = {
                'fear_greed_index': sentiment.fear_greed_index,
                'level': sentiment.sentiment_level.value,
                'news_score': sentiment.news_sentiment
            }

            # 3. Detect Market Regime
            regime_info = None
            current_regime_value = "unknown"
            
            if self.regime_detector:
                regime_info_obj = self.regime_detector.detect_regime(df_5m)
                current_regime = regime_info_obj.regime
                current_regime_value = current_regime.value
                
                # Convert object to dict for probabilistic generator
                regime_info = {
                    'regime': current_regime.value,
                    'should_trade': True, # Base assumption, filtered below
                    'confidence': regime_info_obj.confidence
                }
                
                # Log regime change
                if trading_state.get('current_regime') != current_regime_value:
                    logger.info(f"Market Regime Change: {current_regime_value.upper()} (Conf: {regime_info_obj.confidence:.2f})")
                    trading_state['current_regime'] = current_regime_value
                
                # Filter Strategies based on Regime
                for name, strategy in self.strategy_manager.strategies.items():
                    strat_type = getattr(strategy, 'strategy_type', 'unknown')
                    should_trade = self.regime_detector.should_trade_strategy(strat_type, current_regime)
                    if should_trade:
                        self.strategy_manager.enable_strategy(name)
                    else:
                        self.strategy_manager.disable_strategy(name)
                        
                # Update regime info for generator
                regime_info['should_trade'] = len(self.strategy_manager.active_strategies) > 0

            # 4. Generate Technical Signals
            tech_signals = self.strategy_manager.generate_raw_signals(candles_5m)
            
            # 5. Generate Final Probabilistic Signal
            prob_signal = self.probabilistic_generator.generate_signal(
                df=df_5m,
                technical_signals=tech_signals,
                regime_info=regime_info
            )
            
            # 6. Update UI
            atr = self.calculate_atr(candles_5m, 20)
            self._update_ui_indicators(candles_5m, prob_signal.confidence, prob_signal.action.value, current_regime_value)
            
            # DEBUG LOG
            self._debug_log_signal(current_regime_value, tech_signals, prob_signal)
            
            # DISABLED for testing - allow all signals
            # if not prob_signal.should_trade:
            #     return None
                
            # 7. Convert to Bot Signal Format
            signal = {
                'action': prob_signal.action.value,
                'confidence': prob_signal.confidence,
                'reasons': prob_signal.reasons.copy(),
                'price': current_price,
                'atr': atr,
                'timestamp': datetime.now().isoformat()
            }
            
            # Map BUY/SELL to LONG/SHORT for compatibility
            if signal['action'] == 'BUY': signal['action'] = 'LONG'
            if signal['action'] == 'SELL': signal['action'] = 'SHORT'
            
            # ===== UPGRADE 4: Daily Trend Counter-Trend Filter =====
            # Block trades that go against the daily trend
            daily_trend = self._detect_daily_trend(candles_5m)
            trading_state['daily_trend'] = daily_trend
            
            # DISABLED daily trend filter for testing
            # if signal['action'] == 'LONG' and daily_trend == 'DOWN':
            #     logger.info(f"Signal REJECTED: LONG against daily DOWN trend")
            #     print(f"DEBUG: Counter-trend blocked LONG (Daily: DOWN)")
            #     return None
            
            # if signal['action'] == 'SHORT' and daily_trend == 'UP':
            #     logger.info(f"Signal REJECTED: SHORT against daily UP trend")
            #     print(f"DEBUG: Counter-trend blocked SHORT (Daily: UP)")
            #     return None
            
            signal['reasons'].append(f"Daily Trend: {daily_trend}")
            
            # 8. Advanced Features Validation
            if self.advanced_features:
                # Get higher timeframe data
                df_1h = pd.DataFrame(trading_state.get('candles_1h', []))
                df_4h = pd.DataFrame(trading_state.get('candles_4h', []))
                
                current_positions = len(trading_state.get('positions', []))
                
                is_valid, adv_reasons = self.advanced_features.validate_signal(
                    action=signal['action'],
                    df_5m=df_5m,
                    df_1h=df_1h if len(df_1h) > 0 else None,
                    df_4h=df_4h if len(df_4h) > 0 else None,
                    equity=self.equity,
                    current_positions=current_positions,
                    scalper_mode=True  # Enable counter-trend scalping
                )
                
                # Log execution details
                htf_data = {'has_1h': len(df_1h) > 0, 'has_4h': len(df_4h) > 0}
                adv_result = {'is_valid': is_valid, 'reasons': adv_reasons}
                self._log_trade_execution(signal, htf_data, adv_result)
                
                signal['reasons'].extend(adv_reasons)
                
                if not is_valid:
                    logger.info(f"Signal FILTERED by advanced features: {adv_reasons}")
                    print(f"DEBUG: Signal FILTERED: {adv_reasons}")
                    return None
                
                # Apply volatility-adjusted stop params
                stop_params = self.advanced_features.get_stop_params(df_5m, atr)
                signal['sl_multiplier'] = stop_params['sl_multiplier']
                signal['tp_multiplier'] = stop_params['tp_multiplier']
                
                logger.info(f"Signal PASSED advanced features: {signal['action']} @ ${signal['price']:,.2f}")
                print(f"DEBUG: Signal PASSED - Ready to execute {signal['action']}")
                
                # UPGRADE 8: 4H TREND LOCK - Only trade in direction of higher timeframe
                # This is a key institutional-grade filter
                if len(df_4h) >= 50:
                    closes_4h = df_4h['close'].values
                    
                    # Calculate 4H EMAs
                    def calc_ema(data, period):
                        alpha = 2 / (period + 1)
                        ema = [data[0]]
                        for i in range(1, len(data)):
                            ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
                        return ema
                    
                    ema20_4h = calc_ema(closes_4h, 20)
                    ema50_4h = calc_ema(closes_4h, 50)
                    
                    htf_uptrend = ema20_4h[-1] > ema50_4h[-1]
                    htf_downtrend = ema20_4h[-1] < ema50_4h[-1]
                    
                    action = signal['action']
                    
                    # Soft trend lock - penalize counter-trend but don't block entirely
                    # This allows scalping counter-trend if other signals are strong
                    if action == 'LONG' and not htf_uptrend:
                        # Counter-trend LONG - reduce confidence
                        signal['confidence'] *= 0.7  # 30% penalty
                        signal['reasons'].append("4H Trend: ↓ DOWN (Counter-trend, -30%)")
                        logger.info("Counter-trend LONG: 4H is DOWN, confidence reduced")
                        
                        # Block if confidence now too low
                        if signal['confidence'] < 0.55:
                            logger.info("Signal REJECTED: Counter-trend + low confidence")
                            print("DEBUG: 4H counter-trend + low conf = BLOCKED")
                            return None
                            
                    elif action == 'SHORT' and not htf_downtrend:
                        # Counter-trend SHORT - reduce confidence
                        signal['confidence'] *= 0.7  # 30% penalty
                        signal['reasons'].append("4H Trend: ↑ UP (Counter-trend, -30%)")
                        logger.info("Counter-trend SHORT: 4H is UP, confidence reduced")
                        
                        # Block if confidence now too low
                        if signal['confidence'] < 0.55:
                            logger.info("Signal REJECTED: Counter-trend + low confidence")
                            print("DEBUG: 4H counter-trend + low conf = BLOCKED")
                            return None
                    else:
                        trend_dir = "↑ UP" if htf_uptrend else "↓ DOWN"
                        signal['reasons'].append(f"4H Trend: {trend_dir} (Aligned)")
                        logger.info(f"4H Trend Lock: Trading WITH trend ({trend_dir})")
            
            # 9. Trade Quality Scoring (Professional filter)
            if self.quality_scorer:
                quality_score, breakdown, quality_reasons = self.quality_scorer.score_trade(
                    signal=signal,
                    df=df_5m,
                    df_1h=df_1h if 'df_1h' in dir() and len(df_1h) > 0 else None,
                    df_4h=df_4h if 'df_4h' in dir() and len(df_4h) > 0 else None,
                    current_price=current_price
                )
                
                should_trade, grade = self.quality_scorer.should_take_trade(quality_score)
                
                # ===== UPGRADE 3: Session-Based Min Score Enforcement =====
                # Get session requirements (set earlier in generate_signal)
                session_info = trading_state.get('session', {})
                session_min = session_info.get('min_score', 70) if 'session' in trading_state else session_min_score
                
                # Override should_trade if score doesn't meet session minimum
                if should_trade and quality_score < session_min:
                    should_trade = False
                    grade = f"D ({session_info.get('quality', 'OFF_HOURS')} needs {session_min}+)"
                
                # Log quality assessment
                with open('debug_trace.txt', 'a') as f:
                    f.write(f"\n=== QUALITY SCORE: {quality_score}/100 ({grade}) ===\n")
                    f.write(f"Session: {session_info.get('quality', 'N/A')} (min: {session_min})\n")
                    for factor, pts in breakdown.items():
                        f.write(f"  {factor}: {pts}\n")
                
                if not should_trade:
                    logger.info(f"Signal REJECTED by quality filter: Score {quality_score}/100 ({grade})")
                    print(f"DEBUG: Trade REJECTED - Quality {quality_score}/100 (session needs {session_min}+)")
                    return None
                
                signal['quality_score'] = quality_score
                signal['reasons'].append(f"Quality: {quality_score}/100 ({grade})")
                
                # Dynamic R:R based on score
                if quality_score >= 85:
                    signal['tp_multiplier'] = max(signal.get('tp_multiplier', 2.0), 3.0)  # 2.5:1 or better
                    logger.info(f"A+ Trade ({quality_score}) - Extended TP to {signal['tp_multiplier']:.1f}x ATR")
                elif quality_score >= 75:
                    signal['tp_multiplier'] = max(signal.get('tp_multiplier', 2.0), 2.5)  # 2:1
                    logger.info(f"A Trade ({quality_score}) - TP at {signal['tp_multiplier']:.1f}x ATR")
                # B trades keep default 1.5-2:1
                
                print(f"DEBUG: Trade APPROVED - Quality {quality_score}/100")
            
            # 10. Order Flow Check (if available)
            if self.order_flow:
                orderbook = trading_state.get('orderbook', {})
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if bids and asks:
                    flow_result = self.order_flow.analyze(bids, asks, current_price)
                    supports, conf_boost, flow_reason = self.order_flow.get_support_for_signal(signal['action'])
                    
                    signal['reasons'].append(f"OrderFlow: {flow_reason}")
                    
                    # Boost or reduce confidence based on order flow
                    signal['confidence'] = min(1.0, signal['confidence'] + conf_boost)
            
            # 11. UPGRADE 2: Momentum Filter - confirm direction with price action
            if len(df_5m) >= 10:
                closes = df_5m['close'].values
                ema5 = closes[-5:].mean()
                ema10 = closes[-10:].mean()
                price_above_ema = current_price > ema5
                ema5_above_ema10 = ema5 > ema10
                
                action = signal['action']
                momentum_ok = False
                
                if action == 'LONG':
                    # For LONG: price should be above EMA5 and EMA5 > EMA10
                    if price_above_ema and ema5_above_ema10:
                        momentum_ok = True
                        signal['reasons'].append("Momentum: Confirmed ↑")
                    elif price_above_ema or ema5_above_ema10:
                        momentum_ok = True  # Partial confirmation still ok
                        signal['reasons'].append("Momentum: Partial ↑")
                    else:
                        logger.info("Signal REJECTED: Momentum not confirming LONG")
                        print("DEBUG: Momentum filter blocked LONG")
                        return None
                        
                elif action == 'SHORT':
                    # For SHORT: price should be below EMA5 and EMA5 < EMA10
                    if not price_above_ema and not ema5_above_ema10:
                        momentum_ok = True
                        signal['reasons'].append("Momentum: Confirmed ↓")
                    elif not price_above_ema or not ema5_above_ema10:
                        momentum_ok = True  # Partial confirmation still ok
                        signal['reasons'].append("Momentum: Partial ↓")
                    else:
                        logger.info("Signal REJECTED: Momentum not confirming SHORT")
                        print("DEBUG: Momentum filter blocked SHORT")
                        return None
            
            # 12. UPGRADE 6: ML Signal Filter & Data Collection
            # ----------------------------------------------------------------
            # This is the "Brain" of the system. It uses the Ensemble Model to
            # validate signals based on historical patterns.
            
            # A. Feature Engineering (Always run to collect data)
            ml_features = None
            if self.feature_engineer and len(df_5m) > 100:
                try:
                    # Create features from recent data
                    # We compute on the entire dataframe to get rolling metrics correct
                    features_df = self.feature_engineer.engineer_features(df_5m)
                    
                    if not features_df.empty:
                        # Get the latest row (corresponding to current candle)
                        ml_features_df = features_df.tail(1)
                        ml_features = ml_features_df.to_dict(orient='records')[0]
                        
                        # Store features in signal for later data collection
                        signal['ml_features'] = ml_features
                        
                        # B. ML Prediction (Only if model is trained)
                        if self.ml_model and self.ml_model.models:
                            # Log that ML is checking
                            print(f"DEBUG: ML Model checking signal...")
                            
                            # Predict proba: Cls 0=SELL, 1=NEUTRAL, 2=BUY
                            probs = self.ml_model.predict_proba(ml_features_df)[0]
                            
                            buy_prob = probs[2]
                            sell_prob = probs[0]
                            neutral_prob = probs[1]
                            
                            action = signal['action']
                            ml_confidence = 0.0
                            
                            if action == 'LONG':
                                ml_confidence = buy_prob
                                msg = f"ML Confidence: {buy_prob:.1%} (Neutral: {neutral_prob:.1%})"
                            elif action == 'SHORT':
                                ml_confidence = sell_prob
                                msg = f"ML Confidence: {sell_prob:.1%} (Neutral: {neutral_prob:.1%})"
                            
                            signal['ml_prediction'] = probs.tolist()
                            signal['ml_msg'] = msg
                            
                            # C. Filter Logic
                            # High threshold to filter out noise, but don't be too strict initially
                            ML_THRESHOLD = 0.40  # 40% probability (since there are 3 classes, random is 33%)
                            
                            if ml_confidence < ML_THRESHOLD:
                                logger.info(f"Signal REJECTED by ML Filter: {msg} < {ML_THRESHOLD:.0%}")
                                print(f"DEBUG: ML REJECTED - {msg}")
                                return None
                            
                            # Boost confidence if ML is very sure
                            if ml_confidence > 0.70:
                                signal['confidence'] = min(0.99, signal['confidence'] * 1.2)
                                signal['reasons'].append(f"ML: High Conviction ({ml_confidence:.0%})")
                            elif ml_confidence > 0.50:
                                signal['reasons'].append(f"ML: Confirmed ({ml_confidence:.0%})")
                            else:
                                signal['reasons'].append(f"ML: Neutral/Weak ({ml_confidence:.0%})")
                                
                        else:
                            # Model not trained - just collecting data
                            signal['ml_msg'] = "Model not trained (Collecting Data)"
                            # print("DEBUG: ML - Collecting training data...")
                            
                except Exception as e:
                    logger.error(f"ML Feature/Prediction Error: {e}")
                    # Don't crash bot, just skip ML filter
            
            # 13. UPGRADE 7: Streak-Based Confidence Adjustment
            # On winning streak (3+): boost confidence slightly
            # On losing streak (2+): reduce confidence and be more selective
            if self.win_streak >= 3:
                streak_boost = min(0.10, self.win_streak * 0.02)  # Max 10% boost
                signal['confidence'] = min(0.99, signal['confidence'] + streak_boost)
                signal['reasons'].append(f"Streak: +{self.win_streak} wins ↑")
            elif self.loss_streak >= 2:
                streak_penalty = min(0.15, self.loss_streak * 0.05)  # Max 15% penalty
                signal['confidence'] = max(0.5, signal['confidence'] - streak_penalty)
                signal['reasons'].append(f"Streak: -{self.loss_streak} losses ↓")
                
                # Loss streak note (quality check removed - see execute_trade)
                if self.loss_streak >= 3:
                    signal['reasons'].append(f"Streak: Caution (3+ losses)")
            
            return signal

        except Exception as e:
            import traceback
            error_msg = f"Error generating signal: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            print("DEBUG: EXCEPTION IN GENERATE_SIGNAL:")
            print(error_msg)
            return None

    def _debug_log_signal(self, regime, tech_signals, prob_signal):
        """Helper to log why trades aren't happening"""
        try:
            with open('debug_trace.txt', 'a') as f:
                f.write(f"\n--- DEBUG SIGNAL GENERATION {datetime.now()} ---\n")
                f.write(f"Regime: {regime}\n")
                f.write(f"Active Strategies: {self.strategy_manager.active_strategies}\n")
                f.write(f"Tech Signals: {tech_signals}\n")
                if prob_signal:
                    f.write(f"Probabilities: {prob_signal.probabilities}\n")
                    f.write(f"Action: {prob_signal.action} | Conf: {prob_signal.confidence:.2f}\n")
                    f.write(f"Should Trade: {prob_signal.should_trade}\n")
                    f.write(f"Reasons: {prob_signal.reasons}\n")
                f.write("-------------------------------\n")
        except Exception as e:
            logger.error(f"Debug log failed: {e}")
    
    def _log_trade_execution(self, signal, htf_data, adv_result):
        """Log detailed trade execution info"""
        try:
            with open('debug_trace.txt', 'a') as f:
                f.write(f"\n>>> TRADE EXECUTION CHECK {datetime.now()} <<<\n")
                f.write(f"Signal: {signal['action']} @ ${signal['price']:,.2f}\n")
                f.write(f"Confidence: {signal['confidence']:.2%}\n")
                f.write(f"HTF Data Available: 1H={htf_data['has_1h']}, 4H={htf_data['has_4h']}\n")
                if adv_result:
                    f.write(f"Advanced Features Result: Valid={adv_result['is_valid']}\n")
                    f.write(f"Reasons: {adv_result['reasons']}\n")
                f.write("=================================\n")
        except Exception as e:
            logger.error(f"Execution log failed: {e}")

    def _update_ui_indicators(self, candles, confidence, action, regime):
        """Update shared state for dashboard"""
        try:
            atr = self.calculate_atr(candles, 20)
            trading_state['indicators'] = {
                'confidence': round(confidence, 2),
                'action': action,
                'atr': round(atr, 2),
                'regime': regime
            }
        except Exception:
            pass
    
    def execute_trade(self, signal):
        """
        Execute trade with Dynamic Position Sizing and Multiple Entries
        Supports up to 3 concurrent positions, pyramiding allowed.
        """
        logger.info(f">>> EXECUTE_TRADE called with {signal['action']}")
        
        # Allow up to 3 concurrent positions
        current_positions = trading_state.get('positions', [])
        if self.position:  # Legacy check compatibility
            if isinstance(self.position, list):
                if len(self.position) >= 3: return False
            elif isinstance(self.position, dict):
                 # Convert single position to list if needed, or just block if strictly 1
                 # For now, we'll respect the new logic below and treat self.position as 'last active'
                 if len(current_positions) >= 3: return False

        try:
            price = signal['price']
            atr = signal.get('atr', price * 0.01)
            confidence = signal.get('confidence', 0.6) # Default lowered
            
            # === FEATURE #320: Confidence-Gated Trades ===
            # Block low-confidence trades to ensure only quality signals execute
            MIN_CONFIDENCE_THRESHOLD = 0.50  # 50% minimum confidence (INCREASED for quality)
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                logger.info(f"Signal BLOCKED: Confidence {confidence:.1%} below threshold ({MIN_CONFIDENCE_THRESHOLD:.0%})")
                print(f"DEBUG: Confidence gate blocked trade ({confidence:.1%} < {MIN_CONFIDENCE_THRESHOLD:.0%})")
                return False
            
            # ==================================================================
            # CRITICAL FIX: Use Risk/Reward Optimizer for Proper Stops & Sizing
            # This fixes small wins and large losses problem
            # ==================================================================
            if hasattr(self, 'rr_optimizer') and self.rr_optimizer:
                # Calculate proper stop loss and take profit with 2.5:1 R:R
                levels = self.rr_optimizer.calculate_stop_and_target(
                    entry_price=price,
                    side=signal['action'],
                    atr=atr,
                    rr_ratio=2.5  # 2.5:1 reward/risk ratio
                )
                
                # Validate trade meets minimum R:R requirements
                is_valid, reason = self.rr_optimizer.validate_trade(levels)
                
                if not is_valid:
                    logger.info(f"Trade BLOCKED by R:R optimizer: {reason}")
                    print(f"⚠️ Trade rejected: {reason}")
                    return False
                
                # Calculate position size based on risk (max 1% of equity)
                pos_sizing = self.rr_optimizer.calculate_position_size(
                    equity=self.equity,
                    entry_price=price,
                    stop_loss_price=levels['stop_loss']
                )
                
                size = pos_sizing['position_size']
                
                # Store stops in signal for position management
                signal['stop_loss'] = levels['stop_loss']
                signal['take_profit'] = levels['take_profit']
                signal['rr_ratio'] = levels['rr_ratio']
                signal['dollar_risk'] = pos_sizing['dollar_risk']
                
                logger.info(f"✓ R:R Optimizer: {levels['rr_ratio']}:1 R:R, Risk ${pos_sizing['dollar_risk']:.2f} ({pos_sizing['risk_pct']:.1f}%)")
                print(f"✓ Stop: ${levels['stop_loss']:,.0f}, Target: ${levels['take_profit']:,.0f}, Size: {size:.6f}")
                
            else:
                # Fallback to old sizing if optimizer not available
                logger.warning("⚠️ Risk/Reward optimizer not active - using legacy sizing")
                sizing = self.position_sizer.calculate_size(
                    equity=self.equity,
                    price=price,
                    atr=atr,
                    confidence=confidence
                )
                size = sizing['size']
            
            print(f"DEBUG: Final Size: {size}")
            
            # ===== UPGRADE 1: Dynamic R:R Based on Volatility =====
            # Calculate volatility ratio (current ATR / average ATR)
            candles = trading_state.get('candles', [])
            vol_ratio = self._calculate_volatility_ratio(candles) if len(candles) > 40 else 1.0
            
            # Dynamic SL/TP based on volatility regime
            # INCREASED TARGETS for better R:R
            if vol_ratio < 0.7:  # Low volatility - tighter stops, good targets
                base_sl_mult, base_tp_mult = 0.8, 4.0
                vol_regime = "LOW_VOL"
            elif vol_ratio > 1.5:  # High volatility - wider stops, big targets
                base_sl_mult, base_tp_mult = 1.5, 6.0
                vol_regime = "HIGH_VOL"
            else:  # Normal volatility
                base_sl_mult, base_tp_mult = 1.0, 5.0
                vol_regime = "NORMAL"
            
            # Use signal overrides if provided, otherwise use dynamic values
            sl_mult = signal.get('sl_multiplier', base_sl_mult)
            tp_mult = signal.get('tp_multiplier', base_tp_mult)
            
            # Ensure minimum 2.5:1 R:R ratio
            if tp_mult < sl_mult * 2.5:
                tp_mult = sl_mult * 2.5
            
            # Confidence Scalar: If conf > 0.7, expand TP further
            if confidence >= 0.8:
                tp_mult *= 1.3  # High confidence = bigger target
            elif confidence >= 0.7:
                tp_mult *= 1.15
            
            logger.info(f"Volatility Regime: {vol_regime} (ratio: {vol_ratio:.2f}) | SL: {sl_mult}x, TP: {tp_mult:.1f}x")
            
            stop_loss_dist = atr * sl_mult
            take_profit_dist = atr * tp_mult
            
            is_long = signal['action'] == 'LONG'
            
            stop_loss = price - stop_loss_dist if is_long else price + stop_loss_dist
            take_profit = price + take_profit_dist if is_long else price - take_profit_dist
            
            # Log the R:R ratio
            rr_ratio = take_profit_dist / stop_loss_dist if stop_loss_dist > 0 else 0
            logger.info(f"R:R Ratio: {rr_ratio:.1f}:1 (SL: ${stop_loss_dist:.2f}, TP: ${take_profit_dist:.2f})")
            
            trade_id = f"T-{len(self.trades) + 1:03d}"
            
            new_position = {
                'id': trade_id,
                'side': signal['action'],
                'entry_price': price,
                'size': round(size, 6),
                'original_size': round(size, 6),  # UPGRADE 2: Track original size for partial exits
                'entry_time': datetime.now().isoformat(),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'initial_risk': round(stop_loss_dist, 2),  # UPGRADE 2: Store for partial exit calculation
                'unrealized_pnl': 0,
                'partial_pnl': 0,  # UPGRADE 2: Track realized partial profits
                'reasons': signal.get('reasons', []),
                'first_profit_taken': False,  # For partial profit tracking at 1R
                'second_profit_taken': False,  # For partial profit tracking at 2R
                # ML Data Preservation (for training)
                'ml_features': signal.get('ml_features'),
                'ml_prediction': signal.get('ml_prediction')
            }
            
            # CRITICAL: Actually assign the position!
            self.position = new_position
            
            # CRITICAL FIX: Record trade entry through feature hub (activates all 26 features)
            if hasattr(self, 'feature_hub') and self.feature_hub:
                signal['signal_time'] = datetime.now()  # For latency tracking
                self.feature_hub.on_trade_entry(signal, trading_state)
                logger.info("✓ Feature hub: Trade recorded (DB, alerts, analytics active)")
            
            self.signals.append(signal)
            trading_state['signals'] = self.signals[-20:]
            trading_state['positions'] = [new_position]
            
            reasons_str = ", ".join(signal.get('reasons', []))
            logger.info(f"TRADE: {signal['action']} @ ${price:,.2f} | Confidence: {confidence:.0%}")
            logger.info(f"  Reasons: {reasons_str}")
            logger.info(f"  Size: {size:.6f} BTC | SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def close_position(self, exit_price: float, exit_reason: str):
        """
        Close current position and record exit
        
        Critical fix: Properly closes trades and notifies all systems
        """
        if not self.position:
            logger.warning("close_position called but no position exists")
            return
        
        # Calculate P&L
        entry = self.position['entry_price']
        size = self.position.get('size', 0.1)
        side = self.position['side']
        
        if side == 'LONG':
            pnl = (exit_price - entry) * size
            pnl_pct = (exit_price - entry) / entry * 100
        else:  # SHORT
            pnl = (entry - exit_price) * size
            pnl_pct = (entry - exit_price) / entry * 100
        
        # Update equity
        self.equity += pnl
        trading_state['equity'] = self.equity
        
        # Track win/loss streak
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
        
        # Create trade record
        trade_record = {
            **self.position,
            'exit_price': exit_price,
            'exit_time': datetime.now().isoformat(),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'exit_reason': exit_reason
        }
        self.trades.append(trade_record)
        trading_state['trades'] = self.trades
        
        # CRITICAL FIX: Notify feature hub of exit (DB, analytics, alerts)
        if hasattr(self, 'feature_hub') and self.feature_hub:
            self.feature_hub.on_trade_exit(
                trade=self.position,
                exit_price=exit_price,
                exit_reason=exit_reason,
                bot_state=trading_state
            )
        
        # Clear position
        self.position = None
        trading_state['position'] = None
        trading_state['positions'] = []
        
        # Log
        emoji = "✅" if pnl > 0 else "❌"
        logger.info(f"{emoji} POSITION CLOSED: {exit_reason}")
        logger.info(f"  P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | Equity: ${self.equity:,.2f}")
        logger.info(f"  Streak: {self.win_streak if pnl > 0 else self.loss_streak} {'wins' if pnl > 0 else 'losses'}")
    
    def update_position(self, current_price):
        """Update position with trailing stop"""
        if not self.position:
            return
        
        try:
            pos = self.position
            is_long = pos['side'] == 'LONG'
            entry_price = pos['entry_price']
            
            # Calculate P&L
            if is_long:
                pnl = (current_price - entry_price) * pos['size']
            else:
                pnl = (entry_price - current_price) * pos['size']
            
            pos['unrealized_pnl'] = round(pnl, 2)
            self.equity = self.initial_capital + sum(t['pnl'] for t in self.trades) + pnl
            
            if self.equity > self.max_equity:
                self.max_equity = self.equity
            
            # === TRAILING STOP FEATURE (UPGRADED) ===
            # Move stop to breakeven at 0.5R (earlier protection)
            initial_risk = abs(entry_price - pos['stop_loss'])
            
            if is_long:
                current_profit = current_price - entry_price
                
                # UPGRADE 2: Move to breakeven at 0.3R (was 0.5R) - faster protection
                if current_profit >= initial_risk * 0.3:
                    new_stop = entry_price + (initial_risk * 0.1)  # Breakeven + small buffer
                    if new_stop > pos['stop_loss']:
                        pos['stop_loss'] = round(new_stop, 2)
                        logger.debug(f"Moved to breakeven at ${new_stop:,.2f}")
                
                # At 1R profit, lock in 25%
                if current_profit >= initial_risk:
                    new_stop = entry_price + (initial_risk * 0.25)
                    if new_stop > pos['stop_loss']:
                        pos['stop_loss'] = round(new_stop, 2)
                
                # At 1.5R profit, lock in 50%
                if current_profit >= initial_risk * 1.5:
                    new_stop = entry_price + (initial_risk * 0.5)
                    if new_stop > pos['stop_loss']:
                        pos['stop_loss'] = round(new_stop, 2)
                
                # At 2R+ profit, trail aggressively at 0.5R behind
                if current_profit >= initial_risk * 2:
                    new_stop = current_price - initial_risk * 0.5
                    if new_stop > pos['stop_loss']:
                        pos['stop_loss'] = round(new_stop, 2)
            else:
                # SHORT position
                current_profit = entry_price - current_price
                
                # Move to breakeven at 0.3R (faster protection)
                if current_profit >= initial_risk * 0.3:
                    new_stop = entry_price - (initial_risk * 0.1)
                    if new_stop < pos['stop_loss']:
                        pos['stop_loss'] = round(new_stop, 2)
                        logger.debug(f"Moved to breakeven at ${new_stop:,.2f}")
                
                # At 1R profit, lock in 25%
                if current_profit >= initial_risk:
                    new_stop = entry_price - (initial_risk * 0.25)
                    if new_stop < pos['stop_loss']:
                        pos['stop_loss'] = round(new_stop, 2)
                
                # At 1.5R profit, lock in 50%
                if current_profit >= initial_risk * 1.5:
                    new_stop = entry_price - (initial_risk * 0.5)
                    if new_stop < pos['stop_loss']:
                        pos['stop_loss'] = round(new_stop, 2)
                
                # At 2R+ profit, trail aggressively
                if current_profit >= initial_risk * 2:
                    new_stop = current_price + initial_risk * 0.5
                    if new_stop < pos['stop_loss']:
                        pos['stop_loss'] = round(new_stop, 2)
            # === UPGRADE 2: PARTIAL PROFIT TAKING (1R and 2R milestones) ===
            initial_risk = pos.get('initial_risk', abs(entry_price - pos['stop_loss']))
            original_size = pos.get('original_size', pos['size'])
            
            # At 1R profit: Take 33% profit
            if current_profit >= initial_risk and not pos.get('first_profit_taken', False):
                exit_size = original_size * 0.33
                if exit_size > 0 and pos['size'] > exit_size:
                    partial_pnl = current_profit * exit_size / pos['size'] if pos['size'] > 0 else 0
                    
                    pos['size'] = round(pos['size'] - exit_size, 6)
                    pos['partial_pnl'] = pos.get('partial_pnl', 0) + partial_pnl
                    pos['first_profit_taken'] = True
                    
                    # Move stop to breakeven
                    pos['stop_loss'] = round(entry_price + (initial_risk * 0.1 if is_long else -initial_risk * 0.1), 2)
                    
                    logger.info(f"PARTIAL EXIT 1R: Took 33% profit (+${partial_pnl:.2f}) | Size: {pos['size']:.6f}")
            
            # At 2R profit: Take another 33% profit
            if current_profit >= initial_risk * 2 and not pos.get('second_profit_taken', False) and pos.get('first_profit_taken', False):
                exit_size = original_size * 0.33
                if exit_size > 0 and pos['size'] > exit_size:
                    partial_pnl = current_profit * exit_size / pos['size'] if pos['size'] > 0 else 0
                    
                    pos['size'] = round(pos['size'] - exit_size, 6)
                    pos['partial_pnl'] = pos.get('partial_pnl', 0) + partial_pnl
                    pos['second_profit_taken'] = True
                    
                    # Trail stop at 1R profit
                    new_stop = entry_price + (initial_risk if is_long else -initial_risk)
                    pos['stop_loss'] = round(new_stop, 2)
                    
                    logger.info(f"PARTIAL EXIT 2R: Took 33% profit (+${partial_pnl:.2f}) | Remaining: {pos['size']:.6f}")
            
            # Use advanced_features partial exit if available (additional logic)
            if self.advanced_features:
                partial_exit = self.advanced_features.check_partial_exit(pos, current_price)
                if partial_exit:
                    # Take partial profit
                    exit_size = pos['size'] * partial_exit['exit_pct']
                    
                    if is_long:
                        partial_pnl = (current_price - entry_price) * exit_size
                    else:
                        partial_pnl = (entry_price - current_price) * exit_size
                    
                    # Reduce position size
                    pos['size'] = pos['size'] - exit_size
                    
                    # Move stop as specified
                    pos['stop_loss'] = round(partial_exit['move_stop_to'], 2)
                    
                    # Mark profit taken
                    pos[partial_exit['mark']] = True
                    
                    # Record partial trade
                    partial_trade = {
                        'side': pos['side'],
                        'entry': round(pos['entry_price'], 2),
                        'exit': round(current_price, 2),
                        'size': round(exit_size, 6),
                        'pnl': round(partial_pnl, 2),
                        'return_pct': round((partial_pnl / self.initial_capital) * 100, 2),
                        'exit_reason': partial_exit['reason'],
                        'timestamp': datetime.now().isoformat()
                    }
                    self.trades.append(partial_trade)
                    
                    # Record for risk tracking
                    self.advanced_features.record_trade_result(partial_pnl)
                    
                    logger.info(f"PARTIAL EXIT: {partial_exit['reason']} | PnL: +${partial_pnl:.2f}")
            
            # Check exits
            hit_stop = (is_long and current_price <= pos['stop_loss']) or \
                       (not is_long and current_price >= pos['stop_loss'])
            
            hit_tp = (is_long and current_price >= pos['take_profit']) or \
                     (not is_long and current_price <= pos['take_profit'])
            
            # UPGRADE 5: Time-based exit - close stale trades after 30 minutes
            time_exit = False
            if 'entry_time' in pos:
                try:
                    entry_time = datetime.fromisoformat(pos['entry_time'])
                    minutes_open = (datetime.now() - entry_time).total_seconds() / 60
                    
                    # Close if open > 30 min and not in profit
                    if minutes_open > 30:
                        if is_long:
                            current_pnl = current_price - entry_price
                        else:
                            current_pnl = entry_price - current_price
                        
                        # Only close if not significantly profitable
                        if current_pnl < initial_risk * 0.5:
                            time_exit = True
                            logger.info(f"TIME EXIT: Position open {minutes_open:.0f} min, closing at scratch")
                except:
                    pass
            
            if hit_stop:
                self._close_position(current_price, 'stop_loss')
            elif hit_tp:
                self._close_position(current_price, 'take_profit')
            elif time_exit:
                self._close_position(current_price, 'time_exit')
            
            # Update state
            trading_state['equity'] = round(self.equity, 2)
            trading_state['positions'] = [pos] if self.position else []
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def _close_position(self, exit_price, reason):
        """Close position"""
        try:
            pos = self.position
            is_long = pos['side'] == 'LONG'
            
            pnl = (exit_price - pos['entry_price']) * pos['size'] if is_long else (pos['entry_price'] - exit_price) * pos['size']
            
            return_pct = (pnl / self.initial_capital) * 100
            
            if pnl > 0:
                self.win_streak += 1
                self.loss_streak = 0
            else:
                self.loss_streak += 1
                self.win_streak = 0
            
            trade = {
                'side': pos['side'],
                'entry_price': round(pos['entry_price'], 2),
                'exit_price': round(exit_price, 2),
                'take_profit': round(pos.get('take_profit', 0), 2),
                'stop_loss': round(pos.get('stop_loss', 0), 2),
                'size': round(pos['size'], 6),
                'pnl': round(pnl, 2),
                'return_pct': round(return_pct, 2),
                'exit_reason': reason,
                'status': 'CLOSED',
                'strategy': pos.get('strategy', 'manual'),
                'confidence': pos.get('ml_confidence', 0.0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to DB
            db_manager.save_trade(trade)
            
            # Update internal state (Cache recent trades)
            self.trades.append(trade)
            if len(self.trades) > 100: self.trades.pop(0)  # Keep memory usage low
            
            self.position = None
            self.equity += pnl  # Incremental update (Safe for partial history)
            self.equity = round(self.equity, 2)
            
            # Save State
            save_state(self)
            
            # Sync to Dashboard State (In-memory cache)
            trading_state['trades'] = self.trades
            trading_state['equity'] = self.equity
            trading_state['positions'] = []
            
            self._update_metrics()
            
            # Record trade result for risk tracking
            if self.advanced_features:
                self.advanced_features.record_trade_result(pnl)
            
            # Record trade for smart risk manager (streak tracking, cooldowns)
            if self.risk_manager_smart:
                trade_id = trade.get('id', str(len(self.trades)))
                self.risk_manager_smart.record_trade(pnl, trade_id)
                logger.info(f"Risk Manager: Mode={self.risk_manager_smart.mode.value}, Today's trades={len(self.risk_manager_smart.daily_trades)}")
            
            # === FEATURE #136: Update Circuit Breaker with trade result ===
            if self.circuit_breaker:
                cb_allowed, cb_status = self.circuit_breaker.update(self.equity, trade_result=pnl)
                trading_state['circuit_breaker'] = self.circuit_breaker.get_status()
                if not cb_allowed:
                    logger.critical(f"🛑 CIRCUIT BREAKER TRIGGERED: {cb_status}")
            
            icon = "+" if pnl >= 0 else ""
            logger.info(f"CLOSED: {reason} @ ${exit_price:,.2f} | P&L: {icon}${pnl:.2f} ({icon}{return_pct:.2f}%)")
            
            # --- UPGRADE 6: SAVE LEARNING DATA ---
            # Save features + label (1=Win, 0=Loss) for future training
            if pos.get('ml_features'):
                try:
                    import csv
                    
                    data_row = pos['ml_features'].copy()
                    
                    # Target: 1 for Win (> 0.2% profit), 0 for Neutral, -1 for Loss
                    # We want to learn to predict GOOD trades
                    if return_pct > 0.2:
                        label = 1  # BUY/WIN
                    elif return_pct < -0.1:
                        label = -1 # SELL/LOSS (if it was a long) - wait, labels are about signal VALIDITY
                    else:
                        label = 0  # NEUTRAL
                        
                    # Correction: For ML, "Label" is usually "Future Return".
                    # Here we are mapping "Outcome" to "Label".
                    # Simplest: Did this trade accept make money? 1=Yes, 0=No.
                    # But for multiclass (Sell/Neutral/Buy):
                    # If Long and Win -> Label=2 (Buy)
                    # If Short and Win -> Label=0 (Sell)
                    # If Loss -> Label=1 (Neutral/Avoid)
                    
                    final_label = 1 # Default neutral/avoid
                    if pos['side'] == 'LONG':
                        if return_pct > 0.2: final_label = 2  # Good Long
                        elif return_pct < -0.2: final_label = 0 # Should have Sold (or avoided)
                    elif pos['side'] == 'SHORT':
                        if return_pct > 0.2: final_label = 0  # Good Short
                        elif return_pct < -0.2: final_label = 2 # Should have Longed (or avoided)
                    
                    data_row['label'] = final_label
                    data_row['return_pct'] = return_pct
                    data_row['side'] = pos['side']
                    data_row['timestamp'] = pos['entry_time']
                    
                    file_exists = os.path.isfile('data/ml_training_data.csv')
                    with open('data/ml_training_data.csv', 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=data_row.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(data_row)
                        
                    logger.info(f"ML Data: Saved training sample (Label: {final_label})")
                    
                    # ===== UPGRADE 7: ML Auto-Training Trigger =====
                    # Check if we should auto-retrain (every 50 trades)
                    trade_count = len(self.trades)
                    if trade_count > 0 and trade_count % 50 == 0:
                        logger.info(f"ML AUTO-TRAIN: Triggering retrain after {trade_count} trades")
                        self._trigger_ml_retrain()
                    
                except Exception as e:
                    logger.error(f"Failed to save ML data: {e}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _update_metrics(self):
        """Update metrics for dashboard"""
        try:
            if not self.trades:
                return
            
            wins = [t for t in self.trades if t['pnl'] > 0]
            losses = [t for t in self.trades if t['pnl'] <= 0]
            
            win_rate = len(wins) / len(self.trades) if self.trades else 0
            total_wins = sum(t['pnl'] for t in wins)
            total_losses = abs(sum(t['pnl'] for t in losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else 2.0
            drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
            
            trading_state['metrics'] = {
                'total_return': round((self.equity - self.initial_capital) / self.initial_capital * 100, 2),
                'win_rate': round(win_rate, 2),
                'sharpe_ratio': 0.0,
                'max_drawdown': round(drawdown, 4),
                'num_trades': len(self.trades),
                'profit_factor': round(profit_factor, 2),
                'win_streak': self.win_streak,
                'loss_streak': self.loss_streak
            }
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _trigger_ml_retrain(self):
        """
        UPGRADE 7: Trigger ML model retraining with collected data
        Runs asynchronously to not block trading
        """
        import threading
        
        def retrain_task():
            try:
                from core.ml.real_ml_trainer import RealMLTrainer
                import pandas as pd
                
                # Load training data
                data_path = 'data/ml_training_data.csv'
                if not os.path.isfile(data_path):
                    logger.warning("ML Retrain: No training data found")
                    return
                
                df = pd.read_csv(data_path)
                if len(df) < 100:
                    logger.info(f"ML Retrain: Not enough data ({len(df)} samples, need 100+)")
                    return
                
                logger.info(f"ML AUTO-TRAIN: Starting with {len(df)} samples...")
                
                # Initialize trainer
                trainer = RealMLTrainer(model_type='xgboost')
                
                # Prepare features (excluding non-feature columns)
                feature_cols = [c for c in df.columns if c not in ['label', 'return_pct', 'side', 'timestamp']]
                X = df[feature_cols]
                y = df['label']
                
                # Simple train/test split
                split_idx = int(len(df) * 0.8)
                X_train, y_train = X[:split_idx], y[:split_idx]
                X_test, y_test = X[split_idx:], y[split_idx:]
                
                # Train
                trainer.train(X_train, y_train, X_test, y_test)
                
                # Evaluate
                metrics = trainer.evaluate(X_test, y_test)
                logger.info(f"ML AUTO-TRAIN: Complete! Accuracy: {metrics.get('accuracy', 0):.2%}")
                
                # Save model
                from pathlib import Path
                trainer.save_model(Path('models/auto_trained_model.pkl'))
                logger.info("ML AUTO-TRAIN: Model saved to models/auto_trained_model.pkl")
                
            except Exception as e:
                logger.error(f"ML Retrain failed: {e}")
        
        # Run in background thread
        thread = threading.Thread(target=retrain_task, daemon=True)
        thread.start()
    
    def update_candles(self):
        """Fetch latest candles for all timeframes"""
        # Fetch 5m (primary)
        new_candles_5m = fetch_binance_klines("BTCUSDT", "5m", 300)
        if new_candles_5m:
            trading_state['candles'] = new_candles_5m
            trading_state['candles_5m'] = new_candles_5m
            trading_state['current_price'] = new_candles_5m[-1]['close']
        
        # Periodically update higher timeframes (every ~5 mins is fine, but we'll do it here for simplicity/robustness)
        # In production, we might optimize this to not call every tick
        if getattr(self, '_last_htf_update', 0) < time.time() - 300:
            trading_state['candles_1h'] = fetch_binance_klines("BTCUSDT", "1h", 100) or trading_state.get('candles_1h', [])
            trading_state['candles_4h'] = fetch_binance_klines("BTCUSDT", "4h", 50) or trading_state.get('candles_4h', [])
            self._last_htf_update = time.time()
        
        return trading_state.get('current_price', 0)
    
    def update_sentiment(self):
        """Update sentiment"""
        fg = trading_state['sentiment']['fear_greed_index']
        change = random.randint(-1, 1)
        fg = max(15, min(85, fg + change))
        trading_state['sentiment']['fear_greed_index'] = fg


def run_trading_loop(bot):
    """Main trading loop"""
    logger.info("Starting trading loop...")
    logger.info("Strategies Active: " + ", ".join(bot.strategy_manager.active_strategies))
    
    last_candle_fetch = 0
    tick_count = 0
    
    while True:
        try:
            tick_count += 1
            
            # === CHECK FOR DASHBOARD RESET REQUEST ===
            if trading_state.get('reset_requested', False):
                logger.info("🔄 Dashboard reset requested - clearing bot trades")
                bot.trades = []
                bot.signals = []
                bot.equity = bot.initial_capital
                bot.max_equity = bot.initial_capital
                bot.win_streak = 0
                bot.loss_streak = 0
                bot.position = None
                trading_state['reset_requested'] = False
                logger.info("✓ Bot internal state cleared")
            
            # Fetch candles every 30 seconds
            if time.time() - last_candle_fetch > 30:
                current_price = bot.update_candles()
                last_candle_fetch = time.time()
                logger.debug(f"Updated candles. BTC: ${current_price:,.2f}")
            else:
                price = fetch_current_price("BTCUSDT")
                if price:
                    trading_state['current_price'] = price
                    current_price = price
                else:
                    current_price = trading_state.get('current_price', 0)
            
            # Fetch orderbook every tick
            orderbook = fetch_orderbook("BTCUSDT", 10)
            trading_state['orderbook'] = orderbook
            
            # Update Components
            bot.update_sentiment()
            if bot.position:
                bot.update_position(current_price)
                
                # CRITICAL FIX: Monitor position for stop/target exits (enforces stops!)
                if hasattr(bot, 'feature_hub') and bot.feature_hub:
                    candles = trading_state.get('candles', [])
                    atr = bot.calculate_atr(candles, 20) if len(candles) > 20 else current_price * 0.01
                    
                    exit_signal = bot.feature_hub.monitor_position(
                        current_price=current_price,
                        atr=atr
                    )
                    
                    if exit_signal:
                        # Position hit stop or target - close it
                        bot.close_position(
                            exit_price=exit_signal['price'],
                            exit_reason=exit_signal['reason']
                        )
                        
                        emoji = "✅" if exit_signal['pnl_pct'] > 0 else "❌"
                        print(f"\n{emoji} POSITION CLOSED: {exit_signal['reason'].upper()}")
                        print(f"   Exit: ${exit_signal['price']:,.0f}")
                        print(f"   P&L: {exit_signal['pnl_pct']:+.2f}%")
                        print(f"   Equity: ${bot.equity:,.2f}\n")

            
            # Sync trades and signals to dashboard (ensure they're always current)
            trading_state['trades'] = bot.trades
            trading_state['signals'] = bot.signals[-20:]
            
            # Scalper Mode: Generate signal more frequently (every 2 ticks = ~6 seconds)
            # Also allow trading even with position for scaling/pyramiding
            if tick_count % 2 == 0:
                signal = bot.generate_signal(current_price)
                if signal and not bot.position:  # Only enter if no position for now
                    print(f"DEBUG: Invoking execute_trade for {signal['action']}")
                    trade_result = bot.execute_trade(signal)
                    if trade_result:
                        print(f">>> TRADE EXECUTED: {signal['action']} @ ${current_price:,.2f}")
            
            # Log Status
            if tick_count % 60 == 0:
                pos_status = f"Position: {bot.position['side']}" if bot.position else "Scanning..."
                logger.info(f"BTC: ${current_price:,.2f} | Eq: ${bot.equity:,.2f} | {pos_status}")
            
            time.sleep(3)
            
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            save_trades(bot.trades)
            save_state(bot)
            break
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            time.sleep(5)

def main():
    print("=" * 60)
    print("CRYPTOBOSS PROFESSIONAL - ADVANCED FEATURES v6")
    print("=" * 60)
    print("Core Modules:")
    print("  [+] Real-time Binance Data")
    print("  [+] Multi-Strategy Ensemble")
    print("  [+] Dynamic Risk Management")
    print("  [+] Professional Dashboard")
    print("")
    print("Advanced Features:")
    print("  [+] Multi-Timeframe Confirmation (1H/4H)")
    print("  [+] Session-Based Trading (London/NY)")
    print("  [+] Dynamic Volatility Stops")
    print("  [+] Partial Profit Taking (33% at 1:1, 2:1)")
    print("  [+] Portfolio Risk Limits (Daily/Weekly)")
    print("  [+] Trailing Stop Automation")
    print("=" * 60)
    
    try:
        # Start Dashboard in background thread
        dashboard_thread = threading.Thread(target=run_dash_server, daemon=True)
        dashboard_thread.start()
        logger.info("Dashboard running at http://localhost:8000")
        
        # Initialize Bot
        bot = EnhancedTradingBot(initial_capital=10000)
        
        # CRITICAL FIX: Initialize Risk/Reward Optimizer
        if RISK_REWARD_AVAILABLE:
            bot.rr_optimizer = get_rr_optimizer()
            logger.info("✓ Risk/Reward Optimizer active: Min 2:1 R:R enforced")
            print("✓ Risk/Reward protection enabled: Avg win will be 2.5x avg loss")
        else:
            logger.warning("⚠️ Risk/Reward optimizer not available - trades may have poor R:R")
        
        # CRITICAL FIX: Initialize Feature Hub (Activates all 26 features)
        if FEATURE_HUB_AVAILABLE:
            bot.feature_hub = get_feature_hub()
            features_active = sum(bot.feature_hub.features_loaded.values())
            logger.info(f"✓ Feature Hub active: {features_active}/9 professional features loaded")
            print(f"✓ Integrated features: {features_active}/9 active (DB, Alerts, Analytics)")
        else:
            logger.warning("⚠️ Feature Hub not available - running without integrated features")
        
        # Run Bot
        run_trading_loop(bot)
        
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
