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

# Add project root to path
sys.path.insert(0, '.')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
TRADES_FILE = DATA_DIR / 'trades.json'
STATE_FILE = DATA_DIR / 'bot_state.json'

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

# Import Trade Quality and Order Flow Analysis
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

class DateTimeEncoder(json.JSONEncoder):
    """Handle datetime objects in JSON"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def save_trades(trades):
    """Save all trades to JSON file"""
    try:
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades, f, indent=4, cls=DateTimeEncoder)
        logger.info(f"Saved {len(trades)} trades to disk")
    except Exception as e:
        logger.error(f"Failed to save trades: {e}")

def load_trades():
    """Load trades from JSON file"""
    if TRADES_FILE.exists():
        try:
            with open(TRADES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trades: {e}")
    return []

def save_state(bot):
    """Save bot state"""
    try:
        state = {
            'equity': bot.equity,
            'initial_capital': bot.initial_capital,
            'max_equity': bot.max_equity,
            'win_streak': bot.win_streak,
            'loss_streak': bot.loss_streak,
            'position': bot.position
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4, cls=DateTimeEncoder)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")

def load_state():
    """Load bot state"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    return None

# ============ FETCH REAL BTC DATA ============

def fetch_binance_klines(symbol="BTCUSDT", interval="5m", limit=300):
    """Fetch real candlestick data from Binance"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        candles = []
        for k in data:
            # Binance returns milliseconds, convert to seconds for chart
            candles.append({
                'time': int(k[0] / 1000),  # Convert ms to seconds
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })
        
        return candles
    except Exception as e:
        logger.error(f"Failed to fetch Binance data: {e}")
        return None

def fetch_current_price(symbol="BTCUSDT"):
    """Fetch current BTC price"""
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": symbol}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        logger.error(f"Failed to fetch price: {e}")
        return None

def fetch_orderbook(symbol="BTCUSDT", limit=10):
    """Fetch order book from Binance"""
    try:
        url = "https://api.binance.com/api/v3/depth"
        params = {"symbol": symbol, "limit": limit}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        bids = [[float(b[0]), float(b[1])] for b in data.get('bids', [])]
        asks = [[float(a[0]), float(a[1])] for a in data.get('asks', [])]
        
        spread = 0
        if bids and asks:
            spread = asks[0][0] - bids[0][0]
        
        return {
            'bids': bids,
            'asks': asks,
            'spread': spread
        }
    except Exception as e:
        logger.error(f"Failed to fetch orderbook: {e}")
        return {'bids': [], 'asks': [], 'spread': 0}

# ============ ENHANCED TRADING BOT ============

class EnhancedTradingBot:
    """Professional Trading Bot with Multi-Strategy Ensemble"""
    
    def __init__(self, initial_capital=10000):
        # Load previous state if exists
        saved_state = load_state()
        
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
        self.position_sizer = DynamicPositionSizer(risk_per_trade=0.015)
        self.sentiment_analyzer = SentimentAnalyzer(use_mock_data=True)  # Use mock for now until real keys
        self.probabilistic_generator = ProbabilisticSignalGenerator(buy_threshold=0.55, sell_threshold=0.55)
        
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
            self.quality_scorer = TradeQualityScorer(min_score=75)  # UPGRADED: 75 from 70
            self.order_flow = OrderFlowAnalyzer(imbalance_threshold=0.3)
            logger.info("Trade Quality Scoring ENABLED 🎯 (min score: 75 - UPGRADED)")
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

    def generate_signal(self, current_price):
        """Generate trading signal using Probabilistic AI Engine"""
        
        # 1. Prepare Dataframes
        candles_5m = trading_state.get('candles', [])
        if not candles_5m or len(candles_5m) < 100:
            return None
            
        df_5m = pd.DataFrame(candles_5m)
        
        try:
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
            
            if not prob_signal.should_trade:
                return None
                
            # 7. Convert to Bot Signal Format
            signal = {
                'action': prob_signal.action.value, # 'BUY' -> 'LONG' mapping needed if enum differs
                'confidence': prob_signal.confidence,
                'reasons': prob_signal.reasons.copy(),
                'price': current_price,
                'atr': atr,
                'timestamp': datetime.now().isoformat()
            }
            
            # Map BUY/SELL to LONG/SHORT for compatibility
            if signal['action'] == 'BUY': signal['action'] = 'LONG'
            if signal['action'] == 'SELL': signal['action'] = 'SHORT'
            
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
                
                # Log quality assessment
                with open('debug_trace.txt', 'a') as f:
                    f.write(f"\n=== QUALITY SCORE: {quality_score}/100 ({grade}) ===\n")
                    for factor, pts in breakdown.items():
                        f.write(f"  {factor}: {pts}\n")
                
                if not should_trade:
                    logger.info(f"Signal REJECTED by quality filter: Score {quality_score}/100 ({grade})")
                    print(f"DEBUG: Trade REJECTED - Quality {quality_score}/100 (need 75+)")
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
                
                # On 3+ loss streak, require higher quality score
                if self.loss_streak >= 3 and signal.get('quality_score', 100) < 80:
                    logger.info(f"Signal REJECTED: Loss streak ({self.loss_streak}) requires 80+ quality")
                    print(f"DEBUG: Loss streak protection - need 80+ quality")
                    return None
            
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
            
            # Use Dynamic Position Sizer
            sizing = self.position_sizer.calculate_size(
                equity=self.equity,
                price=price,
                atr=atr,
                confidence=confidence
            )
            
            size = sizing['size']
            print(f"DEBUG: Initial Size: {size}")
            
            # Apply session and risk adjustments from advanced features
            if self.advanced_features:
                size = self.advanced_features.get_size_adjustment(size, self.equity)
                print(f"DEBUG: Adjusted Size: {size}")
            
            if size <= 0:
                msg = f"Signal ignored: Size too small (Size: {size}, Risk: ${sizing.get('risk_amount', 0):.2f})"
                logger.info(msg)
                print(f"DEBUG: {msg}")
                return False
            
            # Dynamic Targets based on Volatility Regime
            # Use dynamic multipliers from signal if available
            # FIXED: Tighter stops (1.0x ATR) and larger targets (2.5-3x ATR)
            # This ensures minimum 2.5:1 R:R ratio
            sl_mult = signal.get('sl_multiplier', 1.0)  # Tighter stop: 1x ATR
            tp_mult = signal.get('tp_multiplier', 2.5)  # Larger target: 2.5x ATR
            
            # Ensure minimum 2.5:1 R:R ratio
            if tp_mult < sl_mult * 2.5:
                tp_mult = sl_mult * 2.5
            
            # Confidence Scalar: If conf > 0.7, expand TP further
            if confidence >= 0.8:
                tp_mult *= 1.3  # High confidence = bigger target
            elif confidence >= 0.7:
                tp_mult *= 1.15
            
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
                'entry_time': datetime.now().isoformat(),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'unrealized_pnl': 0,
                'reasons': signal.get('reasons', []),
                'first_profit_taken': False,  # For partial profit tracking
                'second_profit_taken': False,
                # ML Data Preservation (for training)
                'ml_features': signal.get('ml_features'),
                'ml_prediction': signal.get('ml_prediction')
            }
            
            # CRITICAL: Actually assign the position!
            self.position = new_position
            
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
                
                # UPGRADE 1: Move to breakeven at 0.5R (was 1R)
                if current_profit >= initial_risk * 0.5:
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
                
                # Move to breakeven at 0.5R
                if current_profit >= initial_risk * 0.5:
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
            # === PARTIAL PROFIT TAKING ===
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
                'entry': round(pos['entry_price'], 2),
                'exit': round(exit_price, 2),
                'size': round(pos['size'], 6),
                'pnl': round(pnl, 2),
                'return_pct': round(return_pct, 2),
                'exit_reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            
            self.trades.append(trade)
            self.position = None
            self.equity = self.initial_capital + sum(t['pnl'] for t in self.trades)
            
            save_trades(self.trades)
            save_state(self)
            
            trading_state['trades'] = self.trades
            trading_state['equity'] = round(self.equity, 2)
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
        
        # Run Bot
        run_trading_loop(bot)
        
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
