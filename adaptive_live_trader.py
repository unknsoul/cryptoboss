"""
Adaptive Live Trader with Intelligent Strategy Selection
Automatically selects and switches strategies based on market conditions
"""

import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from core.exchange.binance_client import BinanceClient
from core.strategies.adaptive_selector import AdaptiveStrategySelector, MarketRegime
from core.strategies.factory import AdvancedStrategyFactory
from core.ml.predictor import MLPredictor
from core.ml.regime_detector import RegimeDetector
from core.risk.risk_manager import RiskManager
from core.sentiment.gemini_analyst import GeminiNewsAnalyst
from core.monitoring.logger import get_logger
from core.monitoring.metrics import get_metrics
from core.monitoring.alerting import get_alerts
from core.config import get_settings
from core.ml.signal_filter import SignalQualityFilter
from core.exchange.position_reconciler import PositionReconciler


logger = get_logger()
metrics = get_metrics()
alerts = get_alerts()
settings = get_settings()


class AdaptiveLiveTrader:
    """
    Advanced Live Trader with Adaptive Strategy Selection
    
    Features:
    - Automatic strategy selection based on market regime
    - Real-time performance tracking
    - Multiple strategy support
    - Signal quality filtering
    - Position reconciliation
    """
    
    def __init__(self, symbol: str = "BTCUSDT", capital: float = 10000):
        self.symbol = symbol
        self.capital = capital
        
        logger.info(
            f"Initializing Adaptive Live Trader",
            symbol=symbol,
            capital=capital
        )
        
        # Core components
        self.exchange = BinanceClient()
        self.risk_manager = RiskManager(capital=capital)
        
        # Strategy system
        self.strategy_selector = AdaptiveStrategySelector()
        self.available_strategies = [
            'statistical_arbitrage',
            'breakout_momentum',
            'volume_profile_trading',
            'enhanced_momentum',
            'mean_reversion',
            'liquidity_grab'
        ]
        
        # Create strategy instances
        self.strategies = {}
        for strategy_name in self.available_strategies:
            self.strategies[strategy_name] = AdvancedStrategyFactory.create(strategy_name)
        
        # AI components
        self.ml_predictor = MLPredictor()
        self.regime_detector = RegimeDetector()
        self.sentiment_analyzer = GeminiNewsAnalyst()
        
        # Quality control
        self.signal_filter = SignalQualityFilter(min_quality_score=70.0)
        self.position_reconciler = PositionReconciler(
            exchange=self.exchange,
            risk_manager=self.risk_manager,
            auto_correct=True
        )
        
        # State
        self.current_strategy_name = None
        self.current_position = None
        self.is_running = False
        self.last_reconciliation = None
        
        logger.info("✅ Adaptive Live Trader initialized successfully")
    
    def start(self):
        """Start the adaptive trading system"""
        logger.info("🚀 Starting Adaptive Live Trading")
        alerts.send_alert(
            "trading_started",
            f"Adaptive trading started on {self.symbol}",
            {"capital": self.capital, "strategies": len(self.available_strategies)}
        )
        
        try:
            # Connect to exchange
            self.exchange.connect()
            self.exchange.subscribe_ticker(self.symbol)
            
            # Set up ticker callback
            self.exchange.on('ticker', self._on_ticker_update)
            
            self.is_running = True
            
            # Main loop
            while self.is_running:
                try:
                    # Periodic position reconciliation (every 5 minutes)
                    if self.position_reconciler.should_reconcile(interval_seconds=300):
                        reconciliation_result = self.position_reconciler.reconcile()
                        logger.info("Position reconciliation complete", **reconciliation_result)
                    
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received, shutting down...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", error=str(e))
                    time.sleep(5)
        
        finally:
            self.stop()
    
    def _on_ticker_update(self, ticker_data: Dict[str, Any]):
        """Handle ticker updates"""
        try:
            # Get market data
            market_data = self._prepare_market_data(ticker_data)
            
            if market_data is None:
                return
            
            # Select best strategy for current conditions
            best_strategy_name = self.strategy_selector.select_best_strategy(
                market_data,
                self.available_strategies
            )
            
            # Use selected strategy
            current_strategy = self.strategies[best_strategy_name]
            self.current_strategy_name = best_strategy_name
            
            # Generate signal
            signal = self._generate_combined_signal(current_strategy, market_data)
            
            if signal is None:
                return
            
            # Filter signal quality
            quality = self.signal_filter.calculate_quality_score(signal)
            
            if not quality['should_trade']:
                logger.debug(
                    f"Signal filtered out (low quality)",
                    quality_score=quality['quality_score'],
                    grade=quality['grade']
                )
                return
            
            # Adjust position size based on quality
            base_size = self.risk_manager.calculate_position_size(
                signal['stop'],
                ticker_data['price']
            )
            adjusted_size = base_size * quality['position_multiplier']
            
            # Execute trade
            if signal['action'] in ['LONG', 'SHORT']:
                self._execute_trade(
                    signal,
                    adjusted_size,
                    ticker_data['price'],
                    quality
                )
        
        except Exception as e:
            logger.error(f"Error processing ticker update: {e}", error=str(e))
    
    def _prepare_market_data(self, ticker_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare market data for strategy selection"""
        try:
            # Fetch recent OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1h', limit=200)
            
            if not ohlcv or len(ohlcv) < 100:
                return None
            
            # Extract arrays
            closes = np.array([x[4] for x in ohlcv])
            highs = np.array([x[2] for x in ohlcv])
            lows = np.array([x[3] for x in ohlcv])
            volumes = np.array([x[5] for x in ohlcv])
            
            # Calculate metrics
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
            avg_volatility = np.std(returns[-50:]) if len(returns) >= 50 else 0.02
            
            # Trend strength (R-squared)
            from scipy import stats
            x = np.arange(len(closes[-50:]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, closes[-50:])
            trend_strength = r_value ** 2
            
            # ADX calculation (simplified)
            adx = self._calculate_adx(highs, lows, closes)
            
            # Get sentiment
            sentiment_score = 0
            try:
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(self.symbol)
                sentiment_score = sentiment_result.get('score', 0)
            except:
                pass
            
            # Get regime
            regime_info = self.regime_detector.predict_regime(closes, volumes)
            
            market_data = {
                'prices': closes,
                'highs': highs,
                'lows': lows,
                'volumes': volumes,
                'volume': volumes[-1],
                'avg_volume': np.mean(volumes[-20:]),
                'volatility': volatility,
                'avg_volatility': avg_volatility,
                'sentiment_score': sentiment_score,
                'trend_strength': trend_strength,
                'adx': adx,
                'regime': regime_info.get('regime'),
                'current_price': ticker_data['price']
            }
            
            return market_data
        
        except Exception as e:
            logger.error(f"Error preparing market data: {e}")
            return None
    
    def _calculate_adx(self, highs, lows, closes, period: int = 14):
        """Calculate ADX indicator"""
        if len(closes) < period + 1:
            return 20  # Default
        
        # Simplified ADX calculation
        tr_list = []
        for i in range(-period, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        
        atr = np.mean(tr_list)
        dm_plus = max(highs[-1] - highs[-2], 0)
        dm_minus = max(lows[-2] - lows[-1], 0)
        
        di_plus = (dm_plus / atr * 100) if atr > 0 else 0
        di_minus = (dm_minus / atr * 100) if atr > 0 else 0
        
        dx = abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8) * 100
        
        return dx
    
    def _generate_combined_signal(self, strategy, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate signal combining strategy + ML + filters"""
        try:
            # Get base strategy signal
            signal = strategy.signal(
                market_data['highs'],
                market_data['lows'],
                market_data['prices'],
                market_data['volumes']
            )
            
            if signal is None or signal['action'] == 'HOLD':
                return None
            
            # Enhance with additional data
            signal['ml_confidence'] = signal.get('confidence', 0.5)
            signal['direction'] = signal['action']
            signal['volume'] = market_data['volume']
            signal['avg_volume'] = market_data['avg_volume']
            signal['volatility'] = market_data['volatility']
            signal['avg_volatility'] = market_data['avg_volatility']
            signal['sentiment_score'] = market_data['sentiment_score']
            signal['trend_strength'] = market_data['trend_strength']
            signal['timeframe_alignment'] = 2  # Simplified
            
            # Get order book data if available
            try:
                ticker = self.exchange.get_ticker(self.symbol)
                # Simplified - would use real order book
                signal['orderbook_imbalance'] = 0.1 if signal['action'] == 'LONG' else -0.1
            except:
                signal['orderbook_imbalance'] = 0
            
            return signal
        
        except Exception as e:
            logger.error(f"Error generating combined signal: {e}")
            return None
    
    def _execute_trade(self, signal: Dict[str, Any], size: float, 
                      price: float, quality: Dict[str, Any]):
        """Execute a trade"""
        try:
            # Check risk limits
            if not self.risk_manager.can_open_position(self.symbol, size, price):
                logger.warning("Trade rejected by risk manager")
                return
            
            # Place order
            side = "BUY" if signal['action'] == 'LONG' else "SELL"
            
            logger.info(
                f"🎯 Executing {side} trade",
                strategy=self.current_strategy_name,
                size=size,
                price=price,
                quality_score=quality['quality_score'],
                quality_grade=quality['grade']
            )
            
            order = self.exchange.place_order(
                symbol=self.symbol,
                side=side,
                order_type='MARKET',
                quantity=size
            )
            
            # Update risk manager
            self.risk_manager.update_position(
                self.symbol,
                size if side == "BUY" else -size,
                order['price']
            )
            
            # Record trade
            trade_data = {
                'symbol': self.symbol,
                'strategy': self.current_strategy_name,
                'side': side,
                'size': size,
                'price': order['price'],
                'quality_score': quality['quality_score'],
                'signal_confidence': signal.get('ml_confidence', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.log_trade(trade_data)
            metrics.increment(f"trades_{self.current_strategy_name}")
            
            alerts.send_alert(
                "trade_executed",
                f"{side} {size} {self.symbol} @ {order['price']}",
                trade_data
            )
        
        except Exception as e:
            logger.error(f"Trade execution failed: {e}", error=str(e))
            alerts.send_alert(
                "trade_failed",
                f"Failed to execute {signal['action']} trade",
                {"error": str(e), "signal": signal}
            )
    
    def stop(self):
        """Stop trading"""
        logger.info("🛑 Stopping Adaptive Live Trader")
        
        self.is_running = False
        
        # Disconnect from exchange
        if self.exchange:
            self.exchange.disconnect()
        
        # Final reconciliation
        if self.position_reconciler:
            self.position_reconciler.reconcile()
        
        # Get statistics
        stats = self.strategy_selector.get_statistics()
        
        logger.info("Trading stopped", strategy_switches=stats['total_switches'])
        alerts.send_alert(
            "trading_stopped",
            "Adaptive trading system stopped",
            stats
        )


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 ADAPTIVE LIVE TRADER")
    print("=" * 70)
    print("\nStarting adaptive trading system with intelligent strategy selection...")
    print("\nAvailable strategies:")
    
    strategies = AdvancedStrategyFactory.get_all_strategies()
    for i, name in enumerate(strategies[:6], 1):  # Show first 6
        print(f"  {i}. {name}")
    
    print(f"\n... and {len(strategies) - 6} more strategies")
    
    print("\n" + "⚠️  WARNING " * 7)
    print("This will connect to Binance and execute REAL trades!")
    print("Make sure you're using TESTNET or small capital")
    print("=" * 70)
    
    # Uncomment to run
    # trader = AdaptiveLiveTrader(symbol="BTCUSDT", capital=10000)
    # trader.start()
    
    print("\n✅ Adaptive Live Trader ready (commented out for safety)")
    print("Uncomment the last 2 lines in the script to run")
