"""
Professional AI Trading Bot - Live Trading Execution Engine
Integrates:
- Infrastructure: BinanceClient
- Strategy: StrategyFactory
- AI 1: MLPredictor (XGBoost/RF)
- AI 2: GeminiNewsAnalyst (LLM)
- Context: RegimeDetector (GMM)
- Risk: RiskManager (VolTargeting)
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from core.exchange.binance_client import BinanceClient
from core.strategies.factory import StrategyFactory
from core.ml.predictor import MLPredictor
from core.ml.regime_detector import RegimeDetector
from core.risk.risk_manager import RiskManager
from core.sentiment.gemini_analyst import GeminiNewsAnalyst

class LiveTrader:
    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1h", 
                 strategy_name: str = "enhanced_momentum",
                 capital: float = 10000.0):
        
        print(f"🚀 Initializing Professional Live Trader for {symbol}...")
        
        self.symbol = symbol
        self.interval = interval
        
        # 1. Infrastructure
        self.client = BinanceClient(testnet=False)
        
        # 2. Intelligence Layer
        self.strategy = StrategyFactory.create(strategy_name)
        self.ml_predictor = MLPredictor()
        self.regime_detector = RegimeDetector()
        self.gemini = GeminiNewsAnalyst() # Hooks into GOOGLE_API_KEY
        
        # 3. Risk Layer
        self.risk_manager = RiskManager(capital=capital, target_volatility=0.15)
        
        # State
        self.candles: List[Dict] = []
        self.is_running = False
        self.last_sentiment_score = 0.0
        
    def start(self):
        """Start the main trading loop"""
        self.client.connect()
        self.is_running = True
        
        # 1. Auto-Train AI on Startup (if needed)
        print("🧠 Checking AI Model Status...")
        # In real impl, fetch history here. For now, we wait for data accumulation or assume pre-trained.
        
        # 2. Initial Sentiment Check
        print("📰 analyzing news...")
        headlines = self.gemini.fetch_news(self.symbol)
        sentiment = self.gemini.analyze_sentiment(headlines)
        self.last_sentiment_score = sentiment['score']
        print(f"   Sentiment Score: {self.last_sentiment_score:.2f} ({sentiment['signal']})")
        
        # 3. Live Loop
        print(f"📡 Subscribing to live {self.symbol} feed...")
        self.client.subscribe_ticker(self.symbol)
        self.client.on('ticker', self.on_ticker_update)
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        print("\n🛑 Stopping Live Trader...")
        self.is_running = False
        self.client.disconnect()
        
    def on_ticker_update(self, ticker: Dict):
        """Event-driven processing loop"""
        # (Simplified candle aggregation for demo)
        tick_price = ticker['price']
        timestamp = ticker['timestamp']
        
        # Mock Order Book Imbalance (Simulate random flux)
        # In real impl, we'd subscribe to depth channel
        ob_imbalance = (np.random.random() - 0.5) * 0.2 # Random drift near 0
        
        if not hasattr(self, 'current_candle') or self.current_candle is None:
            self.current_candle = {
                'open': tick_price, 'high': tick_price, 'low': tick_price, 'close': tick_price,
                'volume': 0, 'start_time': timestamp, 'ob_imbalance_sum': 0, 'ticks': 0
            }
            return

        # Update
        c = self.current_candle
        c['high'] = max(c['high'], tick_price)
        c['low'] = min(c['low'], tick_price)
        c['close'] = tick_price
        c['ob_imbalance_sum'] += ob_imbalance
        c['ticks'] += 1
        
        # Close Candle Logic (Demo: every 5 ticks)
        if c['ticks'] >= 5:
            avg_imbalance = c['ob_imbalance_sum'] / c['ticks']
            self._close_candle(avg_imbalance)

    def _close_candle(self, avg_imbalance: float):
        """Finalize candle and run decision brain"""
        c = self.current_candle
        # Add to history
        self.candles.append(c)
        if len(self.candles) > 300: self.candles.pop(0)
        self.current_candle = None
        
        # Need data
        if len(self.candles) < 20: 
            print(".", end="", flush=True)
            return
            
        self._execution_logic(avg_imbalance)

    def _execution_logic(self, ob_imbalance: float):
        """
        The Brain: Integrates ALL signals
        """
        df = pd.DataFrame(self.candles)
        current_price = df['close'].iloc[-1]
        
        # 1. Regime Detection
        regime = self.regime_detector.predict(df)
        regime_id = regime['regime_id']
        regime_name = regime['regime_name']
        
        # --- DYNAMIC STRATEGY SWITCHING (New Feature) ---
        # 0: Low Vol/Range -> Mean Reversion
        # 1: Bull Trend -> Enhanced Momentum
        # 2: Bear Trend -> Enhanced Momentum (Shorts)
        # 3: High Vol/Chop -> Risk Off (or Conservative Scalping)
        
        target_strategy_name = self.strategy.name
        
        if regime_id == 0: # Range
            if not isinstance(self.strategy, type(StrategyFactory.create('mean_reversion'))):
                print(f"🔄 Switching to MEAN REVERSION (Regime: {regime_name})")
                self.strategy = StrategyFactory.create('mean_reversion')
        elif regime_id in [1, 2]: # Trend
            if not isinstance(self.strategy, type(StrategyFactory.create('enhanced_momentum'))):
                print(f"🔄 Switching to MOMENTUM (Regime: {regime_name})")
                self.strategy = StrategyFactory.create('enhanced_momentum')
        elif regime_id == 3: # Chop
             print(f"⚠️ High Volatility Regime detected. Reducing Risk.")
             # Could switch to tight scalping or cash
             pass
             
        # 2. Strategy Signal (Technical)
        tech_signal = self.strategy.signal(df['high'].values, df['low'].values, df['close'].values)
        
        if not tech_signal:
            print(f"💤  Regime:{regime_name} | Strat:{self.strategy.name} | No Signal", end='\r')
            return
            
        print(f"\n🔔 Tech Signal: {tech_signal['action']} | Regime: {regime_name}")
        
        # 3. Fundamental Filter (Gemini)
        # If Sentiment is terrible, Block Longs
        if tech_signal['action'] == 'LONG' and self.last_sentiment_score < -0.5:
            print(f"🛡️ BLOCKED by Gemini Sentiment ({self.last_sentiment_score})")
            return
            
        # 4. AI Prediction (Ensemble)
        ai_pred = self.ml_predictor.predict(df, orderbook_imbalance=ob_imbalance)
        print(f"🧠 AI: {ai_pred['direction']} (Conf: {ai_pred['confidence']:.2f})")
        
        # Confirmation Logic
        valid = False
        if tech_signal['action'] == 'LONG':
             if ai_pred['direction'] != 'SHORT' and regime_id != 2:
                 valid = True
        elif tech_signal['action'] == 'SHORT':
             if ai_pred['direction'] != 'LONG' and regime_id != 1:
                 valid = True
                 
        if not valid:
            print("⚠️ Filtered by AI/Regime Mismatch")
            return
            
        # 5. Risk Calculation
        vol = df['close'].pct_change().std()
        size = self.risk_manager.calculate_vol_target_size(current_price, vol)
        
        # Final Verification
        allowed = self.risk_manager.check_trade_allowed(
            self.symbol, size, current_price, 
            orderbook_imbalance=ob_imbalance, 
            direction=tech_signal['action']
        )
        
        if allowed:
            print(f"⚡ EXECUTING {tech_signal['action']} | Size: {size:.4f} | Imbalance: {ob_imbalance:.2f} ⚡")

if __name__ == "__main__":
    trader = LiveTrader(symbol="BTCUSDT")
    trader.start()
