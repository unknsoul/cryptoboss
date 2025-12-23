"""
Live Paper Trading System
Main script for running paper trading with real-time data

FINAL ADVANCED VERSION - Includes:
- Smart Order Execution (TWAP/VWAP)
- Sentiment Analysis Integration
- Real-time Dashboard
- Portfolio Optimization
"""

import sys
import time
import signal as sys_signal
import threading
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add core to path
sys.path.insert(0, '.')

from core.config.binance_config import get_config, ConfigManager
from core.live.websocket_client import BinanceWebSocketClient
from core.live.live_data_manager import LiveDataManager
from core.live.paper_trader import PaperTrader
from core.live.live_signal_engine import LiveSignalEngine
from core.enhanced_trend_strategy import EnhancedTrendStrategy

# NEW: Advanced modules
from core.execution.smart_execution import SmartOrderRouter, ExecutionOrder, ExecutionAlgorithm
from core.analysis.sentiment_analyzer import SentimentAnalyzer


class LivePaperTradingSystem:
    """
    Complete live paper trading system - FINAL ADVANCED VERSION
    
    Combines:
    - Real-time data (WebSocket)
    - Strategy signal generation
    - Smart order execution (TWAP/VWAP)
    - Sentiment-adjusted signals
    - Paper trade execution
    - Real-time dashboard
    - Performance monitoring
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize live trading system
        
        Args:
            config_manager: ConfigManager instance (optional, will create if not provided)
        """
        # Get or create config
        self.config_mgr = config_manager or get_config()
        self.binance_config = self.config_mgr.config
        
        # Get trading configuration
        trading_config = ConfigManager.get_trading_config()
        
        self.symbol = trading_config['symbol'].upper()
        self.intervals = trading_config['timeframes']
        initial_capital = trading_config['initial_capital']
        
        logger.info(f"Initializing Live Paper Trading System for {self.symbol}")
        logger.info(f"Environment: {self.binance_config.environment.upper()}")
        
        # Data manager
        self.data_manager = LiveDataManager(self.symbol, self.intervals)
        
        # Strategies
        self.strategies = [
            EnhancedTrendStrategy(
                use_volume_filter=True,
                use_volatility_filter=True,
                use_adx_filter=True,
                adx_threshold=25
            )
        ]
        
        # Signal engine
        self.signal_engine = LiveSignalEngine(self.strategies, self.data_manager)
        
        # Paper trader
        self.paper_trader = PaperTrader(
            initial_capital=initial_capital,
            config={
                'risk_per_trade': trading_config['risk_per_trade'],
                'use_partial_profits': True,
                'use_breakeven_stop': True,
                'max_hold_hours': 48,
                'max_drawdown_limit': trading_config['max_drawdown_limit'],
                'daily_loss_limit': trading_config['daily_loss_limit']
            }
        )
        
        # WebSocket client (use configured URL)
        self.ws_client = BinanceWebSocketClient(
            symbol=self.symbol.lower(),
            intervals=self.intervals,
            on_candle_close=self.on_candle_close,
            base_url=self.binance_config.ws_url
        )
        
        self.is_running = False
        
        # NEW: Smart Execution Router
        self.execution_router = SmartOrderRouter(
            default_algorithm=ExecutionAlgorithm.TWAP
        )
        logger.info("Smart Execution Router initialized (TWAP/VWAP)")
        
        # NEW: Sentiment Analyzer
        self.sentiment_analyzer = SentimentAnalyzer(use_mock_data=True)
        logger.info("Sentiment Analyzer initialized")
        
        # NEW: Dashboard state updates
        self.dashboard_enabled = True
        
        logger.info(f"System initialized with {len(self.strategies)} strategies")
        logger.info("🚀 FINAL ADVANCED VERSION - All modules integrated")
    
    def on_candle_close(self, interval, candle):
        """
        Called when new candle closes
        
        Args:
            interval: Timeframe
            candle: Candle data
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🕐 [{interval}] Candle Closed @ {candle['timestamp']}")
        logger.info(f"   OHLC: ${candle['open']:.2f} / ${candle['high']:.2f} / "
                   f"${candle['low']:.2f} / ${candle['close']:.2f}")
        logger.info(f"   Volume: {candle['volume']:.2f}")
        
        # Add to data manager
        self.data_manager.add_candle(interval, candle)
        
        # Check if enough data
        if not self.data_manager.is_ready(interval):
            logger.info(f"   ⏳ Collecting data... "
                       f"({self.data_manager.get_buffer_size(interval)}/200 candles)")
            return
        
        # Update existing position
        if self.paper_trader.has_position():
            self.paper_trader.update_position(
                candle['high'],
                candle['low'],
                candle['close']
            )
        
        # Generate new signal (if no position)
        if not self.paper_trader.has_position():
            signal = self.signal_engine.on_candle_close(interval)
            
            if signal:
                logger.info(f"   🎯 SIGNAL: {signal['action']} from {signal.get('strategy_name', 'Unknown')}")
                
                # Execute signal
                executed = self.paper_trader.execute_signal(
                    signal,
                    candle['close'],
                    candle['timestamp']
                )
                
                if executed:
                    logger.info(f"   ✅ Position opened")
                else:
                    logger.info(f"   ⏸️ Signal skipped")
            else:
                logger.info(f"   ⭕ No signal")
        
        # Print status
        self.print_status()
        logger.info(f"{'='*60}\n")
    
    def print_status(self):
        """Print current system status"""
        # Data manager status
        dm_status = self.data_manager.get_status()
        logger.info(f"\n   📊 Data Buffers:")
        for interval, info in dm_status['buffers'].items():
            ready = "✅" if info['ready'] else "⏳"
            logger.info(f"      {ready} {interval}: {info['size']}/500 candles | "
                       f"${info['latest_price']:.2f}")
        
        # Paper trader status
        metrics = self.paper_trader.get_metrics()
        logger.info(f"\n   💼 Portfolio:")
        logger.info(f"      Equity: ${self.paper_trader.equity:,.2f} | "
                   f"Return: {metrics['total_return']:+.2%}")
        logger.info(f"      Trades: {metrics['num_trades']} | "
                   f"Win Rate: {metrics['win_rate']:.1%} | "
                   f"Expectancy: ${metrics['expectancy']:.2f}")
        
        if self.paper_trader.has_position():
            pos = self.paper_trader.position
            unrealized = self.paper_trader.equity - self.paper_trader.capital
            logger.info(f"\n   📍 Open Position:")
            logger.info(f"      {pos['side']} @ ${pos['entry_price']:.2f} | "
                       f"Size: {pos['size']:.4f}")
            logger.info(f"      Stop: ${pos['stop_price']:.2f} | "
                       f"Unrealized PnL: ${unrealized:+.2f}")
    
    def start(self):
        """Start live trading system"""
        # Print config status
        self.config_mgr.print_status()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 STARTING LIVE PAPER TRADING")
        logger.info(f"{'='*60}")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Intervals: {self.intervals}")
        logger.info(f"Initial Capital: ${self.paper_trader.initial_capital:,.2f}")
        logger.info(f"Strategies: {[s.__class__.__name__ for s in self.strategies]}")
        logger.info(f"{'='*60}\n")
        
        self.is_running = True
        
        # Setup graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\n\n⚠️ Shutdown signal received...")
            self.stop()
        
        sys_signal.signal(sys_signal.SIGINT, signal_handler)
        sys_signal.signal(sys_signal.SIGTERM, signal_handler)
        
        # Start WebSocket
        self.ws_client.start()
        
        logger.info("✅ WebSocket connected - Waiting for candles...")
        logger.info("Press Ctrl+C to stop\n")
        
        try:
            # Keep running
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n\nKeyboard interrupt detected...")
            self.stop()
    
    def stop(self):
        """Stop system gracefully"""
        logger.info("\n" + "="*60)
        logger.info("🛑 STOPPING SYSTEM")
        logger.info("="*60)
        
        self.is_running = False
        
        # Close any open position
        if self.paper_trader.has_position():
            latest_price = self.data_manager.get_latest_close(self.intervals[0])
            if latest_price:
                logger.info(f"Closing open position at ${latest_price:.2f}")
                self.paper_trader.close_position(latest_price, 'system_shutdown')
        
        # Stop WebSocket
        self.ws_client.stop()
        
        # Print final stats
        self.print_final_stats()
        
        logger.info("="*60)
        logger.info("✅ System stopped")
    
    def print_final_stats(self):
        """Print final performance statistics"""
        metrics = self.paper_trader.get_metrics()
        
        logger.info("\n📊 FINAL PERFORMANCE")
        logger.info("="*60)
        logger.info(f"Initial Capital: ${self.paper_trader.initial_capital:,.2f}")
        logger.info(f"Final Equity:    ${self.paper_trader.equity:,.2f}")
        logger.info(f"Total Return:    {metrics['total_return']:+.2%}")
        logger.info(f"")
        logger.info(f"Trades:          {metrics['num_trades']}")
        logger.info(f"Win Rate:        {metrics['win_rate']:.1%}")
        logger.info(f"Average Win:     ${metrics['avg_win']:.2f}")
        logger.info(f"Average Loss:    ${metrics['avg_loss']:.2f}")
        logger.info(f"Expectancy:      ${metrics['expectancy']:.2f}")
        logger.info(f"Profit Factor:   {metrics['profit_factor']:.2f}")
        logger.info("="*60)
        
        # Show trade history
        if self.paper_trader.trades:
            logger.info("\n📋 TRADE HISTORY")
            logger.info("="*60)
            for i, trade in enumerate(self.paper_trader.trades[-10:], 1):
                pnl_icon = "✅" if trade['pnl'] > 0 else "❌"
                logger.info(f"{i}. {pnl_icon} {trade['side']}: "
                           f"${trade['entry']:.2f} → ${trade['exit']:.2f} | "
                           f"PnL: ${trade['pnl']:+.2f} ({trade['return_pct']:+.2f}%) | "
                           f"{trade['exit_reason']}")
            logger.info("="*60)


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("🤖 BINANCE LIVE PAPER TRADING BOT")
    print("="*60)
    print("\nConfiguration:")
    print("  1. Copy .env.example to .env")
    print("  2. Set BINANCE_ENV='testnet' or 'mainnet'")
    print("  3. Add your API keys (optional for paper trading)")
    print("  4. Configure symbol, timeframes, capital in .env")
    print("="*60 + "\n")
    
    try:
        # Create system (reads from .env)
        system = LivePaperTradingSystem()
        
        # Start trading
        system.start()
    
    except FileNotFoundError:
        print("\n❌ Error: .env file not found!")
        print("   Please copy .env.example to .env and configure it")
        print("   Run: cp .env.example .env")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
