"""
Live Data Manager
Maintains real-time OHLCV data buffers for multiple timeframes
"""

import numpy as np
from collections import deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class LiveDataManager:
    """
    Manages real-time OHLCV data buffers for trading strategies
    
    Features:
    - Multiple timeframe support
    - Fixed-size buffers (last 500 candles)
    - Thread-safe updates
    - Numpy array conversion for strategies
    """
    
    def __init__(self, symbol, intervals=None, buffer_size=500):
        """
        Initialize data manager
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            intervals: List of timeframes (e.g., ['1h', '4h'])
            buffer_size: Maximum candles to store per timeframe
        """
        self.symbol = symbol
        self.intervals = intervals or ['1h']
        self.buffer_size = buffer_size
        
        # Data buffers for each timeframe
        self.buffers = {}
        for interval in self.intervals:
            self.buffers[interval] = deque(maxlen=buffer_size)
        
        # Callbacks for new candles
        self.callbacks = {}
        
        # Latest price tracking
        self.latest_prices = {}
        
        logger.info(f"Initialized data manager for {symbol} - {intervals}")
    
    def add_candle(self, interval, candle):
        """
        Add new candle to buffer
        
        Args:
            interval: Timeframe (e.g., '1h')
            candle: Dict with OHLCV data
        """
        if interval not in self.buffers:
            logger.warning(f"Unknown interval: {interval}")
            return
        
        # Add to buffer
        self.buffers[interval].append(candle)
        
        # Update latest price
        self.latest_prices[interval] = candle['close']
        
        logger.debug(f"Added candle to {interval} buffer (size: {len(self.buffers[interval])})")
        
        # Trigger callbacks
        if interval in self.callbacks:
            for callback in self.callbacks[interval]:
                try:
                    callback(interval, candle)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
    
    def register_callback(self, interval, callback):
        """Register callback for new candles"""
        if interval not in self.callbacks:
            self.callbacks[interval] = []
        self.callbacks[interval].append(callback)
        logger.info(f"Registered callback for {interval}")
    
    def get_data(self, interval, lookback=None):
        """
        Get data arrays for strategy
        
        Args:
            interval: Timeframe
            lookback: Number of candles (None = all available)
        
        Returns:
            dict: {'highs': array, 'lows': array, 'closes': array, 'volumes': array, 'timestamps': array}
        """
        if interval not in self.buffers:
            return None
        
        buffer = self.buffers[interval]
        if len(buffer) == 0:
            return None
        
        # Apply lookback
        if lookback and lookback < len(buffer):
            data = list(buffer)[-lookback:]
        else:
            data = list(buffer)
        
        # Convert to numpy arrays
        return {
            'highs': np.array([c['high'] for c in data]),
            'lows': np.array([c['low'] for c in data]),
            'closes': np.array([c['close'] for c in data]),
            'volumes': np.array([c['volume'] for c in data]),
            'timestamps': [c['timestamp'] for c in data],
            'open': np.array([c['open'] for c in data])
        }
    
    def get_latest_close(self, interval):
        """Get latest close price"""
        return self.latest_prices.get(interval)
    
    def get_buffer_size(self, interval):
        """Get current buffer size"""
        return len(self.buffers.get(interval, []))
    
    def is_ready(self, interval, min_candles=200):
        """Check if buffer has enough data for strategies"""
        return self.get_buffer_size(interval) >= min_candles
    
    def get_all_data(self):
        """Get data for all timeframes (for multi-timeframe strategies)"""
        all_data = {}
        for interval in self.intervals:
            all_data[interval] = self.get_data(interval)
        return all_data
    
    def clear(self, interval=None):
        """Clear buffer(s)"""
        if interval:
            if interval in self.buffers:
                self.buffers[interval].clear()
                logger.info(f"Cleared {interval} buffer")
        else:
            for interval in self.buffers:
                self.buffers[interval].clear()
            logger.info("Cleared all buffers")
    
    def get_status(self):
        """Get current status of all buffers"""
        status = {
            'symbol': self.symbol,
            'buffers': {}
        }
        
        for interval in self.intervals:
            buffer_size = self.get_buffer_size(interval)
            latest_price = self.get_latest_close(interval)
            ready = self.is_ready(interval)
            
            status['buffers'][interval] = {
                'size': buffer_size,
                'latest_price': latest_price,
                'ready': ready,
                'max_size': self.buffer_size
            }
        
        return status
    
    def print_status(self):
        """Print status to console"""
        status = self.get_status()
        print(f"\n📊 Data Manager Status - {status['symbol']}")
        print("=" * 60)
        for interval, info in status['buffers'].items():
            ready_icon = "✅" if info['ready'] else "⏳"
            print(f"{ready_icon} {interval}: {info['size']:>3}/{info['max_size']} candles | "
                  f"Latest: ${info['latest_price']:.2f}" if info['latest_price'] else "N/A")
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    from websocket_client import BinanceWebSocketClient
    import time
    
    # Create data manager
    data_mgr = LiveDataManager('BTCUSDT', intervals=['1m'])
    
    # Register callback
    def on_candle_update(interval, candle):
        print(f"\n[Callback] {interval} candle: ${candle['close']:.2f}")
        
        # Get strategy data
        data = data_mgr.get_data(interval, lookback=50)
        if data:
            print(f"  Buffer has {len(data['closes'])} candles")
            print(f"  Avg close: ${np.mean(data['closes']):.2f}")
        
        # Print status
        data_mgr.print_status()
    
    data_mgr.register_callback('1m', on_candle_update)
    
    # Create WebSocket and connect to data manager
    ws_client = BinanceWebSocketClient(
        symbol='btcusdt',
        intervals=['1m'],
        on_candle_close=data_mgr.add_candle
    )
    
    # Start streaming
    ws_client.start()
    
    print("Streaming data... Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        ws_client.stop()
