"""
Binance WebSocket Client
Real-time candlestick (kline) data streaming with auto-reconnection
"""

import json
import time
import threading
from datetime import datetime
from websocket import WebSocketApp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """
    WebSocket client for Binance real-time kline (candlestick) data
    
    Features:
    - Multiple timeframe support (1m, 5m, 15m, 1h, 4h, 1d)
    - Auto-reconnection with exponential backoff
    - Heartbeat monitoring
    - Thread-safe callbacks
    - Testnet/Mainnet support
    """
    
    def __init__(self, symbol='btcusdt', intervals=None, on_candle_close=None, base_url=None):
        """
        Initialize WebSocket client
        
        Args:
            symbol: Trading pair (lowercase, e.g., 'btcusdt')
            intervals: List of timeframes (e.g., ['1h', '4h'])
            on_candle_close: Callback function(interval, candle_data)
            base_url: WebSocket base URL (default: Binance mainnet)
        """
        self.symbol = symbol.lower()
        self.intervals = intervals or ['1h']
        self.on_candle_close = on_candle_close
        self.base_url = base_url or "wss://stream.binance.com:9443/ws"
        
        # WebSocket management
        self.ws = None
        self.ws_thread = None
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_delay = 60  # seconds
        
        # Heartbeat
        self.last_message_time = time.time()
        self.heartbeat_timeout = 30  # seconds
        
        # Build stream URL
        self.stream_url = self._build_stream_url()
        
        logger.info(f"Initialized WebSocket for {symbol} - Intervals: {intervals}")
    
    def _build_stream_url(self):
        """Build WebSocket URL for combined streams"""
        streams = [f"{self.symbol}@kline_{interval}" for interval in self.intervals]
        combined = '/'.join(streams)
        return f"{self.base_url}/{combined}"
    
    def start(self):
        """Start WebSocket connection in separate thread"""
        if self.is_running:
            logger.warning("WebSocket already running")
            return
        
        self.is_running = True
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()
        
        # Start heartbeat monitor
        heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        heartbeat_thread.start()
        
        logger.info("WebSocket started")
    
    def stop(self):
        """Stop WebSocket connection"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        logger.info("WebSocket stopped")
    
    def _run_websocket(self):
        """Run WebSocket connection with auto-reconnection"""
        while self.is_running:
            try:
                self.ws = WebSocketApp(
                    self.stream_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                
                # Run forever (blocks until connection closes)
                self.ws.run_forever()
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            
            # Reconnection logic
            if self.is_running:
                delay = min(2 ** self.reconnect_attempts, self.max_reconnect_delay)
                logger.info(f"Reconnecting in {delay}s (attempt {self.reconnect_attempts + 1})...")
                time.sleep(delay)
                self.reconnect_attempts += 1
    
    def _on_open(self, ws):
        """Called when WebSocket connection opens"""
        logger.info("✅ WebSocket connected")
        self.reconnect_attempts = 0
        self.last_message_time = time.time()
    
    def _on_message(self, ws, message):
        """Called when message received"""
        self.last_message_time = time.time()
        
        try:
            data = json.loads(message)
            
            # Handle combined stream format
            if 'stream' in data:
                stream = data['stream']
                kline_data = data['data']
            else:
                kline_data = data
            
            # Extract kline info
            kline = kline_data.get('k')
            if not kline:
                return
            
            # Only process closed candles
            if kline['x']:  # is_final
                candle = self._parse_kline(kline)
                interval = kline['i']
                
                logger.info(f"📊 Candle closed [{interval}]: {candle['close']:.2f}")
                
                # Trigger callback
                if self.on_candle_close:
                    try:
                        self.on_candle_close(interval, candle)
                    except Exception as e:
                        logger.error(f"Error in candle callback: {e}")
        
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _parse_kline(self, kline):
        """Parse kline data into OHLCV format"""
        return {
            'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'close_time': datetime.fromtimestamp(kline['T'] / 1000),
            'trades': kline['n']
        }
    
    def _on_error(self, ws, error):
        """Called on WebSocket error"""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Called when WebSocket connection closes"""
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
    
    def _heartbeat_monitor(self):
        """Monitor connection health via message timestamps"""
        while self.is_running:
            time.sleep(10)
            
            elapsed = time.time() - self.last_message_time
            if elapsed > self.heartbeat_timeout:
                logger.warning(f"⚠️ No messages for {elapsed:.0f}s - connection may be stale")
                
                # Force reconnection
                if self.ws:
                    self.ws.close()


# Example usage
if __name__ == "__main__":
    def on_new_candle(interval, candle):
        print(f"\n[{interval}] New Candle:")
        print(f"  Time: {candle['timestamp']}")
        print(f"  OHLC: {candle['open']:.2f} / {candle['high']:.2f} / {candle['low']:.2f} / {candle['close']:.2f}")
        print(f"  Volume: {candle['volume']:.2f}")
    
    # Create client
    client = BinanceWebSocketClient(
        symbol='btcusdt',
        intervals=['1m'],  # Use 1m for testing (faster)
        on_candle_close=on_new_candle
    )
    
    # Start streaming
    client.start()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        client.stop()
