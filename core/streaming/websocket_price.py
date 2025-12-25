"""
WebSocket Price Stream Manager
Real-time price updates via WebSocket (replaces 30s polling).
"""
import logging
import json
import websocket
import threading
from typing import Callable, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketPriceStream:
    """
    WebSocket manager for real-time Binance price streams.
    
    Features:
    - Real-time price updates
    - Auto-reconnect on disconnect
    - Thread-safe callback handling
    - Connection health monitoring
    """
    
    def __init__(self, symbol: str = "btcusdt", callback: Optional[Callable] = None):
        """
        Initialize WebSocket stream.
        
        Args:
            symbol: Trading pair (lowercase)
            callback: Function to call on price updates
        """
        self.symbol = symbol.lower()
        self.callback = callback
        
        self.ws = None
        self.ws_thread = None
        self.is_running = False
        self.is_connected = False
        
        self.last_price = None
        self.last_update = None
        self.reconnect_count = 0
        
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"
        
        logger.info(f"WebSocket Price Stream initialized for {symbol.upper()}")
    
    def start(self):
        """Start WebSocket connection in background thread."""
        if self.is_running:
            logger.warning("WebSocket already running")
            return
        
        self.is_running = True
        self.ws_thread = threading.Thread(target=self._run, daemon=True)
        self.ws_thread.start()
        logger.info(f"WebSocket stream started for {self.symbol.upper()}")
    
    def stop(self):
        """Stop WebSocket connection."""
        self.is_running = False
        if self.ws:
            self.ws.close()
        logger.info("WebSocket stream stopped")
    
    def _run(self):
        """Run WebSocket connection with auto-reconnect."""
        while self.is_running:
            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                
                self.ws.run_forever()
                
                # Reconnect if still running
                if self.is_running:
                    self.reconnect_count += 1
                    logger.warning(f"WebSocket disconnected. Reconnecting... (attempt {self.reconnect_count})")
                    import time
                    time.sleep(min(self.reconnect_count * 2, 30))  # Exponential backoff, max 30s
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.is_running:
                    import time
                    time.sleep(5)
    
    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        self.is_connected = True
        self.reconnect_count = 0
        logger.info(f"✅ WebSocket connected to {self.symbol.upper()} stream")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            # Extract price from trade message
            price = float(data['p'])
            timestamp = int(data['T'])
            
            self.last_price = price
            self.last_update = datetime.fromtimestamp(timestamp / 1000)
            
            # Call callback if registered
            if self.callback:
                self.callback(price, timestamp)
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        self.is_connected = False
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
    
    def get_latest_price(self) -> Optional[float]:
        """Get latest price from stream."""
        return self.last_price
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        if not self.is_connected:
            return False
        
        # Check if we've received updates recently (last 60s)
        if self.last_update:
            elapsed = (datetime.now() - self.last_update).total_seconds()
            return elapsed < 60
        
        return False
