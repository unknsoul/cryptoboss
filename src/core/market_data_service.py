"""
Market Data Service - Real-time Price Feed

CRYPTOBOSS vFINAL: WebSocket-based price feed from Binance.

Rules:
- NO REST for live prices
- NO fallback/cached prices
- Testnet trading uses MAINNET market data
- Auto-reconnect on disconnect
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger("MarketData")

# Binance WebSocket endpoints
BINANCE_WS_MAINNET = "wss://stream.binance.com:9443/ws"
BINANCE_WS_TESTNET = "wss://testnet.binance.vision/ws"

# Supported symbols
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]


@dataclass
class PriceTick:
    """Single price update."""
    symbol: str
    price: float
    change_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    timestamp: datetime
    source: str = "BINANCE_MAINNET"
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change24h": self.change_24h,
            "high24h": self.high_24h,
            "low24h": self.low_24h,
            "volume24h": self.volume_24h,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }


@dataclass
class MarketDataState:
    """Current state of market data."""
    prices: Dict[str, PriceTick] = field(default_factory=dict)
    connected: bool = False
    last_update: Optional[datetime] = None
    reconnect_count: int = 0


class MarketDataService:
    """
    Real-time market data service using Binance WebSocket.
    
    IMPORTANT: Always uses MAINNET for price data.
    Testnet has no real market data, so we use mainnet prices
    for display while testnet is used for order execution.
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or SUPPORTED_SYMBOLS
        self.state = MarketDataState()
        self._ws = None
        self._running = False
        self._subscribers: Set[Callable] = set()
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        
    @property
    def is_connected(self) -> bool:
        return self.state.connected
    
    def get_price(self, symbol: str) -> Optional[PriceTick]:
        """Get current price for symbol."""
        return self.state.prices.get(symbol.upper())
    
    def get_all_prices(self) -> Dict[str, PriceTick]:
        """Get all current prices."""
        return self.state.prices.copy()
    
    def subscribe(self, callback: Callable[[PriceTick], None]):
        """Subscribe to price updates."""
        self._subscribers.add(callback)
        
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from price updates."""
        self._subscribers.discard(callback)
    
    async def start(self):
        """Start the market data service."""
        if self._running:
            logger.warning("MarketDataService already running")
            return
            
        self._running = True
        logger.info(f"🚀 Starting MarketDataService for {len(self.symbols)} symbols")
        
        # Start connection loop
        asyncio.create_task(self._connection_loop())
        
    async def stop(self):
        """Stop the market data service."""
        self._running = False
        if self._ws:
            await self._ws.close()
        self.state.connected = False
        logger.info("⏹️ MarketDataService stopped")
        
    async def _connection_loop(self):
        """Main connection loop with auto-reconnect."""
        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.state.connected = False
                
                if self._running:
                    # Exponential backoff
                    delay = min(self._reconnect_delay * (2 ** self.state.reconnect_count), 
                               self._max_reconnect_delay)
                    self.state.reconnect_count += 1
                    logger.info(f"🔄 Reconnecting in {delay:.1f}s (attempt {self.state.reconnect_count})")
                    await asyncio.sleep(delay)
                    
    async def _connect_and_listen(self):
        """Connect to Binance WebSocket and listen for updates."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. Run: pip install websockets")
            return
            
        # Build stream URL - ALWAYS use mainnet for real prices
        streams = "/".join([f"{s.lower()}@ticker" for s in self.symbols])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        
        logger.info(f"📡 Connecting to Binance WebSocket: {url[:60]}...")
        
        async with websockets.connect(url) as ws:
            self._ws = ws
            self.state.connected = True
            self.state.reconnect_count = 0
            self._reconnect_delay = 1.0
            
            logger.info(f"✅ Connected to Binance WebSocket")
            
            async for message in ws:
                await self._process_message(message)
                
    async def _process_message(self, message: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            # Combined stream format: {"stream": "btcusdt@ticker", "data": {...}}
            if "data" in data:
                ticker_data = data["data"]
            else:
                ticker_data = data
                
            # Parse 24hr ticker
            symbol = ticker_data.get("s", "").upper()
            if not symbol:
                return
                
            tick = PriceTick(
                symbol=symbol,
                price=float(ticker_data.get("c", 0)),  # Current price
                change_24h=float(ticker_data.get("P", 0)),  # Price change percent
                high_24h=float(ticker_data.get("h", 0)),
                low_24h=float(ticker_data.get("l", 0)),
                volume_24h=float(ticker_data.get("q", 0)),  # Quote volume
                timestamp=datetime.now(),
                source="BINANCE_MAINNET"
            )
            
            self.state.prices[symbol] = tick
            self.state.last_update = tick.timestamp
            
            # Notify subscribers
            for callback in self._subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(tick)
                    else:
                        callback(tick)
                except Exception as e:
                    logger.error(f"Subscriber error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to process message: {e}")


# Singleton instance
_market_data_service: Optional[MarketDataService] = None


def get_market_data_service() -> MarketDataService:
    """Get the singleton market data service."""
    global _market_data_service
    if _market_data_service is None:
        _market_data_service = MarketDataService()
    return _market_data_service


async def start_market_data():
    """Start the market data service."""
    service = get_market_data_service()
    await service.start()
    return service


async def stop_market_data():
    """Stop the market data service."""
    global _market_data_service
    if _market_data_service:
        await _market_data_service.stop()
        _market_data_service = None
