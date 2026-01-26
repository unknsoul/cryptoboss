"""
Data Validation Layer - Upgrade I

Pydantic models for all data inputs.
Compatible with both pydantic v1 and v2.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try pydantic, fall back to dataclasses
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
    PYDANTIC_V2 = hasattr(BaseModel, 'model_validate')
except ImportError:
    PYDANTIC_AVAILABLE = False
    PYDANTIC_V2 = False


# === Enums ===

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# === Data Models (using dataclasses for compatibility) ===

@dataclass
class OHLCVCandle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    
    def __post_init__(self):
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError("prices must be > 0")


@dataclass
class PriceTick:
    """Real-time price tick."""
    symbol: str
    price: float
    timestamp: datetime
    quantity: float = 0.0
    side: Optional[str] = None
    trade_id: Optional[str] = None


@dataclass
class OrderBookLevel:
    """Single orderbook level."""
    price: float
    quantity: float


@dataclass
class OrderBook:
    """Orderbook snapshot."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel] = None
    asks: List[OrderBookLevel] = None
    
    def __post_init__(self):
        self.bids = self.bids or []
        self.asks = self.asks or []


@dataclass
class OrderRequest:
    """Order request from strategy."""
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market' or 'limit'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None
    strategy_id: Optional[str] = None
    
    def __post_init__(self):
        if self.order_type == 'limit' and self.price is None:
            raise ValueError("price required for limit orders")


@dataclass
class OrderResponse:
    """Order response from exchange."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    status: str
    timestamp: datetime
    client_order_id: Optional[str] = None
    filled_quantity: float = 0
    price: Optional[float] = None
    average_price: Optional[float] = None
    fee: float = 0
    fee_currency: Optional[str] = None


@dataclass
class TradeExecution:
    """Trade execution record."""
    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    fee: float = 0
    fee_currency: Optional[str] = None
    is_maker: bool = False


@dataclass
class Balance:
    """Account balance for a currency."""
    currency: str
    free: float = 0
    locked: float = 0
    
    @property
    def total(self) -> float:
        return self.free + self.locked


@dataclass
class Position:
    """Trading position."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    liquidation_price: Optional[float] = None
    leverage: int = 1


@dataclass
class Signal:
    """Trading signal from strategy."""
    strategy_id: str
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float = 0.5
    quantity: Optional[float] = None
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    strategy_id: str
    strategy_type: str
    symbol: str
    allocation: float
    enabled: bool = True
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


# === Validation Utilities ===

class DataValidator:
    """
    Utility for validating data with logging.
    
    Usage:
        validator = DataValidator()
        
        candle, error = validator.validate_candle(raw_data)
        if error:
            logger.error(f"Invalid candle: {error}")
        else:
            process(candle)
    """
    
    def __init__(self, quarantine_path: str = "data/quarantine"):
        self.quarantine_path = quarantine_path
        self.error_counts: Dict[str, int] = {}
    
    def validate_candle(self, data: Dict) -> tuple:
        """Validate OHLCV candle data."""
        try:
            candle = OHLCVCandle(**data)
            return candle, None
        except Exception as e:
            self._log_error("candle", data, str(e))
            return None, str(e)
    
    def validate_tick(self, data: Dict) -> tuple:
        """Validate price tick data."""
        try:
            tick = PriceTick(**data)
            return tick, None
        except Exception as e:
            self._log_error("tick", data, str(e))
            return None, str(e)
    
    def validate_order_request(self, data: Dict) -> tuple:
        """Validate order request."""
        try:
            order = OrderRequest(**data)
            return order, None
        except Exception as e:
            self._log_error("order_request", data, str(e))
            return None, str(e)
    
    def validate_signal(self, data: Dict) -> tuple:
        """Validate trading signal."""
        try:
            signal = Signal(**data)
            return signal, None
        except Exception as e:
            self._log_error("signal", data, str(e))
            return None, str(e)
    
    def _log_error(self, data_type: str, data: Dict, error: str):
        """Log validation error."""
        self.error_counts[data_type] = self.error_counts.get(data_type, 0) + 1
        logger.warning(f"Validation failed for {data_type}: {error}")
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of validation errors by type."""
        return self.error_counts.copy()


# Singleton validator
_validator: Optional[DataValidator] = None

def get_validator() -> DataValidator:
    global _validator
    if _validator is None:
        _validator = DataValidator()
    return _validator
