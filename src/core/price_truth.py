"""
CryptoBoss 1.0.1 - Price Truth Enforcement

CRITICAL: All prices must be validated, tagged with source, and properly isolated.

This module provides:
- PriceData model with mandatory fields
- PriceValidator for staleness, environment, and source validation
- Price source tagging
- Account-scoped price isolation
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PriceSource(str, Enum):
    """Allowed price sources with strict environment constraints."""
    LIVE_EXCHANGE_TICKER = "LIVE_EXCHANGE_TICKER"     # Real-time, LIVE only
    TESTNET_TICKER = "TESTNET_TICKER"                  # Synthetic, TESTNET only
    DERIVED_PRICE = "DERIVED_PRICE"                    # Calculated indicators
    REPLAY_PRICE = "REPLAY_PRICE"                      # Historical replay only
    UNKNOWN = "UNKNOWN"                                # Rejected immediately


# Max age in milliseconds before price is considered stale
MAX_AGE_MS = {
    PriceSource.LIVE_EXCHANGE_TICKER: 2000,   # 2 seconds
    PriceSource.TESTNET_TICKER: 5000,          # 5 seconds
    PriceSource.DERIVED_PRICE: 10000,          # 10 seconds
    PriceSource.REPLAY_PRICE: 60000,           # 1 minute (historical)
    PriceSource.UNKNOWN: 0,                    # Always stale
}


# Environment restrictions for each source
SOURCE_ENVIRONMENT_ALLOWED = {
    PriceSource.LIVE_EXCHANGE_TICKER: {"live", "LIVE"},
    PriceSource.TESTNET_TICKER: {"testnet", "TESTNET"},
    PriceSource.DERIVED_PRICE: {"live", "LIVE", "testnet", "TESTNET"},
    PriceSource.REPLAY_PRICE: {"replay", "REPLAY"},
}


@dataclass
class PriceData:
    """
    Validated price data with mandatory fields.
    
    HARD RULE: No price rendering without all mandatory fields.
    """
    symbol: str
    price: float
    source: PriceSource
    exchange: str
    environment: str
    timestamp_ms: int
    
    # Optional metadata
    exchange_account_id: Optional[str] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume_24h: Optional[float] = None
    change_24h_pct: Optional[float] = None
    
    # Validation state
    is_valid: bool = True
    rejection_reason: Optional[str] = None
    
    def age_ms(self) -> int:
        """Get age of this price in milliseconds."""
        now_ms = int(datetime.now().timestamp() * 1000)
        return now_ms - self.timestamp_ms
    
    def is_stale(self) -> bool:
        """Check if price is stale based on source max age."""
        max_age = MAX_AGE_MS.get(self.source, 0)
        return self.age_ms() > max_age
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "source": self.source.value,
            "exchange": self.exchange,
            "environment": self.environment,
            "timestamp_ms": self.timestamp_ms,
            "exchange_account_id": self.exchange_account_id,
            "bid": self.bid,
            "ask": self.ask,
            "age_ms": self.age_ms(),
            "is_stale": self.is_stale(),
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason
        }
    
    @classmethod
    def create_empty(cls, symbol: str, environment: str) -> 'PriceData':
        """Create an empty/unavailable price placeholder."""
        return cls(
            symbol=symbol,
            price=0.0,
            source=PriceSource.UNKNOWN,
            exchange="NONE",
            environment=environment,
            timestamp_ms=0,
            is_valid=False,
            rejection_reason="Price unavailable"
        )


class PriceValidator:
    """
    Validates prices against strict rules.
    
    HARD RULES:
    1. Reject price if timestamp too old
    2. Reject price if environment mismatch
    3. Reject price if source not allowed
    4. Reject price if symbol not subscribed
    """
    
    def __init__(self, environment: str, exchange_account_id: str = None):
        self.environment = environment.upper()
        self.exchange_account_id = exchange_account_id
        self.subscribed_symbols: Set[str] = set()
        
        logger.info(f"🔍 PriceValidator initialized for {self.environment}")
    
    def subscribe(self, symbol: str):
        """Subscribe to a symbol's price feed."""
        self.subscribed_symbols.add(symbol.upper())
        logger.debug(f"Subscribed to {symbol}")
    
    def unsubscribe(self, symbol: str):
        """Unsubscribe from a symbol's price feed."""
        self.subscribed_symbols.discard(symbol.upper())
        logger.debug(f"Unsubscribed from {symbol}")
    
    def clear_subscriptions(self):
        """Clear all subscriptions (on account switch)."""
        self.subscribed_symbols.clear()
        logger.info("🔄 All price subscriptions cleared")
    
    def validate(self, price: PriceData) -> PriceData:
        """
        Validate a price against all rules.
        
        Returns the price with is_valid and rejection_reason set.
        """
        # Rule 1: Check source is not UNKNOWN
        if price.source == PriceSource.UNKNOWN:
            price.is_valid = False
            price.rejection_reason = "Unknown price source"
            logger.warning(f"❌ Price rejected: unknown source for {price.symbol}")
            return price
        
        # Rule 2: Check environment matches
        allowed_envs = SOURCE_ENVIRONMENT_ALLOWED.get(price.source, set())
        if self.environment not in allowed_envs and self.environment.lower() not in allowed_envs:
            price.is_valid = False
            price.rejection_reason = f"Source {price.source.value} not allowed in {self.environment}"
            logger.warning(f"❌ Price rejected: {price.source.value} not allowed in {self.environment}")
            return price
        
        # Rule 3: Check staleness
        if price.is_stale():
            price.is_valid = False
            price.rejection_reason = f"Price stale: age {price.age_ms()}ms > max {MAX_AGE_MS.get(price.source, 0)}ms"
            logger.warning(f"❌ Price rejected: stale ({price.age_ms()}ms old)")
            return price
        
        # Rule 4: Check symbol subscription (if subscriptions enabled)
        if self.subscribed_symbols and price.symbol.upper() not in self.subscribed_symbols:
            price.is_valid = False
            price.rejection_reason = f"Symbol {price.symbol} not subscribed"
            logger.warning(f"❌ Price rejected: {price.symbol} not subscribed")
            return price
        
        # Rule 5: Check account scope (if set)
        if self.exchange_account_id and price.exchange_account_id:
            if price.exchange_account_id != self.exchange_account_id:
                price.is_valid = False
                price.rejection_reason = "Account mismatch"
                logger.warning(f"❌ Price rejected: account mismatch")
                return price
        
        # All rules passed
        price.is_valid = True
        price.rejection_reason = None
        return price
    
    def create_price(
        self,
        symbol: str,
        price: float,
        source: PriceSource,
        exchange: str,
        bid: float = None,
        ask: float = None
    ) -> PriceData:
        """Create and validate a new price."""
        price_data = PriceData(
            symbol=symbol.upper(),
            price=price,
            source=source,
            exchange=exchange,
            environment=self.environment,
            timestamp_ms=int(datetime.now().timestamp() * 1000),
            exchange_account_id=self.exchange_account_id,
            bid=bid,
            ask=ask
        )
        return self.validate(price_data)


class PriceFeedManager:
    """
    Manages price feeds per exchange account.
    
    CRITICAL:
    - Price feed must restart on account switch
    - Replay feed isolated from live feed
    - Backtest prices never exposed to UI
    """
    
    _instance = None
    _validators: Dict[str, PriceValidator] = {}
    _price_cache: Dict[str, Dict[str, PriceData]] = {}  # account_id -> symbol -> price
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_validator(self, exchange_account_id: str, environment: str) -> PriceValidator:
        """Get or create validator for an account."""
        if exchange_account_id not in self._validators:
            self._validators[exchange_account_id] = PriceValidator(
                environment=environment,
                exchange_account_id=exchange_account_id
            )
            self._price_cache[exchange_account_id] = {}
        return self._validators[exchange_account_id]
    
    def switch_account(self, new_account_id: str, new_environment: str):
        """
        Switch to a new account - clears all cached prices.
        
        MANDATORY ACTIONS:
        1. Clear price store
        2. Create new validator
        """
        # Clear old caches
        self._price_cache.clear()
        
        # Get/create validator for new account
        validator = self.get_validator(new_account_id, new_environment)
        validator.clear_subscriptions()
        
        logger.info(f"🔄 Price feed switched to account {new_account_id[:8]}...")
        return validator
    
    def update_price(self, exchange_account_id: str, price: PriceData):
        """Update cached price for an account."""
        if exchange_account_id not in self._price_cache:
            self._price_cache[exchange_account_id] = {}
        
        if price.is_valid:
            self._price_cache[exchange_account_id][price.symbol] = price
    
    def get_price(self, exchange_account_id: str, symbol: str) -> Optional[PriceData]:
        """Get cached price for a symbol."""
        account_cache = self._price_cache.get(exchange_account_id, {})
        price = account_cache.get(symbol.upper())
        
        # Re-validate staleness
        if price and price.is_stale():
            price.is_valid = False
            price.rejection_reason = f"Price became stale: age {price.age_ms()}ms"
        
        return price
    
    def get_all_prices(self, exchange_account_id: str) -> Dict[str, PriceData]:
        """Get all cached prices for an account."""
        return self._price_cache.get(exchange_account_id, {})


# Singleton accessor
def get_price_manager() -> PriceFeedManager:
    return PriceFeedManager()
