"""
Symbol Validation Layer - CryptoBoss

Centralized symbol and asset validation.
Only whitelisted symbols and assets are processed.

This module is applied BEFORE data reaches:
- Trading logic
- Dashboard UI
- Risk calculations
"""

import logging
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# === Symbol Whitelist Configuration ===

# Core trading pairs we support
ALLOWED_SYMBOLS: Set[str] = {
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "MATICUSDT",
    "DOTUSDT",
    "LTCUSDT",
}

# Alternative representations
ALLOWED_SYMBOLS_SLASH: Set[str] = {
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "MATIC/USDT",
    "DOT/USDT",
    "LTC/USDT",
}

# Assets we recognize (base and quote)
ALLOWED_ASSETS: Set[str] = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "MATIC", "DOT", "LTC",
    "USDT", "USDC", "BUSD",  # Stablecoins
}

# Quote currencies (what we trade against)
QUOTE_CURRENCIES: Set[str] = {"USDT", "USDC", "BUSD"}


@dataclass
class ValidationStats:
    """Statistics from validation operations."""
    total_received: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    rejected_items: List[str] = None
    
    def __post_init__(self):
        if self.rejected_items is None:
            self.rejected_items = []


class SymbolValidator:
    """
    Validates and filters symbols and assets.
    
    Usage:
        validator = SymbolValidator()
        
        # Check single symbol
        if validator.is_valid_symbol("BTCUSDT"):
            process(symbol)
        
        # Filter balances
        clean_balances = validator.filter_balances(raw_balances)
        
        # Filter symbols list
        clean_symbols = validator.filter_symbols(raw_symbols)
    """
    
    def __init__(
        self,
        allowed_symbols: Set[str] = None,
        allowed_assets: Set[str] = None,
        quote_currencies: Set[str] = None
    ):
        self.allowed_symbols = allowed_symbols or ALLOWED_SYMBOLS
        self.allowed_symbols_slash = ALLOWED_SYMBOLS_SLASH
        self.allowed_assets = allowed_assets or ALLOWED_ASSETS
        self.quote_currencies = quote_currencies or QUOTE_CURRENCIES
        
        # Stats tracking
        self._stats = ValidationStats()
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to standard format (e.g., BTC/USDT -> BTCUSDT)."""
        if "/" in symbol:
            return symbol.replace("/", "")
        return symbol.upper()
    
    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is in the allowed list."""
        normalized = self.normalize_symbol(symbol)
        is_valid = normalized in self.allowed_symbols or symbol in self.allowed_symbols_slash
        
        if not is_valid:
            logger.debug(f"Symbol rejected: {symbol}")
        
        return is_valid
    
    def is_valid_asset(self, asset: str) -> bool:
        """Check if asset is in the allowed list."""
        is_valid = asset.upper() in self.allowed_assets
        
        if not is_valid:
            logger.debug(f"Asset rejected: {asset}")
            
        return is_valid
    
    def has_tradable_pair(self, asset: str) -> bool:
        """Check if asset has a tradable pair with a quote currency."""
        asset_upper = asset.upper()
        
        # Quote currencies are always valid
        if asset_upper in self.quote_currencies:
            return True
        
        # Check if there's a valid pair
        for quote in self.quote_currencies:
            potential_symbol = f"{asset_upper}{quote}"
            if potential_symbol in self.allowed_symbols:
                return True
        
        return False
    
    def filter_balances(self, raw_balances: Dict[str, float]) -> Dict[str, float]:
        """
        Filter balances to only include valid, tradable assets.
        
        Args:
            raw_balances: Dict of {asset: amount}
            
        Returns:
            Filtered dict with only valid assets
        """
        self._stats = ValidationStats(total_received=len(raw_balances))
        filtered = {}
        
        for asset, amount in raw_balances.items():
            # Skip zero balances
            if amount <= 0:
                continue
            
            # Check if asset is valid and tradable
            if self.is_valid_asset(asset) and self.has_tradable_pair(asset):
                filtered[asset] = amount
                self._stats.total_accepted += 1
            else:
                self._stats.total_rejected += 1
                self._stats.rejected_items.append(f"{asset}={amount}")
        
        if self._stats.total_rejected > 0:
            logger.info(
                f"Balance filter: {self._stats.total_accepted} accepted, "
                f"{self._stats.total_rejected} rejected"
            )
            logger.debug(f"Rejected assets: {self._stats.rejected_items[:10]}")  # Show first 10
        
        return filtered
    
    def filter_symbols(self, raw_symbols: List[str]) -> List[str]:
        """
        Filter symbol list to only include valid symbols.
        
        Args:
            raw_symbols: List of symbol strings
            
        Returns:
            Filtered list with only valid symbols
        """
        self._stats = ValidationStats(total_received=len(raw_symbols))
        filtered = []
        
        for symbol in raw_symbols:
            if self.is_valid_symbol(symbol):
                filtered.append(symbol)
                self._stats.total_accepted += 1
            else:
                self._stats.total_rejected += 1
                self._stats.rejected_items.append(symbol)
        
        if self._stats.total_rejected > 0:
            logger.info(
                f"Symbol filter: {self._stats.total_accepted} accepted, "
                f"{self._stats.total_rejected} rejected"
            )
        
        return filtered
    
    def get_stats(self) -> ValidationStats:
        """Get statistics from last validation operation."""
        return self._stats
    
    def add_symbol(self, symbol: str):
        """Dynamically add a symbol to the whitelist."""
        normalized = self.normalize_symbol(symbol)
        self.allowed_symbols.add(normalized)
        
        # Extract and add base asset
        for quote in self.quote_currencies:
            if normalized.endswith(quote):
                base = normalized[:-len(quote)]
                self.allowed_assets.add(base)
                break
        
        logger.info(f"Added symbol to whitelist: {normalized}")
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from the whitelist."""
        normalized = self.normalize_symbol(symbol)
        self.allowed_symbols.discard(normalized)
        logger.info(f"Removed symbol from whitelist: {normalized}")


# === Singleton Instance ===

_validator: Optional[SymbolValidator] = None


def get_symbol_validator() -> SymbolValidator:
    """Get the singleton SymbolValidator instance."""
    global _validator
    if _validator is None:
        _validator = SymbolValidator()
    return _validator


# === Convenience Functions ===

def filter_balances(raw_balances: Dict[str, float]) -> Dict[str, float]:
    """Filter balances using the global validator."""
    return get_symbol_validator().filter_balances(raw_balances)


def filter_symbols(raw_symbols: List[str]) -> List[str]:
    """Filter symbols using the global validator."""
    return get_symbol_validator().filter_symbols(raw_symbols)


def is_valid_symbol(symbol: str) -> bool:
    """Check if symbol is valid using the global validator."""
    return get_symbol_validator().is_valid_symbol(symbol)


def is_valid_asset(asset: str) -> bool:
    """Check if asset is valid using the global validator."""
    return get_symbol_validator().is_valid_asset(asset)
