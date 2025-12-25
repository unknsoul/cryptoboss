"""
Multi-Symbol Trading Manager
Enables simultaneous trading across multiple symbols.
"""
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SymbolPosition:
    """Position for a single symbol."""
    symbol: str
    side: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    strategy: str
    
    def get_value(self, current_price: float) -> float:
        """Calculate current position value."""
        return self.size * current_price
    
    def get_pnl(self, current_price: float) -> float:
        """Calculate current P&L."""
        if self.side == 'LONG':
            return (current_price - self.entry_price) * self.size
        else:  # SHORT
            return (self.entry_price - current_price) * self.size


class MultiSymbolManager:
    """
    Manages trading across multiple symbols.
    
    Features:
    - Per-symbol position tracking
    - Portfolio-level risk management
    - Correlation-aware position sizing
    - Symbol-specific strategies
    """
    
    def __init__(self, symbols: List[str], equity: float):
        """
        Initialize multi-symbol manager.
        
        Args:
            symbols: List of symbols to trade (e.g., ['BTCUSDT', 'ETHUSDT'])
            equity: Total portfolio equity
        """
        self.symbols = symbols
        self.equity = equity
        
        # Position tracking per symbol
        self.positions: Dict[str, Optional[SymbolPosition]] = {s: None for s in symbols}
        
        # Current prices
        self.prices: Dict[str, float] = {s: 0.0 for s in symbols}
        
        # Per-symbol equity allocation
        self.allocations: Dict[str, float] = self._calculate_allocations()
        
        logger.info(f"Multi-Symbol Manager initialized for {len(symbols)} symbols: {', '.join(symbols)}")
    
    def _calculate_allocations(self) -> Dict[str, float]:
        """Calculate equal allocation across symbols."""
        equal_split = self.equity / len(self.symbols)
        return {s: equal_split for s in self.symbols}
    
    def get_open_positions(self) -> List[SymbolPosition]:
        """Get all open positions."""
        return [pos for pos in self.positions.values() if pos is not None]
    
    def can_open_position(self, symbol: str) -> bool:
        """Check if can open position for symbol."""
        # Check if position already exists
        if self.positions.get(symbol) is not None:
            return False
        
        # Check portfolio heat (max concurrent positions)
        open_count = len(self.get_open_positions())
        max_positions = min(3, len(self.symbols))  # Max 3 concurrent positions
        
        if open_count >= max_positions:
            logger.warning(f"Max concurrent positions reached ({max_positions})")
            return False
        
        return True
    
    def open_position(self, position: SymbolPosition):
        """Open a new position."""
        if not self.can_open_position(position.symbol):
            logger.warning(f"Cannot open position for {position.symbol}")
            return False
        
        self.positions[position.symbol] = position
        logger.info(f"Position opened: {position.symbol} {position.side} @ ${position.entry_price:.2f}")
        return True
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[Dict]:
        """Close position for symbol."""
        position = self.positions.get(symbol)
        if not position:
            return None
        
        pnl = position.get_pnl(exit_price)
        pnl_pct = (pnl / (position.entry_price * position.size)) * 100
        
        trade_result = {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'strategy': position.strategy,
            'timestamp': datetime.now().isoformat()
        }
        
        self.positions[symbol] = None
        logger.info(f"Position closed: {symbol} - PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        
        return trade_result
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all symbols."""
        self.prices.update(prices)
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        total = self.equity
        
        for symbol, position in self.positions.items():
            if position:
                current_price = self.prices.get(symbol, position.entry_price)
                pnl = position.get_pnl(current_price)
                total += pnl
        
        return total
    
    def get_portfolio_stats(self) -> Dict:
        """Get portfolio statistics."""
        open_positions = self.get_open_positions()
        portfolio_value = self.get_portfolio_value()
        
        total_exposure = sum(
            pos.size * self.prices.get(pos.symbol, pos.entry_price)
            for pos in open_positions
        )
        
        return {
            'equity': self.equity,
            'portfolio_value': portfolio_value,
            'total_return_pct': ((portfolio_value - self.equity) / self.equity * 100),
            'open_positions': len(open_positions),
            'total_exposure': total_exposure,
            'symbols': list(self.symbols)
        }
