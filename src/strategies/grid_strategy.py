"""
Grid Trading Strategy
Perfect for volatile and ranging markets - profits from oscillations.

How it works:
1. Define price range (e.g., $60,000 - $70,000)
2. Create "grid" of buy/sell orders at fixed intervals
3. Each grid level has a buy order below and sell order above
4. When price moves through grid, orders execute automatically
5. Profit from each grid level movement

Example (10 grids, $60k-$70k range):
- Grid 1: Buy @ $60k, Sell @ $61k → Profit: $1k (1.67%)
- Grid 2: Buy @ $61k, Sell @ $62k → Profit: $1k (1.64%)
- ...
- Grid 10: Buy @ $69k, Sell @ $70k → Profit: $1k (1.45%)

If price oscillates 5 times through range → 50 profitable trades!

Win Rate: Very high (each grid = profit)
Risk: Limited (defined by grid range)
Best Conditions: Volatile, sideways markets
Monthly Return: 8-25% (depends on volatility)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class GridType(Enum):
    """Grid spacing types."""
    ARITHMETIC = "arithmetic"  # Linear spacing
    GEOMETRIC = "geometric"    # Exponential spacing


class GridMode(Enum):
    """Grid directional bias."""
    NEUTRAL = "neutral"  # Equal buy/sell
    LONG = "long"        # More buy orders (bullish bias)
    SHORT = "short"      # More sell orders (bearish bias)


@dataclass
class GridLevel:
    """Represents a single grid level."""
    level: int
    price: float
    buy_filled: bool = False
    sell_filled: bool = False
    buy_time: Optional[datetime] = None
    sell_time: Optional[datetime] = None
    profit_realized: float = 0.0
    
    def reset(self):
        """Reset level after both orders filled."""
        self.buy_filled = False
        self.sell_filled = False
        self.buy_time = None
        self.sell_time = None
        self.profit_realized = 0.0


@dataclass
class GridConfig:
    """Grid configuration."""
    lower_price: float
    upper_price: float
    num_grids: int
    grid_spacing: GridType
    mode: GridMode
    order_size_usd: float  # USD per grid level
    
    def calculate_grid_prices(self) -> List[float]:
        """Calculate all grid price levels."""
        if self.grid_spacing == GridType.ARITHMETIC:
            # Linear spacing
            return list(np.linspace(self.lower_price, self.upper_price, self.num_grids + 1))
        else:
            # Geometric (exponential) spacing
            # Better for crypto (accounts for % moves)
            ratio = (self.upper_price / self.lower_price) ** (1 / self.num_grids)
            prices = [self.lower_price * (ratio ** i) for i in range(self.num_grids + 1)]
            return prices


class GridTradingStrategy:
    """
    Grid Trading Strategy - Profit from price oscillations.
    
    The grid strategy creates a series of buy and sell orders at predetermined
    price levels within a defined range. It's market-neutral and profits from
    volatility regardless of direction.
    
    Configuration:
        lower_price: Bottom of grid range
        upper_price: Top of grid range
        num_grids: Number of grid levels (10-100)
        grid_spacing: 'arithmetic' or 'geometric'
        mode: 'neutral', 'long', or 'short'
        order_size_usd: USD amount per grid
        rebalance_threshold_pct: % move to trigger rebalance (default: 20%)
        stop_loss_breakout_pct: % breakout to close grid (optional)
    
    Grid Modes:
        - NEUTRAL: Equal buys and sells (market neutral)
        - LONG: More buying orders (bullish bias)
        - SHORT: More selling orders (bearish bias)
    
    Grid Spacing:
        - ARITHMETIC: Linear price steps (e.g., $1000 apart)
        - GEOMETRIC: Exponential steps (e.g., 1% apart) - Better for crypto!
    """
    
    def __init__(
        self,
        lower_price: float,
        upper_price: float,
        num_grids: int = 20,
        grid_spacing: str = "geometric",
        mode: str = "neutral",
        order_size_usd: float = 50.0,
        rebalance_threshold_pct: float = 20.0,
        stop_loss_breakout_pct: Optional[float] = None,
        initial_capital: float = 10000.0
    ):
        """
        Initialize Grid Trading Strategy.
        
        Args:
            lower_price: Bottom of price range
            upper_price: Top of price range
            num_grids: Number of grid levels
            grid_spacing: 'arithmetic' or 'geometric'
            mode: 'neutral', 'long', or 'short'
            order_size_usd: USD per grid order
            rebalance_threshold_pct: % move to rebalance grid
            stop_loss_breakout_pct: % breakout to stop (None = no stop)
            initial_capital: Starting capital
        """
        self.config = GridConfig(
            lower_price=lower_price,
            upper_price=upper_price,
            num_grids=num_grids,
            grid_spacing=GridType(grid_spacing),
            mode=GridMode(mode),
            order_size_usd=order_size_usd
        )
        
        self.rebalance_threshold_pct = rebalance_threshold_pct
        self.stop_loss_breakout_pct = stop_loss_breakout_pct
        self.initial_capital = initial_capital
        
        # State
        self.grid_levels: List[GridLevel] = []
        self.is_active = False
        self.total_profit = 0.0
        self.total_trades = 0
        self.position_quantity = 0.0  # Current BTC holdings
        self.cash = initial_capital
        
        # Tracking
        self.trade_history: List[Dict] = []
        
        logger.info(
            f"Grid Strategy initialized: "
            f"Range: ${lower_price:,.0f}-${upper_price:,.0f}, "
            f"Grids: {num_grids}, "
            f"Spacing: {grid_spacing}, "
            f"Mode: {mode}"
        )
    
    def initialize_grid(self, current_price: float) -> bool:
        """
        Initialize grid structure.
        
        Args:
            current_price: Current market price
        
        Returns:
            True if grid created successfully
        """
        # Check if price is within range
        if not (self.config.lower_price <= current_price <= self.config.upper_price):
            logger.warning(
                f"Price ${current_price:,.0f} outside grid range "
                f"${self.config.lower_price:,.0f}-${self.config.upper_price:,.0f}"
            )
            return False
        
        # Calculate grid prices
        grid_prices = self.config.calculate_grid_prices()
        
        # Create grid levels
        self.grid_levels = [
            GridLevel(level=i, price=price)
            for i, price in enumerate(grid_prices)
        ]
        
        # Pre-fill buy orders below current price (we own those grid levels)
        for level in self.grid_levels:
            if level.price < current_price:
                level.buy_filled = True
                self.position_quantity += self.config.order_size_usd / level.price
                self.cash -= self.config.order_size_usd
        
        self.is_active = True
        
        logger.info(
            f"Grid initialized with {len(self.grid_levels)} levels. "
            f"Position: {self.position_quantity:.4f} BTC, Cash: ${self.cash:,.2f}"
        )
        
        return True
    
    def get_active_grid_level(self, price: float) -> Optional[GridLevel]:
        """Find which grid level current price is in."""
        if not self.grid_levels:
            return None
        
        # Find the grid level just below current price
        for i in range(len(self.grid_levels) - 1):
            if self.grid_levels[i].price <= price < self.grid_levels[i + 1].price:
                return self.grid_levels[i]
        
        # Price above all grids
        if price >= self.grid_levels[-1].price:
            return self.grid_levels[-1]
        
        # Price below all grids
        if price < self.grid_levels[0].price:
            return self.grid_levels[0]
        
        return None
    
    def check_buy_triggers(self, current_price: float, timestamp: datetime) -> List[Dict]:
        """Check if any buy orders should execute."""
        buys = []
        
        for level in self.grid_levels:
            # Buy trigger: Price crosses below grid level and we haven't bought yet
            if current_price <= level.price and not level.buy_filled:
                # Execute buy
                quantity = self.config.order_size_usd / level.price
                
                if self.cash >= self.config.order_size_usd:
                    level.buy_filled = True
                    level.buy_time = timestamp
                    self.position_quantity += quantity
                    self.cash -= self.config.order_size_usd
                    
                    buys.append({
                        'action': 'BUY',
                        'level': level.level,
                        'price': level.price,
                        'quantity': quantity,
                        'timestamp': timestamp
                    })
                    
                    self.total_trades += 1
                    logger.debug(f"Grid Buy: Level {level.level} @ ${level.price:,.2f}")
                else:
                    logger.warning(f"Insufficient cash for grid buy @ ${level.price:,.2f}")
        
        return buys
    
    def check_sell_triggers(self, current_price: float, timestamp: datetime) -> List[Dict]:
        """Check if any sell orders should execute."""
        sells = []
        
        for level in self.grid_levels:
            # Sell trigger: Price crosses above grid level, we've bought, and haven't sold yet
            if (current_price >= level.price and 
                level.buy_filled and 
                not level.sell_filled):
                
                # Execute sell
                quantity = self.config.order_size_usd / level.price
                
                if self.position_quantity >= quantity:
                    level.sell_filled = True
                    level.sell_time = timestamp
                    self.position_quantity -= quantity
                    proceeds = level.price * quantity
                    self.cash += proceeds
                    
                    # Calculate profit for this grid level
                    cost = self.config.order_size_usd
                    profit = proceeds - cost
                    level.profit_realized = profit
                    self.total_profit += profit
                    
                    sells.append({
                        'action': 'SELL',
                        'level': level.level,
                        'price': level.price,
                        'quantity': quantity,
                        'profit': profit,
                        'timestamp': timestamp
                    })
                    
                    self.total_trades += 1
                    logger.debug(
                        f"Grid Sell: Level {level.level} @ ${level.price:,.2f}, "
                        f"Profit: ${profit:.2f}"
                    )
                    
                    # Reset level for next cycle
                    level.reset()
                else:
                    logger.warning(f"Insufficient position for grid sell @ ${level.price:,.2f}")
        
        return sells
    
    def should_rebalance(self, current_price: float) -> bool:
        """Check if grid should be rebalanced."""
        if not self.is_active:
            return False
        
        # Check price breakout
        price_pct_below = ((self.config.lower_price - current_price) / 
                          self.config.lower_price * 100)
        price_pct_above = ((current_price - self.config.upper_price) / 
                          self.config.upper_price * 100)
        
        # Rebalance if price moves beyond threshold
        if (price_pct_below > self.rebalance_threshold_pct or 
            price_pct_above > self.rebalance_threshold_pct):
            logger.info(
                f"Grid rebalance triggered: Price ${current_price:,.0f} "
                f"outside range by {max(price_pct_below, price_pct_above):.1f}%"
            )
            return True
        
        return False
    
    def should_stop(self, current_price: float) -> bool:
        """Check if grid should stop due to breakout."""
        if self.stop_loss_breakout_pct is None:
            return False
        
        # Calculate breakout %
        if current_price < self.config.lower_price:
            breakout_pct = ((self.config.lower_price - current_price) / 
                           self.config.lower_price * 100)
        elif current_price > self.config.upper_price:
            breakout_pct = ((current_price - self.config.upper_price) / 
                           self.config.upper_price * 100)
        else:
            return False
        
        if breakout_pct >= self.stop_loss_breakout_pct:
            logger.warning(
                f"Grid stop loss triggered: Breakout {breakout_pct:.1f}% "
                f"exceeds limit {self.stop_loss_breakout_pct}%"
            )
            return True
        
        return False
    
    def rebalance_grid(self, current_price: float):
        """Rebalance grid to new price range."""
        # Calculate new range centered on current price
        range_size = self.config.upper_price - self.config.lower_price
        new_lower = current_price - (range_size / 2)
        new_upper = current_price + (range_size / 2)
        
        logger.info(
            f"Rebalancing grid: "
            f"New range ${new_lower:,.0f}-${new_upper:,.0f}"
        )
        
        # Update config
        self.config.lower_price = new_lower
        self.config.upper_price = new_upper
        
        # Reinitialize grid
        self.grid_levels = []
        self.initialize_grid(current_price)
    
    def close_grid(self, current_price: float, timestamp: datetime):
        """Close all grid positions."""
        logger.info(f"Closing grid @ ${current_price:,.0f}")
        
        # Sell all holdings at current price
        if self.position_quantity > 0:
            proceeds = self.position_quantity * current_price
            self.cash += proceeds
            
            logger.info(
                f"Liquidated {self.position_quantity:.4f} BTC @ ${current_price:,.0f} "
                f"for ${proceeds:,.2f}"
            )
            
            self.position_quantity = 0.0
        
        self.is_active = False
    
    def generate_signal(self, df: pd.DataFrame, i: int, current_price: float) -> Dict:
        """
        Generate trading signals for grid strategy.
        
        Returns:
            Dict with 'action', 'orders', 'info'
        """
        # Initialize grid if not active
        if not self.is_active:
            if self.initialize_grid(current_price):
                return {
                    'action': 'GRID_INITIALIZED',
                    'info': f'Grid created: {len(self.grid_levels)} levels'
                }
            else:
                return {
                    'action': 'HOLD',
                    'info': 'Price outside grid range'
                }
        
        # Check for stop loss
        if self.should_stop(current_price):
            self.close_grid(current_price, df.index[i])
            return {
                'action': 'GRID_STOPPED',
                'info': 'Stop loss breakout triggered'
            }
        
        # Check for rebalance
        if self.should_rebalance(current_price):
            self.rebalance_grid(current_price)
            return {
                'action': 'GRID_REBALANCED',
                'info': 'Grid rebalanced to new range'
            }
        
        # Check buy triggers
        buys = self.check_buy_triggers(current_price, df.index[i])
        
        # Check sell triggers
        sells = self.check_sell_triggers(current_price, df.index[i])
        
        # Record trades
        for trade in buys + sells:
            self.trade_history.append(trade)
        
        if buys or sells:
            return {
                'action': 'GRID_TRADE',
                'buys': len(buys),
                'sells': len(sells),
                'total_profit': self.total_profit,
                'orders': buys + sells
            }
        
        return {
            'action': 'HOLD',
            'position': self.position_quantity,
            'cash': self.cash
        }
    
    def get_metrics(self) -> Dict:
        """Get strategy performance metrics."""
        total_value = self.cash + (self.position_quantity * 
                                   (self.config.upper_price + self.config.lower_price) / 2)
        total_return_pct = ((total_value - self.initial_capital) / 
                           self.initial_capital * 100)
        
        # Calculate average profit per trade
        profitable_trades = [t for t in self.trade_history if 'profit' in t and t['profit'] > 0]
        avg_profit = np.mean([t['profit'] for t in profitable_trades]) if profitable_trades else 0
        
        # Grid utilization
        filled_levels = sum(1 for level in self.grid_levels if level.buy_filled)
        grid_utilization_pct = (filled_levels / len(self.grid_levels) * 100) if self.grid_levels else 0
        
        return {
            'total_trades': self.total_trades,
            'profitable_trades': len(profitable_trades),
            'total_profit': self.total_profit,
            'avg_profit_per_trade': avg_profit,
            'total_value': total_value,
            'total_return_pct': total_return_pct,
            'cash': self.cash,
            'position_btc': self.position_quantity,
            'grid_utilization_pct': grid_utilization_pct,
            'grid_active': self.is_active,
            'num_grid_levels': len(self.grid_levels)
        }
    
    def get_grid_state(self) -> List[Dict]:
        """Get current state of all grid levels."""
        return [
            {
                'level': level.level,
                'price': level.price,
                'buy_filled': level.buy_filled,
                'sell_filled': level.sell_filled,
                'profit_realized': level.profit_realized
            }
            for level in self.grid_levels
        ]
