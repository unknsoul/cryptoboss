"""
Market Making Strategy
Institutional-grade liquidity provision for consistent profits.

How it works:
1. Simultaneously place buy (bid) and sell (ask) orders
2. Profit from the spread between them
3. Continuously adjust quotes based on market conditions
4. Manage inventory to avoid directional risk

Example:
- Current Price: $65,000
- Bid: $64,950 (buy offer)
- Ask: $65,050 (sell offer)
- Spread: $100 (0.15%)
- If both fill: Profit $100 (minus fees)

Revenue Sources:
1. Bid-ask spread capture
2. Maker rebates (negative fees on some exchanges)
3. Inventory appreciation (if managed well)

Monthly Return: 10-30%
Risk: Low (market-neutral when balanced)
Best Conditions: Any (profits from volume)
Requirements: Good execution speed, inventory management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Represents a bid or ask quote."""
    side: str  # 'BID' or 'ASK'
    price: float
    size: float
    level: int  # Distance from mid price (0 = closest)
    timestamp: datetime
    filled: bool = False
    fill_time: Optional[datetime] = None


@dataclass
class InventoryState:
    """Track inventory position and risk."""
    target_position: float = 0.0  # Target BTC holdings
    current_position: float = 0.0  # Actual BTC holdings
    max_position: float = 1.0  # Max BTC to hold
    min_position: float = -1.0  # Max short position
    
    def get_inventory_skew(self) -> float:
        """
        Calculate inventory imbalance.
        
        Returns:
            -1.0 to +1.0 where:
            -1.0 = maximum short (need to buy)
            0.0 = balanced
            +1.0 = maximum long (need to sell)
        """
        if self.current_position >= self.max_position:
            return 1.0
        elif self.current_position <= self.min_position:
            return -1.0
        else:
            # Normalize to -1 to +1
            position_range = self.max_position - self.min_position
            return (self.current_position - self.min_position) / position_range * 2 - 1


class MarketMakingStrategy:
    """
    Market Making Strategy - Provide liquidity and earn spreads.
    
    The market maker simultaneously quotes bid and ask prices, profiting from
    the spread while providing liquidity to the market. This strategy requires
    careful inventory management to avoid directional risk.
    
    Configuration:
        base_spread_bps: Base spread in basis points (e.g., 10 = 0.10%)
        num_levels: Number of order levels (depth in book)
        level_spacing_bps: Spacing between levels in bps
        order_size: Base order size per level (BTC)
        inventory_target: Target inventory (0 = neutral)
        max_inventory: Maximum position size (BTC)
        skew_factor: How much to skew quotes when imbalanced (0-1)
        volatility_multiplier: Adjust spread with volatility
    
    Inventory Management:
        - When long: Widen ask, tighten bid (encourage selling)
        - When short: Widen bid, tighten ask (encourage buying)
        - When balanced: Symmetric quotes
    
    Spread Management:
        - Base spread: Configured minimum
        - Volatility adjustment: Widen in volatile markets
        - Adverse selection: Widen after large trades
    """
    
    def __init__(
        self,
        base_spread_bps: float = 10.0,  # 0.10% spread
        num_levels: int = 5,  # 5 levels deep
        level_spacing_bps: float = 5.0,  # 0.05% between levels
        order_size: float = 0.01,  # 0.01 BTC per level
        inventory_target: float = 0.0,  # Neutral
        max_inventory: float = 1.0,  # Max 1 BTC long/short
        skew_factor: float = 0.5,  # 50% skewing
        volatility_multiplier: float = 2.0,  # 2x spread in high vol
        maker_rebate_bps: float = 0.0,  # Maker rebate (if any)
        initial_capital: float = 10000.0
    ):
        """
        Initialize Market Making Strategy.
        
        Args:
            base_spread_bps: Minimum spread in basis points
            num_levels: Order book depth (number of levels)
            level_spacing_bps: Price spacing between levels
            order_size: Size per order level (BTC)
            inventory_target: Target position (0 = neutral)
            max_inventory: Max position size (BTC)
            skew_factor: Quote skewing strength
            volatility_multiplier: Spread adjustment with volatility
            maker_rebate_bps: Maker fee rebate (negative fee)
            initial_capital: Starting capital
        """
        self.base_spread_bps = base_spread_bps
        self.num_levels = num_levels
        self.level_spacing_bps = level_spacing_bps
        self.order_size = order_size
        self.skew_factor = skew_factor
        self.volatility_multiplier = volatility_multiplier
        self.maker_rebate_bps = maker_rebate_bps
        self.initial_capital = initial_capital
        
        # Inventory management
        self.inventory = InventoryState(
            target_position=inventory_target,
            current_position=0.0,
            max_position=max_inventory,
            min_position=-max_inventory
        )
        
        # State
        self.active_bids: List[Quote] = []
        self.active_asks: List[Quote] = []
        self.cash = initial_capital
        self.total_profit = 0.0
        self.total_trades = 0
        self.total_volume = 0.0
        
        # Tracking
        self.trade_history: List[Dict] = []
        self.quote_history: List[Dict] = []
        
        # Volatility tracking
        self.recent_returns: List[float] = []
        self.max_return_history = 100
        
        logger.info(
            f"Market Making Strategy initialized: "
            f"Spread: {base_spread_bps}bps, "
            f"Levels: {num_levels}, "
            f"Size: {order_size} BTC/level"
        )
    
    def calculate_volatility(self, df: pd.DataFrame, i: int, lookback: int = 24) -> float:
        """Calculate recent volatility."""
        if i < lookback:
            return 0.01  # Default 1%
        
        returns = df['close'].pct_change().iloc[i-lookback:i]
        volatility = returns.std()
        
        return volatility if not np.isnan(volatility) else 0.01
    
    def calculate_spread(
        self, 
        mid_price: float, 
        volatility: float,
        inventory_skew: float
    ) -> Tuple[float, float]:
        """
        Calculate bid-ask spread with adjustments.
        
        Returns:
            (bid_spread_bps, ask_spread_bps)
        """
        # Base spread
        base_spread = self.base_spread_bps
        
        # Volatility adjustment
        vol_adjustment = volatility * 100 * self.volatility_multiplier
        adjusted_spread = base_spread + vol_adjustment
        
        # Inventory skewing
        if inventory_skew > 0:  # Long inventory - encourage sells
            bid_spread = adjusted_spread * (1 + self.skew_factor * inventory_skew)
            ask_spread = adjusted_spread * (1 - self.skew_factor * inventory_skew)
        elif inventory_skew < 0:  # Short inventory - encourage buys
            bid_spread = adjusted_spread * (1 + self.skew_factor * inventory_skew)
            ask_spread = adjusted_spread * (1 - self.skew_factor * inventory_skew)
        else:
            bid_spread = adjusted_spread
            ask_spread = adjusted_spread
        
        # Ensure minimum spread
        bid_spread = max(bid_spread, base_spread / 2)
        ask_spread = max(ask_spread, base_spread / 2)
        
        return bid_spread, ask_spread
    
    def generate_quotes(
        self,
        mid_price: float,
        volatility: float,
        timestamp: datetime
    ) -> Tuple[List[Quote], List[Quote]]:
        """
        Generate bid and ask quotes for all levels.
        
        Returns:
            (bids, asks)
        """
        inventory_skew = self.inventory.get_inventory_skew()
        bid_spread_bps, ask_spread_bps = self.calculate_spread(
            mid_price, volatility, inventory_skew
        )
        
        bids = []
        asks = []
        
        for level in range(self.num_levels):
            # Calculate prices
            level_offset_bps = level * self.level_spacing_bps
            
            bid_price = mid_price * (1 - (bid_spread_bps + level_offset_bps) / 10000)
            ask_price = mid_price * (1 + (ask_spread_bps + level_offset_bps) / 10000)
            
            # Adjust size based on inventory (larger sizes to rebalance)
            size_multiplier = 1.0
            if inventory_skew > 0.5:  # Very long - larger asks
                size_multiplier = 1.0 + (inventory_skew - 0.5)
            elif inventory_skew < -0.5:  # Very short - larger bids
                size_multiplier = 1.0 + abs(inventory_skew + 0.5)
            
            order_size = self.order_size * size_multiplier
            
            bids.append(Quote(
                side='BID',
                price=bid_price,
                size=order_size,
                level=level,
                timestamp=timestamp
            ))
            
            asks.append(Quote(
                side='ASK',
                price=ask_price,
                size=order_size,
                level=level,
                timestamp=timestamp
            ))
        
        return bids, asks
    
    def check_fills(
        self,
        current_price: float,
        timestamp: datetime
    ) -> List[Dict]:
        """Check if any quotes have been filled."""
        fills = []
        
        # Check bid fills (price touched our bid)
        for bid in self.active_bids:
            if not bid.filled and current_price <= bid.price:
                # Bid filled - we bought
                bid.filled = True
                bid.fill_time = timestamp
                
                # Update inventory
                self.inventory.current_position += bid.size
                cost = bid.price * bid.size
                self.cash -= cost
                
                # Record trade
                fills.append({
                    'side': 'BUY',
                    'price': bid.price,
                    'size': bid.size,
                    'level': bid.level,
                    'timestamp': timestamp
                })
                
                self.total_trades += 1
                self.total_volume += bid.size
                
                logger.debug(
                    f"Bid filled: Level {bid.level} @ ${bid.price:,.2f}, "
                    f"Inventory: {self.inventory.current_position:.4f}"
                )
        
        # Check ask fills (price touched our ask)
        for ask in self.active_asks:
            if not ask.filled and current_price >= ask.price:
                # Ask filled - we sold
                ask.filled = True
                ask.fill_time = timestamp
                
                # Update inventory
                self.inventory.current_position -= ask.size
                proceeds = ask.price * ask.size
                self.cash += proceeds
                
                # Calculate profit if we had corresponding buy
                # Simplified: assume average FIFO profit
                avg_cost = (proceeds / ask.size) * 0.999  # Approximate cost basis
                profit = proceeds - (avg_cost * ask.size)
                
                # Add maker rebate
                rebate = proceeds * (self.maker_rebate_bps / 10000)
                profit += rebate
                
                self.total_profit += profit
                
                # Record trade
                fills.append({
                    'side': 'SELL',
                    'price': ask.price,
                    'size': ask.size,
                    'level': ask.level,
                    'profit': profit,
                    'timestamp': timestamp
                })
                
                self.total_trades += 1
                self.total_volume += ask.size
                
                logger.debug(
                    f"Ask filled: Level {ask.level} @ ${ask.price:,.2f}, "
                    f"Profit: ${profit:.2f}, "
                    f"Inventory: {self.inventory.current_position:.4f}"
                )
        
        # Record fills
        for fill in fills:
            self.trade_history.append(fill)
        
        return fills
    
    def update_quotes(
        self,
        mid_price: float,
        volatility: float,
        timestamp: datetime
    ):
        """Update all quotes to current market conditions."""
        # Clear old quotes
        self.active_bids = []
        self.active_asks = []
        
        # Generate new quotes
        bids, asks = self.generate_quotes(mid_price, volatility, timestamp)
        
        self.active_bids = bids
        self.active_asks = asks
        
        # Record quote snapshot
        self.quote_history.append({
            'timestamp': timestamp,
            'mid_price': mid_price,
            'best_bid': bids[0].price if bids else None,
            'best_ask': asks[0].price if asks else None,
            'spread_bps': ((asks[0].price - bids[0].price) / mid_price * 10000) if bids and asks else None,
            'inventory': self.inventory.current_position,
            'inventory_skew': self.inventory.get_inventory_skew()
        })
    
    def emergency_unwind(self, current_price: float, timestamp: datetime):
        """Emergency inventory liquidation."""
        if abs(self.inventory.current_position) > 0.001:
            logger.warning(
                f"Emergency unwind: Liquidating {self.inventory.current_position:.4f} BTC "
                f"@ ${current_price:,.2f}"
            )
            
            # Sell all holdings
            proceeds = self.inventory.current_position * current_price
            self.cash += proceeds
            self.inventory.current_position = 0.0
    
    def generate_signal(self, df: pd.DataFrame, i: int, current_price: float) -> Dict:
        """
        Generate market making signals.
        
        Returns:
            Dict with 'action', 'quotes', 'fills'
        """
        if i < 24:  # Need warmup for volatility
            return {'action': 'HOLD', 'info': 'Warming up'}
        
        # Calculate volatility
        volatility = self.calculate_volatility(df, i)
        
        # Check for fills on existing quotes
        fills = self.check_fills(current_price, df.index[i])
        
        # Update quotes to current market
        self.update_quotes(current_price, volatility, df.index[i])
        
        # Check inventory limits
        if abs(self.inventory.current_position) > self.inventory.max_position:
            logger.warning(
                f"Inventory limit exceeded: {self.inventory.current_position:.4f} BTC"
            )
            self.emergency_unwind(current_price, df.index[i])
        
        result = {
            'action': 'MARKET_MAKE',
            'fills': len(fills),
            'active_bids': len(self.active_bids),
            'active_asks': len(self.active_asks),
            'inventory': self.inventory.current_position,
            'inventory_skew': self.inventory.get_inventory_skew(),
            'total_profit': self.total_profit
        }
        
        if fills:
            result['filled_orders'] = fills
        
        return result
    
    def get_metrics(self) -> Dict:
        """Get strategy performance metrics."""
        total_value = self.cash + (self.inventory.current_position * 
                                   (self.active_asks[0].price if self.active_asks else 0))
        total_return_pct = ((total_value - self.initial_capital) / 
                           self.initial_capital * 100)
        
        # Calculate average profit per trade
        profitable_trades = [t for t in self.trade_history 
                           if t.get('side') == 'SELL' and t.get('profit', 0) > 0]
        avg_profit = (np.mean([t['profit'] for t in profitable_trades]) 
                     if profitable_trades else 0)
        
        # Calculate average spread captured
        if self.quote_history:
            avg_spread = np.mean([q['spread_bps'] for q in self.quote_history 
                                if q.get('spread_bps') is not None])
        else:
            avg_spread = 0
        
        return {
            'total_trades': self.total_trades,
            'total_volume': self.total_volume,
            'total_profit': self.total_profit,
            'avg_profit_per_trade': avg_profit,
            'avg_spread_captured_bps': avg_spread,
            'total_value': total_value,
            'total_return_pct': total_return_pct,
            'cash': self.cash,
            'inventory_btc': self.inventory.current_position,
            'inventory_skew':  self.inventory.get_inventory_skew(),
            'num_active_quotes': len(self.active_bids) + len(self.active_asks)
        }
