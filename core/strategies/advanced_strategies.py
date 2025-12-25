"""
Advanced Trading Strategies - Enterprise Features #82, #85, #88, #92
Market Making, Grid Trading, DCA, and Momentum Scalping.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class MarketMakingStrategy:
    """
    Feature #82: Market Making Strategy
    
    Places bid/ask orders to capture spread.
    """
    
    def __init__(
        self,
        spread_pct: float = 0.1,           # 0.1% spread
        order_size_pct: float = 5.0,       # 5% of equity per side
        max_inventory: float = 3.0,        # Max 3x base position
        refresh_seconds: int = 30
    ):
        """
        Initialize market making strategy.
        
        Args:
            spread_pct: Target spread percentage
            order_size_pct: Order size as % of equity
            max_inventory: Maximum inventory multiplier
            refresh_seconds: Order refresh interval
        """
        self.spread_pct = spread_pct
        self.order_size_pct = order_size_pct
        self.max_inventory = max_inventory
        self.refresh_seconds = refresh_seconds
        
        self.inventory = 0
        self.realized_pnl = 0
        self.trades = 0
        self.active_orders: Dict[str, Dict] = {}
        
        logger.info(f"Market Making initialized - Spread: {spread_pct}%")
    
    def generate_quotes(self, mid_price: float, equity: float) -> Dict:
        """
        Generate bid/ask quotes.
        
        Args:
            mid_price: Current mid price
            equity: Available equity
            
        Returns:
            Bid and ask quotes
        """
        half_spread = mid_price * (self.spread_pct / 100) / 2
        
        # Calculate order size
        size_usd = equity * (self.order_size_pct / 100)
        size = size_usd / mid_price
        
        # Skew quotes based on inventory
        inventory_skew = self.inventory / self.max_inventory if self.max_inventory > 0 else 0
        skew_adjustment = half_spread * inventory_skew * 0.5
        
        bid_price = mid_price - half_spread - skew_adjustment
        ask_price = mid_price + half_spread - skew_adjustment
        
        # Reduce sizes if inventory limits reached
        bid_size = size if self.inventory < self.max_inventory else 0
        ask_size = size if self.inventory > -self.max_inventory else 0
        
        return {
            'bid': {'price': round(bid_price, 2), 'size': round(bid_size, 6)},
            'ask': {'price': round(ask_price, 2), 'size': round(ask_size, 6)},
            'mid': mid_price,
            'spread_bps': round((ask_price - bid_price) / mid_price * 10000, 2),
            'inventory': self.inventory
        }
    
    def on_fill(self, side: str, price: float, size: float):
        """Handle order fill."""
        if side == 'BUY':
            self.inventory += size
        else:
            self.inventory -= size
        
        self.trades += 1
        logger.debug(f"MM Fill: {side} {size} @ {price}, Inventory: {self.inventory}")
    
    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        return {
            'inventory': round(self.inventory, 6),
            'realized_pnl': round(self.realized_pnl, 2),
            'total_trades': self.trades,
            'is_hedged': abs(self.inventory) < 0.001
        }


class GridTradingSystem:
    """
    Feature #85: Grid Trading System
    
    Places orders at fixed price intervals.
    """
    
    def __init__(
        self,
        grid_levels: int = 10,
        grid_spacing_pct: float = 1.0,
        order_size_usd: float = 100
    ):
        """
        Initialize grid trading.
        
        Args:
            grid_levels: Number of grid levels each direction
            grid_spacing_pct: Space between levels (%)
            order_size_usd: Order size per level
        """
        self.grid_levels = grid_levels
        self.grid_spacing = grid_spacing_pct / 100
        self.order_size_usd = order_size_usd
        
        self.grid_orders: Dict[float, Dict] = {}
        self.filled_levels: List[Dict] = []
        self.base_price: Optional[float] = None
        self.total_profit = 0
        
        logger.info(f"Grid Trading initialized - {grid_levels} levels, {grid_spacing_pct}% spacing")
    
    def setup_grid(self, base_price: float) -> List[Dict]:
        """
        Set up grid around a base price.
        
        Args:
            base_price: Center price for grid
            
        Returns:
            List of grid orders to place
        """
        self.base_price = base_price
        self.grid_orders.clear()
        orders = []
        
        for i in range(1, self.grid_levels + 1):
            # Buy orders below
            buy_price = base_price * (1 - i * self.grid_spacing)
            buy_size = self.order_size_usd / buy_price
            
            orders.append({
                'side': 'BUY',
                'price': round(buy_price, 2),
                'size': round(buy_size, 6),
                'level': -i
            })
            self.grid_orders[round(buy_price, 2)] = {'side': 'BUY', 'level': -i}
            
            # Sell orders above
            sell_price = base_price * (1 + i * self.grid_spacing)
            sell_size = self.order_size_usd / sell_price
            
            orders.append({
                'side': 'SELL',
                'price': round(sell_price, 2),
                'size': round(sell_size, 6),
                'level': i
            })
            self.grid_orders[round(sell_price, 2)] = {'side': 'SELL', 'level': i}
        
        return orders
    
    def on_level_filled(self, price: float, side: str) -> Optional[Dict]:
        """
        Handle grid level fill.
        
        Returns:
            Counter order to place
        """
        self.filled_levels.append({'price': price, 'side': side, 'time': datetime.now()})
        
        # Place counter order at adjacent level
        if side == 'BUY':
            counter_price = price * (1 + self.grid_spacing)
            counter_side = 'SELL'
        else:
            counter_price = price * (1 - self.grid_spacing)
            counter_side = 'BUY'
        
        profit = abs(price * self.grid_spacing) * (self.order_size_usd / price)
        self.total_profit += profit
        
        return {
            'side': counter_side,
            'price': round(counter_price, 2),
            'size': round(self.order_size_usd / counter_price, 6)
        }
    
    def get_stats(self) -> Dict:
        """Get grid statistics."""
        return {
            'base_price': self.base_price,
            'grid_levels': self.grid_levels,
            'active_orders': len(self.grid_orders),
            'filled_count': len(self.filled_levels),
            'total_profit': round(self.total_profit, 2)
        }


class DCAAccumulator:
    """
    Feature #88: DCA Accumulator
    
    Dollar-cost averaging for position building.
    """
    
    def __init__(
        self,
        total_budget: float = 1000,
        num_orders: int = 10,
        interval_hours: float = 24,
        dip_threshold_pct: float = 2.0
    ):
        """
        Initialize DCA accumulator.
        
        Args:
            total_budget: Total budget to deploy
            num_orders: Number of DCA orders
            interval_hours: Hours between orders
            dip_threshold_pct: Extra buy on dips
        """
        self.total_budget = total_budget
        self.num_orders = num_orders
        self.order_size = total_budget / num_orders
        self.interval_hours = interval_hours
        self.dip_threshold = dip_threshold_pct / 100
        
        self.executed_orders: List[Dict] = []
        self.total_spent = 0
        self.total_accumulated = 0
        self.last_price: Optional[float] = None
        self.highest_price: Optional[float] = None
        
        logger.info(f"DCA initialized - ${total_budget} in {num_orders} orders")
    
    def should_buy(self, current_price: float) -> Dict:
        """
        Check if DCA order should execute.
        
        Args:
            current_price: Current market price
            
        Returns:
            Buy decision and size
        """
        if self.total_spent >= self.total_budget:
            return {'should_buy': False, 'reason': 'Budget exhausted'}
        
        if len(self.executed_orders) >= self.num_orders:
            return {'should_buy': False, 'reason': 'Order count reached'}
        
        # Update price tracking
        if self.highest_price is None:
            self.highest_price = current_price
        else:
            self.highest_price = max(self.highest_price, current_price)
        
        # Check for dip opportunity
        dip_pct = (self.highest_price - current_price) / self.highest_price
        
        if dip_pct >= self.dip_threshold:
            # Bonus buy on dip
            bonus_size = self.order_size * min(dip_pct / self.dip_threshold, 2)
            return {
                'should_buy': True,
                'size_usd': round(min(bonus_size, self.total_budget - self.total_spent), 2),
                'reason': f'Dip detected: {dip_pct:.1%}',
                'is_dip_buy': True
            }
        
        return {
            'should_buy': True,
            'size_usd': round(min(self.order_size, self.total_budget - self.total_spent), 2),
            'reason': 'Regular DCA',
            'is_dip_buy': False
        }
    
    def record_buy(self, price: float, size_usd: float):
        """Record an executed buy."""
        size = size_usd / price
        
        self.executed_orders.append({
            'price': price,
            'size_usd': size_usd,
            'size': size,
            'timestamp': datetime.now().isoformat()
        })
        
        self.total_spent += size_usd
        self.total_accumulated += size
        self.last_price = price
    
    def get_average_price(self) -> float:
        """Get average buy price."""
        if self.total_accumulated == 0:
            return 0
        return self.total_spent / self.total_accumulated
    
    def get_stats(self) -> Dict:
        """Get DCA statistics."""
        avg_price = self.get_average_price()
        return {
            'total_spent': round(self.total_spent, 2),
            'total_accumulated': round(self.total_accumulated, 6),
            'average_price': round(avg_price, 2),
            'orders_executed': len(self.executed_orders),
            'orders_remaining': self.num_orders - len(self.executed_orders),
            'budget_remaining': round(self.total_budget - self.total_spent, 2)
        }


class MomentumScalper:
    """
    Feature #92: Momentum Scalper
    
    Quick trades based on short-term momentum.
    """
    
    def __init__(
        self,
        rsi_period: int = 7,
        overbought: float = 70,
        oversold: float = 30,
        target_pct: float = 0.3,
        stop_pct: float = 0.15,
        max_hold_minutes: int = 30
    ):
        """
        Initialize momentum scalper.
        
        Args:
            rsi_period: RSI calculation period
            overbought: RSI overbought level
            oversold: RSI oversold level
            target_pct: Take profit %
            stop_pct: Stop loss %
            max_hold_minutes: Maximum hold time
        """
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.target_pct = target_pct / 100
        self.stop_pct = stop_pct / 100
        self.max_hold_minutes = max_hold_minutes
        
        self.position: Optional[Dict] = None
        self.trades: List[Dict] = []
        
        logger.info(f"Momentum Scalper initialized - RSI({rsi_period}) {oversold}/{overbought}")
    
    def calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI from price list."""
        if len(prices) < self.rsi_period + 1:
            return 50
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        changes = changes[-self.rsi_period:]
        
        gains = [c if c > 0 else 0 for c in changes]
        losses = [-c if c < 0 else 0 for c in changes]
        
        avg_gain = sum(gains) / self.rsi_period
        avg_loss = sum(losses) / self.rsi_period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    def generate_signal(self, prices: List[float], volume: float = 0) -> Optional[Dict]:
        """Generate scalping signal."""
        if self.position:
            return None  # Already in position
        
        rsi = self.calculate_rsi(prices)
        current_price = prices[-1]
        
        # Check momentum at extremes
        if rsi <= self.oversold:
            return {
                'side': 'LONG',
                'entry': current_price,
                'target': current_price * (1 + self.target_pct),
                'stop': current_price * (1 - self.stop_pct),
                'rsi': rsi,
                'reason': f'RSI oversold: {rsi}'
            }
        
        if rsi >= self.overbought:
            return {
                'side': 'SHORT',
                'entry': current_price,
                'target': current_price * (1 - self.target_pct),
                'stop': current_price * (1 + self.stop_pct),
                'rsi': rsi,
                'reason': f'RSI overbought: {rsi}'
            }
        
        return None
    
    def enter_position(self, signal: Dict):
        """Enter a position."""
        self.position = {
            **signal,
            'entry_time': datetime.now()
        }
    
    def check_exit(self, current_price: float) -> Optional[Dict]:
        """Check if position should exit."""
        if not self.position:
            return None
        
        pos = self.position
        
        # Check target
        if pos['side'] == 'LONG' and current_price >= pos['target']:
            return self._close_position(current_price, 'target')
        if pos['side'] == 'SHORT' and current_price <= pos['target']:
            return self._close_position(current_price, 'target')
        
        # Check stop
        if pos['side'] == 'LONG' and current_price <= pos['stop']:
            return self._close_position(current_price, 'stop')
        if pos['side'] == 'SHORT' and current_price >= pos['stop']:
            return self._close_position(current_price, 'stop')
        
        # Check time
        hold_time = (datetime.now() - pos['entry_time']).total_seconds() / 60
        if hold_time >= self.max_hold_minutes:
            return self._close_position(current_price, 'timeout')
        
        return None
    
    def _close_position(self, exit_price: float, reason: str) -> Dict:
        """Close current position."""
        pos = self.position
        
        if pos['side'] == 'LONG':
            pnl_pct = (exit_price - pos['entry']) / pos['entry'] * 100
        else:
            pnl_pct = (pos['entry'] - exit_price) / pos['entry'] * 100
        
        trade = {
            'side': pos['side'],
            'entry': pos['entry'],
            'exit': exit_price,
            'pnl_pct': round(pnl_pct, 3),
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        self.trades.append(trade)
        self.position = None
        
        return trade
    
    def get_stats(self) -> Dict:
        """Get scalper statistics."""
        if not self.trades:
            return {'total_trades': 0}
        
        wins = [t for t in self.trades if t['pnl_pct'] > 0]
        
        return {
            'total_trades': len(self.trades),
            'win_rate': round(len(wins) / len(self.trades) * 100, 1),
            'avg_pnl_pct': round(sum(t['pnl_pct'] for t in self.trades) / len(self.trades), 3),
            'best_trade': max(t['pnl_pct'] for t in self.trades),
            'worst_trade': min(t['pnl_pct'] for t in self.trades),
            'in_position': self.position is not None
        }


# Singletons
_market_maker: Optional[MarketMakingStrategy] = None
_grid: Optional[GridTradingSystem] = None
_dca: Optional[DCAAccumulator] = None
_scalper: Optional[MomentumScalper] = None


def get_market_maker() -> MarketMakingStrategy:
    global _market_maker
    if _market_maker is None:
        _market_maker = MarketMakingStrategy()
    return _market_maker


def get_grid_trader() -> GridTradingSystem:
    global _grid
    if _grid is None:
        _grid = GridTradingSystem()
    return _grid


def get_dca() -> DCAAccumulator:
    global _dca
    if _dca is None:
        _dca = DCAAccumulator()
    return _dca


def get_scalper() -> MomentumScalper:
    global _scalper
    if _scalper is None:
        _scalper = MomentumScalper()
    return _scalper


if __name__ == '__main__':
    # Test market maker
    mm = MarketMakingStrategy()
    quotes = mm.generate_quotes(50000, 10000)
    print(f"MM Quotes: {quotes}")
    
    # Test grid
    grid = GridTradingSystem(grid_levels=5, grid_spacing_pct=1.0)
    orders = grid.setup_grid(50000)
    print(f"Grid orders: {len(orders)}")
    
    # Test DCA
    dca = DCAAccumulator(total_budget=500, num_orders=5)
    decision = dca.should_buy(50000)
    print(f"DCA decision: {decision}")
    
    # Test scalper
    scalper = MomentumScalper()
    prices = [50000 + i*10 for i in range(20)] + [50200 - i*20 for i in range(15)]
    signal = scalper.generate_signal(prices)
    print(f"Scalper signal: {signal}")
