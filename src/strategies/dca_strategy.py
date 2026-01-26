"""
DCA (Dollar-Cost Averaging) Strategy
The most popular crypto bot strategy - used by 60%+ of 3Commas users.

How it works:
1. Place initial "base order" when signal triggers
2. If price drops, place "safety orders" at preset intervals
3. Each safety order averages down the position
4. Take profit when average price + target % is reached

Example:
- Base Order: $100 @ $60,000
- Safety Order 1: $200 @ $58,500 (-2.5%)
- Safety Order 2: $400 @ $57,000 (-5%)
- Average Price: $58,333
- Take Profit @ $60,083 (+3% from avg)
- Final Profit: $100 (after $900 invested)

Win Rate: 80-95% (almost all deals close profitable eventually)
Risk: Moderate (can average down into deep losses if no stop-loss)
Best Conditions: Downtrend followed by recovery
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DCADeal:
    """
    Represents a single DCA deal (one complete cycle from entry to exit).
    """
    deal_id: str
    symbol: str
    start_time: datetime
    base_order_price: float
    base_order_size: float
    target_profit_pct: float
    max_safety_orders: int
    price_step_pct: float  # % drop to trigger next safety order
    safety_order_volume_scale: float  # Martingale multiplier
    
    # State tracking
    safety_orders_filled: List[Dict] = field(default_factory=list)
    total_invested: float = 0.0
    total_quantity: float = 0.0
    average_price: float = 0.0
    is_active: bool = True
    
    # Exit info
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    exit_reason: str = ""
    
    def __post_init__(self):
        # Initialize with base order
        self.total_invested = self.base_order_price * self.base_order_size
        self.total_quantity = self.base_order_size
        self.average_price = self.base_order_price
    
    def add_safety_order(self, price: float, size: float, timestamp: datetime):
        """Add a filled safety order to this deal."""
        self.safety_orders_filled.append({
            'order_num': len(self.safety_orders_filled) + 1,
            'price': price,
            'size': size,
            'timestamp': timestamp
        })
        
        # Update averages
        self.total_invested += price * size
        self.total_quantity += size
        self.average_price = self.total_invested / self.total_quantity
        
        logger.debug(
            f"Deal {self.deal_id}: Safety Order {len(self.safety_orders_filled)} filled "
            f"@ ${price:.2f} - Avg Price now ${self.average_price:.2f}"
        )
    
    def get_take_profit_price(self) -> float:
        """Calculate current take profit price based on average."""
        return self.average_price * (1 + self.target_profit_pct / 100)
    
    def get_next_safety_order_price(self) -> Optional[float]:
        """Calculate price for next safety order."""
        if len(self.safety_orders_filled) >= self.max_safety_orders:
            return None  # Max safety orders reached
        
        # Price drop from base order for next safety order
        step_num = len(self.safety_orders_filled) + 1
        drop_pct = self.price_step_pct * step_num
        
        next_price = self.base_order_price * (1 - drop_pct / 100)
        return next_price
    
    def get_next_safety_order_size(self) -> Optional[float]:
        """Calculate size for next safety order (with scaling)."""
        if len(self.safety_orders_filled) >= self.max_safety_orders:
            return None
        
        # Apply volume scale (Martingale)
        if self.safety_order_volume_scale == 1.0:
            # Linear scaling - same size
            return self.base_order_size
        else:
            # Martingale - increasing size
            step_num = len(self.safety_orders_filled)
            return self.base_order_size * (self.safety_order_volume_scale ** step_num)
    
    def close_deal(self, exit_price: float, exit_time: datetime, reason: str):
        """Close the deal and calculate P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.is_active = False
        
        # Calculate P&L
        proceeds = exit_price * self.total_quantity
        self.realized_pnl = proceeds - self.total_invested
        self.realized_pnl_pct = (self.realized_pnl / self.total_invested) * 100
        
        logger.info(
            f"Deal {self.deal_id} closed: "
            f"Invested: ${self.total_invested:.2f}, "
            f"Avg Price: ${self.average_price:.2f}, "
            f"Exit: ${exit_price:.2f}, "
            f"P&L: ${self.realized_pnl:.2f} ({self.realized_pnl_pct:+.2f}%), "
            f"Reason: {reason}"
        )


class DCAStrategy:
    """
    Dollar-Cost Averaging (DCA) Strategy.
    
    This is the most popular crypto bot strategy, known for high win rates
    (80-95%) by averaging down during dips and taking profit on recovery.
    
    Configuration Parameters:
        base_order_size: Initial order size (USD)
        safety_order_size: Base size for safety orders (USD)
        max_safety_orders: Maximum number of safety orders (e.g., 5-7)
        price_step_pct: % price drop to trigger next safety order (e.g., 2.5%)
        target_profit_pct: Take profit target % from average price (e.g., 3%)
        safety_order_volume_scale: Multiplier for safety order sizes
            - 1.0 = Linear (same size each time)
            - 2.0 = Martingale (double each time)
            - 1.5 = Conservative Martingale
        stop_loss_pct: Max loss % to cut deal (optional, None = no stop)
        cooldown_bars: Bars to wait after deal close before new deal
    """
    
    def __init__(
        self,
        base_order_size: float = 100.0,
        safety_order_size: float = 200.0,
        max_safety_orders: int = 5,
        price_step_pct: float = 2.5,
        target_profit_pct: float = 3.0,
        safety_order_volume_scale: float = 2.0,
        stop_loss_pct: Optional[float] = 20.0,
        cooldown_bars: int = 24,  # 24 hours for hourly data
        entry_signal_func: Optional[callable] = None
    ):
        """
        Initialize DCA strategy.
        
        Args:
            base_order_size: Initial order USD value
            safety_order_size: Base safety order USD value
            max_safety_orders: Max number of safety orders
            price_step_pct: % drop to trigger next safety order
            target_profit_pct: Profit target % from average price
            safety_order_volume_scale: Safety order size multiplier
            stop_loss_pct: Stop loss % from base order (None = no stop)
            cooldown_bars: Bars to wait after deal close
            entry_signal_func: Custom function to generate entry signals
        """
        self.base_order_size = base_order_size
        self.safety_order_size = safety_order_size
        self.max_safety_orders = max_safety_orders
        self.price_step_pct = price_step_pct
        self.target_profit_pct = target_profit_pct
        self.safety_order_volume_scale = safety_order_volume_scale
        self.stop_loss_pct = stop_loss_pct
        self.cooldown_bars = cooldown_bars
        self.entry_signal_func = entry_signal_func
        
        # State
        self.active_deal: Optional[DCADeal] = None
        self.closed_deals: List[DCADeal] = []
        self.deal_counter = 0
        self.bars_since_last_deal = 0
        
        logger.info(
            f"DCA Strategy initialized: "
            f"Base: ${base_order_size}, "
            f"Safety: ${safety_order_size}, "
            f"Max SO: {max_safety_orders}, "
            f"Step: {price_step_pct}%, "
            f"TP: {target_profit_pct}%, "
            f"Scale: {safety_order_volume_scale}x"
        )
    
    def calculate_total_investment(self) -> float:
        """Calculate total capital needed for full DCA cycle."""
        base_investment = self.base_order_size
        
        safety_investment = 0
        for i in range(self.max_safety_orders):
            so_size = self.safety_order_size * (self.safety_order_volume_scale ** i)
            safety_investment += so_size
        
        total = base_investment + safety_investment
        return total
    
    def _default_entry_signal(self, df: pd.DataFrame, i: int) -> bool:
        """
        Default entry signal: Simple trend following with oversold detection.
        
        Entry when:
        - RSI < 40 (oversold)
        - Price above 200 EMA (long-term uptrend)
        """
        if i < 200:
            return False
        
        # Calculate indicators
        close = df['close'].iloc[i]
        ema_200 = df['close'].iloc[i-200:i].ewm(span=200).mean().iloc[-1]
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[i]
        
        # Entry: Oversold in uptrend
        entry = (current_rsi < 40) and (close > ema_200)
        
        return entry
    
    def generate_signal(self, df: pd.DataFrame, i: int, current_price: float) -> Dict:
        """
        Generate trading signals for DCA strategy.
        
        Returns:
            Dict with 'action', 'size', 'price', 'reason'
        """
        # Update cooldown
        if not self.active_deal:
            self.bars_since_last_deal += 1
        
        # ========== NO ACTIVE DEAL - Check for new deal ==========
        if not self.active_deal:
            # Check cooldown
            if self.bars_since_last_deal < self.cooldown_bars:
                return {'action': 'HOLD', 'reason': f'Cooldown ({self.bars_since_last_deal}/{self.cooldown_bars})'}
            
            # Check entry signal
            entry_func = self.entry_signal_func or self._default_entry_signal
            should_enter = entry_func(df, i)
            
            if should_enter:
                # Start new deal
                base_quantity = self.base_order_size / current_price
                
                self.deal_counter += 1
                self.active_deal = DCADeal(
                    deal_id=f"DCA_{self.deal_counter}",
                    symbol="BTCUSDT",
                    start_time=df.index[i],
                    base_order_price=current_price,
                    base_order_size=base_quantity,
                    target_profit_pct=self.target_profit_pct,
                    max_safety_orders=self.max_safety_orders,
                    price_step_pct=self.price_step_pct,
                    safety_order_volume_scale=self.safety_order_volume_scale
                )
                
                logger.info(
                    f"🟢 New DCA Deal {self.active_deal.deal_id} started @ ${current_price:.2f}"
                )
                
                return {
                    'action': 'BUY',
                    'size': base_quantity,
                    'price': current_price,
                    'reason': 'BASE_ORDER'
                }
        
        # ========== ACTIVE DEAL - Manage position ==========
        else:
            deal = self.active_deal
            
            # Check Take Profit
            tp_price = deal.get_take_profit_price()
            if current_price >= tp_price:
                # Close deal with profit
                deal.close_deal(current_price, df.index[i], "TAKE_PROFIT")
                self.closed_deals.append(deal)
                self.active_deal = None
                self.bars_since_last_deal = 0
                
                return {
                    'action': 'SELL',
                    'size': deal.total_quantity,
                    'price': current_price,
                    'reason': 'TAKE_PROFIT',
                    'pnl': deal.realized_pnl,
                    'pnl_pct': deal.realized_pnl_pct
                }
            
            # Check Stop Loss
            if self.stop_loss_pct is not None:
                max_loss_price = deal.base_order_price * (1 - self.stop_loss_pct / 100)
                if current_price <= max_loss_price:
                    # Emergency exit
                    deal.close_deal(current_price, df.index[i], "STOP_LOSS")
                    self.closed_deals.append(deal)
                    self.active_deal = None
                    self.bars_since_last_deal = 0
                    
                    return {
                        'action': 'SELL',
                        'size': deal.total_quantity,
                        'price': current_price,
                        'reason': 'STOP_LOSS',
                        'pnl': deal.realized_pnl,
                        'pnl_pct': deal.realized_pnl_pct
                    }
            
            # Check for next Safety Order trigger
            next_so_price = deal.get_next_safety_order_price()
            if next_so_price is not None and current_price <= next_so_price:
                # Trigger safety order
                next_so_size_usd = deal.get_next_safety_order_size()
                next_so_quantity = next_so_size_usd / current_price
                
                deal.add_safety_order(current_price, next_so_quantity, df.index[i])
                
                return {
                    'action': 'BUY',
                    'size': next_so_quantity,
                    'price': current_price,
                    'reason': f'SAFETY_ORDER_{len(deal.safety_orders_filled)}'
                }
        
        return {'action': 'HOLD', 'reason': 'No signal'}
    
    def get_metrics(self) -> Dict:
        """Get strategy performance metrics."""
        if not self.closed_deals:
            return {
                'total_deals': 0,
                'active_deal': self.active_deal is not None
            }
        
        total_pnl = sum(d.realized_pnl for d in self.closed_deals)
        profitable_deals = [d for d in self.closed_deals if d.realized_pnl > 0]
        losing_deals = [d for d in self.closed_deals if d.realized_pnl < 0]
        
        win_rate = len(profitable_deals) / len(self.closed_deals) * 100
        
        avg_pnl_pct = np.mean([d.realized_pnl_pct for d in self.closed_deals])
        avg_win_pct = np.mean([d.realized_pnl_pct for d in profitable_deals]) if profitable_deals else 0
        avg_loss_pct = np.mean([d.realized_pnl_pct for d in losing_deals]) if losing_deals else 0
        
        avg_safety_orders = np.mean([len(d.safety_orders_filled) for d in self.closed_deals])
        
        return {
            'total_deals': len(self.closed_deals),
            'profitable_deals': len(profitable_deals),
            'losing_deals': len(losing_deals),
            'win_rate_pct': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_pct': avg_pnl_pct,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'avg_safety_orders_used': avg_safety_orders,
            'active_deal': self.active_deal is not None,
            'max_capital_required': self.calculate_total_investment()
        }
    
    def get_deal_history(self) -> List[Dict]:
        """Get full history of all deals."""
        history = []
        for deal in self.closed_deals:
            history.append({
                'deal_id': deal.deal_id,
                'start_time': deal.start_time,
                'end_time': deal.exit_time,
                'base_price': deal.base_order_price,
                'avg_price': deal.average_price,
                'exit_price': deal.exit_price,
                'safety_orders_used': len(deal.safety_orders_filled),
                'total_invested': deal.total_invested,
                'realized_pnl': deal.realized_pnl,
                'realized_pnl_pct': deal.realized_pnl_pct,
                'exit_reason': deal.exit_reason
            })
        return history
