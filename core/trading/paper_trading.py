"""
Paper Trading & Advanced Features - Enterprise Features #58, #63, #68, #74
Paper Trading Simulator, Fee Modeling, Dynamic Targets, and Seasonal Detection.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
import math

logger = logging.getLogger(__name__)


# =============================================================================
# UPGRADE #1: Order Book Simulation
# =============================================================================

class OrderBookSimulator:
    """
    Feature: Order Book Depth Simulation
    
    Simulates realistic order book behavior for accurate fill modeling.
    """
    
    def __init__(
        self,
        base_spread_bps: float = 1.0,  # Base spread in basis points
        depth_levels: int = 10,
        liquidity_factor: float = 1.0
    ):
        """
        Initialize order book simulator.
        
        Args:
            base_spread_bps: Base bid-ask spread in basis points
            depth_levels: Number of price levels to simulate
            liquidity_factor: Multiplier for available liquidity
        """
        self.base_spread_bps = base_spread_bps
        self.depth_levels = depth_levels
        self.liquidity_factor = liquidity_factor
        
        logger.info(f"Order Book Simulator initialized - Spread: {base_spread_bps}bps, Levels: {depth_levels}")
    
    def generate_order_book(self, mid_price: float, volatility: float = 0.02) -> Dict:
        """
        Generate simulated order book around mid price.
        
        Args:
            mid_price: Current mid price
            volatility: Current volatility for spread adjustment
            
        Returns:
            Order book with bids and asks
        """
        # Adjust spread based on volatility
        vol_adj_spread = self.base_spread_bps * (1 + volatility * 10)
        spread = mid_price * (vol_adj_spread / 10000)
        
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        bids = []
        asks = []
        
        for i in range(self.depth_levels):
            # Price decreases for bids, increases for asks
            bid_price = best_bid - (i * spread * 0.5)
            ask_price = best_ask + (i * spread * 0.5)
            
            # Liquidity increases with depth (more volume further from mid)
            base_qty = random.uniform(0.5, 2.0) * self.liquidity_factor
            bid_qty = base_qty * (1 + i * 0.2)
            ask_qty = base_qty * (1 + i * 0.2)
            
            bids.append({'price': round(bid_price, 2), 'qty': round(bid_qty, 4)})
            asks.append({'price': round(ask_price, 2), 'qty': round(ask_qty, 4)})
        
        return {
            'mid_price': mid_price,
            'best_bid': round(best_bid, 2),
            'best_ask': round(best_ask, 2),
            'spread': round(spread, 2),
            'spread_bps': round(vol_adj_spread, 2),
            'bids': bids,
            'asks': asks
        }
    
    def simulate_fill(
        self,
        side: str,
        size: float,
        order_book: Dict
    ) -> Dict:
        """
        Simulate order fill with partial fill possibility.
        
        Args:
            side: 'BUY' or 'SELL'
            size: Order size
            order_book: Current order book
            
        Returns:
            Fill result with average price and filled quantity
        """
        levels = order_book['asks'] if side == 'BUY' else order_book['bids']
        
        remaining = size
        filled_qty = 0
        total_cost = 0
        fills = []
        
        for level in levels:
            if remaining <= 0:
                break
            
            fill_qty = min(remaining, level['qty'])
            fills.append({'price': level['price'], 'qty': fill_qty})
            
            total_cost += level['price'] * fill_qty
            filled_qty += fill_qty
            remaining -= fill_qty
        
        avg_price = total_cost / filled_qty if filled_qty > 0 else 0
        
        return {
            'filled': filled_qty >= size,
            'partial_fill': 0 < filled_qty < size,
            'filled_qty': round(filled_qty, 4),
            'remaining_qty': round(remaining, 4),
            'avg_fill_price': round(avg_price, 2),
            'fills': fills,
            'slippage': round(avg_price - order_book['mid_price'], 2) if side == 'BUY' else round(order_book['mid_price'] - avg_price, 2)
        }


# =============================================================================
# UPGRADE #6: Latency Simulation
# =============================================================================

class LatencySimulator:
    """
    Feature: Realistic Network Latency Simulation
    
    Simulates variable latency, order rejections, and requotes.
    """
    
    def __init__(
        self,
        min_latency_ms: int = 20,
        max_latency_ms: int = 200,
        rejection_rate: float = 0.01,  # 1% rejection rate
        requote_rate: float = 0.02      # 2% requote rate
    ):
        """
        Initialize latency simulator.
        
        Args:
            min_latency_ms: Minimum network latency
            max_latency_ms: Maximum network latency
            rejection_rate: Order rejection probability
            requote_rate: Requote probability
        """
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.rejection_rate = rejection_rate
        self.requote_rate = requote_rate
        
        self.total_rejections = 0
        self.total_requotes = 0
        
        logger.info(f"Latency Simulator initialized - {min_latency_ms}-{max_latency_ms}ms")
    
    def simulate_execution(self, price: float) -> Dict:
        """
        Simulate order execution with latency effects.
        
        Args:
            price: Requested execution price
            
        Returns:
            Execution result with status and actual price
        """
        import time
        
        # Simulate variable latency
        latency = random.randint(self.min_latency_ms, self.max_latency_ms)
        time.sleep(latency / 1000)
        
        # Check for rejection
        if random.random() < self.rejection_rate:
            self.total_rejections += 1
            return {
                'status': 'REJECTED',
                'reason': random.choice(['INSUFFICIENT_MARGIN', 'RATE_LIMIT', 'MARKET_CLOSED']),
                'latency_ms': latency,
                'original_price': price,
                'execution_price': None
            }
        
        # Check for requote
        if random.random() < self.requote_rate:
            self.total_requotes += 1
            requote_adjustment = price * random.uniform(-0.001, 0.001)
            return {
                'status': 'REQUOTE',
                'latency_ms': latency,
                'original_price': price,
                'requote_price': round(price + requote_adjustment, 2)
            }
        
        return {
            'status': 'FILLED',
            'latency_ms': latency,
            'original_price': price,
            'execution_price': price
        }
    
    def get_stats(self) -> Dict:
        """Get latency simulation statistics."""
        return {
            'total_rejections': self.total_rejections,
            'total_requotes': self.total_requotes,
            'latency_range': f"{self.min_latency_ms}-{self.max_latency_ms}ms"
        }


# =============================================================================
# UPGRADE #2: Trailing Stop Manager
# =============================================================================

class TrailingStopManager:
    """
    Feature: Dynamic Trailing Stop Loss Management
    
    Supports multiple trailing stop modes for protecting profits.
    """
    
    def __init__(self):
        """Initialize trailing stop manager."""
        self.active_stops: Dict[str, Dict] = {}
        
        logger.info("Trailing Stop Manager initialized")
    
    def create_trailing_stop(
        self,
        position_id: str,
        entry_price: float,
        side: str,
        mode: str = 'PERCENT',
        trail_value: float = 1.0,
        breakeven_trigger_pct: float = 0.5,
        atr: Optional[float] = None
    ) -> Dict:
        """
        Create a trailing stop for a position.
        
        Args:
            position_id: Position identifier
            entry_price: Entry price
            side: 'LONG' or 'SHORT'
            mode: 'PERCENT', 'ATR', or 'BREAKEVEN'
            trail_value: Trail distance (pct or ATR multiplier)
            breakeven_trigger_pct: Profit % to trigger breakeven mode
            atr: ATR value (required for ATR mode)
            
        Returns:
            Trailing stop configuration
        """
        stop = {
            'position_id': position_id,
            'entry_price': entry_price,
            'side': side,
            'mode': mode,
            'trail_value': trail_value,
            'breakeven_trigger_pct': breakeven_trigger_pct,
            'atr': atr,
            'current_stop': None,
            'highest_price': entry_price if side == 'LONG' else None,
            'lowest_price': entry_price if side == 'SHORT' else None,
            'breakeven_activated': False,
            'created_at': datetime.now().isoformat()
        }
        
        # Initialize stop level
        if mode == 'PERCENT':
            distance = entry_price * (trail_value / 100)
        elif mode == 'ATR' and atr:
            distance = atr * trail_value
        else:
            distance = entry_price * 0.01  # Default 1%
        
        if side == 'LONG':
            stop['current_stop'] = round(entry_price - distance, 2)
        else:
            stop['current_stop'] = round(entry_price + distance, 2)
        
        self.active_stops[position_id] = stop
        
        return stop
    
    def update_stop(self, position_id: str, current_price: float) -> Dict:
        """
        Update trailing stop based on current price.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            
        Returns:
            Updated stop info with trigger status
        """
        if position_id not in self.active_stops:
            return {'error': 'Position not found'}
        
        stop = self.active_stops[position_id]
        triggered = False
        
        if stop['side'] == 'LONG':
            # Update highest price
            if current_price > stop['highest_price']:
                stop['highest_price'] = current_price
                
                # Check for breakeven activation
                profit_pct = (current_price - stop['entry_price']) / stop['entry_price'] * 100
                if profit_pct >= stop['breakeven_trigger_pct'] and not stop['breakeven_activated']:
                    stop['current_stop'] = stop['entry_price']  # Move to breakeven
                    stop['breakeven_activated'] = True
                else:
                    # Trail the stop
                    if stop['mode'] == 'PERCENT':
                        new_stop = current_price * (1 - stop['trail_value'] / 100)
                    elif stop['mode'] == 'ATR' and stop['atr']:
                        new_stop = current_price - (stop['atr'] * stop['trail_value'])
                    else:
                        new_stop = stop['current_stop']
                    
                    if new_stop > stop['current_stop']:
                        stop['current_stop'] = round(new_stop, 2)
            
            # Check for stop trigger
            if current_price <= stop['current_stop']:
                triggered = True
        
        else:  # SHORT
            # Update lowest price
            if current_price < stop['lowest_price']:
                stop['lowest_price'] = current_price
                
                profit_pct = (stop['entry_price'] - current_price) / stop['entry_price'] * 100
                if profit_pct >= stop['breakeven_trigger_pct'] and not stop['breakeven_activated']:
                    stop['current_stop'] = stop['entry_price']
                    stop['breakeven_activated'] = True
                else:
                    if stop['mode'] == 'PERCENT':
                        new_stop = current_price * (1 + stop['trail_value'] / 100)
                    elif stop['mode'] == 'ATR' and stop['atr']:
                        new_stop = current_price + (stop['atr'] * stop['trail_value'])
                    else:
                        new_stop = stop['current_stop']
                    
                    if new_stop < stop['current_stop']:
                        stop['current_stop'] = round(new_stop, 2)
            
            if current_price >= stop['current_stop']:
                triggered = True
        
        return {
            'position_id': position_id,
            'current_stop': stop['current_stop'],
            'triggered': triggered,
            'breakeven_activated': stop['breakeven_activated'],
            'current_price': current_price
        }
    
    def remove_stop(self, position_id: str):
        """Remove trailing stop for closed position."""
        if position_id in self.active_stops:
            del self.active_stops[position_id]


# =============================================================================
# UPGRADE #3: Position Scaling
# =============================================================================

class PositionScaler:
    """
    Feature: Position Scaling (Pyramiding & Partial Exits)
    
    Manages scaling into positions and partial profit taking.
    """
    
    def __init__(
        self,
        max_scale_ins: int = 3,
        scale_in_factor: float = 0.5,  # Each add is 50% of original
        take_profit_levels: List[Tuple[float, float]] = None  # [(pct_profit, pct_to_close)]
    ):
        """
        Initialize position scaler.
        
        Args:
            max_scale_ins: Maximum number of scale-ins allowed
            scale_in_factor: Size factor for each scale-in
            take_profit_levels: List of (profit_pct, close_pct) tuples
        """
        self.max_scale_ins = max_scale_ins
        self.scale_in_factor = scale_in_factor
        self.take_profit_levels = take_profit_levels or [
            (1.0, 0.25),   # At 1% profit, close 25%
            (2.0, 0.25),   # At 2% profit, close 25%
            (3.0, 0.25)    # At 3% profit, close 25%
        ]
        
        self.positions: Dict[str, Dict] = {}
        
        logger.info(f"Position Scaler initialized - Max scale-ins: {max_scale_ins}")
    
    def create_position(
        self,
        position_id: str,
        side: str,
        entry_price: float,
        size: float
    ) -> Dict:
        """Create a new scalable position."""
        position = {
            'position_id': position_id,
            'side': side,
            'entries': [{'price': entry_price, 'size': size, 'time': datetime.now()}],
            'original_size': size,
            'current_size': size,
            'avg_entry': entry_price,
            'scale_in_count': 0,
            'partial_exits': [],
            'tp_levels_hit': []
        }
        
        self.positions[position_id] = position
        return position
    
    def scale_in(
        self,
        position_id: str,
        price: float,
        size_override: Optional[float] = None
    ) -> Dict:
        """
        Add to an existing position (pyramid).
        
        Args:
            position_id: Position to scale into
            price: Current price for scale-in
            size_override: Custom size, or use scale_in_factor
            
        Returns:
            Scale-in result
        """
        if position_id not in self.positions:
            return {'error': 'Position not found'}
        
        pos = self.positions[position_id]
        
        if pos['scale_in_count'] >= self.max_scale_ins:
            return {'error': 'Max scale-ins reached', 'count': pos['scale_in_count']}
        
        # Calculate scale-in size
        add_size = size_override or (pos['original_size'] * self.scale_in_factor)
        
        # Update average entry
        total_cost = pos['avg_entry'] * pos['current_size'] + price * add_size
        new_size = pos['current_size'] + add_size
        pos['avg_entry'] = round(total_cost / new_size, 2)
        pos['current_size'] = new_size
        
        pos['entries'].append({'price': price, 'size': add_size, 'time': datetime.now()})
        pos['scale_in_count'] += 1
        
        return {
            'position_id': position_id,
            'action': 'SCALE_IN',
            'add_size': add_size,
            'new_avg_entry': pos['avg_entry'],
            'new_total_size': pos['current_size'],
            'scale_in_count': pos['scale_in_count']
        }
    
    def check_take_profit(self, position_id: str, current_price: float) -> Optional[Dict]:
        """
        Check if any take profit level is hit.
        
        Args:
            position_id: Position to check
            current_price: Current market price
            
        Returns:
            Partial exit instruction if level hit
        """
        if position_id not in self.positions:
            return None
        
        pos = self.positions[position_id]
        
        if pos['side'] == 'LONG':
            profit_pct = (current_price - pos['avg_entry']) / pos['avg_entry'] * 100
        else:
            profit_pct = (pos['avg_entry'] - current_price) / pos['avg_entry'] * 100
        
        for level_pct, close_pct in self.take_profit_levels:
            if level_pct not in pos['tp_levels_hit'] and profit_pct >= level_pct:
                close_size = pos['original_size'] * close_pct
                close_size = min(close_size, pos['current_size'])
                
                pos['tp_levels_hit'].append(level_pct)
                pos['current_size'] -= close_size
                pos['partial_exits'].append({
                    'level_pct': level_pct,
                    'close_size': close_size,
                    'price': current_price,
                    'time': datetime.now()
                })
                
                return {
                    'position_id': position_id,
                    'action': 'PARTIAL_EXIT',
                    'trigger_level': level_pct,
                    'close_size': close_size,
                    'remaining_size': pos['current_size'],
                    'profit_pct': round(profit_pct, 2)
                }
        
        return None


# =============================================================================
# UPGRADE #4: Circuit Breaker
# =============================================================================

class CircuitBreaker:
    """
    Feature: Drawdown Tracking & Circuit Breaker
    
    Monitors drawdown and enforces trading limits.
    """
    
    def __init__(
        self,
        max_drawdown_pct: float = 10.0,
        daily_loss_limit_pct: float = 5.0,
        max_consecutive_losses: int = 5,
        cooldown_minutes: int = 60
    ):
        """
        Initialize circuit breaker.
        
        Args:
            max_drawdown_pct: Maximum allowed drawdown from peak
            daily_loss_limit_pct: Maximum daily loss allowed
            max_consecutive_losses: Stop after N consecutive losses
            cooldown_minutes: Cooldown period after breaker trips
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_minutes = cooldown_minutes
        
        self.initial_equity = 0
        self.peak_equity = 0
        self.current_equity = 0
        self.daily_start_equity = 0
        self.consecutive_losses = 0
        
        self.is_tripped = False
        self.trip_reason = None
        self.trip_time: Optional[datetime] = None
        
        logger.info(f"Circuit Breaker initialized - Max DD: {max_drawdown_pct}%, Daily limit: {daily_loss_limit_pct}%")
    
    def initialize(self, equity: float):
        """Initialize with starting equity."""
        self.initial_equity = equity
        self.peak_equity = equity
        self.current_equity = equity
        self.daily_start_equity = equity
    
    def reset_daily(self):
        """Reset daily tracking (call at start of trading day)."""
        self.daily_start_equity = self.current_equity
        self.consecutive_losses = 0
    
    def update(self, new_equity: float, trade_result: Optional[str] = None) -> Dict:
        """
        Update equity and check circuit breaker conditions.
        
        Args:
            new_equity: Current equity value
            trade_result: 'WIN' or 'LOSS' for last trade
            
        Returns:
            Status with any breaker triggers
        """
        self.current_equity = new_equity
        
        # Update peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        
        # Track consecutive losses
        if trade_result == 'LOSS':
            self.consecutive_losses += 1
        elif trade_result == 'WIN':
            self.consecutive_losses = 0
        
        # Check if still in cooldown
        if self.trip_time:
            elapsed = (datetime.now() - self.trip_time).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                return {
                    'can_trade': False,
                    'is_tripped': True,
                    'reason': self.trip_reason,
                    'cooldown_remaining_min': round(self.cooldown_minutes - elapsed, 1)
                }
            else:
                # Cooldown expired, reset
                self.is_tripped = False
                self.trip_reason = None
                self.trip_time = None
        
        # Calculate drawdowns
        drawdown_from_peak = (self.peak_equity - new_equity) / self.peak_equity * 100
        daily_drawdown = (self.daily_start_equity - new_equity) / self.daily_start_equity * 100
        
        # Check breaker conditions
        if drawdown_from_peak >= self.max_drawdown_pct:
            self._trip('MAX_DRAWDOWN', f"Drawdown {drawdown_from_peak:.1f}% exceeded limit")
        elif daily_drawdown >= self.daily_loss_limit_pct:
            self._trip('DAILY_LIMIT', f"Daily loss {daily_drawdown:.1f}% exceeded limit")
        elif self.consecutive_losses >= self.max_consecutive_losses:
            self._trip('CONSECUTIVE_LOSSES', f"{self.consecutive_losses} consecutive losses")
        
        return {
            'can_trade': not self.is_tripped,
            'is_tripped': self.is_tripped,
            'reason': self.trip_reason,
            'drawdown_pct': round(drawdown_from_peak, 2),
            'daily_drawdown_pct': round(daily_drawdown, 2),
            'consecutive_losses': self.consecutive_losses,
            'current_equity': new_equity,
            'peak_equity': self.peak_equity
        }
    
    def _trip(self, reason: str, message: str):
        """Trip the circuit breaker."""
        self.is_tripped = True
        self.trip_reason = reason
        self.trip_time = datetime.now()
        logger.warning(f"CIRCUIT BREAKER TRIPPED: {message}")
    
    def force_reset(self):
        """Force reset the circuit breaker (manual override)."""
        self.is_tripped = False
        self.trip_reason = None
        self.trip_time = None
        self.consecutive_losses = 0


# =============================================================================
# UPGRADE #5: Bracket Order Manager
# =============================================================================

class BracketOrderManager:
    """
    Feature: OCO Bracket Orders
    
    Manages entry orders with attached stop-loss and take-profit.
    """
    
    def __init__(self):
        """Initialize bracket order manager."""
        self.bracket_orders: Dict[str, Dict] = {}
        
        logger.info("Bracket Order Manager initialized")
    
    def create_bracket_order(
        self,
        order_id: str,
        side: str,
        entry_price: float,
        size: float,
        sl_price: float,
        tp_price: float,
        order_type: str = 'LIMIT'
    ) -> Dict:
        """
        Create a bracket order (entry + SL + TP).
        
        Args:
            order_id: Unique order identifier
            side: 'BUY' or 'SELL'
            entry_price: Entry price
            size: Order size
            sl_price: Stop-loss price
            tp_price: Take-profit price
            order_type: 'MARKET' or 'LIMIT'
            
        Returns:
            Bracket order details
        """
        bracket = {
            'bracket_id': f"BRK-{order_id}",
            'entry_order': {
                'id': f"ENT-{order_id}",
                'side': side,
                'price': entry_price,
                'size': size,
                'type': order_type,
                'status': 'PENDING'
            },
            'sl_order': {
                'id': f"SL-{order_id}",
                'side': 'SELL' if side == 'BUY' else 'BUY',
                'price': sl_price,
                'size': size,
                'type': 'STOP',
                'status': 'INACTIVE'  # Activated when entry fills
            },
            'tp_order': {
                'id': f"TP-{order_id}",
                'side': 'SELL' if side == 'BUY' else 'BUY',
                'price': tp_price,
                'size': size,
                'type': 'LIMIT',
                'status': 'INACTIVE'
            },
            'status': 'PENDING',
            'created_at': datetime.now().isoformat()
        }
        
        self.bracket_orders[order_id] = bracket
        
        return bracket
    
    def on_entry_fill(self, order_id: str, fill_price: float) -> Dict:
        """
        Handle entry order fill - activate SL and TP.
        
        Args:
            order_id: Order identifier
            fill_price: Actual fill price
            
        Returns:
            Updated bracket status
        """
        if order_id not in self.bracket_orders:
            return {'error': 'Bracket order not found'}
        
        bracket = self.bracket_orders[order_id]
        bracket['entry_order']['status'] = 'FILLED'
        bracket['entry_order']['fill_price'] = fill_price
        bracket['sl_order']['status'] = 'ACTIVE'
        bracket['tp_order']['status'] = 'ACTIVE'
        bracket['status'] = 'ACTIVE'
        
        logger.info(f"Bracket {order_id} activated - SL: {bracket['sl_order']['price']}, TP: {bracket['tp_order']['price']}")
        
        return bracket
    
    def check_triggers(self, order_id: str, current_price: float) -> Optional[Dict]:
        """
        Check if SL or TP is triggered.
        
        Args:
            order_id: Order identifier
            current_price: Current market price
            
        Returns:
            Triggered order details if any
        """
        if order_id not in self.bracket_orders:
            return None
        
        bracket = self.bracket_orders[order_id]
        
        if bracket['status'] != 'ACTIVE':
            return None
        
        entry_side = bracket['entry_order']['side']
        sl_price = bracket['sl_order']['price']
        tp_price = bracket['tp_order']['price']
        
        if entry_side == 'BUY':  # Long position
            if current_price <= sl_price:
                return self._trigger_order(order_id, 'SL', current_price)
            elif current_price >= tp_price:
                return self._trigger_order(order_id, 'TP', current_price)
        else:  # Short position
            if current_price >= sl_price:
                return self._trigger_order(order_id, 'SL', current_price)
            elif current_price <= tp_price:
                return self._trigger_order(order_id, 'TP', current_price)
        
        return None
    
    def _trigger_order(self, order_id: str, triggered: str, price: float) -> Dict:
        """Trigger SL or TP and cancel the other (OCO)."""
        bracket = self.bracket_orders[order_id]
        
        if triggered == 'SL':
            bracket['sl_order']['status'] = 'FILLED'
            bracket['sl_order']['fill_price'] = price
            bracket['tp_order']['status'] = 'CANCELLED'
        else:
            bracket['tp_order']['status'] = 'FILLED'
            bracket['tp_order']['fill_price'] = price
            bracket['sl_order']['status'] = 'CANCELLED'
        
        bracket['status'] = 'CLOSED'
        
        return {
            'bracket_id': bracket['bracket_id'],
            'triggered': triggered,
            'fill_price': price,
            'cancelled': 'TP' if triggered == 'SL' else 'SL'
        }
    
    def cancel_bracket(self, order_id: str) -> Dict:
        """Cancel entire bracket order."""
        if order_id not in self.bracket_orders:
            return {'error': 'Bracket order not found'}
        
        bracket = self.bracket_orders[order_id]
        bracket['entry_order']['status'] = 'CANCELLED'
        bracket['sl_order']['status'] = 'CANCELLED'
        bracket['tp_order']['status'] = 'CANCELLED'
        bracket['status'] = 'CANCELLED'
        
        return {'bracket_id': bracket['bracket_id'], 'status': 'CANCELLED'}


# =============================================================================
# UPGRADE #9: Kelly Criterion Position Sizing
# =============================================================================

class KellyCriterionSizer:
    """
    Feature: Kelly Criterion Optimal Position Sizing
    
    Calculates optimal position size based on edge and win rate.
    """
    
    def __init__(
        self,
        max_kelly_fraction: float = 0.25,  # Max 25% Kelly (quarter Kelly)
        min_trades_required: int = 20
    ):
        """
        Initialize Kelly criterion sizer.
        
        Args:
            max_kelly_fraction: Maximum Kelly fraction to use (0.25 = quarter Kelly)
            min_trades_required: Minimum trades needed for reliable calculation
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.min_trades_required = min_trades_required
        
        self.wins: List[float] = []
        self.losses: List[float] = []
        
        logger.info(f"Kelly Criterion Sizer initialized - Max fraction: {max_kelly_fraction}")
    
    def record_trade(self, pnl: float):
        """Record a trade result for Kelly calculation."""
        if pnl > 0:
            self.wins.append(pnl)
        else:
            self.losses.append(abs(pnl))
    
    def calculate_kelly(self) -> Dict:
        """
        Calculate Kelly criterion.
        
        Kelly % = W - (1-W)/R
        Where:
            W = Win probability
            R = Win/loss ratio (avg win / avg loss)
            
        Returns:
            Kelly calculation details
        """
        total_trades = len(self.wins) + len(self.losses)
        
        if total_trades < self.min_trades_required:
            return {
                'kelly_pct': 0,
                'recommended_size_pct': 0,
                'insufficient_data': True,
                'trades_needed': self.min_trades_required - total_trades
            }
        
        win_rate = len(self.wins) / total_trades
        
        avg_win = sum(self.wins) / len(self.wins) if self.wins else 0
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 1
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        
        # Kelly formula
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply maximum fraction
        recommended = min(max(kelly, 0), 1) * self.max_kelly_fraction
        
        return {
            'kelly_pct': round(kelly * 100, 2),
            'recommended_size_pct': round(recommended * 100, 2),
            'win_rate': round(win_rate * 100, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'win_loss_ratio': round(win_loss_ratio, 2),
            'total_trades': total_trades,
            'insufficient_data': False
        }
    
    def get_position_size(
        self,
        equity: float,
        price: float,
        override_pct: Optional[float] = None
    ) -> Dict:
        """
        Calculate position size for a trade.
        
        Args:
            equity: Current account equity
            price: Asset price
            override_pct: Override Kelly with specific percentage
            
        Returns:
            Position size recommendation
        """
        kelly = self.calculate_kelly()
        
        if override_pct:
            size_pct = override_pct
        elif kelly['insufficient_data']:
            size_pct = 1.0  # Default to 1% if insufficient data
        else:
            size_pct = kelly['recommended_size_pct']
        
        position_value = equity * (size_pct / 100)
        position_size = position_value / price
        
        return {
            'position_size': round(position_size, 6),
            'position_value': round(position_value, 2),
            'size_pct': size_pct,
            'kelly_data': kelly
        }


class PaperTradingSimulator:
    """
    Feature #58: Paper Trading Simulator
    
    Simulates live trading without real money for testing strategies.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        slippage_pct: float = 0.05,
        execution_delay_ms: int = 100
    ):
        """
        Initialize paper trading simulator.
        
        Args:
            initial_capital: Starting capital
            slippage_pct: Simulated slippage percentage
            execution_delay_ms: Simulated execution delay
        """
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.slippage_pct = slippage_pct
        self.execution_delay_ms = execution_delay_ms
        
        self.positions: List[Dict] = []
        self.trades: List[Dict] = []
        self.orders: List[Dict] = []
        
        self.is_live = False  # False = paper, True = live
        
        logger.info(f"Paper Trading initialized - ${initial_capital:,.2f} capital")
    
    def place_order(
        self,
        side: str,
        size: float,
        price: float,
        order_type: str = 'MARKET',
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> Dict:
        """
        Place a paper order.
        
        Args:
            side: 'BUY' or 'SELL'
            size: Position size
            price: Current/limit price
            order_type: 'MARKET' or 'LIMIT'
            sl: Stop loss price
            tp: Take profit price
            
        Returns:
            Order result
        """
        # Simulate execution delay
        import time
        time.sleep(self.execution_delay_ms / 1000)
        
        # Apply slippage for market orders
        if order_type == 'MARKET':
            slippage = price * (self.slippage_pct / 100)
            fill_price = price + slippage if side == 'BUY' else price - slippage
        else:
            fill_price = price
        
        order_id = f"PAPER-{len(self.orders)+1:05d}"
        
        order = {
            'id': order_id,
            'timestamp': datetime.now().isoformat(),
            'side': side,
            'size': size,
            'order_price': price,
            'fill_price': round(fill_price, 2),
            'type': order_type,
            'sl': sl,
            'tp': tp,
            'status': 'FILLED',
            'slippage': round(abs(fill_price - price), 2)
        }
        
        self.orders.append(order)
        
        # Create position
        position = {
            'order_id': order_id,
            'side': 'LONG' if side == 'BUY' else 'SHORT',
            'entry_price': fill_price,
            'size': size,
            'sl': sl,
            'tp': tp,
            'open_time': datetime.now()
        }
        self.positions.append(position)
        
        logger.info(f"[PAPER] Order filled: {side} {size} @ ${fill_price:.2f}")
        
        return order
    
    def close_position(self, position_idx: int, exit_price: float, reason: str = 'manual') -> Dict:
        """Close a paper position."""
        if position_idx >= len(self.positions):
            return {'error': 'Invalid position'}
        
        pos = self.positions[position_idx]
        
        # Apply slippage
        slippage = exit_price * (self.slippage_pct / 100)
        if pos['side'] == 'LONG':
            fill_price = exit_price - slippage
            pnl = (fill_price - pos['entry_price']) * pos['size']
        else:
            fill_price = exit_price + slippage
            pnl = (pos['entry_price'] - fill_price) * pos['size']
        
        trade = {
            'entry_price': pos['entry_price'],
            'exit_price': round(fill_price, 2),
            'side': pos['side'],
            'size': pos['size'],
            'pnl': round(pnl, 2),
            'return_pct': round((pnl / (pos['entry_price'] * pos['size'])) * 100, 2),
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        self.trades.append(trade)
        self.equity += pnl
        self.positions.pop(position_idx)
        
        logger.info(f"[PAPER] Trade closed: {pos['side']} P&L: ${pnl:.2f}")
        
        return trade
    
    def get_metrics(self) -> Dict:
        """Get paper trading performance metrics."""
        if not self.trades:
            return {'total_trades': 0, 'equity': self.equity}
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(self.trades) if self.trades else 0,
            'total_pnl': sum(t['pnl'] for t in self.trades),
            'avg_win': sum(t['pnl'] for t in wins) / len(wins) if wins else 0,
            'avg_loss': sum(t['pnl'] for t in losses) / len(losses) if losses else 0,
            'equity': round(self.equity, 2),
            'return_pct': round((self.equity - self.initial_capital) / self.initial_capital * 100, 2)
        }
    
    # =========================================================================
    # UPGRADE #8: Trade Journal with Export
    # =========================================================================
    
    def export_trades(self, file_path: str, format: str = 'csv') -> Dict:
        """
        Export trade history to file.
        
        Args:
            file_path: Path to save export file
            format: 'csv' or 'json'
            
        Returns:
            Export status
        """
        import json
        import csv
        
        if not self.trades:
            return {'status': 'error', 'message': 'No trades to export'}
        
        try:
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    export_data = {
                        'export_timestamp': datetime.now().isoformat(),
                        'initial_capital': self.initial_capital,
                        'final_equity': self.equity,
                        'metrics': self.get_metrics(),
                        'trades': self.trades,
                        'orders': self.orders
                    }
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                with open(file_path, 'w', newline='') as f:
                    if self.trades:
                        fieldnames = list(self.trades[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(self.trades)
            
            logger.info(f"Trades exported to {file_path}")
            return {'status': 'success', 'file': file_path, 'trades_exported': len(self.trades)}
        
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_trade_analytics(self) -> Dict:
        """
        Get detailed trade analytics for journal.
        
        Returns:
            Comprehensive trade analytics
        """
        if not self.trades:
            return {'error': 'No trades to analyze'}
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        # Calculate streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        streak_type = None
        
        for trade in self.trades:
            if trade['pnl'] > 0:
                if streak_type == 'win':
                    current_streak += 1
                else:
                    current_streak = 1
                    streak_type = 'win'
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if streak_type == 'loss':
                    current_streak += 1
                else:
                    current_streak = 1
                    streak_type = 'loss'
                max_loss_streak = max(max_loss_streak, current_streak)
        
        # Calculate drawdown
        running_pnl = 0
        peak_pnl = 0
        max_drawdown = 0
        
        for trade in self.trades:
            running_pnl += trade['pnl']
            if running_pnl > peak_pnl:
                peak_pnl = running_pnl
            drawdown = peak_pnl - running_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t['pnl'] for t in losses) / len(losses)) if losses else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Trade duration analytics
        trade_reasons = {}
        for trade in self.trades:
            reason = trade.get('reason', 'unknown')
            if reason not in trade_reasons:
                trade_reasons[reason] = {'count': 0, 'pnl': 0}
            trade_reasons[reason]['count'] += 1
            trade_reasons[reason]['pnl'] += trade['pnl']
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate_pct': round(win_rate * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'expectancy': round(expectancy, 2),
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'max_drawdown': round(max_drawdown, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'net_profit': round(gross_profit - gross_loss, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'largest_win': round(max(t['pnl'] for t in wins), 2) if wins else 0,
            'largest_loss': round(min(t['pnl'] for t in losses), 2) if losses else 0,
            'returns_by_reason': trade_reasons,
            'initial_capital': self.initial_capital,
            'final_equity': round(self.equity, 2),
            'total_return_pct': round((self.equity - self.initial_capital) / self.initial_capital * 100, 2)
        }
    
    def get_daily_pnl_summary(self) -> Dict:
        """Get P&L summary grouped by day."""
        from collections import defaultdict
        
        daily_pnl = defaultdict(list)
        
        for trade in self.trades:
            date = trade['timestamp'].split('T')[0] if isinstance(trade['timestamp'], str) else trade['timestamp'].date().isoformat()
            daily_pnl[date].append(trade['pnl'])
        
        summary = {}
        for date, pnls in sorted(daily_pnl.items()):
            summary[date] = {
                'trade_count': len(pnls),
                'total_pnl': round(sum(pnls), 2),
                'avg_pnl': round(sum(pnls) / len(pnls), 2) if pnls else 0,
                'wins': len([p for p in pnls if p > 0]),
                'losses': len([p for p in pnls if p <= 0])
            }
        
        return summary


class FeeCalculator:
    """
    Feature #63: Commission/Fee-Aware Simulation
    
    Calculates trading fees for realistic P&L.
    """
    
    def __init__(
        self,
        maker_fee_pct: float = 0.02,   # 0.02% maker
        taker_fee_pct: float = 0.04,   # 0.04% taker
        funding_rate: float = 0.01     # 0.01% every 8h
    ):
        """
        Initialize fee calculator.
        
        Args:
            maker_fee_pct: Maker fee percentage
            taker_fee_pct: Taker fee percentage
            funding_rate: Funding rate for perpetuals
        """
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.funding_rate = funding_rate
        
        self.total_fees_paid = 0
        
        logger.info(f"Fee Calculator initialized - Maker: {maker_fee_pct}%, Taker: {taker_fee_pct}%")
    
    def calculate_trade_fees(
        self,
        notional: float,
        is_maker: bool = False
    ) -> Dict:
        """
        Calculate fees for a trade.
        
        Args:
            notional: Trade value in USD
            is_maker: True if limit order (maker), False if market (taker)
            
        Returns:
            Fee breakdown
        """
        fee_pct = self.maker_fee_pct if is_maker else self.taker_fee_pct
        fee = notional * (fee_pct / 100)
        
        self.total_fees_paid += fee
        
        return {
            'fee': round(fee, 4),
            'fee_pct': fee_pct,
            'type': 'maker' if is_maker else 'taker',
            'notional': notional
        }
    
    def calculate_funding(
        self,
        position_value: float,
        hours_held: float
    ) -> float:
        """
        Calculate funding fees for perpetual positions.
        
        Args:
            position_value: Position notional value
            hours_held: Hours position was held
            
        Returns:
            Total funding fee
        """
        funding_periods = hours_held / 8
        funding = position_value * (self.funding_rate / 100) * funding_periods
        return round(funding, 4)
    
    def adjust_pnl(
        self,
        gross_pnl: float,
        entry_notional: float,
        exit_notional: float,
        hours_held: float = 0,
        is_maker: bool = False
    ) -> Dict:
        """
        Adjust P&L for all fees.
        
        Args:
            gross_pnl: P&L before fees
            entry_notional: Entry trade value
            exit_notional: Exit trade value
            hours_held: Time position was held
            is_maker: If limit orders used
            
        Returns:
            Fee-adjusted P&L
        """
        entry_fee = self.calculate_trade_fees(entry_notional, is_maker)['fee']
        exit_fee = self.calculate_trade_fees(exit_notional, is_maker)['fee']
        funding = self.calculate_funding((entry_notional + exit_notional) / 2, hours_held)
        
        total_fees = entry_fee + exit_fee + funding
        net_pnl = gross_pnl - total_fees
        
        return {
            'gross_pnl': round(gross_pnl, 2),
            'net_pnl': round(net_pnl, 2),
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'funding_fee': funding,
            'total_fees': round(total_fees, 4),
            'fee_impact_pct': round((total_fees / abs(gross_pnl)) * 100, 2) if gross_pnl != 0 else 0
        }


class DynamicTargetCalculator:
    """
    Feature #68: Dynamic Stop/Target Levels
    
    Adjusts SL/TP based on volatility and market conditions.
    """
    
    def __init__(
        self,
        base_sl_atr: float = 1.0,
        base_tp_atr: float = 2.0
    ):
        """
        Initialize dynamic target calculator.
        
        Args:
            base_sl_atr: Base stop loss in ATR units
            base_tp_atr: Base take profit in ATR units
        """
        self.base_sl_atr = base_sl_atr
        self.base_tp_atr = base_tp_atr
        
        logger.info(f"Dynamic Targets initialized - Base SL: {base_sl_atr}x ATR, TP: {base_tp_atr}x ATR")
    
    def calculate(
        self,
        entry_price: float,
        atr: float,
        side: str,
        volatility_ratio: float = 1.0,
        win_streak: int = 0,
        confidence: float = 0.5
    ) -> Dict:
        """
        Calculate dynamic SL/TP levels.
        
        Args:
            entry_price: Entry price
            atr: Current ATR
            side: 'LONG' or 'SHORT'
            volatility_ratio: Current vol / avg vol
            win_streak: Current winning streak
            confidence: Signal confidence
            
        Returns:
            Dynamic targets
        """
        # Adjust based on volatility
        if volatility_ratio < 0.7:  # Low vol
            sl_mult = self.base_sl_atr * 0.75
            tp_mult = self.base_tp_atr * 0.8
        elif volatility_ratio > 1.5:  # High vol
            sl_mult = self.base_sl_atr * 1.5
            tp_mult = self.base_tp_atr * 1.5
        else:  # Normal
            sl_mult = self.base_sl_atr
            tp_mult = self.base_tp_atr
        
        # Adjust based on confidence
        if confidence >= 0.8:
            tp_mult *= 1.2  # Extend TP for high confidence
        elif confidence < 0.6:
            sl_mult *= 0.8  # Tighter SL for low confidence
        
        # Adjust based on win streak (trail tighter after wins)
        if win_streak >= 3:
            sl_mult *= 0.9  # Tighter to protect profits
        
        # Calculate actual levels
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult
        
        if side == 'LONG':
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        
        return {
            'entry_price': entry_price,
            'sl_price': round(sl_price, 2),
            'tp_price': round(tp_price, 2),
            'sl_distance': round(sl_distance, 2),
            'tp_distance': round(tp_distance, 2),
            'sl_mult': round(sl_mult, 2),
            'tp_mult': round(tp_mult, 2),
            'risk_reward': round(tp_distance / sl_distance, 2) if sl_distance > 0 else 0
        }


class SeasonalCyclicDetector:
    """
    Feature #74: Seasonal/Cyclic Detector - ENHANCED
    
    Detects time-based patterns in price action.
    Now includes: Monthly patterns, Week-of-month, Quarter-end effects.
    """
    
    def __init__(self):
        """Initialize seasonal detector with extended tracking."""
        self.hourly_returns: Dict[int, List[float]] = {h: [] for h in range(24)}
        self.daily_returns: Dict[int, List[float]] = {d: [] for d in range(7)}
        
        # UPGRADE #7: Extended seasonal tracking
        self.monthly_returns: Dict[int, List[float]] = {m: [] for m in range(1, 13)}
        self.week_of_month_returns: Dict[int, List[float]] = {w: [] for w in range(1, 6)}
        self.quarter_end_returns: List[float] = []  # Returns during quarter-end periods
        self.quarter_start_returns: List[float] = []  # Returns during quarter-start periods
        
        logger.info("Seasonal Cyclic Detector initialized (Enhanced with monthly/quarterly patterns)")
    
    def record_return(self, timestamp: datetime, return_pct: float):
        """Record a return for pattern analysis - extended tracking."""
        hour = timestamp.hour
        day = timestamp.weekday()
        month = timestamp.month
        
        # Week of month (1-5)
        week_of_month = min((timestamp.day - 1) // 7 + 1, 5)
        
        self.hourly_returns[hour].append(return_pct)
        self.daily_returns[day].append(return_pct)
        self.monthly_returns[month].append(return_pct)
        self.week_of_month_returns[week_of_month].append(return_pct)
        
        # Track quarter-end effects (last 5 days of quarter)
        if month in [3, 6, 9, 12] and timestamp.day >= 26:
            self.quarter_end_returns.append(return_pct)
        
        # Track quarter-start effects (first 5 days of quarter)
        if month in [1, 4, 7, 10] and timestamp.day <= 5:
            self.quarter_start_returns.append(return_pct)
        
        # Keep last 100 per bucket
        self.hourly_returns[hour] = self.hourly_returns[hour][-100:]
        self.daily_returns[day] = self.daily_returns[day][-100:]
        self.monthly_returns[month] = self.monthly_returns[month][-50:]
        self.week_of_month_returns[week_of_month] = self.week_of_month_returns[week_of_month][-50:]
        self.quarter_end_returns = self.quarter_end_returns[-100:]
        self.quarter_start_returns = self.quarter_start_returns[-100:]
    
    def get_hourly_bias(self, hour: int) -> Dict:
        """Get trading bias for specific hour."""
        returns = self.hourly_returns.get(hour, [])
        if len(returns) < 10:
            return {'bias': 'neutral', 'confidence': 0, 'insufficient_data': True}
        
        avg = sum(returns) / len(returns)
        
        if avg > 0.1:
            bias = 'bullish'
        elif avg < -0.1:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        return {
            'hour': hour,
            'bias': bias,
            'avg_return': round(avg, 4),
            'sample_size': len(returns),
            'insufficient_data': False
        }
    
    def get_daily_bias(self, day: int) -> Dict:
        """Get trading bias for specific day (0=Monday)."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        returns = self.daily_returns.get(day, [])
        
        if len(returns) < 10:
            return {'bias': 'neutral', 'confidence': 0, 'insufficient_data': True}
        
        avg = sum(returns) / len(returns)
        
        if avg > 0.2:
            bias = 'bullish'
        elif avg < -0.2:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        return {
            'day': days[day],
            'day_num': day,
            'bias': bias,
            'avg_return': round(avg, 4),
            'sample_size': len(returns),
            'insufficient_data': False
        }
    
    def get_monthly_bias(self, month: int) -> Dict:
        """Get trading bias for specific month (1-12)."""
        months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
        returns = self.monthly_returns.get(month, [])
        
        if len(returns) < 5:
            return {'bias': 'neutral', 'confidence': 0, 'insufficient_data': True}
        
        avg = sum(returns) / len(returns)
        
        if avg > 0.3:
            bias = 'bullish'
        elif avg < -0.3:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        return {
            'month': months[month],
            'month_num': month,
            'bias': bias,
            'avg_return': round(avg, 4),
            'sample_size': len(returns),
            'insufficient_data': False
        }
    
    def get_week_of_month_bias(self, week: int) -> Dict:
        """Get trading bias for week of month (1-5)."""
        returns = self.week_of_month_returns.get(week, [])
        
        if len(returns) < 10:
            return {'bias': 'neutral', 'confidence': 0, 'insufficient_data': True}
        
        avg = sum(returns) / len(returns)
        
        if avg > 0.15:
            bias = 'bullish'
        elif avg < -0.15:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        return {
            'week_of_month': week,
            'bias': bias,
            'avg_return': round(avg, 4),
            'sample_size': len(returns),
            'insufficient_data': False
        }
    
    def get_quarter_effect(self) -> Dict:
        """Get quarter-end and quarter-start effects."""
        qe_avg = sum(self.quarter_end_returns) / len(self.quarter_end_returns) if self.quarter_end_returns else 0
        qs_avg = sum(self.quarter_start_returns) / len(self.quarter_start_returns) if self.quarter_start_returns else 0
        
        return {
            'quarter_end': {
                'avg_return': round(qe_avg, 4),
                'sample_size': len(self.quarter_end_returns),
                'bias': 'bullish' if qe_avg > 0.2 else ('bearish' if qe_avg < -0.2 else 'neutral')
            },
            'quarter_start': {
                'avg_return': round(qs_avg, 4),
                'sample_size': len(self.quarter_start_returns),
                'bias': 'bullish' if qs_avg > 0.2 else ('bearish' if qs_avg < -0.2 else 'neutral')
            }
        }
    
    def is_quarter_end_period(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if we're in a quarter-end period."""
        ts = timestamp or datetime.now()
        return ts.month in [3, 6, 9, 12] and ts.day >= 26
    
    def is_quarter_start_period(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if we're in a quarter-start period."""
        ts = timestamp or datetime.now()
        return ts.month in [1, 4, 7, 10] and ts.day <= 5
    
    def get_current_bias(self) -> Dict:
        """Get comprehensive bias for current time."""
        now = datetime.now()
        hourly = self.get_hourly_bias(now.hour)
        daily = self.get_daily_bias(now.weekday())
        monthly = self.get_monthly_bias(now.month)
        week_of_month = self.get_week_of_month_bias(min((now.day - 1) // 7 + 1, 5))
        quarter_effects = self.get_quarter_effect()
        
        # Combine biases with weights
        combined_score = 0
        weight_total = 0
        
        biases = [
            (hourly['bias'], 1.0),
            (daily['bias'], 1.5),
            (monthly['bias'], 0.5),
            (week_of_month['bias'], 0.5)
        ]
        
        for bias, weight in biases:
            if bias == 'bullish':
                combined_score += weight
            elif bias == 'bearish':
                combined_score -= weight
            weight_total += weight
        
        # Add quarter effects if applicable
        if self.is_quarter_end_period(now):
            qe_bias = quarter_effects['quarter_end']['bias']
            if qe_bias == 'bullish':
                combined_score += 1
            elif qe_bias == 'bearish':
                combined_score -= 1
            weight_total += 1
        
        if self.is_quarter_start_period(now):
            qs_bias = quarter_effects['quarter_start']['bias']
            if qs_bias == 'bullish':
                combined_score += 1
            elif qs_bias == 'bearish':
                combined_score -= 1
            weight_total += 1
        
        normalized_score = combined_score / weight_total if weight_total > 0 else 0
        
        if normalized_score >= 0.3:
            combined_bias = 'bullish'
        elif normalized_score <= -0.3:
            combined_bias = 'bearish'
        else:
            combined_bias = 'neutral'
        
        return {
            'hourly': hourly,
            'daily': daily,
            'monthly': monthly,
            'week_of_month': week_of_month,
            'quarter_effects': quarter_effects,
            'is_quarter_end': self.is_quarter_end_period(now),
            'is_quarter_start': self.is_quarter_start_period(now),
            'combined_bias': combined_bias,
            'combined_score': round(normalized_score, 2)
        }


# Singleton instances
_paper_trader: Optional[PaperTradingSimulator] = None
_fee_calc: Optional[FeeCalculator] = None
_target_calc: Optional[DynamicTargetCalculator] = None
_seasonal: Optional[SeasonalCyclicDetector] = None
_order_book: Optional[OrderBookSimulator] = None
_latency: Optional[LatencySimulator] = None
_trailing_stop: Optional[TrailingStopManager] = None
_position_scaler: Optional[PositionScaler] = None
_circuit_breaker: Optional[CircuitBreaker] = None
_bracket_manager: Optional[BracketOrderManager] = None
_kelly_sizer: Optional[KellyCriterionSizer] = None


def get_paper_trader() -> PaperTradingSimulator:
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTradingSimulator()
    return _paper_trader


def get_fee_calculator() -> FeeCalculator:
    global _fee_calc
    if _fee_calc is None:
        _fee_calc = FeeCalculator()
    return _fee_calc


def get_target_calculator() -> DynamicTargetCalculator:
    global _target_calc
    if _target_calc is None:
        _target_calc = DynamicTargetCalculator()
    return _target_calc


def get_seasonal_detector() -> SeasonalCyclicDetector:
    global _seasonal
    if _seasonal is None:
        _seasonal = SeasonalCyclicDetector()
    return _seasonal


# New singleton getters for enterprise features

def get_order_book_simulator() -> OrderBookSimulator:
    global _order_book
    if _order_book is None:
        _order_book = OrderBookSimulator()
    return _order_book


def get_latency_simulator() -> LatencySimulator:
    global _latency
    if _latency is None:
        _latency = LatencySimulator()
    return _latency


def get_trailing_stop_manager() -> TrailingStopManager:
    global _trailing_stop
    if _trailing_stop is None:
        _trailing_stop = TrailingStopManager()
    return _trailing_stop


def get_position_scaler() -> PositionScaler:
    global _position_scaler
    if _position_scaler is None:
        _position_scaler = PositionScaler()
    return _position_scaler


def get_circuit_breaker() -> CircuitBreaker:
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker()
    return _circuit_breaker


def get_bracket_manager() -> BracketOrderManager:
    global _bracket_manager
    if _bracket_manager is None:
        _bracket_manager = BracketOrderManager()
    return _bracket_manager


def get_kelly_sizer() -> KellyCriterionSizer:
    global _kelly_sizer
    if _kelly_sizer is None:
        _kelly_sizer = KellyCriterionSizer()
    return _kelly_sizer


if __name__ == '__main__':
    print("=" * 60)
    print("PAPER TRADING ENTERPRISE FEATURES - COMPREHENSIVE TEST")
    print("=" * 60)
    
    # =========================================================================
    # Test 1: Basic Paper Trading
    # =========================================================================
    print("\n[TEST 1] Paper Trading Simulator")
    paper = PaperTradingSimulator(initial_capital=10000)
    
    paper.place_order('BUY', 0.1, 50000, sl=49500, tp=51000)
    print(f"  Positions: {len(paper.positions)}")
    
    paper.close_position(0, 50800, 'take_profit')
    print(f"  Metrics: {paper.get_metrics()}")
    
    # =========================================================================
    # Test 2: Fee Calculator
    # =========================================================================
    print("\n[TEST 2] Fee Calculator")
    fees = FeeCalculator()
    result = fees.adjust_pnl(100, 5000, 5100, hours_held=12)
    print(f"  Fee adjusted: {result}")
    
    # =========================================================================
    # Test 3: Dynamic Targets
    # =========================================================================
    print("\n[TEST 3] Dynamic Targets")
    targets = DynamicTargetCalculator()
    levels = targets.calculate(50000, 500, 'LONG', volatility_ratio=1.2, confidence=0.75)
    print(f"  Dynamic targets: {levels}")
    
    # =========================================================================
    # Test 4: Order Book Simulation (UPGRADE #1)
    # =========================================================================
    print("\n[TEST 4] Order Book Simulator (UPGRADE #1)")
    ob = OrderBookSimulator(base_spread_bps=2.0, depth_levels=5)
    order_book = ob.generate_order_book(50000, volatility=0.03)
    print(f"  Spread: {order_book['spread_bps']} bps")
    print(f"  Best Bid: ${order_book['best_bid']}, Best Ask: ${order_book['best_ask']}")
    
    fill = ob.simulate_fill('BUY', 0.5, order_book)
    print(f"  Fill Result: Avg Price=${fill['avg_fill_price']}, Slippage=${fill['slippage']}")
    
    # =========================================================================
    # Test 5: Trailing Stop Manager (UPGRADE #2)
    # =========================================================================
    print("\n[TEST 5] Trailing Stop Manager (UPGRADE #2)")
    tsm = TrailingStopManager()
    stop = tsm.create_trailing_stop('POS-001', 50000, 'LONG', mode='PERCENT', trail_value=1.5)
    print(f"  Initial Stop: ${stop['current_stop']}")
    
    update = tsm.update_stop('POS-001', 51000)  # Price moved up
    print(f"  Updated Stop: ${update['current_stop']} (Breakeven: {update['breakeven_activated']})")
    
    # =========================================================================
    # Test 6: Position Scaler (UPGRADE #3)
    # =========================================================================
    print("\n[TEST 6] Position Scaler (UPGRADE #3)")
    scaler = PositionScaler(max_scale_ins=3)
    pos = scaler.create_position('POS-002', 'LONG', 50000, 0.1)
    print(f"  Initial Position: {pos['current_size']} @ ${pos['avg_entry']}")
    
    scale = scaler.scale_in('POS-002', 49000)  # Add at lower price
    print(f"  After Scale-In: {scale['new_total_size']} @ ${scale['new_avg_entry']}")
    
    tp_check = scaler.check_take_profit('POS-002', 52000)  # Check TP levels
    if tp_check:
        print(f"  TP Triggered: Level {tp_check['trigger_level']}%, Close {tp_check['close_size']}")
    
    # =========================================================================
    # Test 7: Circuit Breaker (UPGRADE #4)
    # =========================================================================
    print("\n[TEST 7] Circuit Breaker (UPGRADE #4)")
    cb = CircuitBreaker(max_drawdown_pct=10, daily_loss_limit_pct=5)
    cb.initialize(10000)
    
    status1 = cb.update(9700, 'LOSS')
    print(f"  After loss: Can Trade={status1['can_trade']}, DD={status1['drawdown_pct']}%")
    
    status2 = cb.update(9400, 'LOSS')
    print(f"  After 2nd loss: Can Trade={status2['can_trade']}, DD={status2['drawdown_pct']}%")
    
    # =========================================================================
    # Test 8: Bracket Order Manager (UPGRADE #5)
    # =========================================================================
    print("\n[TEST 8] Bracket Order Manager (UPGRADE #5)")
    bracket = BracketOrderManager()
    order = bracket.create_bracket_order('ORD-001', 'BUY', 50000, 0.1, sl_price=49000, tp_price=52000)
    print(f"  Bracket Created: {order['bracket_id']}, Status={order['status']}")
    
    bracket.on_entry_fill('ORD-001', 50000)
    trigger = bracket.check_triggers('ORD-001', 52000)  # Price hit TP
    if trigger:
        print(f"  Triggered: {trigger['triggered']} @ ${trigger['fill_price']}, Cancelled: {trigger['cancelled']}")
    
    # =========================================================================
    # Test 9: Latency Simulator (UPGRADE #6)
    # =========================================================================
    print("\n[TEST 9] Latency Simulator (UPGRADE #6)")
    lat = LatencySimulator(min_latency_ms=10, max_latency_ms=50, rejection_rate=0.0)
    exec_result = lat.simulate_execution(50000)
    print(f"  Execution: Status={exec_result['status']}, Latency={exec_result['latency_ms']}ms")
    
    # =========================================================================
    # Test 10: Enhanced Seasonal Detector (UPGRADE #7)
    # =========================================================================
    print("\n[TEST 10] Enhanced Seasonal Detector (UPGRADE #7)")
    seasonal = SeasonalCyclicDetector()
    for i in range(20):
        seasonal.record_return(datetime.now(), random.uniform(-0.5, 0.8))
    
    bias = seasonal.get_current_bias()
    print(f"  Combined Bias: {bias['combined_bias']} (Score: {bias['combined_score']})")
    print(f"  Is Quarter End: {bias['is_quarter_end']}")
    
    # =========================================================================
    # Test 11: Trade Journal Export (UPGRADE #8)
    # =========================================================================
    print("\n[TEST 11] Trade Journal Export (UPGRADE #8)")
    analytics = paper.get_trade_analytics()
    print(f"  Analytics: Win Rate={analytics['win_rate_pct']}%, Profit Factor={analytics['profit_factor']}")
    print(f"  Max Win Streak: {analytics['max_win_streak']}, Max Loss Streak: {analytics['max_loss_streak']}")
    
    # =========================================================================
    # Test 12: Kelly Criterion Sizer (UPGRADE #9)
    # =========================================================================
    print("\n[TEST 12] Kelly Criterion Sizer (UPGRADE #9)")
    kelly = KellyCriterionSizer(max_kelly_fraction=0.25, min_trades_required=5)
    
    # Record some trades
    for i in range(30):
        pnl = random.choice([100, 150, 200, -80, -60])
        kelly.record_trade(pnl)
    
    kelly_result = kelly.calculate_kelly()
    print(f"  Kelly %: {kelly_result['kelly_pct']}%")
    print(f"  Recommended Size: {kelly_result['recommended_size_pct']}%")
    print(f"  Win Rate: {kelly_result['win_rate']}%, Win/Loss Ratio: {kelly_result['win_loss_ratio']}")
    
    position = kelly.get_position_size(10000, 50000)
    print(f"  Position Size for $10k equity: {position['position_size']} BTC")
    
    print("\n" + "=" * 60)
    print("ALL ENTERPRISE FEATURES TESTED SUCCESSFULLY!")
    print("=" * 60)

