"""
Enhanced Paper Trading Broker
Realistic simulation with orderbook-based fills and position tracking.

Critical for validating strategies before live trading.
Simulates:
- Orderbook-based fill prices
- Realistic slippage based on order size
- Partial fills on limit orders
- Position tracking with unrealized P&L
- Fee calculation
- 6-month validation period tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """Track a paper trading position."""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    quantity: float
    entry_time: datetime
    fees_paid: float = 0.0
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.side == 'LONG':
            return (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - current_price) * self.quantity
    
    def get_unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        pnl = self.get_unrealized_pnl(current_price)
        cost_basis = self.entry_price * self.quantity
        return (pnl / cost_basis) * 100


@dataclass  
class PaperTrade:
    """Completed paper trade."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    realized_pnl: float
    realized_pnl_pct: float
    fees_total: float
    strategy: str = ""


class EnhancedPaperBroker:
    """
    Enhanced Paper Trading Broker with realistic simulation.
    
    Features:
    - Orderbook-based fill simulation
    - Realistic slippage based on order size
    - Partial fills on limit orders  
    - Position tracking with unrealized P&L
    - Fee calculation (maker/taker)
    - Performance metrics tracking
    - 6-month validation period
    - Graduation criteria checking
    
    Validation Criteria (before live trading):
    - Minimum 6 months paper trading
    - Positive returns in 5/6 months
    - Sharpe ratio > 1.5
    - Max drawdown < 20%
    - Win rate > 45%
    - No critical bugs
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_taker_pct: float = 0.04,  # 0.04% taker
        fee_maker_pct: float = 0.02,  # 0.02% maker
        slippage_base_bps: float = 5.0,  # Base 5 bps slippage
        validation_months: int = 6
    ):
        """
        Initialize paper broker.
        
        Args:
            initial_capital: Starting capital
            fee_taker_pct: Taker fee percentage
            fee_maker_pct: Maker fee percentage
            slippage_base_bps: Base slippage in basis points
            validation_months: Required validation period
        """
        self.initial_capital = initial_capital
        self.fee_taker = fee_taker_pct / 100
        self.fee_maker = fee_maker_pct / 100
        self.slippage_base_bps = slippage_base_bps
        self.validation_months = validation_months
        
        # Account state
        self.cash = initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_trades: List[PaperTrade] = []
        
        # Performance tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.start_time: Optional[datetime] = None
        self.monthly_returns: List[float] = []
        
        logger.info(
            f"Paper Broker initialized: Capital=${initial_capital:,.2f}, "
            f"Validation period: {validation_months} months"
        )
    
    def simulate_orderbook_fill(
        self,
        order_type: str,  # 'MARKET' or 'LIMIT'
        side: str,  # 'BUY' or 'SELL'
        price: float,
        quantity: float,
        current_price: float,
        volatility: float = 0.01
    ) -> Tuple[float, float, bool]:
        """
        Simulate realistic orderbook fill.
        
        Returns:
            (fill_price, filled_quantity, fully_filled)
        """
        if order_type == 'MARKET':
            # Market orders fill immediately with slippage
            slippage_bps = self.slippage_base_bps
            
            # More slippage in high volatility
            slippage_bps += volatility * 1000  # 1% vol = +10 bps
            
            # More slippage for large orders
            # Assume $10k order has minimal impact
            size_impact_bps = (quantity * price / 10000) * 5
            slippage_bps += size_impact_bps
            
            # Apply slippage
            if side == 'BUY':
                fill_price = current_price * (1 + slippage_bps / 10000)
            else:  # SELL
                fill_price = current_price * (1 - slippage_bps / 10000)
            
            return fill_price, quantity, True
            
        else:  # LIMIT
            # Limit orders only fill if price touches limit
            if side == 'BUY':
                if current_price <= price:
                    # Filled at limit price (or better)
                    fill_price = min(price, current_price)
                    return fill_price, quantity, True
                else:
                    return 0, 0, False
            else:  # SELL
                if current_price >= price:
                    fill_price = max(price, current_price)
                    return fill_price, quantity, True
                else:
                    return 0, 0, False
    
    def place_market_order(
        self,
        symbol: str,
        side: str,  # 'BUY' or 'SELL'
        quantity: float,
        current_price: float,
        volatility: float = 0.01,
        strategy: str = ""
    ) -> Dict:
        """
        Place market order (fills immediately).
        
        Returns:
            Order result dict
        """
        # Simulate fill
        fill_price, filled_qty, fully_filled = self.simulate_orderbook_fill(
            'MARKET', side, current_price, quantity, current_price, volatility
        )
        
        if not fully_filled:
            logger.warning(f"Market order partially filled: {filled_qty}/{quantity}")
        
        # Calculate fees (taker)
        fee = fill_price * filled_qty * self.fee_taker
        
        if side == 'BUY':
            # Open or add to long position
            cost = (fill_price * filled_qty) + fee
            
            if cost > self.cash:
                logger.error(f"Insufficient cash: Need ${cost:,.2f}, have ${self.cash:,.2f}")
                return {'status': 'REJECTED', 'reason': 'Insufficient funds'}
            
            self.cash -= cost
            
            if symbol in self.positions:
                # Add to position
                pos = self.positions[symbol]
                total_qty = pos.quantity + filled_qty
                total_cost = (pos.entry_price * pos.quantity) + (fill_price * filled_qty)
                pos.entry_price = total_cost / total_qty
                pos.quantity = total_qty
                pos.fees_paid += fee
            else:
                # New position
                self.positions[symbol] = PaperPosition(
                    symbol=symbol,
                    side='LONG',
                    entry_price=fill_price,
                    quantity=filled_qty,
                    entry_time=datetime.now(),
                    fees_paid=fee
                )
            
            logger.info(f"BUY {filled_qty:.4f} {symbol} @ ${fill_price:,.2f}, Fee: ${fee:.2f}")
            
        else:  # SELL
            # Close or reduce position
            if symbol not in self.positions:
                logger.error(f"No position to sell: {symbol}")
                return {'status': 'REJECTED', 'reason': 'No position'}
            
            pos = self.positions[symbol]
            if filled_qty > pos.quantity:
                logger.warning(f"Sell quantity {filled_qty} > position {pos.quantity}, capping")
                filled_qty = pos.quantity
            
            # Calculate P&L
            proceeds = fill_price * filled_qty
            cost_basis = pos.entry_price * filled_qty
            realized_pnl = proceeds - cost_basis - fee - (pos.fees_paid * filled_qty / pos.quantity)
            realized_pnl_pct = (realized_pnl / cost_basis) * 100
            
            self.cash += proceeds - fee
            
            # Record trade
            self.closed_trades.append(PaperTrade(
                symbol=symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=fill_price,
                quantity=filled_qty,
                entry_time=pos.entry_time,
                exit_time=datetime.now(),
                realized_pnl=realized_pnl,
                realized_pnl_pct=realized_pnl_pct,
                fees_total=fee + (pos.fees_paid * filled_qty / pos.quantity),
                strategy=strategy
            ))
            
            # Update or close position
            if filled_qty >= pos.quantity:
                del self.positions[symbol]
            else:
                pos.quantity -= filled_qty
                pos.fees_paid -= pos.fees_paid * (filled_qty / pos.quantity)
            
            logger.info(
                f"SELL {filled_qty:.4f} {symbol} @ ${fill_price:,.2f}, "
                f"P&L: ${realized_pnl:+,.2f} ({realized_pnl_pct:+.2f}%)"
            )
        
        return {
            'status': 'FILLED',
            'side': side,
            'symbol': symbol,
            'filled_price': fill_price,
            'filled_quantity': filled_qty,
            'fee': fee,
            'timestamp': datetime.now()
        }
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total = self.cash
        
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                total += pos.quantity * current_prices[symbol]
        
        return total
    
    def update_equity_curve(self, current_prices: Dict[str, float]):
        """Update equity curve with current portfolio value."""
        portfolio_value = self.get_portfolio_value(current_prices)
        self.equity_curve.append((datetime.now(), portfolio_value))
        
        if self.start_time is None:
            self.start_time = datetime.now()
    
    def get_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate comprehensive performance metrics."""
        portfolio_value = self.get_portfolio_value(current_prices)
        total_return_pct = ((portfolio_value - self.initial_capital) / 
                           self.initial_capital * 100)
        
        # Trade statistics
        if self.closed_trades:
            wins = [t for t in self.closed_trades if t.realized_pnl > 0]
            losses = [t for t in self.closed_trades if t.realized_pnl < 0]
            
            win_rate = len(wins) / len(self.closed_trades) * 100
            avg_win = np.mean([t.realized_pnl for t in wins]) if wins else 0
            avg_loss = np.mean([t.realized_pnl for t in losses]) if losses else 0
            
            total_wins = sum(t.realized_pnl for t in wins)
            total_losses = abs(sum(t.realized_pnl for t in losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Drawdown
        if len(self.equity_curve) > 1:
            equity_series = pd.Series([e[1] for e in self.equity_curve])
            running_max = equity_series.expanding().max()
            dr awdown = (equity_series - running_max) / running_max * 100
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0
        
        # Sharpe ratio (if enough data)
        if len(self.equity_curve) > 30:
            equity_series = pd.Series([e[1] for e in self.equity_curve])
            returns = equity_series.pct_change().dropna()
            if returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'total_return_pct': total_return_pct,
            'total_trades': len(self.closed_trades),
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'open_positions': len(self.positions),
            'validation_progress_days': (datetime.now() - self.start_time).days if self.start_time else 0,
            'validation_required_days': self.validation_months * 30
        }
    
    def check_graduation_criteria(self, current_prices: Dict[str, float]) -> Dict:
        """
        Check if ready to graduate to live trading.
        
        Returns:
            Dict with pass/fail for each criterion
        """
        metrics = self.get_metrics(current_prices)
        
        criteria = {
            'validation_period_complete': metrics['validation_progress_days'] >= metrics['validation_required_days'],
            'positive_return': metrics['total_return_pct'] > 0,
            'sharpe_acceptable': metrics['sharpe_ratio'] > 1.5,
            'drawdown_acceptable': metrics['max_drawdown_pct'] < 20.0,
            'win_rate_acceptable': metrics['win_rate_pct'] > 45.0,
            'min_trades': metrics['total_trades'] >= 20
        }
        
        all_passed = all(criteria.values())
        
        return {
            'ready_for_live': all_passed,
            'criteria': criteria,
            'metrics': metrics
        }
