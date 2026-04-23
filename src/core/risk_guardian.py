"""
Global Risk Guardian - The "Kill Switch" Layer

Every order MUST pass through this guardian before execution.
Prevents catastrophic losses from bugs, fat fingers, or market crashes.

Protection Layers:
1. Per-Order Limits (max size, max value)
2. Per-Strategy Limits (allocation, drawdown)
3. Portfolio Limits (daily loss, concentration)
4. Circuit Breakers (halt trading on errors)
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimits:
    """Configuration for risk limits."""
    # Per-Order Limits
    max_order_size_btc: float = 1.0
    max_order_value_usd: float = 50000.0
    
    # Per-Strategy Limits
    max_strategy_allocation_pct: float = 25.0  # Max 25% of portfolio per strategy
    max_strategy_drawdown_pct: float = 20.0  # Stop strategy if down 20%
    max_open_orders_per_strategy: int = 10
    
    # Portfolio Limits
    max_daily_loss_usd: float = 500.0
    max_daily_loss_pct: float = 5.0
    max_weekly_loss_usd: float = 1500.0
    max_weekly_loss_pct: float = 15.0
    max_position_concentration_pct: float = 40.0  # Max 40% in any single asset
    max_total_open_orders: int = 50
    max_concurrent_trades: int = 5  # Max open positions at same time
    risk_per_trade_pct: float = 2.0  # Max 2% of portfolio per trade
    
    # Circuit Breakers
    max_consecutive_errors: int = 5
    error_cooldown_minutes: int = 15
    max_orders_per_minute: int = 30
    
    # Emergency
    emergency_stop_enabled: bool = True


@dataclass
class RiskState:
    """Current risk state tracking."""
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    consecutive_errors: int = 0
    last_error_time: Optional[datetime] = None
    orders_this_minute: int = 0
    last_minute_reset: datetime = None
    emergency_stop_active: bool = False
    
    def reset_daily(self):
        self.daily_pnl = 0.0
    
    def reset_weekly(self):
        self.weekly_pnl = 0.0


class RiskGuardian:
    """
    Global risk management guardian.
    
    Usage:
        guardian = RiskGuardian(portfolio_value=100000)
        
        # Check before every order
        approved, reason = guardian.approve_order(order_intent)
        if not approved:
            logger.error(f"Order blocked: {reason}")
            return
        
        # After order execution
        guardian.record_trade(pnl=+150.0)
        
        # On errors
        guardian.record_error("Connection timeout")
    """
    
    def __init__(
        self,
        portfolio_value: float = 10000.0,
        limits: RiskLimits = None,
        on_emergency_stop: callable = None
    ):
        self.portfolio_value = portfolio_value
        self.limits = limits or RiskLimits()
        self.state = RiskState(last_minute_reset=datetime.now())
        self.on_emergency_stop = on_emergency_stop
        
        # Strategy tracking
        self.strategy_allocations: Dict[str, float] = {}
        self.strategy_pnls: Dict[str, float] = {}
        self.strategy_open_orders: Dict[str, int] = {}
        
        # Position tracking
        self.positions: Dict[str, float] = {}  # symbol -> value in USD
        self._open_positions: Dict[str, float] = {}  # symbol -> quantity (for concurrent check)
        
        # Order tracking
        self.recent_orders: List[datetime] = []
        
        self._lock = threading.Lock()
        
        logger.info(f"RiskGuardian initialized with portfolio value ${portfolio_value:,.2f}")
    
    def approve_order(self, order_intent) -> Tuple[bool, str]:
        """
        Check if an order should be approved.
        
        Returns:
            (approved, reason)
        """
        with self._lock:
            # Emergency stop check
            if self.state.emergency_stop_active:
                return False, "Emergency stop is active"
            
            # Circuit breaker check
            if self.state.consecutive_errors >= self.limits.max_consecutive_errors:
                if self.state.last_error_time:
                    cooldown_end = self.state.last_error_time + timedelta(minutes=self.limits.error_cooldown_minutes)
                    if datetime.now() < cooldown_end:
                        return False, f"Circuit breaker active, cooldown until {cooldown_end}"
                    else:
                        self.state.consecutive_errors = 0
            
            # Rate limit check
            self._update_rate_limit()
            if self.state.orders_this_minute >= self.limits.max_orders_per_minute:
                return False, f"Rate limit exceeded ({self.limits.max_orders_per_minute}/min)"
            
            # Order size check
            order_value = order_intent.quantity * (order_intent.price or self._get_estimated_price(order_intent.symbol))
            
            if order_intent.quantity > self.limits.max_order_size_btc:
                return False, f"Order size {order_intent.quantity} exceeds max {self.limits.max_order_size_btc}"
            
            if order_value > self.limits.max_order_value_usd:
                return False, f"Order value ${order_value:,.0f} exceeds max ${self.limits.max_order_value_usd:,.0f}"
            
            # Per-trade risk check
            max_risk_value = self.portfolio_value * (self.limits.risk_per_trade_pct / 100)
            if order_value > max_risk_value:
                return False, (
                    f"Order value ${order_value:,.0f} exceeds {self.limits.risk_per_trade_pct}% "
                    f"per-trade risk limit (${max_risk_value:,.0f})"
                )
            
            # Daily loss check
            if abs(self.state.daily_pnl) >= self.limits.max_daily_loss_usd:
                return False, f"Daily loss limit reached (${abs(self.state.daily_pnl):,.0f})"
            
            daily_loss_pct = abs(self.state.daily_pnl) / self.portfolio_value * 100
            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                return False, f"Daily loss {daily_loss_pct:.1f}% exceeds max {self.limits.max_daily_loss_pct}%"
            
            # Weekly loss check
            if abs(self.state.weekly_pnl) >= self.limits.max_weekly_loss_usd:
                return False, f"Weekly loss limit reached (${abs(self.state.weekly_pnl):,.0f})"
            
            weekly_loss_pct = abs(self.state.weekly_pnl) / self.portfolio_value * 100
            if weekly_loss_pct >= self.limits.max_weekly_loss_pct:
                return False, f"Weekly loss {weekly_loss_pct:.1f}% exceeds max {self.limits.max_weekly_loss_pct}%"
            
            # Max concurrent trades check
            active_positions = len([q for q in self._open_positions.values() if abs(q) > 0])
            if active_positions >= self.limits.max_concurrent_trades:
                return False, (
                    f"Max concurrent trades reached ({active_positions}/{self.limits.max_concurrent_trades})"
                )
            
            # Strategy allocation check
            strategy_id = order_intent.strategy_id
            if strategy_id:
                current_allocation = self.strategy_allocations.get(strategy_id, 0)
                new_allocation = current_allocation + order_value
                allocation_pct = new_allocation / self.portfolio_value * 100
                
                if allocation_pct > self.limits.max_strategy_allocation_pct:
                    return False, f"Strategy allocation {allocation_pct:.1f}% exceeds max {self.limits.max_strategy_allocation_pct}%"
                
                # Strategy drawdown check
                strategy_pnl = self.strategy_pnls.get(strategy_id, 0)
                if current_allocation > 0:
                    strategy_drawdown_pct = abs(min(0, strategy_pnl)) / current_allocation * 100
                    if strategy_drawdown_pct >= self.limits.max_strategy_drawdown_pct:
                        return False, f"Strategy drawdown {strategy_drawdown_pct:.1f}% exceeds max {self.limits.max_strategy_drawdown_pct}%"
            
            # Position concentration check
            symbol = order_intent.symbol.split("/")[0] if "/" in order_intent.symbol else order_intent.symbol.replace("USDT", "")
            current_position = self.positions.get(symbol, 0)
            if order_intent.side.value == "buy":
                new_position = current_position + order_value
            else:
                new_position = current_position - order_value
            
            concentration_pct = abs(new_position) / self.portfolio_value * 100
            if concentration_pct > self.limits.max_position_concentration_pct:
                return False, f"Position concentration {concentration_pct:.1f}% exceeds max {self.limits.max_position_concentration_pct}%"
            
            # All checks passed
            self.state.orders_this_minute += 1
            self.recent_orders.append(datetime.now())
            
            return True, "Approved"
    
    def _update_rate_limit(self):
        """Update rate limit counter."""
        now = datetime.now()
        if self.state.last_minute_reset is None or (now - self.state.last_minute_reset).total_seconds() >= 60:
            self.state.orders_this_minute = 0
            self.state.last_minute_reset = now
    
    def _get_estimated_price(self, symbol: str) -> float:
        """Get estimated price for a symbol."""
        # Fallback prices (should be replaced with real prices)
        prices = {"BTC/USDT": 65000, "ETH/USDT": 3500, "BTC": 65000, "ETH": 3500}
        return prices.get(symbol, 65000)
    
    def record_trade(self, pnl: float, strategy_id: str = None):
        """Record a completed trade's P&L."""
        with self._lock:
            self.state.daily_pnl += pnl
            self.state.weekly_pnl += pnl
            
            if strategy_id:
                self.strategy_pnls[strategy_id] = self.strategy_pnls.get(strategy_id, 0) + pnl
            
            # Reset consecutive errors on successful trade
            self.state.consecutive_errors = 0
            
            logger.debug(f"Recorded trade P&L: ${pnl:+,.2f}, Daily total: ${self.state.daily_pnl:+,.2f}")
    
    def record_error(self, error_message: str):
        """Record an error for circuit breaker tracking."""
        with self._lock:
            self.state.consecutive_errors += 1
            self.state.last_error_time = datetime.now()
            
            logger.warning(f"Error recorded ({self.state.consecutive_errors}/{self.limits.max_consecutive_errors}): {error_message}")
            
            if self.state.consecutive_errors >= self.limits.max_consecutive_errors:
                logger.error("Circuit breaker triggered!")
    
    def update_position(self, symbol: str, value_usd: float):
        """Update position value for concentration tracking."""
        with self._lock:
            self.positions[symbol] = value_usd
    
    def update_strategy_allocation(self, strategy_id: str, allocated_usd: float):
        """Update strategy capital allocation."""
        with self._lock:
            self.strategy_allocations[strategy_id] = allocated_usd
    
    def emergency_stop(self, reason: str = "Manual trigger"):
        """Activate emergency stop - halt all trading."""
        with self._lock:
            self.state.emergency_stop_active = True
            logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
            
            if self.on_emergency_stop:
                self.on_emergency_stop(reason)
    
    def resume_trading(self):
        """Resume trading after emergency stop."""
        with self._lock:
            self.state.emergency_stop_active = False
            self.state.consecutive_errors = 0
            logger.info("Trading resumed after emergency stop")
    
    def record_position_open(self, symbol: str, quantity: float):
        """Record that a position was opened."""
        with self._lock:
            self._open_positions[symbol] = self._open_positions.get(symbol, 0) + quantity
            logger.debug(f"Position opened: {symbol} qty={quantity}")
    
    def record_position_close(self, symbol: str, quantity: float = 0):
        """Record that a position was closed."""
        with self._lock:
            if quantity > 0:
                self._open_positions[symbol] = self._open_positions.get(symbol, 0) - quantity
            else:
                self._open_positions.pop(symbol, None)
            logger.debug(f"Position closed: {symbol}")
    
    def get_risk_report(self) -> Dict:
        """Generate current risk status report."""
        return {
            "emergency_stop_active": self.state.emergency_stop_active,
            "daily_pnl": self.state.daily_pnl,
            "daily_pnl_pct": self.state.daily_pnl / self.portfolio_value * 100,
            "weekly_pnl": self.state.weekly_pnl,
            "consecutive_errors": self.state.consecutive_errors,
            "orders_this_minute": self.state.orders_this_minute,
            "portfolio_value": self.portfolio_value,
            "total_allocated": sum(self.strategy_allocations.values()),
            "strategy_count": len(self.strategy_allocations),
            "open_positions": len([q for q in self._open_positions.values() if abs(q) > 0]),
            "limits": {
                "max_daily_loss_pct": self.limits.max_daily_loss_pct,
                "max_order_value_usd": self.limits.max_order_value_usd,
                "max_concurrent_trades": self.limits.max_concurrent_trades,
                "risk_per_trade_pct": self.limits.risk_per_trade_pct,
                "circuit_breaker_threshold": self.limits.max_consecutive_errors
            }
        }
    
    def reset_daily(self):
        """Reset daily counters (call at midnight)."""
        with self._lock:
            self.state.reset_daily()
            logger.info("Daily risk counters reset")
    
    def reset_weekly(self):
        """Reset weekly counters (call on Sunday)."""
        with self._lock:
            self.state.reset_weekly()
            logger.info("Weekly risk counters reset")


# Singleton instance
_risk_guardian: Optional[RiskGuardian] = None


def get_risk_guardian(portfolio_value: float = 10000.0) -> RiskGuardian:
    """Get the global RiskGuardian instance."""
    global _risk_guardian
    if _risk_guardian is None:
        _risk_guardian = RiskGuardian(portfolio_value=portfolio_value)
    return _risk_guardian
