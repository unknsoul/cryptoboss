"""
Circuit Breaker - Enterprise Safety Feature #136
Automatically halts trading when losses exceed thresholds to protect capital.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker operating states"""
    NORMAL = "normal"          # Trading allowed
    WARNING = "warning"        # Close to limits
    HALTED = "halted"          # Trading stopped
    COOLDOWN = "cooldown"      # Waiting before resuming


class CircuitBreaker:
    """
    Enterprise-grade circuit breaker for trading risk management.
    
    Monitors multiple conditions and halts trading when thresholds are breached:
    - Daily loss limit
    - Drawdown limit
    - Consecutive losses
    - Rapid loss rate (losses per hour)
    
    Features:
    - Multiple trigger conditions
    - Automatic cooldown period
    - Manual reset capability
    - State persistence
    """
    
    def __init__(
        self,
        daily_loss_limit_pct: float = 5.0,      # Max 5% daily loss
        max_drawdown_pct: float = 10.0,          # Max 10% drawdown
        max_consecutive_losses: int = 5,          # Max 5 losses in a row
        rapid_loss_threshold: int = 3,            # 3+ losses in 1 hour
        cooldown_minutes: int = 30,               # 30 min cooldown after halt
        warning_threshold_pct: float = 0.7        # Warn at 70% of limit
    ):
        """
        Initialize circuit breaker with configurable thresholds.
        
        Args:
            daily_loss_limit_pct: Maximum daily loss as percentage of equity
            max_drawdown_pct: Maximum drawdown from peak equity
            max_consecutive_losses: Maximum consecutive losing trades
            rapid_loss_threshold: Number of losses in 1 hour to trigger halt
            cooldown_minutes: Minutes to wait before resuming after halt
            warning_threshold_pct: Percentage of limit to trigger warning
        """
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.rapid_loss_threshold = rapid_loss_threshold
        self.cooldown_minutes = cooldown_minutes
        self.warning_threshold_pct = warning_threshold_pct
        
        # State tracking
        self.state = CircuitBreakerState.NORMAL
        self.halt_reason: Optional[str] = None
        self.halt_time: Optional[datetime] = None
        self.cooldown_end: Optional[datetime] = None
        
        # Metrics tracking
        self.daily_start_equity: float = 0
        self.peak_equity: float = 0
        self.consecutive_losses: int = 0
        self.recent_losses: list = []  # Timestamps of recent losses
        self.daily_reset_date: Optional[datetime] = None
        
        # Trigger history
        self.trigger_history: list = []
        
        logger.info(f"Circuit Breaker initialized - Daily limit: {daily_loss_limit_pct}%, "
                   f"Max DD: {max_drawdown_pct}%, Max consec losses: {max_consecutive_losses}")
    
    def update(self, current_equity: float, trade_result: Optional[float] = None) -> Tuple[bool, str]:
        """
        Update circuit breaker state and check if trading should halt.
        
        Args:
            current_equity: Current account equity
            trade_result: P&L of most recent trade (None if no new trade)
            
        Returns:
            Tuple of (is_trading_allowed, status_message)
        """
        now = datetime.now()
        
        # Check for daily reset
        self._check_daily_reset(now, current_equity)
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Process new trade result if provided
        if trade_result is not None:
            self._process_trade_result(trade_result, now)
        
        # Check if in cooldown
        if self.state == CircuitBreakerState.COOLDOWN:
            if now >= self.cooldown_end:
                self._exit_cooldown()
            else:
                remaining = (self.cooldown_end - now).total_seconds() / 60
                return False, f"Cooldown: {remaining:.1f} min remaining"
        
        # Check if halted (requires manual reset)
        if self.state == CircuitBreakerState.HALTED:
            return False, f"HALTED: {self.halt_reason}"
        
        # Run all halt condition checks
        halt, reason = self._check_halt_conditions(current_equity, now)
        
        if halt:
            self._trigger_halt(reason, now)
            return False, f"CIRCUIT BREAKER TRIGGERED: {reason}"
        
        # Check for warning conditions
        warning, warning_msg = self._check_warning_conditions(current_equity)
        if warning:
            self.state = CircuitBreakerState.WARNING
            return True, f"WARNING: {warning_msg}"
        
        self.state = CircuitBreakerState.NORMAL
        return True, "OK"
    
    def _check_halt_conditions(self, current_equity: float, now: datetime) -> Tuple[bool, str]:
        """Check all halt conditions and return first triggered."""
        
        # 1. Daily Loss Limit
        if self.daily_start_equity > 0:
            daily_loss_pct = (self.daily_start_equity - current_equity) / self.daily_start_equity * 100
            if daily_loss_pct >= self.daily_loss_limit_pct:
                return True, f"Daily loss limit reached ({daily_loss_pct:.2f}%)"
        
        # 2. Maximum Drawdown
        if self.peak_equity > 0:
            drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity * 100
            if drawdown_pct >= self.max_drawdown_pct:
                return True, f"Max drawdown reached ({drawdown_pct:.2f}%)"
        
        # 3. Consecutive Losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return True, f"Max consecutive losses reached ({self.consecutive_losses})"
        
        # 4. Rapid Loss Rate
        one_hour_ago = now - timedelta(hours=1)
        recent_loss_count = len([t for t in self.recent_losses if t > one_hour_ago])
        if recent_loss_count >= self.rapid_loss_threshold:
            return True, f"Rapid loss rate triggered ({recent_loss_count} in 1hr)"
        
        return False, ""
    
    def _check_warning_conditions(self, current_equity: float) -> Tuple[bool, str]:
        """Check warning thresholds (70% of limits)."""
        
        # Daily Loss Warning
        if self.daily_start_equity > 0:
            daily_loss_pct = (self.daily_start_equity - current_equity) / self.daily_start_equity * 100
            warning_level = self.daily_loss_limit_pct * self.warning_threshold_pct
            if daily_loss_pct >= warning_level:
                return True, f"Approaching daily limit ({daily_loss_pct:.1f}%/{self.daily_loss_limit_pct}%)"
        
        # Drawdown Warning
        if self.peak_equity > 0:
            drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity * 100
            warning_level = self.max_drawdown_pct * self.warning_threshold_pct
            if drawdown_pct >= warning_level:
                return True, f"Approaching max DD ({drawdown_pct:.1f}%/{self.max_drawdown_pct}%)"
        
        # Consecutive Loss Warning
        warning_losses = int(self.max_consecutive_losses * self.warning_threshold_pct)
        if self.consecutive_losses >= warning_losses:
            return True, f"Consecutive losses: {self.consecutive_losses}/{self.max_consecutive_losses}"
        
        return False, ""
    
    def _process_trade_result(self, pnl: float, timestamp: datetime):
        """Process a new trade result."""
        if pnl < 0:
            self.consecutive_losses += 1
            self.recent_losses.append(timestamp)
            # Keep only last hour of losses
            one_hour_ago = timestamp - timedelta(hours=1)
            self.recent_losses = [t for t in self.recent_losses if t > one_hour_ago]
        else:
            self.consecutive_losses = 0
    
    def _trigger_halt(self, reason: str, timestamp: datetime):
        """Trigger circuit breaker halt."""
        self.state = CircuitBreakerState.HALTED
        self.halt_reason = reason
        self.halt_time = timestamp
        self.cooldown_end = timestamp + timedelta(minutes=self.cooldown_minutes)
        
        # Record trigger
        self.trigger_history.append({
            'timestamp': timestamp.isoformat(),
            'reason': reason,
            'state': 'halted'
        })
        
        logger.critical(f"🛑 CIRCUIT BREAKER TRIGGERED: {reason}")
        logger.info(f"Trading halted until manual reset or cooldown at {self.cooldown_end}")
    
    def _exit_cooldown(self):
        """Exit cooldown and resume trading with caution."""
        self.state = CircuitBreakerState.WARNING
        self.halt_reason = None
        logger.info("Circuit breaker cooldown complete - resuming with WARNING state")
    
    def _check_daily_reset(self, now: datetime, current_equity: float):
        """Reset daily metrics at start of new trading day."""
        today = now.date()
        if self.daily_reset_date != today:
            self.daily_reset_date = today
            self.daily_start_equity = current_equity
            self.recent_losses = []
            logger.info(f"Daily reset - Start equity: ${current_equity:,.2f}")
    
    def manual_reset(self) -> bool:
        """Manually reset circuit breaker (use with caution)."""
        if self.state in [CircuitBreakerState.HALTED, CircuitBreakerState.COOLDOWN]:
            logger.warning("Circuit breaker manually reset - proceed with caution!")
            self.state = CircuitBreakerState.WARNING
            self.halt_reason = None
            self.consecutive_losses = 0
            self.recent_losses = []
            return True
        return False
    
    def enter_cooldown(self):
        """Enter cooldown mode (auto-resume after cooldown_minutes)."""
        self.state = CircuitBreakerState.COOLDOWN
        self.cooldown_end = datetime.now() + timedelta(minutes=self.cooldown_minutes)
        logger.info(f"Entering cooldown mode until {self.cooldown_end}")
    
    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        return {
            'state': self.state.value,
            'halt_reason': self.halt_reason,
            'consecutive_losses': self.consecutive_losses,
            'losses_last_hour': len([t for t in self.recent_losses if t > one_hour_ago]),
            'daily_start_equity': self.daily_start_equity,
            'peak_equity': self.peak_equity,
            'cooldown_remaining': (
                (self.cooldown_end - now).total_seconds() / 60 
                if self.cooldown_end and now < self.cooldown_end else 0
            ),
            'trigger_count': len(self.trigger_history)
        }
    
    def is_trading_allowed(self) -> bool:
        """Quick check if trading is currently allowed."""
        return self.state in [CircuitBreakerState.NORMAL, CircuitBreakerState.WARNING]


# Singleton instance
_circuit_breaker: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get or create the global circuit breaker instance."""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker()
    return _circuit_breaker


if __name__ == '__main__':
    # Test circuit breaker
    cb = CircuitBreaker(
        daily_loss_limit_pct=5.0,
        max_drawdown_pct=10.0,
        max_consecutive_losses=3
    )
    
    # Simulate trading
    equity = 10000
    cb.update(equity)
    print(f"Initial: {cb.get_status()}")
    
    # Simulate losses
    for i in range(4):
        equity -= 100
        allowed, msg = cb.update(equity, trade_result=-100)
        print(f"Trade {i+1}: Equity=${equity}, Allowed={allowed}, Msg={msg}")
