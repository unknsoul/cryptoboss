"""
Trade Budget Manager - Live Readiness Component

Limits trade activity like a professional trader:
- Maximum trades per day
- Maximum trades per context state
- Maximum losses per bias period
- Total exposure limits

Budgets reset on defined schedules.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class BudgetType(Enum):
    """Types of trade budgets."""
    DAILY_TRADES = "daily_trades"
    CONTEXT_TRADES = "context_trades"
    BIAS_LOSSES = "bias_losses"
    TOTAL_EXPOSURE = "total_exposure"


@dataclass
class BudgetLimits:
    """Configuration for trade budgets."""
    max_trades_per_day: int = 10
    max_trades_per_context: int = 3
    max_losses_per_bias: int = 2
    max_total_exposure_pct: float = 50.0
    
    # Warning thresholds (trigger alerts before exhaustion)
    warning_threshold_pct: float = 80.0


@dataclass
class BudgetStatus:
    """Current status of all budgets."""
    # Remaining counts
    trades_remaining_day: int
    trades_remaining_context: int
    losses_remaining_bias: int
    
    # Usage
    trades_used_day: int
    trades_used_context: int
    losses_used_bias: int
    current_exposure_pct: float
    
    # Exhausted budgets
    exhausted_budgets: List[BudgetType]
    warning_budgets: List[BudgetType]
    
    # Overall status
    trading_allowed: bool
    block_reason: Optional[str]
    
    def to_dict(self) -> Dict:
        return {
            'trades_remaining_day': self.trades_remaining_day,
            'trades_remaining_context': self.trades_remaining_context,
            'losses_remaining_bias': self.losses_remaining_bias,
            'trades_used_day': self.trades_used_day,
            'trades_used_context': self.trades_used_context,
            'losses_used_bias': self.losses_used_bias,
            'current_exposure_pct': self.current_exposure_pct,
            'exhausted_budgets': [b.value for b in self.exhausted_budgets],
            'warning_budgets': [b.value for b in self.warning_budgets],
            'trading_allowed': self.trading_allowed,
            'block_reason': self.block_reason
        }


class TradeBudgetManager:
    """
    Manages trade budgets and limits.
    
    Professional traders don't trade endlessly. This component
    enforces discipline through configurable limits that reset
    on defined schedules.
    
    Usage:
        budget_manager = TradeBudgetManager()
        
        # Check before trading
        status = budget_manager.get_status()
        if not status.trading_allowed:
            logger.info(f"Trading blocked: {status.block_reason}")
            return
        
        # Record activity
        budget_manager.record_trade()
        budget_manager.record_loss()  # On losing trade
    """
    
    def __init__(
        self,
        limits: Optional[BudgetLimits] = None,
        portfolio_value: float = 10000.0
    ):
        self.limits = limits or BudgetLimits()
        self.portfolio_value = portfolio_value
        
        # Current usage counters
        self._trades_today = 0
        self._trades_in_context = 0
        self._losses_in_bias = 0
        self._current_exposure = 0.0
        
        # Tracking dates for resets
        self._daily_reset_date = date.today()
        self._context_start_time: Optional[datetime] = None
        self._bias_start_time: Optional[datetime] = None
        
        logger.info(
            f"TradeBudgetManager initialized: "
            f"daily={self.limits.max_trades_per_day}, "
            f"context={self.limits.max_trades_per_context}, "
            f"bias_losses={self.limits.max_losses_per_bias}, "
            f"exposure={self.limits.max_total_exposure_pct}%"
        )
    
    def get_status(self) -> BudgetStatus:
        """
        Get current budget status.
        
        Checks for date resets before returning status.
        """
        self._check_daily_reset()
        
        exhausted = []
        warnings = []
        
        # Check daily trades
        trades_remaining_day = self.limits.max_trades_per_day - self._trades_today
        if trades_remaining_day <= 0:
            exhausted.append(BudgetType.DAILY_TRADES)
        elif trades_remaining_day <= self.limits.max_trades_per_day * 0.2:
            warnings.append(BudgetType.DAILY_TRADES)
        
        # Check context trades  
        trades_remaining_context = self.limits.max_trades_per_context - self._trades_in_context
        if trades_remaining_context <= 0:
            exhausted.append(BudgetType.CONTEXT_TRADES)
        elif trades_remaining_context <= 1:
            warnings.append(BudgetType.CONTEXT_TRADES)
        
        # Check bias losses
        losses_remaining_bias = self.limits.max_losses_per_bias - self._losses_in_bias
        if losses_remaining_bias <= 0:
            exhausted.append(BudgetType.BIAS_LOSSES)
        elif losses_remaining_bias <= 1:
            warnings.append(BudgetType.BIAS_LOSSES)
        
        # Check exposure
        exposure_pct = (self._current_exposure / self.portfolio_value) * 100
        if exposure_pct >= self.limits.max_total_exposure_pct:
            exhausted.append(BudgetType.TOTAL_EXPOSURE)
        elif exposure_pct >= self.limits.max_total_exposure_pct * 0.8:
            warnings.append(BudgetType.TOTAL_EXPOSURE)
        
        # Determine if trading allowed
        trading_allowed = len(exhausted) == 0
        block_reason = None
        if not trading_allowed:
            reasons = [b.value for b in exhausted]
            block_reason = f"Budget exhausted: {', '.join(reasons)}"
        
        return BudgetStatus(
            trades_remaining_day=max(0, trades_remaining_day),
            trades_remaining_context=max(0, trades_remaining_context),
            losses_remaining_bias=max(0, losses_remaining_bias),
            trades_used_day=self._trades_today,
            trades_used_context=self._trades_in_context,
            losses_used_bias=self._losses_in_bias,
            current_exposure_pct=exposure_pct,
            exhausted_budgets=exhausted,
            warning_budgets=warnings,
            trading_allowed=trading_allowed,
            block_reason=block_reason
        )
    
    def can_trade(self) -> tuple[bool, Optional[str]]:
        """
        Quick check if trading is allowed.
        
        Returns: (allowed, reason if not allowed)
        """
        status = self.get_status()
        return status.trading_allowed, status.block_reason
    
    def record_trade(self, position_size: float = 0.0):
        """
        Record a trade execution.
        
        Args:
            position_size: Size of position opened (for exposure tracking)
        """
        self._check_daily_reset()
        
        self._trades_today += 1
        self._trades_in_context += 1
        self._current_exposure += position_size
        
        logger.debug(
            f"Trade recorded: day={self._trades_today}/{self.limits.max_trades_per_day}, "
            f"context={self._trades_in_context}/{self.limits.max_trades_per_context}"
        )
    
    def record_loss(self):
        """Record a losing trade."""
        self._losses_in_bias += 1
        
        logger.debug(
            f"Loss recorded: bias={self._losses_in_bias}/{self.limits.max_losses_per_bias}"
        )
    
    def record_win(self):
        """Record a winning trade (no action currently)."""
        pass  # Could implement win streaks etc.
    
    def close_position(self, position_size: float):
        """Record position close (reduces exposure)."""
        self._current_exposure = max(0, self._current_exposure - position_size)
    
    def on_context_change(self):
        """
        Reset context-specific budgets.
        
        Called when market context state changes.
        """
        self._trades_in_context = 0
        self._context_start_time = datetime.now()
        
        logger.info("Context budget reset (context changed)")
    
    def on_bias_change(self):
        """
        Reset bias-specific budgets.
        
        Called when directional bias changes.
        """
        self._losses_in_bias = 0
        self._bias_start_time = datetime.now()
        
        logger.info("Bias budget reset (bias changed)")
    
    def set_portfolio_value(self, value: float):
        """Update portfolio value for exposure calculations."""
        self.portfolio_value = value
    
    def force_reset_all(self):
        """Force reset all budgets (admin action)."""
        self._trades_today = 0
        self._trades_in_context = 0
        self._losses_in_bias = 0
        self._current_exposure = 0.0
        self._daily_reset_date = date.today()
        
        logger.warning("All budgets force reset")
    
    def _check_daily_reset(self):
        """Check if daily reset is needed."""
        today = date.today()
        if self._daily_reset_date != today:
            logger.info("New day detected, resetting daily budgets")
            self._trades_today = 0
            self._daily_reset_date = today
    
    def to_dict(self) -> Dict:
        """Serialize current state."""
        return {
            'trades_today': self._trades_today,
            'trades_in_context': self._trades_in_context,
            'losses_in_bias': self._losses_in_bias,
            'current_exposure': self._current_exposure,
            'daily_reset_date': self._daily_reset_date.isoformat(),
            'limits': {
                'max_trades_per_day': self.limits.max_trades_per_day,
                'max_trades_per_context': self.limits.max_trades_per_context,
                'max_losses_per_bias': self.limits.max_losses_per_bias,
                'max_total_exposure_pct': self.limits.max_total_exposure_pct
            }
        }
    
    def from_dict(self, data: Dict):
        """Restore state from dict."""
        self._trades_today = data.get('trades_today', 0)
        self._trades_in_context = data.get('trades_in_context', 0)
        self._losses_in_bias = data.get('losses_in_bias', 0)
        self._current_exposure = data.get('current_exposure', 0.0)
        
        if data.get('daily_reset_date'):
            self._daily_reset_date = date.fromisoformat(data['daily_reset_date'])
        
        self._check_daily_reset()


# Singleton instance
_budget_manager: Optional[TradeBudgetManager] = None


def get_budget_manager(portfolio_value: float = 10000.0) -> TradeBudgetManager:
    """Get global TradeBudgetManager instance."""
    global _budget_manager
    if _budget_manager is None:
        _budget_manager = TradeBudgetManager(portfolio_value=portfolio_value)
    return _budget_manager
