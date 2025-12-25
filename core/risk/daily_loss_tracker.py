"""
Daily/Weekly Loss Tracking
Implements the TODO from advanced_risk.py with timezone-aware tracking.
"""
import logging
from datetime import datetime, time, timedelta
from typing import Dict, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class DailyLossTracker:
    """
    Tracks daily and weekly losses with automatic reset at market open.
    
    Features:
    - Timezone-aware tracking
    - Automatic daily/weekly reset
    - Configurable loss limits
    - Pause trading when limits exceeded
    """
    
    def __init__(
        self,
        daily_loss_limit_pct: float = 5.0,
        weekly_loss_limit_pct: float = 10.0,
        timezone: str = "America/New_York",
        market_open_time: str = "09:30"
    ):
        """
        Initialize daily loss tracker.
        
        Args:
            daily_loss_limit_pct: Maximum daily loss (% of equity)
            weekly_loss_limit_pct: Maximum weekly loss (% of equity)
            timezone: Market timezone (default: NYSE)
            market_open_time: Market open time in HH:MM format
        """
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.weekly_loss_limit_pct = weekly_loss_limit_pct
        self.tz = ZoneInfo(timezone)
        
        # Parse market open time
        hour, minute = map(int, market_open_time.split(":"))
        self.market_open_time = time(hour, minute)
        
        # Tracking variables
        self.daily_start_equity = 0.0
        self.weekly_start_equity = 0.0
        self.current_equity = 0.0
        
        self.last_reset_date: Optional[datetime] = None
        self.week_start_date: Optional[datetime] = None
        
        self.daily_limit_hit = False
        self.weekly_limit_hit = False
        
        logger.info(f"Daily Loss Tracker initialized - Daily: {daily_loss_limit_pct}%, Weekly: {weekly_loss_limit_pct}%")
    
    def initialize(self, equity: float):
        """Initialize tracker with starting equity."""
        now = datetime.now(self.tz)
        self.daily_start_equity = equity
        self.weekly_start_equity = equity
        self.current_equity = equity
        self.last_reset_date = now.date()
        self.week_start_date = now.date()
        logger.info(f"Loss tracker initialized with equity: ${equity:,.2f}")
    
    def _should_reset_daily(self) -> bool:
        """Check if we should reset daily tracking."""
        now = datetime.now(self.tz)
        current_date = now.date()
        
        # Reset if it's a new day and past market open
        if self.last_reset_date != current_date:
            if now.time() >= self.market_open_time:
                return True
        return False
    
    def _should_reset_weekly(self) -> bool:
        """Check if we should reset weekly tracking (Monday)."""
        now = datetime.now(self.tz)
        
        # Reset on Monday
        if now.weekday() == 0 and self.week_start_date.weekday() != 0:
            return True
        return False
    
    def update(self, new_equity: float) -> Dict:
        """
        Update equity and check loss limits.
        
        Args:
            new_equity: Current equity value
            
        Returns:
            Status dict with can_trade flag and loss info
        """
        self.current_equity = new_equity
        
        # Check for resets
        if self._should_reset_daily():
            self.daily_start_equity = new_equity
            self.last_reset_date = datetime.now(self.tz).date()
            self.daily_limit_hit = False
            logger.info(f"Daily loss tracking reset. New baseline: ${new_equity:,.2f}")
        
        if self._should_reset_weekly():
            self.weekly_start_equity = new_equity
            self.week_start_date = datetime.now(self.tz).date()
            self.weekly_limit_hit = False
            logger.info(f"Weekly loss tracking reset. New baseline: ${new_equity:,.2f}")
        
        # Calculate losses
        daily_loss = self.daily_start_equity - new_equity
        weekly_loss = self.weekly_start_equity - new_equity
        
        daily_loss_pct = (daily_loss / self.daily_start_equity * 100) if self.daily_start_equity > 0 else 0
        weekly_loss_pct = (weekly_loss / self.weekly_start_equity * 100) if self.weekly_start_equity > 0 else 0
        
        # Check limits
        if not self.daily_limit_hit and daily_loss_pct >= self.daily_loss_limit_pct:
            self.daily_limit_hit = True
            logger.critical(f"🚨 DAILY LOSS LIMIT HIT: {daily_loss_pct:.2f}% >= {self.daily_loss_limit_pct}%")
        
        if not self.weekly_limit_hit and weekly_loss_pct >= self.weekly_loss_limit_pct:
            self.weekly_limit_hit = True
            logger.critical(f"🚨 WEEKLY LOSS LIMIT HIT: {weekly_loss_pct:.2f}% >= {self.weekly_loss_limit_pct}%")
        
        can_trade = not (self.daily_limit_hit or self.weekly_limit_hit)
        
        return {
            'can_trade': can_trade,
            'daily_loss_pct': round(daily_loss_pct, 2),
            'weekly_loss_pct': round(weekly_loss_pct, 2),
            'daily_limit_hit': self.daily_limit_hit,
            'weekly_limit_hit': self.weekly_limit_hit,
            'current_equity': new_equity,
            'daily_start': self.daily_start_equity,
            'weekly_start': self.weekly_start_equity
        }
    
    def force_reset(self):
        """Force reset all limits (manual override)."""
        self.daily_limit_hit = False
        self.weekly_limit_hit = False
        logger.warning("Loss limits manually reset")
    
    def get_stats(self) -> Dict:
        """Get current loss tracking statistics."""
        daily_loss_pct = (self.daily_start_equity - self.current_equity) / self.daily_start_equity * 100
        weekly_loss_pct = (self.weekly_start_equity - self.current_equity) / self.weekly_start_equity * 100
        
        return {
            'daily_loss_pct': round(daily_loss_pct, 2),
            'weekly_loss_pct': round(weekly_loss_pct, 2),
            'daily_limit': self.daily_loss_limit_pct,
            'weekly_limit': self.weekly_loss_limit_pct,
            'daily_hit': self.daily_limit_hit,
            'weekly_hit': self.weekly_limit_hit,
            'can_trade': not (self.daily_limit_hit or self.weekly_limit_hit)
        }
