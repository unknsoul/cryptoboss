"""
Smart Risk Manager - Professional-Grade Trading Control

Modes:
- NORMAL: Standard trading with full size
- PROFIT_PROTECT: Daily profit target hit, stop trading to protect gains
- RECOVERY: In drawdown, reduced size, only A+ trades
- CIRCUIT_BREAKER: Max daily loss hit, all trading stopped

Also includes:
- Trailing stop management
- Partial profit taking
- Cooldown periods
- Streak management
"""

import os
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    NORMAL = "normal"
    PROFIT_PROTECT = "profit_protect"
    RECOVERY = "recovery"
    CIRCUIT_BREAKER = "circuit_breaker"


class SmartRiskManager:
    """
    Professional risk management with adaptive trading modes.
    """
    
    # Thresholds (as percentage of daily starting equity)
    PROFIT_PROTECT_THRESHOLD = 2.0    # +2% - stop and protect gains
    RECOVERY_THRESHOLD = -1.0          # -1% - enter recovery mode
    MAX_RECOVERY_DRAWDOWN = -2.5       # -2.5% - very careful trading
    CIRCUIT_BREAKER_THRESHOLD = -3.0   # -3% - stop all trading
    
    # Cooldown periods (seconds)
    COOLDOWN_AFTER_WIN = 180          # 3 minutes after win
    COOLDOWN_AFTER_LOSS = 300         # 5 minutes after loss
    
    # Streak limits
    MAX_WIN_STREAK_NORMAL = 3         # After 3 wins, reduce size
    MAX_LOSS_STREAK_BEFORE_RECOVERY = 2  # After 2 losses, enter recovery
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.daily_start_equity = initial_capital
        self.current_equity = initial_capital
        
        self.mode = TradingMode.NORMAL
        self.today = date.today()
        
        # Trade tracking
        self.daily_trades = []
        self.win_streak = 0
        self.loss_streak = 0
        self.last_trade_time = None
        self.last_trade_result = None  # 'win' or 'loss'
        
        # Partial profit tracking
        self.partial_profits_taken = {}  # {trade_id: [1R, 2R taken]}
        
        # Trailing stop tracking
        self.trailing_stops = {}  # {trade_id: current_stop}
        self.max_profits = {}     # {trade_id: max_unrealized_pnl}
        
        logger.info(f"SmartRiskManager initialized with ${initial_capital:,.2f}")
        logger.info(f"  Profit Protection: +{self.PROFIT_PROTECT_THRESHOLD}%")
        logger.info(f"  Recovery Mode: {self.RECOVERY_THRESHOLD}%")
        logger.info(f"  Circuit Breaker: {self.CIRCUIT_BREAKER_THRESHOLD}%")
    
    def reset_daily(self, current_equity: float):
        """Reset for new trading day"""
        if self.today != date.today():
            self.today = date.today()
            self.daily_start_equity = current_equity
            self.daily_trades = []
            self.mode = TradingMode.NORMAL
            self.win_streak = 0
            self.loss_streak = 0
            logger.info(f"New trading day. Starting equity: ${current_equity:,.2f}")
    
    def update_equity(self, new_equity: float):
        """Update current equity and check mode transitions"""
        self.current_equity = new_equity
        self.reset_daily(new_equity)  # Check for new day
        
        # Calculate daily P&L
        daily_pnl = self._get_daily_pnl_percent()
        
        # Check mode transitions
        old_mode = self.mode
        
        if daily_pnl <= self.CIRCUIT_BREAKER_THRESHOLD:
            self.mode = TradingMode.CIRCUIT_BREAKER
        elif daily_pnl >= self.PROFIT_PROTECT_THRESHOLD:
            self.mode = TradingMode.PROFIT_PROTECT
        elif daily_pnl <= self.RECOVERY_THRESHOLD:
            self.mode = TradingMode.RECOVERY
        elif self.loss_streak >= self.MAX_LOSS_STREAK_BEFORE_RECOVERY:
            self.mode = TradingMode.RECOVERY
        else:
            # Can return to normal if recovered
            if self.mode == TradingMode.RECOVERY and daily_pnl > self.RECOVERY_THRESHOLD / 2:
                self.mode = TradingMode.NORMAL
        
        if old_mode != self.mode:
            logger.info(f"Mode change: {old_mode.value} → {self.mode.value}")
            self._log_mode_change(old_mode, self.mode, daily_pnl)
    
    def _get_daily_pnl_percent(self) -> float:
        """Get daily P&L as percentage"""
        if self.daily_start_equity == 0:
            return 0.0
        return ((self.current_equity - self.daily_start_equity) / self.daily_start_equity) * 100
    
    def can_trade(self, quality_score: int = 70) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on current mode and conditions.
        
        Returns:
            (can_trade, reason)
        """
        daily_pnl = self._get_daily_pnl_percent()
        
        # Circuit Breaker - No trading
        if self.mode == TradingMode.CIRCUIT_BREAKER:
            return False, f"CIRCUIT BREAKER: Daily loss {daily_pnl:.1f}% exceeded limit"
        
        # Profit Protection - No more trading today
        if self.mode == TradingMode.PROFIT_PROTECT:
            return False, f"PROFIT PROTECTED: Daily gain {daily_pnl:.1f}% locked in"
        
        # Check cooldown
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            
            if self.last_trade_result == 'loss' and elapsed < self.COOLDOWN_AFTER_LOSS:
                remaining = int(self.COOLDOWN_AFTER_LOSS - elapsed)
                return False, f"COOLDOWN: {remaining}s remaining after loss"
            
            if self.last_trade_result == 'win' and elapsed < self.COOLDOWN_AFTER_WIN:
                remaining = int(self.COOLDOWN_AFTER_WIN - elapsed)
                return False, f"COOLDOWN: {remaining}s remaining after win"
        
        # Recovery Mode - Only A+ trades (score 85+)
        if self.mode == TradingMode.RECOVERY:
            if quality_score < 85:
                return False, f"RECOVERY MODE: Need score 85+, got {quality_score}"
            return True, f"RECOVERY: A+ trade allowed (score {quality_score})"
        
        # Normal mode
        return True, f"NORMAL: Trading allowed (P&L: {daily_pnl:.1f}%)"
    
    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on mode and streaks"""
        base_mult = 1.0
        
        # Mode adjustments
        if self.mode == TradingMode.RECOVERY:
            base_mult = 0.5  # Half size in recovery
        
        # Win streak adjustment (avoid overconfidence)
        if self.win_streak >= self.MAX_WIN_STREAK_NORMAL:
            base_mult *= 0.75
            logger.debug(f"Win streak {self.win_streak}: size reduced to 75%")
        
        # Loss streak adjustment
        if self.loss_streak >= 2:
            base_mult *= 0.5
            logger.debug(f"Loss streak {self.loss_streak}: size reduced to 50%")
        
        return base_mult
    
    def record_trade(self, pnl: float, trade_id: str = None):
        """Record trade result and update streaks"""
        self.last_trade_time = datetime.now()
        
        if pnl > 0:
            self.last_trade_result = 'win'
            self.win_streak += 1
            self.loss_streak = 0
            logger.info(f"WIN recorded: +${pnl:.2f} (streak: {self.win_streak})")
        else:
            self.last_trade_result = 'loss'
            self.loss_streak += 1
            self.win_streak = 0
            logger.info(f"LOSS recorded: -${abs(pnl):.2f} (streak: {self.loss_streak})")
        
        self.daily_trades.append({
            'time': datetime.now().isoformat(),
            'pnl': pnl,
            'result': self.last_trade_result
        })
        
        # Cleanup tracking for completed trade
        if trade_id:
            self.partial_profits_taken.pop(trade_id, None)
            self.trailing_stops.pop(trade_id, None)
            self.max_profits.pop(trade_id, None)
    
    # ============ TRAILING STOP MANAGEMENT ============
    
    def update_trailing_stop(
        self,
        trade_id: str,
        entry_price: float,
        current_price: float,
        current_stop: float,
        atr: float,
        is_long: bool
    ) -> Tuple[float, str]:
        """
        Update trailing stop for a position.
        
        Returns:
            (new_stop, reason)
        """
        # Calculate current P&L in ATR units
        if is_long:
            pnl_atr = (current_price - entry_price) / atr
            unrealized = current_price - entry_price
        else:
            pnl_atr = (entry_price - current_price) / atr
            unrealized = entry_price - current_price
        
        # Track max profit
        if trade_id not in self.max_profits:
            self.max_profits[trade_id] = 0
        self.max_profits[trade_id] = max(self.max_profits[trade_id], unrealized)
        
        new_stop = current_stop
        reason = "No change"
        
        # Move to breakeven at 1R (1 ATR profit)
        if pnl_atr >= 1.0:
            breakeven = entry_price
            if is_long and new_stop < breakeven:
                new_stop = breakeven
                reason = "Moved to breakeven (1R reached)"
            elif not is_long and new_stop > breakeven:
                new_stop = breakeven
                reason = "Moved to breakeven (1R reached)"
        
        # Trail at 0.5 ATR behind price after 1.5R
        if pnl_atr >= 1.5:
            if is_long:
                trail_stop = current_price - (0.5 * atr)
                if trail_stop > new_stop:
                    new_stop = trail_stop
                    reason = f"Trailing at ${new_stop:.2f}"
            else:
                trail_stop = current_price + (0.5 * atr)
                if trail_stop < new_stop:
                    new_stop = trail_stop
                    reason = f"Trailing at ${new_stop:.2f}"
        
        # Lock in 50% of max profit after 2R
        if pnl_atr >= 2.0 and self.max_profits[trade_id] > 0:
            lock_profit = self.max_profits[trade_id] * 0.5
            if is_long:
                lock_stop = entry_price + lock_profit
                if lock_stop > new_stop:
                    new_stop = lock_stop
                    reason = f"Locked 50% profit at ${new_stop:.2f}"
            else:
                lock_stop = entry_price - lock_profit
                if lock_stop < new_stop:
                    new_stop = lock_stop
                    reason = f"Locked 50% profit at ${new_stop:.2f}"
        
        self.trailing_stops[trade_id] = new_stop
        return round(new_stop, 2), reason
    
    # ============ PARTIAL PROFIT TAKING ============
    
    def should_take_partial(
        self,
        trade_id: str,
        entry_price: float,
        current_price: float,
        atr: float,
        is_long: bool
    ) -> Tuple[bool, float, str]:
        """
        Check if should take partial profit.
        
        Returns:
            (should_take, percent_to_close, reason)
        """
        if trade_id not in self.partial_profits_taken:
            self.partial_profits_taken[trade_id] = {'1R': False, '2R': False}
        
        taken = self.partial_profits_taken[trade_id]
        
        # Calculate P&L in ATR units
        if is_long:
            pnl_atr = (current_price - entry_price) / atr
        else:
            pnl_atr = (entry_price - current_price) / atr
        
        # Take 33% at 1R
        if pnl_atr >= 1.0 and not taken['1R']:
            taken['1R'] = True
            return True, 0.33, "Taking 33% at 1R"
        
        # Take 33% at 2R
        if pnl_atr >= 2.0 and not taken['2R']:
            taken['2R'] = True
            return True, 0.33, "Taking 33% at 2R"
        
        return False, 0, "Hold position"
    
    # ============ LOGGING ============
    
    def _log_mode_change(self, old_mode: TradingMode, new_mode: TradingMode, pnl: float):
        """Log mode changes to debug trace"""
        try:
            with open('debug_trace.txt', 'a') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"RISK MODE CHANGE: {old_mode.value} → {new_mode.value}\n")
                f.write(f"Daily P&L: {pnl:.2f}%\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"Win Streak: {self.win_streak}, Loss Streak: {self.loss_streak}\n")
                f.write(f"{'='*50}\n")
        except:
            pass
    
    def get_status(self) -> Dict:
        """Get current risk manager status"""
        return {
            'mode': self.mode.value,
            'daily_pnl_percent': self._get_daily_pnl_percent(),
            'win_streak': self.win_streak,
            'loss_streak': self.loss_streak,
            'size_multiplier': self.get_position_size_multiplier(),
            'trades_today': len(self.daily_trades),
            'can_trade': self.can_trade()[0]
        }


# Export
__all__ = ['SmartRiskManager', 'TradingMode']


if __name__ == '__main__':
    print("=" * 60)
    print("SMART RISK MANAGER TEST")
    print("=" * 60)
    
    rm = SmartRiskManager(initial_capital=10000)
    
    # Test mode transitions
    print(f"\nInitial mode: {rm.mode.value}")
    print(f"Can trade: {rm.can_trade()}")
    
    # Simulate profit
    rm.update_equity(10200)  # +2%
    print(f"\nAfter +2%: {rm.mode.value}")
    print(f"Can trade: {rm.can_trade()}")
    
    # Reset and simulate loss
    rm.mode = TradingMode.NORMAL
    rm.update_equity(9700)  # -3%
    print(f"\nAfter -3%: {rm.mode.value}")
    print(f"Can trade: {rm.can_trade()}")
