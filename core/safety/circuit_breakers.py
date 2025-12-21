"""
Circuit Breakers & Kill Switches - Emergency Risk Controls
Automatic position flattening and trading halts

Features:
- Daily loss limits
- Drawdown circuit breakers
- Volatility spike protection
- Manual kill switch
- Emergency position flattening
- Auto-failover to safe mode
- Progressive shutdown
"""

import numpy as np
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum

from core.monitoring.logger import get_logger
from core.monitoring.metrics import get_metrics
from core.monitoring.alerting import get_alerts


logger = get_logger()
metrics = get_metrics()
alerts = get_alerts()


class TradingMode(Enum):
    """Trading system modes"""
    NORMAL = "normal"
    REDUCED_RISK = "reduced_risk"
    SAFE_MODE = "safe_mode"
    HALTED = "halted"
    EMERGENCY_FLATTEN = "emergency_flatten"


class CircuitBreaker:
    """
    Automated circuit breakers to protect capital
    """
    
    def __init__(self,
                 daily_loss_limit: float = 0.05,
                 max_drawdown: float = 0.15,
                 volatility_spike_threshold: float = 3.0,
                 consecutive_losses_limit: int = 5):
        """
        Args:
            daily_loss_limit: Max daily loss before halt (5%)
            max_drawdown: Max drawdown before halt (15%)
            volatility_spike_threshold: Vol spike multiplier to trigger (3x)
            consecutive_losses_limit: Max consecutive losing trades
        """
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown
        self.volatility_spike_threshold = volatility_spike_threshold
        self.consecutive_losses_limit = consecutive_losses_limit
        
        # State tracking
        self.trading_mode = TradingMode.NORMAL
        self.circuit_breaker_triggered = False
        self.trigger_reason = None
        self.trigger_time = None
        
        # Daily tracking
        self.daily_start_capital = None
        self.daily_pnl = 0
        self.today = datetime.now().date()
        
        # Consecutive losses
        self.consecutive_losses = 0
        
        # Callbacks
        self.on_circuit_break: Optional[Callable] = None
        self.on_mode_change: Optional[Callable] = None
    
    def check_circuit_breakers(self,
                              current_capital: float,
                              peak_capital: float,
                              current_volatility: float,
                              avg_volatility: float,
                              last_trade_pnl: Optional[float] = None) -> TradingMode:
        """
        Check all circuit breakers
        
        Args:
            current_capital: Current portfolio value
            peak_capital: Historical peak capital
            current_volatility: Current market volatility
            avg_volatility: Average volatility
            last_trade_pnl: P&L of last trade (if any)
        
        Returns:
            Recommended trading mode
        """
        # Reset daily tracking
        self._check_daily_reset()
        
        # Update daily P&L
        if self.daily_start_capital is None:
            self.daily_start_capital = current_capital
        
        self.daily_pnl = (current_capital - self.daily_start_capital) / self.daily_start_capital
        
        # Track consecutive losses
        if last_trade_pnl is not None:
            if last_trade_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
        
        # Check Circuit Breaker 1: Daily Loss Limit
        if self.daily_pnl <= -self.daily_loss_limit:
            return self._trigger_circuit_breaker(
                TradingMode.HALTED,
                f"Daily loss limit exceeded: {self.daily_pnl:.2%}",
                {
                    'daily_pnl': self.daily_pnl,
                    'limit': self.daily_loss_limit
                }
            )
        
        # Check Circuit Breaker 2: Maximum Drawdown
        current_dd = (current_capital - peak_capital) / peak_capital
        if current_dd <= -self.max_drawdown:
            return self._trigger_circuit_breaker(
                TradingMode.EMERGENCY_FLATTEN,
                f"Max drawdown exceeded: {current_dd:.2%}",
                {
                    'drawdown': current_dd,
                    'limit': self.max_drawdown,
                    'current_capital': current_capital,
                    'peak_capital': peak_capital
                }
            )
        
        # Check Circuit Breaker 3: Volatility Spike
        vol_ratio = current_volatility / (avg_volatility + 1e-8)
        if vol_ratio > self.volatility_spike_threshold:
            return self._trigger_circuit_breaker(
                TradingMode.SAFE_MODE,
                f"Volatility spike detected: {vol_ratio:.1f}x normal",
                {
                    'volatility_ratio': vol_ratio,
                    'threshold': self.volatility_spike_threshold
                }
            )
        
        # Check Circuit Breaker 4: Consecutive Losses
        if self.consecutive_losses >= self.consecutive_losses_limit:
            return self._trigger_circuit_breaker(
                TradingMode.REDUCED_RISK,
                f"{self.consecutive_losses} consecutive losses",
                {
                    'consecutive_losses': self.consecutive_losses,
                    'limit': self.consecutive_losses_limit
                }
            )
        
        # Progressive risk reduction based on daily P&L
        if -0.03 <= self.daily_pnl < -0.02:  # -2% to -3% loss
            return self._set_mode(TradingMode.REDUCED_RISK, "Elevated daily loss")
        
        # All clear - return to normal if currently in reduced mode
        if self.trading_mode == TradingMode.REDUCED_RISK and self.daily_pnl > -0.01:
            return self._set_mode(TradingMode.NORMAL, "Risk levels normalized")
        
        return self.trading_mode
    
    def _trigger_circuit_breaker(self,
                                 mode: TradingMode,
                                 reason: str,
                                 details: Dict[str, Any]) -> TradingMode:
        """Trigger circuit breaker"""
        if not self.circuit_breaker_triggered or mode != self.trading_mode:
            logger.error(
                f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}",
                mode=mode.value,
                **details
            )
            
            alerts.send_alert(
                "circuit_breaker_triggered",
                f"CIRCUIT BREAKER: {reason}",
                {
                    'mode': mode.value,
                    'reason': reason,
                    **details
                },
                severity="CRITICAL"
            )
            
            self.circuit_breaker_triggered = True
            self.trigger_reason = reason
            self.trigger_time = datetime.now()
            
            # Execute callback
            if self.on_circuit_break:
                self.on_circuit_break(mode, reason, details)
            
            metrics.increment(f"circuit_breaker_{mode.value}")
        
        return self._set_mode(mode, reason)
    
    def _set_mode(self, mode: TradingMode, reason: str) -> TradingMode:
        """Set trading mode"""
        if mode != self.trading_mode:
            old_mode = self.trading_mode
            self.trading_mode = mode
            
            logger.warning(
                f"Trading mode changed: {old_mode.value} → {mode.value}",
                reason=reason
            )
            
            if self.on_mode_change:
                self.on_mode_change(old_mode, mode, reason)
            
            metrics.gauge("trading_mode", self._mode_to_int(mode))
        
        return mode
    
    def _mode_to_int(self, mode: TradingMode) -> int:
        """Convert mode to int for metrics"""
        mapping = {
            TradingMode.EMERGENCY_FLATTEN: 0,
            TradingMode.HALTED: 1,
            TradingMode.SAFE_MODE: 2,
            TradingMode.REDUCED_RISK: 3,
            TradingMode.NORMAL: 4
        }
        return mapping.get(mode, 0)
    
    def _check_daily_reset(self):
        """Reset daily tracking at start of new day"""
        current_date = datetime.now().date()
        if current_date != self.today:
            logger.info("Daily reset - new trading day")
            self.today = current_date
            self.daily_start_capital = None
            self.daily_pnl = 0
            
            # Reset circuit breaker if it was triggered yesterday
            if self.circuit_breaker_triggered:
                if self.trigger_time and (datetime.now() - self.trigger_time) > timedelta(days=1):
                    logger.info("Circuit breaker auto-reset after 24h")
                    self.circuit_breaker_triggered = False
                    self.trigger_reason = None
                    self._set_mode(TradingMode.NORMAL, "Daily auto-reset")
    
    def manual_halt(self, reason: str = "Manual halt"):
        """Manual emergency stop"""
        logger.error(f"🛑 MANUAL HALT: {reason}")
        
        alerts.send_alert(
            "manual_halt",
            f"Trading manually halted: {reason}",
            {},
            severity="CRITICAL"
        )
        
        self._set_mode(TradingMode.HALTED, reason)
        metrics.increment("manual_halt")
    
    def manual_resume(self):
        """Resume trading after manual halt"""
        if self.trading_mode == TradingMode.HALTED:
            logger.info("✅ Trading manually resumed")
            
            alerts.send_alert(
                "trading_resumed",
                "Trading manually resumed",
                {},
                severity="WARNING"
            )
            
            self._set_mode(TradingMode.NORMAL, "Manual resume")
            self.circuit_breaker_triggered = False
            metrics.increment("manual_resume")
    
    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on current mode
        
        Returns:
            Multiplier (0.0-1.0)
        """
        multipliers = {
            TradingMode.NORMAL: 1.0,
            TradingMode.REDUCED_RISK: 0.5,
            TradingMode.SAFE_MODE: 0.25,
            TradingMode.HALTED: 0.0,
            TradingMode.EMERGENCY_FLATTEN: 0.0
        }
        
        return multipliers.get(self.trading_mode, 0.0)
    
    def should_trade(self) -> bool:
        """Check if trading is allowed"""
        return self.trading_mode in [TradingMode.NORMAL, TradingMode.REDUCED_RISK, TradingMode.SAFE_MODE]
    
    def should_flatten_positions(self) -> bool:
        """Check if positions should be flattened"""
        return self.trading_mode == TradingMode.EMERGENCY_FLATTEN
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'trading_mode': self.trading_mode.value,
            'circuit_breaker_triggered': self.circuit_breaker_triggered,
            'trigger_reason': self.trigger_reason,
            'trigger_time': self.trigger_time.isoformat() if self.trigger_time else None,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'position_size_multiplier': self.get_position_size_multiplier(),
            'can_trade': self.should_trade(),
            'should_flatten': self.should_flatten_positions()
        }


class KillSwitch:
    """
    Emergency kill switch - immediately halt all trading
    """
    
    def __init__(self):
        self.activated = False
        self.activation_time = None
        self.activation_reason = None
    
    def activate(self, reason: str = "Emergency kill switch activated"):
        """Activate kill switch"""
        if not self.activated:
            self.activated = True
            self.activation_time = datetime.now()
            self.activation_reason = reason
            
            logger.error(
                f"🔴 KILL SWITCH ACTIVATED: {reason}",
                timestamp=self.activation_time.isoformat()
            )
            
            alerts.send_alert(
                "kill_switch_activated",
                f"KILL SWITCH: {reason}",
                {
                    'timestamp': self.activation_time.isoformat(),
                    'reason': reason
                },
                severity="CRITICAL"
            )
            
            metrics.increment("kill_switch_activated")
    
    def deactivate(self, authorization_code: Optional[str] = None):
        """
        Deactivate kill switch (requires authorization)
        
        Args:
            authorization_code: Optional authorization code for safety
        """
        # In production, verify authorization_code
        if self.activated:
            self.activated = False
            
            logger.warning(
                "Kill switch deactivated",
                authorization=bool(authorization_code)
            )
            
            alerts.send_alert(
                "kill_switch_deactivated",
                "Kill switch deactivated - trading can resume",
                {},
                severity="WARNING"
            )
            
            metrics.increment("kill_switch_deactivated")
    
    def is_active(self) -> bool:
        """Check if kill switch is active"""
        return self.activated


if __name__ == "__main__":
    print("=" * 70)
    print("🚨 CIRCUIT BREAKERS TEST")
    print("=" * 70)
    
    cb = CircuitBreaker(
        daily_loss_limit=0.05,
        max_drawdown=0.15,
        volatility_spike_threshold=3.0,
        consecutive_losses_limit=5
    )
    
    # Test 1: Normal conditions
    print("\n1. Normal Trading Conditions:")
    mode = cb.check_circuit_breakers(
        current_capital=10500,
        peak_capital=11000,
        current_volatility=0.02,
        avg_volatility=0.02
    )
    print(f"   Mode: {mode.value}")
    print(f"   Can trade: {cb.should_trade()}")
    print(f"   Position multiplier: {cb.get_position_size_multiplier()}")
    
    # Test 2: Daily loss limit
    print("\n2. Daily Loss Limit Exceeded:")
    mode = cb.check_circuit_breakers(
        current_capital=9400,  # -6% loss
        peak_capital=11000,
        current_volatility=0.02,
        avg_volatility=0.02
    )
    print(f"   Mode: {mode.value}")
    print(f"   Daily P&L: {cb.daily_pnl:.2%}")
    print(f"   Can trade: {cb.should_trade()}")
    
    # Test 3: Kill switch
    print("\n3. Kill Switch:")
    ks = KillSwitch()
    ks.activate("Manual emergency stop")
    print(f"   Kill switch active: {ks.is_active()}")
    
    print("\n✅ Circuit breakers test complete")
