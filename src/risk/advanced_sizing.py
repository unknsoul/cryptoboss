"""
Volatility-Adjusted Position Sizing
Scales positions inversely with market volatility.

Formula:
position_size = (equity * risk_pct) / (atr * multiplier)

High volatility → smaller positions
Low volatility → larger positions
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class VolatilityAdjustedSizing:
    """
    Risk-adjusted position sizing based on volatility.
    
    Uses ATR to measure volatility and adjusts position size accordingly.
    """
    
    def __init__(
        self,
        base_risk_pct: float = 1.0,
        atr_multiplier: float = 2.0,
        min_size_pct: float = 0.1,
        max_size_pct: float = 10.0
    ):
        """
        Initialize volatility-adjusted sizing.
        
        Args:
            base_risk_pct: Base risk per trade (%)
            atr_multiplier: Stop loss as multiple of ATR
            min_size_pct: Minimum position size (% of equity)
            max_size_pct: Maximum position size (% of equity)
        """
        self.base_risk_pct = base_risk_pct
        self.atr_multiplier = atr_multiplier
        self.min_size_pct = min_size_pct
        self.max_size_pct = max_size_pct
        
        logger.info(f"VolatilityAdjustedSizing initialized (risk: {base_risk_pct}%)")
    
    def calculate_position_size(
        self,
        equity: float,
        price: float,
        atr: float,
        volatility_regime: str = "normal"
    ) -> Dict:
        """
        Calculate position size based on volatility.
        
        Args:
            equity: Current account equity
            price: Current asset price
            atr: Average True Range
            volatility_regime: 'low', 'normal', or 'high'
            
        Returns:
            Dict with size, quantity, risk, stop_loss
        """
        # Adjust base risk based on regime
        regime_multipliers = {
            'low': 1.2,      # Increase size in low vol
            'normal': 1.0,
            'high': 0.7      # Decrease size in high vol
        }
        
        adjusted_risk_pct = self.base_risk_pct * regime_multipliers.get(volatility_regime, 1.0)
        
        # Calculate dollar risk
        dollar_risk = equity * (adjusted_risk_pct / 100)
        
        # Stop distance (ATR-based)
        stop_distance = atr * self.atr_multiplier
        stop_loss = price - stop_distance
        
        # Position size in dollar terms
        position_size_dollars = dollar_risk / stop_distance
        
        # Cap at min/max
        min_dollars = equity * (self.min_size_pct / 100)
        max_dollars = equity * (self.max_size_pct / 100)
        position_size_dollars = max(min_dollars, min(position_size_dollars, max_dollars))
        
        # Convert to quantity
        quantity = position_size_dollars / price
        
        # Actual risk (in case capped)
        actual_risk_pct = (stop_distance * quantity) / equity * 100
        
        return {
            'quantity': quantity,
            'position_size_dollars': position_size_dollars,
            'risk_pct': actual_risk_pct,
            'stop_loss': stop_loss,
            'stop_distance': stop_distance,
            'regime': volatility_regime
        }
    
    def detect_volatility_regime(
        self,
        current_atr: float,
        historical_atr_mean: float,
        threshold_pct: float = 20.0
    ) -> str:
        """
        Detect current volatility regime.
        
        Args:
            current_atr: Current ATR
            historical_atr_mean: Historical average ATR
            threshold_pct: Threshold for regime detection (%)
            
        Returns:
            'low', 'normal', or 'high'
        """
        ratio = (current_atr / historical_atr_mean - 1) * 100
        
        if ratio < -threshold_pct:
            return 'low'
        elif ratio > threshold_pct:
            return 'high'
        else:
            return 'normal'


class KillSwitch:
    """
    Emergency trading halt system.
    
    Conditions that trigger halt:
    - Maximum daily loss exceeded
    - Consecutive losses threshold
    - Maximum drawdown from peak
    """
    
    HALT_CONDITIONS = {
        'max_daily_loss_pct': -5.0,          # Stop at -5% daily
        'max_consecutive_losses': 5,          # Stop after 5 losses in a row
        'max_drawdown_pct': -10.0,           # Stop at -10% drawdown
        'min_equity_threshold_pct': 50.0     # Stop if equity drops below 50% of initial
    }
    
    def __init__(self, initial_equity: float):
        """
        Initialize kill switch.
        
        Args:
            initial_equity: Starting equity
        """
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.is_halted = False
        self.halt_reason = None
        
        self.consecutive_losses = 0
        self.daily_start_equity = initial_equity
        
        logger.info("Kill Switch initialized")
    
    def check_halt_conditions(
        self,
        current_equity: float,
        last_trade_pnl: float = None
    ) -> Dict:
        """
        Check if trading should be halted.
        
        Args:
            current_equity: Current account equity
            last_trade_pnl: P&L of last trade (optional)
            
        Returns:
            Dict with should_halt (bool) and reason (str)
        """
        if self.is_halted:
            return {
                'should_halt': True,
                'reason': self.halt_reason,
                'is_active_halt': True
            }
        
        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Update consecutive losses
        if last_trade_pnl is not None:
            if last_trade_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
        
        # Check conditions
        
        # 1. Daily loss
        daily_loss_pct = ((current_equity - self.daily_start_equity) / self.daily_start_equity) * 100
        if daily_loss_pct <= self.HALT_CONDITIONS['max_daily_loss_pct']:
            self._trigger_halt(f"Daily loss exceeded: {daily_loss_pct:.2f}%")
            return {'should_halt': True, 'reason': self.halt_reason}
        
        # 2. Consecutive losses
        if self.consecutive_losses >= self.HALT_CONDITIONS['max_consecutive_losses']:
            self._trigger_halt(f"Consecutive losses: {self.consecutive_losses}")
            return {'should_halt': True, 'reason': self.halt_reason}
        
        # 3. Drawdown
        drawdown_pct = ((current_equity - self.peak_equity) / self.peak_equity) * 100
        if drawdown_pct <= self.HALT_CONDITIONS['max_drawdown_pct']:
            self._trigger_halt(f"Max drawdown exceeded: {drawdown_pct:.2f}%")
            return {'should_halt': True, 'reason': self.halt_reason}
        
        # 4. Minimum equity
        equity_pct = (current_equity / self.initial_equity) * 100
        if equity_pct <= self.HALT_CONDITIONS['min_equity_threshold_pct']:
            self._trigger_halt(f"Equity below threshold: {equity_pct:.1f}%")
            return {'should_halt': True, 'reason': self.halt_reason}
        
        return {'should_halt': False, 'reason': None}
    
    def _trigger_halt(self, reason: str):
        """Trigger emergency halt."""
        self.is_halted = True
        self.halt_reason = reason
        logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
    
    def reset_halt(self):
        """Reset halt (manual override)."""
        logger.warning("Kill Switch reset (manual override)")
        self.is_halted = False
        self.halt_reason = None
        self.consecutive_losses = 0
    
    def reset_daily(self, current_equity: float):
        """Reset daily tracking (call at market open)."""
        self.daily_start_equity = current_equity
        logger.info(f"Kill Switch daily reset - Starting equity: ${current_equity:,.2f}")
    
    def get_status(self) -> Dict:
        """Get current kill switch status."""
        return {
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason,
            'consecutive_losses': self.consecutive_losses,
            'peak_equity': self.peak_equity,
            'daily_start_equity': self.daily_start_equity
        }
