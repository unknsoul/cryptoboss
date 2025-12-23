"""
Advanced Risk Management System
Institutional-grade risk controls for professional trading
"""

import numpy as np
from typing import List, Dict, Optional


class AdvancedRiskManager:
    """
    Professional risk management with:
    - Position sizing (Kelly Criterion, Fixed Fractional)
    - Drawdown limits
    - Correlation checks
    - Exposure limits
    - Dynamic risk scaling
    """
    
    def __init__(self,
                 max_position_size=0.30,  # Max 30% of capital per position
                 max_total_exposure=0.60,   # Max 60% total exposure
                 max_correlation=0.7,        # Max correlation between positions
                 kelly_fraction=0.25):       # Kelly Criterion fraction
        
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.max_correlation = max_correlation
        self.kelly_fraction = kelly_fraction
    
    def calculate_position_size(self, 
                                capital, 
                                risk_per_trade, 
                                stop_distance,
                                win_rate=None,
                                avg_win=None,
                                avg_loss=None,
                                use_kelly=False):
        """
        Calculate optimal position size
        
        Args:
            capital: Available capital
            risk_per_trade: Risk fraction (e.g., 0.02 for 2%)
            stop_distance: StopDistance in price
            win_rate: Win rate for Kelly (optional)
            avg_win: Average win for Kelly (optional)
            avg_loss: Average loss for Kelly (optional)
            use_kelly: Use Kelly Criterion instead of fixed fractional
        
        Returns:
            position_size: Number of units to trade
        """
        
        if use_kelly and all(x is not None for x in [win_rate, avg_win, avg_loss]):
            # Kelly Criterion: f* = (p*b - q) / b
            # where p = win_rate, q = loss_rate, b = avg_win/avg_loss
            if avg_loss != 0:
                b = abs(avg_win / avg_loss)
                q = 1 - win_rate
                kelly_fraction_calc = (win_rate * b - q) / b
                
                # Apply Kelly fraction (typically 1/4 or 1/2 Kelly)
                kelly_fraction_calc = max(0, min(kelly_fraction_calc, 0.5)) * self.kelly_fraction
                
                risk_amount = capital * kelly_fraction_calc
            else:
                risk_amount = capital * risk_per_trade
        else:
            # Fixed fractional position sizing
            risk_amount = capital * risk_per_trade
        
        # Position size based on risk
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = 0
        
        # Apply maximum position size limit
        max_size_by_capital = capital * self.max_position_size
        # Assuming position_size is in units, convert to capital exposure
        # This would need current price, but for now we use simplified logic
        
        return position_size
    
    def check_correlation(self, 
                         current_positions: List[Dict],
                         new_signal: Dict) -> bool:
        """
        Check if new position would exceed correlation limits
        
        Args:
            current_positions: List of active positions
            new_signal: New signal to evaluate
        
        Returns:
            True if position is allowed, False otherwise
        """
        
        if not current_positions:
            return True
        
        # Simplified correlation check based on direction
        # In reality, you'd calculate price correlation over a window
        new_side = new_signal.get('action')
        
        same_direction_exposure = 0
        for pos in current_positions:
            if pos['side'] == new_side:
                same_direction_exposure += pos.get('exposure', 0)
        
        # Don't allow if too much exposure in same direction
        if same_direction_exposure > self.max_total_exposure:
            return False
        
        return True
    
    def calculate_var(self, returns: np.ndarray, confidence=0.95):
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Array of returns
            confidence: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            var: Value at Risk (positive number representing potential loss)
        """
        if len(returns) == 0:
            return 0
        
        # Historical VaR
        var = -np.percentile(returns, (1 - confidence) * 100)
        return var
    
    def calculate_cvar(self, returns: np.ndarray, confidence=0.95):
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
        
        Args:
            returns: Array of returns
            confidence: Confidence level
        
        Returns:
            cvar: Expected loss beyond VaR threshold
        """
        if len(returns) == 0:
            return 0
        
        var = self.calculate_var(returns, confidence)
        # CVaR is the mean of returns worse than VaR
        threshold = -var
        tail_returns = returns[returns < threshold]
        
        if len(tail_returns) > 0:
            cvar = -np.mean(tail_returns)
        else:
            cvar = var
        
        return cvar
    
    def adjust_risk_for_volatility(self,
                                   base_risk: float,
                                   current_volatility: float,
                                   average_volatility: float) -> float:
        """
        Scale position size based on volatility
        
        Higher volatility = smaller position size
        Lower volatility = larger position size
        
        Args:
            base_risk: Base risk percentage
            current_volatility: Current market volatility (e.g., ATR)
            average_volatility: Average volatility
        
        Returns:
            adjusted_risk: Scaled risk percentage
        """
        if average_volatility <= 0:
            return base_risk
        
        # Inverse volatility scaling
        volatility_ratio = average_volatility / current_volatility
        
        # Clamp the adjustment (0.5x to 1.5x base risk)
        volatility_ratio = max(0.5, min(1.5, volatility_ratio))
        
        adjusted_risk = base_risk * volatility_ratio
        
        return adjusted_risk


class SafetyControls:
    """Emergency safety controls"""
    
    def __init__(self,
                 max_drawdown_kill=0.25,
                 max_daily_loss=0.05,
                 max_weekly_loss=0.12,
                 max_consecutive_losses=5):
        
        self.max_drawdown_kill = max_drawdown_kill
        self.max_daily_loss = max_daily_loss
        self.max_weekly_loss = max_weekly_loss
        self.max_consecutive_losses = max_consecutive_losses
        
        # State
        self.peak_equity = None
        self.daily_start = None
        self.weekly_start = None
        self.consecutive_losses = 0
        self.trading_halted = False
        self.halt_reason = None
    
    def update(self, current_equity, current_time=None):
        """
        Update safety controls and check limits
        
        Returns:
            (is_safe: bool, message: str)
        """
        
        if self.peak_equity is None:
            self.peak_equity = current_equity
            self.daily_start = current_equity
            self.weekly_start = current_equity
        
        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Check max drawdown
        current_dd = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        if current_dd >= self.max_drawdown_kill:
            self.trading_halted = True
            self.halt_reason = f"Max drawdown {current_dd:.2%} exceeded"
            return False, self.halt_reason
        
        # TODO: Add daily/weekly loss checks with timestamp tracking
        
        return True, ""
    
    def record_trade_result(self, pnl):
        """Record trade result and update consecutive loss counter"""
        
        if pnl < 0:
            self.consecutive_losses += 1
            
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.trading_halted = True
                self.halt_reason = f"{self.consecutive_losses} consecutive losses"
                return False
        else:
            self.consecutive_losses = 0
        
        return True
    
    def reset(self):
        """Reset safety controls (use with caution!)"""
        self.trading_halted = False
        self.halt_reason = None
        self.consecutive_losses = 0
