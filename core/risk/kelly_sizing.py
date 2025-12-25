"""
Risk-Adjusted Position Sizing
Kelly Criterion and volatility-based sizing.
"""
import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RiskAdjustedSizing:
    """
    Advanced position sizing strategies.
    
    Methods:
    - Kelly Criterion
    - Volatility-adjusted sizing
    - Correlation-aware sizing
    """
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        logger.info("Risk-Adjusted Sizing initialized")
    
    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.5
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Formula: f = (p * b - q) / b
        where:
            p = win probability
            q = loss probability
            b = win/loss ratio
            
        Args:
            win_rate: Win rate (0.0 - 1.0)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive)
            kelly_fraction: Fraction of Kelly to use (0.25 - 1.0, default 0.5 for safety)
            
        Returns:
            Recommended position size as % of capital
        """
        if avg_loss == 0:
            return 0.0
        
        b = abs(avg_win / avg_loss)  # Win/loss ratio
        p = win_rate
        q = 1 - win_rate
        
        # Kelly formula
        kelly_pct = (p * b - q) / b
        
        # Apply fractional Kelly for safety
        kelly_pct *= kelly_fraction
        
        # Cap at reasonable limits (5-25%)
        kelly_pct = max(0.05, min(kelly_pct, 0.25))
        
        logger.debug(f"Kelly sizing: {kelly_pct*100:.1f}% (win_rate={win_rate:.2f}, ratio={b:.2f})")
        
        return kelly_pct
    
    def volatility_adjusted_size(
        self,
        base_risk_pct: float,
        current_volatility: float,
        avg_volatility: float
    ) -> float:
        """
        Adjust position size based on volatility.
        
        Scale down in high volatility, scale up in low volatility.
        
        Args:
            base_risk_pct: Base risk per trade (%)
            current_volatility: Current market volatility (ATR/price)
            avg_volatility: Average historical volatility
            
        Returns:
            Adjusted risk %
        """
        if avg_volatility == 0:
            return base_risk_pct
        
        # Volatility ratio
        vol_ratio = current_volatility / avg_volatility
        
        # Inverse adjustment (high vol = lower size)
        adjustment_factor = 1 / vol_ratio
        
        # Apply bounds (0.5x - 1.5x)
        adjustment_factor = max(0.5, min(adjustment_factor, 1.5))
        
        adjusted_risk = base_risk_pct * adjustment_factor
        
        logger.debug(f"Vol-adjusted risk: {adjusted_risk:.2f}% (vol ratio={vol_ratio:.2f})")
        
        return adjusted_risk
    
    def correlation_adjusted_size(
        self,
        base_risk_pct: float,
        existing_positions: int,
        avg_correlation: float = 0.7
    ) -> float:
        """
        Adjust size based on portfolio correlation.
        
        Reduce size if adding correlated position.
        
        Args:
            base_risk_pct: Base risk per trade (%)
            existing_positions: Number of open positions
            avg_correlation: Average correlation between positions
            
        Returns:
            Adjusted risk %
        """
        if existing_positions == 0:
            return base_risk_pct
        
        # Correlation penalty
        correlation_factor = 1 - (avg_correlation * 0.3)  # Max 30% reduction
        
        # Position count penalty
        position_factor = 1 / np.sqrt(existing_positions + 1)
        
        adjusted_risk = base_risk_pct * correlation_factor * position_factor
        
        logger.debug(f"Correlation-adjusted risk: {adjusted_risk:.2f}% ({existing_positions} positions)")
        
        return adjusted_risk
    
    def calculate_optimal_size(
        self,
        strategy: str = "kelly",
        **kwargs
    ) -> float:
        """
        Calculate optimal position size using specified strategy.
        
        Args:
            strategy: 'kelly', 'volatility', 'correlation', or 'combined'
            **kwargs: Strategy-specific parameters
            
        Returns:
            Recommended position size (% of capital)
        """
        if strategy == "kelly":
            return self.kelly_criterion(
                kwargs.get('win_rate', 0.5),
                kwargs.get('avg_win', 1.0),
                kwargs.get('avg_loss', 1.0),
                kwargs.get('kelly_fraction', 0.5)
            )
        
        elif strategy == "volatility":
            return self.volatility_adjusted_size(
                kwargs.get('base_risk_pct', 1.0),
                kwargs.get('current_volatility', 0.02),
                kwargs.get('avg_volatility', 0.02)
            )
        
        elif strategy == "correlation":
            return self.correlation_adjusted_size(
                kwargs.get('base_risk_pct', 1.0),
                kwargs.get('existing_positions', 0),
                kwargs.get('avg_correlation', 0.7)
            )
        
        else:
            # Combined approach
            kelly = self.kelly_criterion(
                kwargs.get('win_rate', 0.5),
                kwargs.get('avg_win', 1.0),
                kwargs.get('avg_loss', 1.0)
            )
            
            vol_adj = self.volatility_adjusted_size(
                kelly,
                kwargs.get('current_volatility', 0.02),
                kwargs.get('avg_volatility', 0.02)
            )
            
            final = self.correlation_adjusted_size(
                vol_adj,
                kwargs.get('existing_positions', 0)
            )
            
            return final
