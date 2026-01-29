"""
Capital Allocation Governor - v10.0 Component

Dynamic capital allocation based on market context:
- Each context has a maximum allocation %
- High volatility automatically reduces allocation
- NO_TRADE context allocates zero capital
- Allocation adjusts in real-time

Ensures risk-appropriate position sizing across all conditions.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AllocationContext(Enum):
    """Market contexts with allocation limits."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    NO_TRADE = "no_trade"


# Default allocation percentages per context
DEFAULT_CONTEXT_ALLOCATIONS: Dict[AllocationContext, float] = {
    AllocationContext.TRENDING_UP: 1.0,      # 100% available
    AllocationContext.TRENDING_DOWN: 1.0,    # 100% available
    AllocationContext.RANGING: 0.75,         # 75% max
    AllocationContext.HIGH_VOLATILITY: 0.30, # 30% max
    AllocationContext.NO_TRADE: 0.0,         # 0% - no trading
}


@dataclass
class AllocationSnapshot:
    """Current allocation state."""
    timestamp: datetime
    context: AllocationContext
    
    # Allocation factors
    base_allocation: float      # From context table
    volatility_modifier: float  # Reduces in high vol
    drawdown_modifier: float    # Reduces after losses
    health_modifier: float      # Exchange health impact
    
    # Final allocation
    effective_allocation: float
    max_position_size: float
    
    # Limits
    portfolio_value: float
    current_exposure: float
    available_capital: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'context': self.context.value,
            'base_allocation': self.base_allocation,
            'volatility_modifier': self.volatility_modifier,
            'drawdown_modifier': self.drawdown_modifier,
            'health_modifier': self.health_modifier,
            'effective_allocation': self.effective_allocation,
            'max_position_size': self.max_position_size,
            'portfolio_value': self.portfolio_value,
            'current_exposure': self.current_exposure,
            'available_capital': self.available_capital
        }


class CapitalAllocationGovernor:
    """
    Governs dynamic capital allocation based on market conditions.
    
    Allocation is determined by:
    1. Base allocation from context (0-100%)
    2. Volatility modifier (reduces in high vol)
    3. Drawdown modifier (reduces after losses)
    4. Exchange health modifier
    
    Final allocation = base × vol_mod × dd_mod × health_mod
    
    Usage:
        governor = CapitalAllocationGovernor(portfolio_value=10000)
        
        # Get current allocation
        allocation = governor.get_allocation(
            context="ranging",
            volatility_percentile=65,
            daily_drawdown_pct=-2.5,
            exchange_health=0.9
        )
        
        # Use for position sizing
        max_size = allocation.max_position_size
    """
    
    # Volatility thresholds
    VOL_REDUCTION_THRESHOLD = 70  # Start reducing at 70th percentile
    VOL_CRITICAL_THRESHOLD = 90   # Aggressive reduction above 90th
    
    # Drawdown thresholds  
    DD_REDUCTION_THRESHOLD = 3.0  # Start reducing at 3% drawdown
    DD_CRITICAL_THRESHOLD = 5.0   # Aggressive reduction at 5%
    
    def __init__(
        self,
        portfolio_value: float = 10000.0,
        context_allocations: Optional[Dict[AllocationContext, float]] = None,
        min_allocation: float = 0.10  # Never go below 10% unless NO_TRADE
    ):
        self.portfolio_value = portfolio_value
        self.context_allocations = context_allocations or DEFAULT_CONTEXT_ALLOCATIONS.copy()
        self.min_allocation = min_allocation
        
        self._current_exposure: float = 0.0
        self._last_allocation: Optional[AllocationSnapshot] = None
        
        logger.info(
            f"CapitalAllocationGovernor initialized: "
            f"portfolio=${portfolio_value:,.0f}"
        )
    
    def get_allocation(
        self,
        context: str,
        volatility_percentile: float = 50.0,
        daily_drawdown_pct: float = 0.0,
        exchange_health: float = 1.0,
        current_exposure: Optional[float] = None
    ) -> AllocationSnapshot:
        """
        Calculate current allocation based on all factors.
        
        Args:
            context: Current market context
            volatility_percentile: ATR percentile (0-100)
            daily_drawdown_pct: Daily drawdown as positive percentage
            exchange_health: Exchange health score (0.0-1.0)
            current_exposure: Current position exposure in dollars
            
        Returns:
            AllocationSnapshot with effective allocation
        """
        # Parse context
        try:
            ctx = AllocationContext(context.lower())
        except ValueError:
            logger.warning(f"Unknown context '{context}', defaulting to RANGING")
            ctx = AllocationContext.RANGING
        
        # 1. Get base allocation from context
        base_allocation = self.context_allocations.get(ctx, 0.75)
        
        # 2. Calculate volatility modifier
        vol_modifier = self._calculate_volatility_modifier(volatility_percentile)
        
        # 3. Calculate drawdown modifier
        dd_modifier = self._calculate_drawdown_modifier(abs(daily_drawdown_pct))
        
        # 4. Exchange health modifier
        health_modifier = self._calculate_health_modifier(exchange_health)
        
        # 5. v11.0: DrawdownGovernor integration for multi-timeframe control
        drawdown_multiplier = 1.0
        in_defensive_mode = False
        try:
            from .drawdown_governor import get_drawdown_governor
            dd_governor = get_drawdown_governor()
            
            # Update equity in drawdown governor
            dd_governor.update_equity(self.portfolio_value)
            
            # Get size multiplier from drawdown governor
            drawdown_multiplier = dd_governor.get_size_multiplier()
            in_defensive_mode = dd_governor.is_in_defensive_mode()
            
            if in_defensive_mode:
                logger.warning(
                    f"DrawdownGovernor DEFENSIVE MODE active - "
                    f"size multiplier: {drawdown_multiplier:.2f}"
                )
        except ImportError:
            pass  # DrawdownGovernor not available
        except Exception as e:
            logger.debug(f"DrawdownGovernor not available: {e}")
        
        # 6. Calculate effective allocation
        if ctx == AllocationContext.NO_TRADE:
            effective = 0.0
        else:
            # Combine all modifiers including drawdown governor
            effective = base_allocation * vol_modifier * dd_modifier * health_modifier * drawdown_multiplier
            effective = max(self.min_allocation, effective) if not in_defensive_mode else effective
        
        # 7. Calculate max position size
        available = self.portfolio_value * effective
        
        if current_exposure is not None:
            self._current_exposure = current_exposure
        
        remaining = max(0, available - self._current_exposure)
        
        snapshot = AllocationSnapshot(
            timestamp=datetime.now(),
            context=ctx,
            base_allocation=base_allocation,
            volatility_modifier=vol_modifier,
            drawdown_modifier=dd_modifier * drawdown_multiplier,  # Include governor modifier
            health_modifier=health_modifier,
            effective_allocation=effective,
            max_position_size=remaining,
            portfolio_value=self.portfolio_value,
            current_exposure=self._current_exposure,
            available_capital=remaining
        )
        
        self._last_allocation = snapshot
        
        logger.debug(
            f"Allocation: {ctx.value} -> {effective:.1%} "
            f"(base={base_allocation:.0%}, vol={vol_modifier:.2f}, "
            f"dd={dd_modifier:.2f}, health={health_modifier:.2f}, "
            f"ddGov={drawdown_multiplier:.2f})"
        )
        
        return snapshot
    
    def set_portfolio_value(self, value: float):
        """Update portfolio value."""
        self.portfolio_value = value
        logger.info(f"Portfolio value updated: ${value:,.0f}")
    
    def update_exposure(self, exposure: float):
        """Update current exposure."""
        self._current_exposure = exposure
    
    def add_exposure(self, amount: float):
        """Add to current exposure (new position)."""
        self._current_exposure += amount
    
    def remove_exposure(self, amount: float):
        """Remove from exposure (closed position)."""
        self._current_exposure = max(0, self._current_exposure - amount)
    
    def get_max_new_position_size(
        self,
        context: str,
        volatility_percentile: float = 50.0,
        daily_drawdown_pct: float = 0.0,
        exchange_health: float = 1.0
    ) -> float:
        """
        Get maximum size for a new position.
        
        Convenience method that returns just the number.
        """
        allocation = self.get_allocation(
            context, volatility_percentile, daily_drawdown_pct, exchange_health
        )
        return allocation.max_position_size
    
    def can_trade(self, required_size: float) -> tuple[bool, str]:
        """
        Check if a trade of given size is allowed.
        
        Returns: (allowed, reason)
        """
        if self._last_allocation is None:
            return False, "Allocation not calculated yet"
        
        if self._last_allocation.context == AllocationContext.NO_TRADE:
            return False, "NO_TRADE context - trading blocked"
        
        if self._last_allocation.effective_allocation <= 0:
            return False, "Zero allocation - trading blocked"
        
        if required_size > self._last_allocation.available_capital:
            return False, (
                f"Size ${required_size:,.0f} exceeds available "
                f"${self._last_allocation.available_capital:,.0f}"
            )
        
        return True, "Trade size within allocation limits"
    
    def veto_trade(
        self,
        proposed_size: float,
        context: str,
        volatility_percentile: float = 50.0,
        daily_drawdown_pct: float = 0.0,
        exchange_health: float = 1.0
    ) -> tuple[bool, str, float]:
        """
        v10.1-FINAL: VETO trade based on capital constraints.
        
        Returns: (allowed, reason, effective_size)
        
        If effective_size is 0, trade is VETOED.
        """
        # Get fresh allocation
        allocation = self.get_allocation(
            context=context,
            volatility_percentile=volatility_percentile,
            daily_drawdown_pct=daily_drawdown_pct,
            exchange_health=exchange_health
        )
        
        # VETO condition 1: NO_TRADE context
        if allocation.context == AllocationContext.NO_TRADE:
            logger.warning(f"Capital Governor VETO: NO_TRADE context")
            return False, "VETO: NO_TRADE context allocates zero capital", 0.0
        
        # VETO condition 2: Zero effective allocation
        if allocation.effective_allocation <= 0:
            logger.warning(f"Capital Governor VETO: zero allocation")
            return False, "VETO: Effective allocation is zero", 0.0
        
        # VETO condition 3: No available capital
        if allocation.available_capital <= 0:
            logger.warning(f"Capital Governor VETO: no available capital")
            return False, "VETO: No available capital for new positions", 0.0
        
        # Calculate effective size
        effective_size = min(proposed_size, allocation.available_capital)
        
        # VETO condition 4: Effective size too small
        if effective_size <= 0:
            logger.warning(f"Capital Governor VETO: effective size is zero")
            return False, "VETO: Effective size would be zero", 0.0
        
        # Approved with possible reduction
        if effective_size < proposed_size:
            logger.info(
                f"Capital Governor: Size reduced from ${proposed_size:,.0f} "
                f"to ${effective_size:,.0f}"
            )
            return True, f"Approved with reduction to ${effective_size:,.0f}", effective_size
        
        return True, "Approved at full size", effective_size
    
    def _calculate_volatility_modifier(self, vol_percentile: float) -> float:
        """Calculate allocation modifier based on volatility."""
        if vol_percentile < self.VOL_REDUCTION_THRESHOLD:
            return 1.0
        elif vol_percentile < self.VOL_CRITICAL_THRESHOLD:
            # Linear reduction from 1.0 to 0.5
            reduction = (vol_percentile - self.VOL_REDUCTION_THRESHOLD) / 20
            return max(0.5, 1.0 - reduction * 0.5)
        else:
            # Aggressive reduction above critical
            return 0.3
    
    def _calculate_drawdown_modifier(self, dd_pct: float) -> float:
        """Calculate allocation modifier based on drawdown."""
        if dd_pct < self.DD_REDUCTION_THRESHOLD:
            return 1.0
        elif dd_pct < self.DD_CRITICAL_THRESHOLD:
            # Linear reduction
            reduction = (dd_pct - self.DD_REDUCTION_THRESHOLD) / 2
            return max(0.5, 1.0 - reduction * 0.5)
        else:
            # Aggressive reduction
            return 0.3
    
    def _calculate_health_modifier(self, health: float) -> float:
        """Calculate allocation modifier based on exchange health."""
        if health >= 0.9:
            return 1.0
        elif health >= 0.7:
            return 0.7
        elif health >= 0.5:
            return 0.4
        else:
            return 0.0  # Block trading on very low health
    
    def get_current_snapshot(self) -> Optional[AllocationSnapshot]:
        """Get the most recent allocation snapshot."""
        return self._last_allocation
    
    def configure_context_allocation(
        self,
        context: AllocationContext,
        allocation: float
    ):
        """Configure allocation percentage for a context."""
        self.context_allocations[context] = max(0.0, min(1.0, allocation))
        logger.info(f"Context {context.value} allocation set to {allocation:.0%}")


# Singleton instance
_capital_governor: Optional[CapitalAllocationGovernor] = None


def get_capital_governor(portfolio_value: float = 10000.0) -> CapitalAllocationGovernor:
    """Get global CapitalAllocationGovernor instance."""
    global _capital_governor
    if _capital_governor is None:
        _capital_governor = CapitalAllocationGovernor(portfolio_value=portfolio_value)
    return _capital_governor
