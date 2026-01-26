"""
Trade Permission Filter - The Final Gate

This is the last decision point before trade execution.
Even if context is good and bias is established, permission can still be denied.

Checks (all must pass):
1. Spread threshold
2. Liquidity depth
3. Volatility bounds
4. Time-of-day filtering
5. Daily drawdown limit
6. Consecutive loss limit
7. Market context approval
8. Bias alignment

Critical: Any failed check blocks NEW trades but allows managing open positions.
"""

import logging
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum

from .market_context_engine import MarketContext, MarketRegime
from .bias_engine import BiasState, TradeBias
from .risk_guardian import RiskGuardian

logger = logging.getLogger(__name__)


class PermissionDenialReason(Enum):
    """Reasons for permission denial."""
    SPREAD_TOO_WIDE = "spread_too_wide"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"
    VOLATILITY_TOO_HIGH = "volatility_too_high"
    VOLATILITY_TOO_LOW = "volatility_too_low"
    TIME_BLOCKED = "time_blocked"
    DAILY_DRAWDOWN = "daily_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    CONTEXT_BLOCKED = "context_blocked"
    BIAS_NEUTRAL = "bias_neutral"
    BIAS_MISALIGNED = "bias_misaligned"
    RISK_GUARDIAN_BLOCKED = "risk_guardian_blocked"


@dataclass
class PermissionResult:
    """Result of permission check."""
    approved: bool
    reason: str
    denial_category: Optional[PermissionDenialReason]
    checks_passed: Dict[str, bool]
    metadata: Dict


class TradePermissionFilter:
    """
    Trade Permission Filter - Final gate before execution.
    
    All checks must pass for permission to be granted.
    This is the last line of defense before capital is risked.
    
    Usage:
        permission_filter = TradePermissionFilter(risk_guardian)
        
        permission = permission_filter.check_permission(
            context=market_context,
            bias=bias_state,
            direction="LONG",
            orderbook=current_orderbook
        )
        
        if not permission.approved:
            logger.info(f"Trade blocked: {permission.reason}")
            return
    """
    
    def __init__(
        self,
        risk_guardian: RiskGuardian,
        
        # Volatility bounds (ATR percentile)
        max_volatility_percentile: float = 90.0,
        min_volatility_percentile: float = 10.0,
        
        # Time filtering (UTC hours)
        blocked_hours: list = None,  # Hours to block trading (e.g., [0, 1, 2, 3, 4, 5])
        
        # Consecutive losses
        max_consecutive_losses: int = 3,
    ):
        self.risk_guardian = risk_guardian
        self.max_volatility_percentile = max_volatility_percentile
        self.min_volatility_percentile = min_volatility_percentile
        self.blocked_hours = blocked_hours or []
        self.max_consecutive_losses = max_consecutive_losses
        
        # Track consecutive losses
        self.consecutive_losses = 0
        self.last_trade_result: Optional[str] = None
        
        logger.info(
            f"TradePermissionFilter initialized: "
            f"max_vol={max_volatility_percentile}, "
            f"blocked_hours={blocked_hours}"
        )
    
    def check_permission(
        self,
        context: MarketContext,
        bias: BiasState,
        direction: str,
        orderbook: Optional[Dict] = None,
    ) -> PermissionResult:
        """
        Check if new trade is permitted.
        
        Args:
            context: Current market context
            bias: Current directional bias
            direction: Proposed trade direction ('LONG' or 'SHORT')
            orderbook: Current orderbook snapshot
            
        Returns:
            PermissionResult with approval status and reasoning
        """
        checks = {}
        
        # 1. Market context check
        passed, reason = self._check_market_context(context)
        checks['market_context'] = passed
        if not passed:
            return self._create_denial(
                PermissionDenialReason.CONTEXT_BLOCKED,
                reason,
                checks
            )
        
        # 2. Bias check
        passed, reason = self._check_bias(bias, direction)
        checks['bias'] = passed
        if not passed:
            denial_reason = (
                PermissionDenialReason.BIAS_NEUTRAL if bias.bias == TradeBias.NEUTRAL
                else PermissionDenialReason.BIAS_MISALIGNED
            )
            return self._create_denial(denial_reason, reason, checks)
        
        # 3. Spread check (uses context liquidity)
        passed, reason = self._check_spread(context)
        checks['spread'] = passed
        if not passed:
            return self._create_denial(
                PermissionDenialReason.SPREAD_TOO_WIDE,
                reason,
                checks
            )
        
        # 4. Liquidity check
        passed, reason = self._check_liquidity(context)
        checks['liquidity'] = passed
        if not passed:
            return self._create_denial(
                PermissionDenialReason.INSUFFICIENT_LIQUIDITY,
                reason,
                checks
            )
        
        # 5. Volatility bounds check
        passed, reason = self._check_volatility_bounds(context)
        checks['volatility'] = passed
        if not passed:
            denial_reason = (
                PermissionDenialReason.VOLATILITY_TOO_HIGH
                if context.atr_percentile > self.max_volatility_percentile
                else PermissionDenialReason.VOLATILITY_TOO_LOW
            )
            return self._create_denial(denial_reason, reason, checks)
        
        # 6. Time-of-day check
        passed, reason = self._check_time_of_day()
        checks['time_of_day'] = passed
        if not passed:
            return self._create_denial(
                PermissionDenialReason.TIME_BLOCKED,
                reason,
                checks
            )
        
        # 7. Daily drawdown check (via RiskGuardian)
        passed, reason = self._check_daily_drawdown()
        checks['daily_drawdown'] = passed
        if not passed:
            return self._create_denial(
                PermissionDenialReason.DAILY_DRAWDOWN,
                reason,
                checks
            )
        
        # 8. Consecutive losses check
        passed, reason = self._check_consecutive_losses()
        checks['consecutive_losses'] = passed
        if not passed:
            return self._create_denial(
                PermissionDenialReason.CONSECUTIVE_LOSSES,
                reason,
                checks
            )
        
        # All checks passed
        return PermissionResult(
            approved=True,
            reason=f"All permission checks passed for {direction} trade",
            denial_category=None,
            checks_passed=checks,
            metadata={
                'context_regime': context.regime.value,
                'bias': bias.bias.value,
                'bias_conviction': bias.conviction,
                'direction': direction
            }
        )
    
    def record_trade_result(self, won: bool):
        """
        Record trade result for consecutive loss tracking.
        
        Args:
            won: True if trade was profitable, False if loss
        """
        if won:
            self.consecutive_losses = 0
            self.last_trade_result = "win"
            logger.debug("Trade win recorded, consecutive losses reset")
        else:
            self.consecutive_losses += 1
            self.last_trade_result = "loss"
            logger.warning(
                f"Trade loss recorded, consecutive losses: {self.consecutive_losses}"
            )
    
    def _check_market_context(self, context: MarketContext) -> Tuple[bool, str]:
        """Check if market context allows trading."""
        if not context.trading_allowed:
            return False, f"Context blocked: {context.reason}"
        
        if context.regime == MarketRegime.NO_TRADE:
            return False, "Context regime is NO_TRADE"
        
        return True, "Context approved"
    
    def _check_bias(self, bias: BiasState, direction: str) -> Tuple[bool, str]:
        """Check bias alignment with proposed direction."""
        if bias.bias == TradeBias.NEUTRAL:
            return False, "Bias is NEUTRAL - no directional conviction"
        
        direction = direction.upper()
        
        if direction == "LONG" and bias.bias != TradeBias.LONG_BIAS:
            return False, f"Direction LONG conflicts with bias {bias.bias.value}"
        
        if direction == "SHORT" and bias.bias != TradeBias.SHORT_BIAS:
            return False, f"Direction SHORT conflicts with bias {bias.bias.value}"
        
        return True, f"Direction aligned with bias {bias.bias.value}"
    
    def _check_spread(self, context: MarketContext) -> Tuple[bool, str]:
        """Check if spread is acceptable."""
        # Spread already checked in context liquidity
        if not context.liquidity.is_acceptable:
            if "Spread" in context.liquidity.reason:
                return False, context.liquidity.reason
        
        return True, "Spread acceptable"
    
    def _check_liquidity(self, context: MarketContext) -> Tuple[bool, str]:
        """Check if liquidity is sufficient."""
        if not context.liquidity.is_acceptable:
            return False, context.liquidity.reason
        
        return True, "Liquidity acceptable"
    
    def _check_volatility_bounds(self, context: MarketContext) -> Tuple[bool, str]:
        """Check if volatility is within acceptable bounds."""
        atr_pct = context.atr_percentile
        
        if atr_pct > self.max_volatility_percentile:
            return False, (
                f"Volatility too high: ATR at {atr_pct:.1f} percentile "
                f"(max {self.max_volatility_percentile})"
            )
        
        if atr_pct < self.min_volatility_percentile:
            return False, (
                f"Volatility too low: ATR at {atr_pct:.1f} percentile "
                f"(min {self.min_volatility_percentile})"
            )
        
        return True, f"Volatility acceptable ({atr_pct:.1f} percentile)"
    
    def _check_time_of_day(self) -> Tuple[bool, str]:
        """Check if current time allows trading."""
        if not self.blocked_hours:
            return True, "No time restrictions"
        
        current_hour = datetime.utcnow().hour
        
        if current_hour in self.blocked_hours:
            return False, f"Trading blocked during hour {current_hour} UTC"
        
        return True, "Time allowed"
    
    def _check_daily_drawdown(self) -> Tuple[bool, str]:
        """Check if daily drawdown limit has been hit."""
        risk_report = self.risk_guardian.get_risk_report()
        
        if risk_report.get('emergency_stop_active'):
            return False, "Emergency stop is active"
        
        daily_pnl = risk_report.get('daily_pnl', 0)
        daily_pnl_pct = risk_report.get('daily_pnl_pct', 0)
        
        max_loss_pct = self.risk_guardian.limits.max_daily_loss_pct
        
        if daily_pnl_pct <= -max_loss_pct:
            return False, (
                f"Daily drawdown limit hit: {daily_pnl_pct:.2f}% "
                f"(max {max_loss_pct}%)"
            )
        
        return True, f"Daily drawdown OK ({daily_pnl_pct:.2f}%)"
    
    def _check_consecutive_losses(self) -> Tuple[bool, str]:
        """Check if consecutive loss limit has been hit."""
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, (
                f"Consecutive loss limit hit: {self.consecutive_losses} losses "
                f"(max {self.max_consecutive_losses})"
            )
        
        return True, f"Consecutive losses OK ({self.consecutive_losses}/{self.max_consecutive_losses})"
    
    def _create_denial(
        self,
        category: PermissionDenialReason,
        reason: str,
        checks: Dict[str, bool]
    ) -> PermissionResult:
        """Create a permission denial result."""
        return PermissionResult(
            approved=False,
            reason=reason,
            denial_category=category,
            checks_passed=checks,
            metadata={'denial_reason': category.value}
        )
    
    def get_permission_stats(self) -> Dict:
        """Get permission filter statistics."""
        return {
            "consecutive_losses": self.consecutive_losses,
            "last_trade_result": self.last_trade_result,
            "max_consecutive_losses": self.max_consecutive_losses,
            "blocked_hours": self.blocked_hours,
            "volatility_bounds": {
                "min_percentile": self.min_volatility_percentile,
                "max_percentile": self.max_volatility_percentile
            }
        }


# Singleton instance
_permission_filter: Optional[TradePermissionFilter] = None


def get_permission_filter(risk_guardian: RiskGuardian = None) -> TradePermissionFilter:
    """Get the global TradePermissionFilter instance."""
    global _permission_filter
    if _permission_filter is None:
        if risk_guardian is None:
            from .risk_guardian import get_risk_guardian
            risk_guardian = get_risk_guardian()
        _permission_filter = TradePermissionFilter(risk_guardian)
    return _permission_filter
