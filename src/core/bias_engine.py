"""
Bias Engine - Directional Conviction System

Determines market bias BEFORE entry logic executes.
Bias restricts which directions strategies can propose trades.

Key Principles:
- Bias is determined by higher timeframes, not lower timeframes
- NEUTRAL bias disables all entries
- Bias cannot flip frequently (stability window)
- Bias is advisory but enforced

Architecture: Market Context → Bias → Permission → Execution
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from .market_context_engine import MarketContext, MarketRegime

logger = logging.getLogger(__name__)


class TradeBias(Enum):
    """Directional bias for trading."""
    LONG_BIAS = "long_bias"      # Favor long entries
    SHORT_BIAS = "short_bias"    # Favor short entries
    NEUTRAL = "neutral"          # No directional conviction


@dataclass
class BiasState:
    """Current bias state and metadata."""
    bias: TradeBias
    conviction: float  # 0.0 to 1.0 (strength of bias)
    timestamp: datetime
    
    # Reasoning
    higher_tf_trend: str  # 'up', 'down', 'neutral'
    momentum_direction: str  # 'bullish', 'bearish', 'neutral'
    volatility_expansion: bool
    
    # Stability tracking
    last_flip_time: Optional[datetime]
    flips_in_window: int
    
    # Metadata
    reason: str
    metadata: Dict


class BiasEngine:
    """
    Bias Engine - Determines directional preference.
    
    Bias is determined INDEPENDENTLY of strategies and acts as
    a constraint on which directions can be traded.
    
    Usage:
        bias_engine = BiasEngine()
        
        # Get current bias
        bias_state = bias_engine.get_current_bias(df, context)
        
        # Check if direction is allowed
        if bias_state.bias == TradeBias.NEUTRAL:
            logger.info("No directional conviction - no trading")
            return
        
        # Filter strategy proposals by bias
        if proposal.direction == "LONG" and bias_state.bias != TradeBias.LONG_BIAS:
            logger.info("Proposal rejected: direction conflicts with bias")
            return
    """
    
    def __init__(
        self,
        # Bias stability (prevents rapid flipping)
        min_flip_interval_hours: int = 4,
        max_flips_per_day: int = 3,
        
        # Conviction thresholds
        min_conviction_to_bias: float = 0.6,  # Need 60%+ conviction to have bias
        
        # Higher timeframe weight
        htf_weight: float = 0.7,  # 70% weight on higher timeframe
        ltf_weight: float = 0.3,  # 30% weight on lower timeframe
        
        # Momentum thresholds
        momentum_lookback_hours: int = 24,
        momentum_threshold_pct: float = 2.0,  # Need 2%+ move for momentum
    ):
        self.min_flip_interval_hours = min_flip_interval_hours
        self.max_flips_per_day = max_flips_per_day
        self.min_conviction_to_bias = min_conviction_to_bias
        
        self.htf_weight = htf_weight
        self.ltf_weight = ltf_weight
        
        self.momentum_lookback_hours = momentum_lookback_hours
        self.momentum_threshold_pct = momentum_threshold_pct
        
        # State tracking
        self.current_bias: Optional[BiasState] = None
        self.bias_history: list = []
        
        logger.info(
            f"BiasEngine initialized: "
            f"min_flip_interval={min_flip_interval_hours}h, "
            f"min_conviction={min_conviction_to_bias}"
        )
    
    def get_current_bias(
        self,
        df: pd.DataFrame,
        context: MarketContext,
        volume_profile: Optional[Dict] = None
    ) -> BiasState:
        """
        Get current directional bias.
        
        Args:
            df: OHLCV dataframe (1h or higher recommended)
            context: Current market context from MarketContextEngine
            volume_profile: Optional volume analysis
            
        Returns:
            BiasState with current bias and conviction
        """
        timestamp = datetime.now()
        
        # If context blocks trading, return NEUTRAL bias
        if not context.trading_allowed:
            return self._create_neutral_bias(
                timestamp,
                f"Context blocked trading: {context.reason}"
            )
        
        # 1. Analyze higher timeframe trend
        htf_trend = self._analyze_higher_timeframe_trend(context)
        
        # 2. Analyze momentum
        momentum_direction = self._analyze_momentum(df)
        
        # 3. Check for volatility expansion
        vol_expansion = self._check_volatility_expansion(context)
        
        # 4. Calculate conviction score
        conviction = self._calculate_conviction(
            htf_trend, momentum_direction, vol_expansion, context
        )
        
        # 5. Determine bias from conviction
        bias = self._determine_bias(
            htf_trend, momentum_direction, conviction
        )
        
        # 6. Check stability constraints
        bias, reason = self._apply_stability_constraints(
            bias, conviction, timestamp
        )
        
        # 7. Create bias state
        bias_state = BiasState(
            bias=bias,
            conviction=conviction,
            timestamp=timestamp,
            higher_tf_trend=htf_trend,
            momentum_direction=momentum_direction,
            volatility_expansion=vol_expansion,
            last_flip_time=self.current_bias.last_flip_time if self.current_bias else None,
            flips_in_window=self._count_recent_flips(timestamp),
            reason=reason,
            metadata={
                'context_regime': context.regime.value,
                'trend_alignment': f"{context.trend_1h}/{context.trend_4h}/{context.trend_1d}"
            }
        )
        
        # Update state
        self._update_bias_state(bias_state)
        
        logger.debug(
            f"Bias: {bias.value} (conviction: {conviction:.2f}) | "
            f"HTF: {htf_trend} | Momentum: {momentum_direction} | "
            f"Reason: {reason}"
        )
        
        return bias_state
    
    def is_direction_allowed(
        self,
        direction: str,
        bias_state: BiasState
    ) -> Tuple[bool, str]:
        """
        Check if a trade direction is allowed given current bias.
        
        Args:
            direction: 'LONG' or 'SHORT'
            bias_state: Current bias state
            
        Returns:
            (allowed, reason)
        """
        direction = direction.upper()
        
        # Neutral bias blocks all directions
        if bias_state.bias == TradeBias.NEUTRAL:
            return False, "Bias is NEUTRAL - no directional conviction"
        
        # Long bias allows only longs
        if bias_state.bias == TradeBias.LONG_BIAS:
            if direction == "LONG":
                return True, f"Aligned with LONG bias (conviction: {bias_state.conviction:.2f})"
            else:
                return False, f"Direction conflicts with LONG bias"
        
        # Short bias allows only shorts
        if bias_state.bias == TradeBias.SHORT_BIAS:
            if direction == "SHORT":
                return True, f"Aligned with SHORT bias (conviction: {bias_state.conviction:.2f})"
            else:
                return False, f"Direction conflicts with SHORT bias"
        
        return False, "Unknown bias state"
    
    def _analyze_higher_timeframe_trend(self, context: MarketContext) -> str:
        """
        Analyze higher timeframe trend from context.
        
        Returns: 'up', 'down', or 'neutral'
        """
        # Priority: 1D > 4H > 1H
        if context.trend_1d != 'neutral':
            return context.trend_1d
        elif context.trend_4h != 'neutral':
            return context.trend_4h
        else:
            return context.trend_1h
    
    def _analyze_momentum(self, df: pd.DataFrame) -> str:
        """
        Analyze recent momentum/impulse direction.
        
        Returns: 'bullish', 'bearish', or 'neutral'
        """
        if len(df) < self.momentum_lookback_hours:
            return 'neutral'
        
        # Get recent price action
        recent = df.tail(self.momentum_lookback_hours)
        
        # Calculate percentage move
        start_price = recent['close'].iloc[0]
        end_price = recent['close'].iloc[-1]
        pct_change = ((end_price - start_price) / start_price) * 100
        
        # Check for momentum
        if pct_change > self.momentum_threshold_pct:
            return 'bullish'
        elif pct_change < -self.momentum_threshold_pct:
            return 'bearish'
        else:
            return 'neutral'
    
    def _check_volatility_expansion(self, context: MarketContext) -> bool:
        """
        Check if volatility is expanding (often precedes trending moves).
        
        Returns: True if volatility is expanding
        """
        # High ATR percentile suggests expansion
        return context.atr_percentile > 70.0
    
    def _calculate_conviction(
        self,
        htf_trend: str,
        momentum: str,
        vol_expansion: bool,
        context: MarketContext
    ) -> float:
        """
        Calculate conviction score (0.0 to 1.0).
        
        Higher conviction when:
        - HTF and momentum align
        - Strong trend strength (ADX)
        - Volatility expansion
        - Clear market regime
        """
        score = 0.0
        
        # Base score from HTF trend
        if htf_trend != 'neutral':
            score += self.htf_weight
        
        # Bonus if momentum aligns
        if (htf_trend == 'up' and momentum == 'bullish') or \
           (htf_trend == 'down' and momentum == 'bearish'):
            score += self.ltf_weight
        
        # Bonus for strong trend
        if context.trend_strength == 'strong':
            score += 0.1
        
        # Bonus for volatility expansion (momentum building)
        if vol_expansion:
            score += 0.1
        
        # Bonus for high context confidence
        score += context.confidence * 0.1
        
        return min(score, 1.0)
    
    def _determine_bias(
        self,
        htf_trend: str,
        momentum: str,
        conviction: float
    ) -> TradeBias:
        """
        Determine bias from analysis.
        
        Returns: TradeBias
        """
        # Need minimum conviction to have bias
        if conviction < self.min_conviction_to_bias:
            return TradeBias.NEUTRAL
        
        # Determine direction (priority to HTF)
        if htf_trend == 'up':
            return TradeBias.LONG_BIAS
        elif htf_trend == 'down':
            return TradeBias.SHORT_BIAS
        elif momentum == 'bullish':
            return TradeBias.LONG_BIAS
        elif momentum == 'bearish':
            return TradeBias.SHORT_BIAS
        else:
            return TradeBias.NEUTRAL
    
    def _apply_stability_constraints(
        self,
        new_bias: TradeBias,
        conviction: float,
        timestamp: datetime
    ) -> Tuple[TradeBias, str]:
        """
        Prevent rapid bias flipping.
        
        Returns: (final_bias, reason)
        """
        # If no previous bias, allow new bias
        if self.current_bias is None:
            return new_bias, f"Initial bias: {new_bias.value}"
        
        # If bias hasn't changed, allow
        if new_bias == self.current_bias.bias:
            return new_bias, f"Bias unchanged: {new_bias.value}"
        
        # Check if enough time has passed since last flip
        if self.current_bias.last_flip_time:
            time_since_flip = (timestamp - self.current_bias.last_flip_time).total_seconds() / 3600
            
            if time_since_flip < self.min_flip_interval_hours:
                return (
                    self.current_bias.bias,
                    f"Bias flip blocked: only {time_since_flip:.1f}h since last flip "
                    f"(min {self.min_flip_interval_hours}h)"
                )
        
        # Check flip rate
        recent_flips = self._count_recent_flips(timestamp)
        if recent_flips >= self.max_flips_per_day:
            return (
                self.current_bias.bias,
                f"Bias flip blocked: {recent_flips} flips in last 24h "
                f"(max {self.max_flips_per_day})"
            )
        
        # Allow flip
        return new_bias, f"Bias flipped: {self.current_bias.bias.value} → {new_bias.value}"
    
    def _count_recent_flips(self, timestamp: datetime) -> int:
        """Count bias flips in last 24 hours."""
        cutoff = timestamp - timedelta(hours=24)
        
        flips = 0
        prev_bias = None
        
        for state in self.bias_history:
            if state.timestamp < cutoff:
                continue
            
            if prev_bias and state.bias != prev_bias:
                flips += 1
            
            prev_bias = state.bias
        
        return flips
    
    def _update_bias_state(self, new_state: BiasState):
        """Update current bias and history."""
        # Detect flip
        if self.current_bias and new_state.bias != self.current_bias.bias:
            new_state.last_flip_time = new_state.timestamp
        elif self.current_bias:
            new_state.last_flip_time = self.current_bias.last_flip_time
        
        # Update current
        self.current_bias = new_state
        
        # Add to history
        self.bias_history.append(new_state)
        
        # Keep history manageable (last 7 days)
        cutoff = new_state.timestamp - timedelta(days=7)
        self.bias_history = [
            s for s in self.bias_history if s.timestamp > cutoff
        ]
    
    def _create_neutral_bias(
        self,
        timestamp: datetime,
        reason: str
    ) -> BiasState:
        """Create a NEUTRAL bias state."""
        return BiasState(
            bias=TradeBias.NEUTRAL,
            conviction=0.0,
            timestamp=timestamp,
            higher_tf_trend='neutral',
            momentum_direction='neutral',
            volatility_expansion=False,
            last_flip_time=self.current_bias.last_flip_time if self.current_bias else None,
            flips_in_window=0,
            reason=reason,
            metadata={}
        )
    
    def get_bias_report(self) -> Dict:
        """Generate bias statistics report."""
        if not self.current_bias:
            return {"status": "No bias data"}
        
        recent_20 = self.bias_history[-20:] if len(self.bias_history) >= 20 else self.bias_history
        
        bias_counts = {}
        for state in recent_20:
            bias_counts[state.bias.value] = bias_counts.get(state.bias.value, 0) + 1
        
        return {
            "current_bias": self.current_bias.bias.value,
            "current_conviction": self.current_bias.conviction,
            "last_flip": self.current_bias.last_flip_time.isoformat() if self.current_bias.last_flip_time else "Never",
            "flips_24h": self._count_recent_flips(datetime.now()),
            "bias_distribution_recent_20": bias_counts,
            "avg_conviction": np.mean([s.conviction for s in recent_20]) if recent_20 else 0.0
        }


# Singleton instance
_bias_engine: Optional[BiasEngine] = None


def get_bias_engine() -> BiasEngine:
    """Get the global BiasEngine instance."""
    global _bias_engine
    if _bias_engine is None:
        _bias_engine = BiasEngine()
    return _bias_engine
