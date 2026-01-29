"""
Strategy Intent Adapter - v11.0 Bridge

Provides utilities for strategies to produce TradeIntent objects
compatible with the v11.0 execution pipeline.

This adapter:
- Converts legacy proposals to TradeIntent format
- Provides helper functions for common patterns
- Maintains backward compatibility

v11.0 - Production-Grade Platform Upgrade
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class StrategySignalType(str, Enum):
    """Type of signal from strategy."""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


@dataclass
class StrategySignal:
    """
    Signal from a strategy, to be converted to TradeIntent.
    
    This provides a simpler interface for strategies to emit signals
    without needing to know the full TradeIntent structure.
    """
    signal_type: StrategySignalType
    symbol: str
    confidence: float
    reasoning: str
    
    # Optional parameters
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    signal_strength: float = 0.0
    
    # ML features if applicable
    ml_probability: Optional[float] = None
    ml_confidence: Optional[float] = None
    
    # Metadata
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class StrategyIntentAdapter:
    """
    Adapter to convert strategy signals to TradeIntent objects.
    
    Usage:
        adapter = StrategyIntentAdapter(strategy_id="momentum_v2")
        
        # In strategy:
        signal = StrategySignal(
            signal_type=StrategySignalType.ENTRY_LONG,
            symbol="BTC/USDT",
            confidence=0.85,
            reasoning="Breakout above resistance"
        )
        
        intent = adapter.signal_to_intent(signal, context)
        
        # Submit to orchestrator
        decision = orchestrator.process_intent(intent, current_price)
    """
    
    def __init__(
        self,
        strategy_id: str,
        strategy_version: str = "1.0",
        default_ttl_seconds: int = 300
    ):
        """
        Initialize adapter.
        
        Args:
            strategy_id: Unique strategy identifier
            strategy_version: Version string
            default_ttl_seconds: Default TTL for intents
        """
        self.strategy_id = strategy_id
        self.strategy_version = strategy_version
        self.default_ttl_seconds = default_ttl_seconds
        
        logger.info(f"StrategyIntentAdapter initialized for: {strategy_id} v{strategy_version}")
    
    def signal_to_intent(
        self,
        signal: StrategySignal,
        market_context: Optional[Dict] = None,
        ml_features: Optional[Dict] = None
    ) -> "TradeIntent":
        """
        Convert a StrategySignal to a TradeIntent.
        
        Args:
            signal: The strategy signal
            market_context: Optional market context data
            ml_features: Optional ML features
            
        Returns:
            TradeIntent ready for submission
        """
        from src.core.trade_intent import TradeIntent, IntentDirection, IntentPriority
        
        # Determine direction
        if signal.signal_type in (StrategySignalType.ENTRY_LONG, StrategySignalType.SCALE_IN):
            direction = IntentDirection.LONG
        else:
            direction = IntentDirection.SHORT
        
        # Determine priority based on confidence
        if signal.confidence >= 0.9:
            priority = IntentPriority.HIGH
        elif signal.confidence >= 0.7:
            priority = IntentPriority.NORMAL
        else:
            priority = IntentPriority.LOW
        
        # Extract context data
        context = market_context or {}
        
        # Create intent
        intent = TradeIntent.create(
            symbol=signal.symbol,
            direction=direction,
            strategy_id=self.strategy_id,
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            strategy_version=self.strategy_version,
            signal_strength=signal.signal_strength,
            suggested_entry=signal.entry_price,
            suggested_stop=signal.stop_loss,
            suggested_target=signal.take_profit,
            ml_probability=signal.ml_probability,
            ml_confidence=signal.ml_confidence,
            market_regime=context.get('regime', 'unknown'),
            directional_bias=context.get('bias', 'neutral'),
            volatility_regime=context.get('volatility', 'normal'),
            priority=priority,
            ttl_seconds=self.default_ttl_seconds,
            ml_features=ml_features or {},
            tags=[signal.signal_type.value, self.strategy_id],
        )
        
        return intent
    
    def create_long_intent(
        self,
        symbol: str,
        confidence: float,
        reasoning: str,
        entry_price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        **kwargs
    ) -> "TradeIntent":
        """
        Convenience method to create a long intent.
        
        Args:
            symbol: Trading pair
            confidence: Confidence level (0-1)
            reasoning: Human-readable explanation
            entry_price: Optional entry price
            stop_loss: Optional stop loss
            take_profit: Optional take profit
            **kwargs: Additional TradeIntent parameters
            
        Returns:
            TradeIntent
        """
        from src.core.trade_intent import TradeIntent, IntentDirection
        
        return TradeIntent.create(
            symbol=symbol,
            direction=IntentDirection.LONG,
            strategy_id=self.strategy_id,
            confidence=confidence,
            reasoning=reasoning,
            strategy_version=self.strategy_version,
            suggested_entry=entry_price,
            suggested_stop=stop_loss,
            suggested_target=take_profit,
            ttl_seconds=self.default_ttl_seconds,
            **kwargs
        )
    
    def create_short_intent(
        self,
        symbol: str,
        confidence: float,
        reasoning: str,
        entry_price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        **kwargs
    ) -> "TradeIntent":
        """
        Convenience method to create a short intent.
        
        Args:
            symbol: Trading pair
            confidence: Confidence level (0-1)
            reasoning: Human-readable explanation
            entry_price: Optional entry price
            stop_loss: Optional stop loss
            take_profit: Optional take profit
            **kwargs: Additional TradeIntent parameters
            
        Returns:
            TradeIntent
        """
        from src.core.trade_intent import TradeIntent, IntentDirection
        
        return TradeIntent.create(
            symbol=symbol,
            direction=IntentDirection.SHORT,
            strategy_id=self.strategy_id,
            confidence=confidence,
            reasoning=reasoning,
            strategy_version=self.strategy_version,
            suggested_entry=entry_price,
            suggested_stop=stop_loss,
            suggested_target=take_profit,
            ttl_seconds=self.default_ttl_seconds,
            **kwargs
        )
    
    def proposals_to_intents(
        self,
        proposals: List[Dict],
        market_context: Optional[Dict] = None
    ) -> List["TradeIntent"]:
        """
        Convert legacy proposal dicts to TradeIntent objects.
        
        This is useful for migrating existing strategies.
        
        Args:
            proposals: List of legacy proposal dicts
            market_context: Optional market context
            
        Returns:
            List of TradeIntent objects
        """
        from src.core.trade_intent import TradeIntent, IntentDirection, IntentPriority
        
        intents = []
        context = market_context or {}
        
        for p in proposals:
            # Parse direction
            direction_str = p.get('direction', 'long').lower()
            direction = IntentDirection.LONG if direction_str == 'long' else IntentDirection.SHORT
            
            # Parse priority
            priority_str = p.get('priority', 'normal').lower()
            try:
                priority = IntentPriority(priority_str)
            except ValueError:
                priority = IntentPriority.NORMAL
            
            intent = TradeIntent.create(
                symbol=p.get('symbol', 'UNKNOWN'),
                direction=direction,
                strategy_id=p.get('strategy_id', self.strategy_id),
                confidence=p.get('confidence', 0.5),
                reasoning=p.get('reasoning', p.get('reason', 'No reason provided')),
                strategy_version=self.strategy_version,
                signal_strength=p.get('signal_strength', 0.0),
                suggested_entry=p.get('entry_price'),
                suggested_stop=p.get('stop_loss'),
                suggested_target=p.get('take_profit', p.get('target')),
                ml_probability=p.get('ml_probability'),
                ml_confidence=p.get('ml_confidence'),
                market_regime=context.get('regime', 'unknown'),
                directional_bias=context.get('bias', 'neutral'),
                priority=priority,
                ttl_seconds=self.default_ttl_seconds,
            )
            
            intents.append(intent)
        
        return intents


def create_intent(
    symbol: str,
    direction: str,
    strategy_id: str,
    confidence: float,
    reasoning: str,
    **kwargs
) -> "TradeIntent":
    """
    Convenience function to create a TradeIntent.
    
    This is the simplest way for strategies to create intents.
    
    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        direction: "long" or "short"
        strategy_id: Unique strategy identifier
        confidence: Confidence level (0-1)
        reasoning: Human-readable explanation
        **kwargs: Additional optional parameters
        
    Returns:
        TradeIntent ready for submission
        
    Example:
        from src.strategies.intent_adapter import create_intent
        
        intent = create_intent(
            symbol="BTC/USDT",
            direction="long",
            strategy_id="momentum_breakout",
            confidence=0.85,
            reasoning="Price broke above 50-day MA with volume",
            suggested_entry=45000.0,
            suggested_stop=44000.0,
        )
    """
    from src.core.trade_intent import TradeIntent, IntentDirection
    
    dir_enum = IntentDirection.LONG if direction.lower() == 'long' else IntentDirection.SHORT
    
    return TradeIntent.create(
        symbol=symbol,
        direction=dir_enum,
        strategy_id=strategy_id,
        confidence=confidence,
        reasoning=reasoning,
        **kwargs
    )


def submit_intent(
    intent: "TradeIntent",
    current_price: float,
    context_data: Optional[Dict] = None
) -> "TradeDecision":
    """
    Submit a TradeIntent to the execution pipeline.
    
    This is a convenience wrapper around the ExecutionFlowOrchestrator.
    
    Args:
        intent: TradeIntent to process
        current_price: Current market price
        context_data: Optional pre-computed context
        
    Returns:
        TradeDecision with complete audit trail
        
    Example:
        decision = submit_intent(intent, current_price=45000.0)
        
        if decision.is_approved:
            print(f"Trade approved: {decision.execution_params}")
        else:
            print(f"Rejected: {decision.rejection_reason}")
    """
    from src.core.execution_flow import get_execution_flow
    
    orchestrator = get_execution_flow()
    return orchestrator.process_intent(intent, current_price, context_data)
