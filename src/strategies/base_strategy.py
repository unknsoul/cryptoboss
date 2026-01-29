"""
Base Strategy - v11.0 TradeIntent Integration

All strategies should inherit from this base class to automatically
integrate with the TradeIntent/TradeDecision pipeline.

Features:
- Automatic TradeIntent generation from signals
- Strategy versioning and identification
- Metrics collection
- Context propagation

v11.0 - Production-Grade Platform Upgrade
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from .intent_adapter import StrategyIntentAdapter, create_intent

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a strategy."""
    strategy_id: str
    version: str = "1.0"
    symbol: str = "BTC/USDT"
    enabled: bool = True
    max_position_pct: float = 100.0  # Max % of available capital
    min_confidence: float = 0.5  # Minimum confidence to generate intent
    cooldown_seconds: int = 0  # Cooldown between signals
    metadata: Dict = field(default_factory=dict)


@dataclass
class SignalResult:
    """
    Standard signal result from strategies.
    
    This is the common output format that can be converted to TradeIntent.
    """
    action: str  # 'BUY', 'SELL', 'HOLD', 'CLOSE'
    reason: str
    confidence: float = 0.0
    size: float = 0.0
    price: float = 0.0
    
    # Optional fields
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    signal_strength: float = 0.0
    
    # ML features if applicable
    ml_probability: Optional[float] = None
    ml_confidence: Optional[float] = None
    
    # Additional data
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_actionable(self) -> bool:
        """Check if this signal requires action."""
        return self.action in ('BUY', 'SELL', 'CLOSE')
    
    @property
    def is_entry(self) -> bool:
        """Check if this is an entry signal."""
        return self.action == 'BUY'
    
    @property
    def is_exit(self) -> bool:
        """Check if this is an exit signal."""
        return self.action in ('SELL', 'CLOSE')


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    v11.0 Features:
    - Automatic TradeIntent generation
    - Strategy versioning
    - Metrics collection
    - Pipeline integration
    
    Usage:
        class MyStrategy(BaseStrategy):
            def __init__(self):
                super().__init__(
                    StrategyConfig(
                        strategy_id="my_strategy",
                        version="1.0"
                    )
                )
            
            def generate_signal(self, df, i, current_price) -> SignalResult:
                # Your signal logic here
                return SignalResult(action='BUY', reason='...')
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.strategy_id = config.strategy_id
        self.version = config.version
        self.symbol = config.symbol
        
        # v11.0: Intent adapter
        self._intent_adapter = StrategyIntentAdapter(
            strategy_id=self.strategy_id,
            strategy_version=self.version
        )
        
        # Metrics
        self._signals_generated = 0
        self._intents_created = 0
        self._last_signal_time: Optional[datetime] = None
        self._signal_history: List[SignalResult] = []
        
        logger.info(f"Strategy '{self.strategy_id}' v{self.version} initialized")
    
    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        i: int,
        current_price: float
    ) -> SignalResult:
        """
        Generate trading signal based on market data.
        
        Args:
            df: DataFrame with OHLCV data
            i: Current index in DataFrame
            current_price: Current market price
            
        Returns:
            SignalResult with action and reason
        """
        pass
    
    def generate_intent(
        self,
        df: pd.DataFrame,
        i: int,
        current_price: float,
        market_context: Optional[Dict] = None
    ) -> Optional["TradeIntent"]:
        """
        Generate a TradeIntent from the strategy's signal.
        
        This is the primary method for v11.0 pipeline integration.
        
        Args:
            df: DataFrame with OHLCV data
            i: Current index in DataFrame  
            current_price: Current market price
            market_context: Optional market context data
            
        Returns:
            TradeIntent if signal is actionable, None otherwise
        """
        # Generate signal
        signal = self.generate_signal(df, i, current_price)
        self._signals_generated += 1
        self._signal_history.append(signal)
        
        # Check if actionable
        if not signal.is_actionable:
            return None
        
        # Check confidence threshold
        if signal.confidence < self.config.min_confidence:
            logger.debug(
                f"Signal confidence {signal.confidence:.2f} below threshold "
                f"{self.config.min_confidence:.2f}"
            )
            return None
        
        # Check cooldown
        if self._last_signal_time and self.config.cooldown_seconds > 0:
            elapsed = (datetime.now() - self._last_signal_time).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                logger.debug(f"Signal cooldown: {elapsed:.0f}s / {self.config.cooldown_seconds}s")
                return None
        
        # Determine direction
        direction = "long" if signal.is_entry else "short"
        
        # Create TradeIntent
        intent = create_intent(
            symbol=self.symbol,
            direction=direction,
            strategy_id=self.strategy_id,
            confidence=signal.confidence,
            reasoning=signal.reason,
            strategy_version=self.version,
            signal_strength=signal.signal_strength,
            suggested_entry=signal.price or current_price,
            suggested_stop=signal.stop_loss,
            suggested_target=signal.take_profit,
            ml_probability=signal.ml_probability,
            ml_confidence=signal.ml_confidence,
            market_regime=market_context.get('regime', 'unknown') if market_context else 'unknown',
            directional_bias=market_context.get('bias', 'neutral') if market_context else 'neutral',
        )
        
        self._intents_created += 1
        self._last_signal_time = datetime.now()
        
        logger.info(
            f"Strategy '{self.strategy_id}' generated intent: "
            f"{direction.upper()} {self.symbol} @ {current_price:.2f} "
            f"(confidence: {signal.confidence:.2f})"
        )
        
        return intent
    
    def signal_to_legacy_dict(self, signal: SignalResult) -> Dict:
        """
        Convert SignalResult to legacy dict format for backward compatibility.
        
        Args:
            signal: SignalResult object
            
        Returns:
            Dict in legacy format
        """
        return {
            'action': signal.action,
            'size': signal.size,
            'price': signal.price,
            'reason': signal.reason,
            'confidence': signal.confidence,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            **signal.metadata
        }
    
    def get_metrics(self) -> Dict:
        """Get strategy metrics."""
        return {
            'strategy_id': self.strategy_id,
            'version': self.version,
            'symbol': self.symbol,
            'enabled': self.config.enabled,
            'signals_generated': self._signals_generated,
            'intents_created': self._intents_created,
            'intent_conversion_rate': (
                self._intents_created / self._signals_generated
                if self._signals_generated > 0 else 0
            ),
            'last_signal_time': (
                self._last_signal_time.isoformat() 
                if self._last_signal_time else None
            ),
        }
    
    def get_recent_signals(self, limit: int = 100) -> List[SignalResult]:
        """Get recent signals."""
        return self._signal_history[-limit:]
    
    def reset_metrics(self):
        """Reset strategy metrics."""
        self._signals_generated = 0
        self._intents_created = 0
        self._signal_history = []
        logger.info(f"Strategy '{self.strategy_id}' metrics reset")


class LegacyStrategyWrapper(BaseStrategy):
    """
    Wrapper to adapt legacy strategies to v11.0 TradeIntent format.
    
    Use this to wrap existing strategies without modifying them.
    
    Usage:
        legacy_strategy = DCAStrategy(...)  # Old format
        wrapped = LegacyStrategyWrapper(
            legacy_strategy,
            StrategyConfig(strategy_id="dca_btc_1", symbol="BTC/USDT")
        )
        
        intent = wrapped.generate_intent(df, i, current_price)
    """
    
    def __init__(self, legacy_strategy, config: StrategyConfig):
        super().__init__(config)
        self._legacy = legacy_strategy
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        i: int,
        current_price: float
    ) -> SignalResult:
        """Wrap legacy strategy's generate_signal."""
        # Call legacy strategy
        legacy_result = self._legacy.generate_signal(df, i, current_price)
        
        # Convert to SignalResult
        return SignalResult(
            action=legacy_result.get('action', 'HOLD'),
            reason=legacy_result.get('reason', 'Legacy signal'),
            confidence=legacy_result.get('confidence', 0.7),  # Default confidence
            size=legacy_result.get('size', 0.0),
            price=legacy_result.get('price', current_price),
            stop_loss=legacy_result.get('stop_loss'),
            take_profit=legacy_result.get('take_profit'),
            metadata=legacy_result
        )
