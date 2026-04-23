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
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Sequence
import pandas as pd

from .intent_adapter import StrategyIntentAdapter, create_intent

try:
    from ..ml.predator_model import PredatorModel
except Exception:  # pragma: no cover - runtime optional dependency
    PredatorModel = None

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

    # Optional ONNX/externally-served model integration.
    onnx_model_path: Optional[str] = None
    onnx_input_name: Optional[str] = None
    onnx_output_names: List[str] = field(default_factory=list)
    onnx_providers: List[str] = field(default_factory=list)

    # Injection hook for tests or custom runtime adapters.
    external_model: Optional[Any] = None


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
    ml_model_id: Optional[str] = None

    # Optional execution hints from model heads.
    urgency: Optional[float] = None
    order_preference: Optional[str] = None
    
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
        self._model_signals_generated = 0
        self._intents_created = 0
        self._last_signal_time: Optional[datetime] = None
        self._signal_history: List[SignalResult] = []

        # Optional external model bridge (ONNX or custom adapter).
        self._external_model: Optional[Any] = None
        self._external_model_id: Optional[str] = None
        self._initialize_external_model()
        
        logger.info(f"Strategy '{self.strategy_id}' v{self.version} initialized")

    def _initialize_external_model(self) -> None:
        """Attach externally trained model when configured."""
        if self.config.external_model is not None:
            if hasattr(self.config.external_model, "predict"):
                self._external_model = self.config.external_model
                self._external_model_id = getattr(
                    self.config.external_model,
                    "model_id",
                    "external_model",
                )
                logger.info(
                    "Strategy '%s' attached external model adapter: %s",
                    self.strategy_id,
                    self._external_model_id,
                )
            else:
                logger.warning(
                    "Strategy '%s' external_model missing predict() method; ignored",
                    self.strategy_id,
                )
            return

        if not self.config.onnx_model_path:
            return

        if PredatorModel is None:
            logger.warning(
                "Strategy '%s' ONNX model configured (%s) but ONNX runtime adapter is unavailable",
                self.strategy_id,
                self.config.onnx_model_path,
            )
            return

        model = PredatorModel(
            model_path=self.config.onnx_model_path,
            input_name=self.config.onnx_input_name,
            output_names=self.config.onnx_output_names,
            providers=self.config.onnx_providers,
        )
        if not model.is_ready:
            logger.warning(
                "Strategy '%s' could not initialize ONNX model from %s",
                self.strategy_id,
                self.config.onnx_model_path,
            )
            return

        self._external_model = model
        self._external_model_id = os.path.basename(self.config.onnx_model_path)
        logger.info(
            "Strategy '%s' ONNX model loaded: %s",
            self.strategy_id,
            self._external_model_id,
        )

    @property
    def external_model_enabled(self) -> bool:
        """Whether this strategy currently has a live external model adapter."""
        return self._external_model is not None

    def build_external_features(
        self,
        df: pd.DataFrame,
        i: int,
        current_price: float,
    ) -> List[float]:
        """
        Build a fixed-size feature vector for external model inference.

        Strategies can override this for richer order-book or tape features.
        """
        if df is None or len(df) == 0:
            return [float(current_price)] + [0.0] * 23

        end = min(max(i, 0), len(df) - 1) + 1
        window = df.iloc[max(0, end - 32):end].copy()

        if "close" in window:
            close = pd.to_numeric(window["close"], errors="coerce").dropna().astype(float).tolist()
        else:
            close = [float(current_price)]

        if not close:
            close = [float(current_price)]

        returns: List[float] = []
        for prev, cur in zip(close[:-1], close[1:]):
            if prev == 0:
                returns.append(0.0)
            else:
                returns.append((cur - prev) / prev)

        # Keep last 16 returns for short-horizon behavior; left-pad with zeros.
        tail_returns = returns[-16:]
        if len(tail_returns) < 16:
            tail_returns = ([0.0] * (16 - len(tail_returns))) + tail_returns

        last_close = close[-1]
        mean_close = sum(close) / len(close)
        price_vs_mean = (last_close - mean_close) / mean_close if mean_close else 0.0
        short_mom = (last_close - close[-4]) / close[-4] if len(close) >= 4 and close[-4] else 0.0
        long_mom = (last_close - close[0]) / close[0] if close[0] else 0.0
        volatility = (sum((r - (sum(returns) / len(returns) if returns else 0.0)) ** 2 for r in returns) / len(returns)) ** 0.5 if returns else 0.0

        spread_norm = 0.0
        if "high" in window and "low" in window and current_price:
            high_last = float(pd.to_numeric(window["high"], errors="coerce").iloc[-1])
            low_last = float(pd.to_numeric(window["low"], errors="coerce").iloc[-1])
            spread_norm = max(0.0, (high_last - low_last) / current_price)

        volume_delta = 0.0
        if "volume" in window:
            vol = pd.to_numeric(window["volume"], errors="coerce").dropna().astype(float)
            if not vol.empty:
                v_mean = float(vol.mean())
                v_last = float(vol.iloc[-1])
                volume_delta = ((v_last - v_mean) / v_mean) if v_mean else 0.0

        aggregates = [
            float(current_price),
            float(last_close),
            float(mean_close),
            float(price_vs_mean),
            float(short_mom),
            float(long_mom),
            float(volatility),
            float(spread_norm),
            float(volume_delta),
        ]

        # 16 + 9 = 25 features total.
        return [float(x) for x in (tail_returns + aggregates)]

    def infer_external_action(self, feature_vector: Sequence[float]) -> Optional[Dict[str, Any]]:
        """Run inference through ONNX/custom adapter and return normalized action heads."""
        if not self._external_model:
            return None

        try:
            prediction = self._external_model.predict(feature_vector)
        except Exception as exc:
            logger.error(
                "Strategy '%s' external model inference failed: %s",
                self.strategy_id,
                exc,
            )
            return None

        if isinstance(prediction, dict):
            return prediction

        return {"raw_prediction": prediction}

    def model_action_to_signal(
        self,
        model_action: Dict[str, Any],
        current_price: float,
        hold_threshold: float = 0.10,
    ) -> SignalResult:
        """Convert model action heads into a strategy signal contract."""
        direction_score = float(model_action.get("direction", 0.0) or 0.0)
        urgency = float(model_action.get("urgency", 0.0) or 0.0)
        size = float(model_action.get("size", 0.0) or 0.0)
        confidence = float(model_action.get("confidence", abs(direction_score)) or 0.0)

        action = "HOLD"
        if direction_score >= hold_threshold:
            action = "BUY"
        elif direction_score <= -hold_threshold:
            action = "SELL"

        if size < 0:
            size = 0.0

        return SignalResult(
            action=action,
            reason="external_model_inference",
            confidence=max(0.0, min(1.0, confidence)),
            size=size,
            price=current_price,
            signal_strength=direction_score,
            ml_probability=model_action.get("probability"),
            ml_confidence=model_action.get("confidence"),
            ml_model_id=self._external_model_id,
            urgency=urgency,
            order_preference=model_action.get("order_type"),
            metadata={
                "model_action": model_action,
                "ml_model_id": self._external_model_id,
            },
        )

    def _generate_signal_from_external_model(
        self,
        df: pd.DataFrame,
        i: int,
        current_price: float,
    ) -> Optional[SignalResult]:
        """Generate signal from external model when enabled; returns None on unavailable model."""
        if not self.external_model_enabled:
            return None

        features = self.build_external_features(df, i, current_price)
        model_action = self.infer_external_action(features)
        if model_action is None:
            return None

        self._model_signals_generated += 1
        return self.model_action_to_signal(model_action, current_price)
    
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
        # Generate signal. If an external model is attached, it gets first pass.
        signal = self._generate_signal_from_external_model(df, i, current_price)
        if signal is None:
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
            ml_model_id=signal.ml_model_id or signal.metadata.get('ml_model_id') or self._external_model_id,
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
            'urgency': signal.urgency,
            'order_preference': signal.order_preference,
            'ml_model_id': signal.ml_model_id,
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
            'model_signals_generated': self._model_signals_generated,
            'intents_created': self._intents_created,
            'intent_conversion_rate': (
                self._intents_created / self._signals_generated
                if self._signals_generated > 0 else 0
            ),
            'external_model_enabled': self.external_model_enabled,
            'external_model_id': self._external_model_id,
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
        self._model_signals_generated = 0
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
