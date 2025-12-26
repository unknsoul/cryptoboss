"""
Strategy Selection System
Regime detection and adaptive strategy switching.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


class RegimeDetector:
    """
    Detects current market regime.
    
    Uses:
    - ADX for trend strength
    - Volatility for regime classification
    - Price action for direction
    """
    
    def __init__(
        self,
        adx_threshold: float = 25.0,
        vol_threshold_pct: float = 50.0
    ):
        """
        Initialize regime detector.
        
        Args:
            adx_threshold: ADX threshold for trending (>25 = trending)
            vol_threshold_pct: Volatility threshold for high vol regime
        """
        self.adx_threshold = adx_threshold
        self.vol_threshold_pct = vol_threshold_pct
        
        logger.info("RegimeDetector initialized")
    
    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            df: DataFrame with OHLCV + indicators
            
        Returns:
            MarketRegime enum
        """
        # Get latest data
        if len(df) < 50:
            return MarketRegime.RANGING
        
        latest = df.iloc[-1]
        recent = df.iloc[-50:]
        
        # Calculate indicators if not present
        if 'adx' not in df.columns:
            adx = self._calculate_adx(df)
        else:
            adx = latest['adx']
        
        # Volatility (realized vol vs historical)
        current_vol = recent['close'].pct_change().std()
        historical_vol = df['close'].pct_change().std()
        vol_ratio = (current_vol / historical_vol - 1) * 100
        
        # Trend direction (20-period SMA slope)
        sma_20 = recent['close'].rolling(20).mean()
        trend_slope = (sma_20.iloc[-1] - sma_20.iloc[-20]) / sma_20.iloc[-20] * 100
        
        # Regime decision logic
        if vol_ratio > self.vol_threshold_pct:
            regime = MarketRegime.HIGH_VOLATILITY
        elif adx > self.adx_threshold:
            if trend_slope > 0:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
        else:
            regime = MarketRegime.RANGING
        
        logger.debug(f"Regime detected: {regime.value} (ADX: {adx:.1f}, Vol ratio: {vol_ratio:.1f}%)")
        return regime
    
    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX."""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0))
        minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # DI
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1]


class StrategySelector:
    """
    Selects appropriate strategy based on regime.
    
    Strategy mapping:
    - Trending: Momentum strategy
    - Ranging: Mean reversion strategy
    - High Vol: Breakout strategy
    """
    
    STRATEGY_MAP = {
        MarketRegime.TRENDING_UP: "momentum_long",
        MarketRegime.TRENDING_DOWN: "momentum_short",
        MarketRegime.RANGING: "mean_reversion",
        MarketRegime.HIGH_VOLATILITY: "breakout"
    }
    
    def __init__(self):
        """Initialize strategy selector."""
        self.current_strategy = None
        self.strategy_history: List[Dict] = []
        self.regime_detector = RegimeDetector()
        
        logger.info("StrategySelector initialized")
    
    def select_strategy(self, df: pd.DataFrame) -> str:
        """
        Select optimal strategy for current regime.
        
        Args:
            df: Market data DataFrame
            
        Returns:
            Strategy name
        """
        # Detect regime
        regime = self.regime_detector.detect(df)
        
        # Map to strategy
        strategy = self.STRATEGY_MAP.get(regime, "mean_reversion")
        
        # Record switch if changed
        if strategy != self.current_strategy:
            logger.info(f"Strategy switch: {self.current_strategy} → {strategy} (regime: {regime.value})")
            self.strategy_history.append({
                'timestamp': pd.Timestamp.now(),
                'prev_strategy': self.current_strategy,
                'new_strategy': strategy,
                'regime': regime.value
            })
            self.current_strategy = strategy
        
        return strategy
    
    def get_strategy_performance(self) -> Dict:
        """Get performance statistics by strategy."""
        if not self.strategy_history:
            return {}
        
        # Count strategy usage
        strategy_counts = {}
        for record in self.strategy_history:
            strat = record['new_strategy']
            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
        
        return {
            'total_switches': len(self.strategy_history),
            'current_strategy': self.current_strategy,
            'strategy_usage': strategy_counts
        }
