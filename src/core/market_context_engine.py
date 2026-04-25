"""
Market Context Engine - The First Gate

This module determines if trading is allowed at all by analyzing:
1. Multi-timeframe market structure
2. Volatility regime
3. Trend strength
4. Liquidity conditions

If context returns NO_TRADE, all downstream modules must halt.

Key Principle: Never trade without defined market context.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    NO_TRADE = "no_trade"


@dataclass
class LiquidityMetrics:
    """Liquidity condition metrics."""
    spread_bps: float
    bid_size: float
    ask_size: float
    volume_24h: float
    is_acceptable: bool
    reason: str = ""


@dataclass
class MarketContext:
    """Complete market context snapshot."""
    timestamp: datetime
    symbol: str
    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    
    # Multi-timeframe analysis
    trend_1h: str  # 'up', 'down', 'neutral'
    trend_4h: str
    trend_1d: str
    
    # Volatility
    atr_percentile: float
    volatility_regime: str  # 'low', 'normal', 'high', 'extreme'
    
    # Trend strength
    adx_value: float
    trend_strength: str  # 'weak', 'moderate', 'strong'
    
    # Liquidity
    liquidity: LiquidityMetrics
    
    # Context decision
    trading_allowed: bool
    reason: str  # Why trading is allowed or blocked
    
    # Metadata
    metadata: Dict


class MarketContextEngine:
    """
    Market Context Engine - First decision gate.
    
    Determines if trading conditions are acceptable before any
    strategy logic executes.
    
    Usage:
        context_engine = MarketContextEngine()
        context = context_engine.get_current_context(df, price, orderbook)
        
        if context.regime == MarketRegime.NO_TRADE:
            logger.info(f"Trading blocked: {context.reason}")
            return
    """
    
    def __init__(
        self,
        # Volatility thresholds
        extreme_vol_percentile: float = 95.0,
        high_vol_percentile: float = 80.0,
        low_vol_percentile: float = 20.0,
        
        # Trend strength thresholds (ADX)
        strong_trend_adx: float = 30.0,
        moderate_trend_adx: float = 20.0,
        
        # Liquidity thresholds
        max_spread_bps: float = 10.0,  # 0.10%
        min_bid_ask_size_btc: float = 1.0,
        min_volume_24h_btc: float = 100.0,
        
        # Multi-timeframe lookback
        lookback_1h: int = 24,  # Last 24 hours
        lookback_4h: int = 42,  # Last week
        lookback_1d: int = 30,  # Last month
    ):
        self.extreme_vol_percentile = extreme_vol_percentile
        self.high_vol_percentile = high_vol_percentile
        self.low_vol_percentile = low_vol_percentile
        
        self.strong_trend_adx = strong_trend_adx
        self.moderate_trend_adx = moderate_trend_adx
        
        self.max_spread_bps = max_spread_bps
        self.min_bid_ask_size_btc = min_bid_ask_size_btc
        self.min_volume_24h_btc = min_volume_24h_btc
        
        self.lookback_1h = lookback_1h
        self.lookback_4h = lookback_4h
        self.lookback_1d = lookback_1d
        
        logger.info("MarketContextEngine initialized")
    
    def get_current_context(
        self,
        df_1h: pd.DataFrame,
        df_4h: Optional[pd.DataFrame] = None,
        df_1d: Optional[pd.DataFrame] = None,
        current_price: float = None,
        orderbook: Optional[Dict] = None,
        volume_24h: Optional[float] = None
    ) -> MarketContext:
        """
        Analyze current market context across all dimensions.
        
        Args:
            df_1h: 1-hour OHLCV data (required)
            df_4h: 4-hour OHLCV data (optional, recommended)
            df_1d: Daily OHLCV data (optional, recommended)
            current_price: Current market price
            orderbook: Order book snapshot {'bids': [...], 'asks': [...]}
            volume_24h: 24-hour volume
            
        Returns:
            MarketContext with complete analysis
        """
        timestamp = datetime.now()
        symbol = "UNKNOWN"  # Can be passed as parameter
        
        # Get current price
        if current_price is None:
            current_price = df_1h['close'].iloc[-1]
        
        # 1. Multi-timeframe trend analysis
        trend_1h = self._analyze_trend(df_1h, self.lookback_1h)
        trend_4h = self._analyze_trend(df_4h, self.lookback_4h) if df_4h is not None else 'neutral'
        trend_1d = self._analyze_trend(df_1d, self.lookback_1d) if df_1d is not None else 'neutral'
        
        # 2. Volatility regime
        atr_percentile, volatility_regime = self._analyze_volatility(df_1h)
        
        # 3. Trend strength
        adx_value, trend_strength = self._analyze_trend_strength(df_1h)
        
        # 4. Liquidity conditions
        liquidity = self._analyze_liquidity(
            current_price, orderbook, volume_24h
        )
        
        # 5. Classify market regime
        regime, confidence = self._classify_regime(
            trend_1h, trend_4h, trend_1d,
            volatility_regime,
            trend_strength,
            liquidity,
            atr_percentile,
            adx_value
        )
        
        # 6. Determine if trading is allowed
        trading_allowed, reason = self._determine_trading_permission(
            regime, liquidity, volatility_regime
        )
        
        context = MarketContext(
            timestamp=timestamp,
            symbol=symbol,
            regime=regime,
            confidence=confidence,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            trend_1d=trend_1d,
            atr_percentile=atr_percentile,
            volatility_regime=volatility_regime,
            adx_value=adx_value,
            trend_strength=trend_strength,
            liquidity=liquidity,
            trading_allowed=trading_allowed,
            reason=reason,
            metadata={
                'price': current_price,
                'volume_24h': volume_24h
            }
        )
        
        logger.debug(
            f"Context: {regime.value} | Trend: {trend_1h}/{trend_4h}/{trend_1d} | "
            f"Vol: {volatility_regime} | ADX: {adx_value:.1f} | "
            f"Trading: {'✓' if trading_allowed else '✗'} ({reason})"
        )
        
        return context
    
    def _analyze_trend(self, df: pd.DataFrame, lookback: int) -> str:
        """
        Analyze trend direction using EMA crossover.
        
        Returns: 'up', 'down', or 'neutral'
        """
        if df is None or len(df) < lookback:
            return 'neutral'
        
        recent = df.tail(lookback)
        
        # EMA 20 vs EMA 50
        ema_20 = recent['close'].ewm(span=20).mean()
        ema_50 = recent['close'].ewm(span=50).mean()
        
        if ema_20.iloc[-1] > ema_50.iloc[-1] * 1.01:  # 1% buffer
            return 'up'
        elif ema_20.iloc[-1] < ema_50.iloc[-1] * 0.99:
            return 'down'
        else:
            return 'neutral'
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Analyze volatility regime using ATR percentiles.
        
        Returns: (atr_percentile, regime)
        """
        if len(df) < 100:
            return 50.0, 'normal'
        
        # Calculate ATR
        atr = self._calculate_atr(df)
        
        # Get current ATR vs historical
        current_atr = atr.iloc[-1]
        if pd.isna(current_atr):
            return 50.0, 'normal'

        historical_atr = atr.dropna().tail(100)
        if historical_atr.empty:
            return 50.0, 'normal'

        less_than = (historical_atr < current_atr).mean()
        less_or_equal = (historical_atr <= current_atr).mean()
        percentile = ((less_than + less_or_equal) / 2) * 100

        current_close = max(float(df['close'].iloc[-1]), 1e-9)
        atr_pct_of_price = (float(current_atr) / current_close) * 100

        # Absolute intrabar range catches uniformly-volatile datasets that
        # percentile-only ranking can miss.
        if atr_pct_of_price >= 10.0:
            return max(percentile, self.extreme_vol_percentile), 'extreme'
        if atr_pct_of_price >= 5.0:
            return max(percentile, self.high_vol_percentile), 'high'
        
        # Classify regime
        if percentile >= self.extreme_vol_percentile:
            regime = 'extreme'
        elif percentile >= self.high_vol_percentile:
            regime = 'high'
        elif percentile <= self.low_vol_percentile:
            regime = 'low'
        else:
            regime = 'normal'
        
        return percentile, regime
    
    def _analyze_trend_strength(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Analyze trend strength using ADX.
        
        Returns: (adx_value, strength)
        """
        if len(df) < 30:
            return 0.0, 'weak'
        
        adx = self._calculate_adx(df)
        current_adx = adx.iloc[-1]
        
        if current_adx >= self.strong_trend_adx:
            strength = 'strong'
        elif current_adx >= self.moderate_trend_adx:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        return current_adx, strength
    
    def _analyze_liquidity(
        self,
        price: float,
        orderbook: Optional[Dict],
        volume_24h: Optional[float]
    ) -> LiquidityMetrics:
        """
        Analyze liquidity conditions.
        
        Returns: LiquidityMetrics with analysis
        """
        # Default values if no orderbook provided
        if orderbook is None:
            return LiquidityMetrics(
                spread_bps=5.0,
                bid_size=10.0,
                ask_size=10.0,
                volume_24h=volume_24h or 1000.0,
                is_acceptable=True,
                reason="No orderbook data, assuming acceptable"
            )
        
        # Calculate spread
        best_bid = orderbook['bids'][0][0] if orderbook.get('bids') else price * 0.999
        best_ask = orderbook['asks'][0][0] if orderbook.get('asks') else price * 1.001
        spread_bps = ((best_ask - best_bid) / price) * 10000
        
        # Get bid/ask sizes
        bid_size = orderbook['bids'][0][1] if orderbook.get('bids') else 0
        ask_size = orderbook['asks'][0][1] if orderbook.get('asks') else 0
        
        # Check acceptability
        issues = []
        
        if spread_bps > self.max_spread_bps:
            issues.append(f"Spread {spread_bps:.1f}bps > {self.max_spread_bps}bps")
        
        if bid_size < self.min_bid_ask_size_btc:
            issues.append(f"Bid size {bid_size:.2f} < {self.min_bid_ask_size_btc}")
        
        if ask_size < self.min_bid_ask_size_btc:
            issues.append(f"Ask size {ask_size:.2f} < {self.min_bid_ask_size_btc}")
        
        if volume_24h and volume_24h < self.min_volume_24h_btc:
            issues.append(f"Volume {volume_24h:.1f} < {self.min_volume_24h_btc}")
        
        is_acceptable = len(issues) == 0
        reason = "; ".join(issues) if issues else "Acceptable"
        
        return LiquidityMetrics(
            spread_bps=spread_bps,
            bid_size=bid_size,
            ask_size=ask_size,
            volume_24h=volume_24h or 0,
            is_acceptable=is_acceptable,
            reason=reason
        )
    
    def _classify_regime(
        self,
        trend_1h: str,
        trend_4h: str,
        trend_1d: str,
        volatility_regime: str,
        trend_strength: str,
        liquidity: LiquidityMetrics,
        atr_percentile: float,
        adx_value: float
    ) -> Tuple[MarketRegime, float]:
        """
        Classify overall market regime.
        
        Returns: (regime, confidence)
        """
        confidence = 0.7  # Base confidence
        
        # Critical condition: Liquidity
        if not liquidity.is_acceptable:
            return MarketRegime.LOW_LIQUIDITY, 0.9
        
        # Critical condition: Extreme volatility
        if volatility_regime == 'extreme':
            return MarketRegime.HIGH_VOLATILITY, 0.9
        
        # Strong trend conditions
        if trend_strength == 'strong':
            # Align higher timeframes
            if trend_1h == trend_4h == trend_1d == 'up':
                return MarketRegime.TRENDING_UP, 0.95
            elif trend_1h == trend_4h == trend_1d == 'down':
                return MarketRegime.TRENDING_DOWN, 0.95
            elif trend_1h == 'up' and (trend_4h == 'up' or trend_1d == 'up'):
                return MarketRegime.TRENDING_UP, 0.8
            elif trend_1h == 'down' and (trend_4h == 'down' or trend_1d == 'down'):
                return MarketRegime.TRENDING_DOWN, 0.8
        
        # Moderate trend
        if trend_strength == 'moderate':
            if trend_1h == 'up':
                return MarketRegime.TRENDING_UP, 0.6
            elif trend_1h == 'down':
                return MarketRegime.TRENDING_DOWN, 0.6
        
        # High volatility but not extreme
        if volatility_regime == 'high':
            return MarketRegime.HIGH_VOLATILITY, 0.7
        
        # Default to ranging
        return MarketRegime.RANGING, 0.6
    
    def _determine_trading_permission(
        self,
        regime: MarketRegime,
        liquidity: LiquidityMetrics,
        volatility_regime: str
    ) -> Tuple[bool, str]:
        """
        Determine if trading should be allowed.
        
        Returns: (allowed, reason)
        """
        # Block: Low liquidity
        if regime == MarketRegime.LOW_LIQUIDITY:
            return False, f"Low liquidity: {liquidity.reason}"
        
        # Block: No established regime (shouldn't happen but safety)
        if regime == MarketRegime.NO_TRADE:
            return False, "No tradeable regime identified"
        
        # Warning: Extreme volatility (block by default, can be configured)
        if regime == MarketRegime.HIGH_VOLATILITY and volatility_regime == 'extreme':
            return False, "Extreme volatility - risk too high"
        
        # Allow: All other regimes
        return True, f"Trading allowed in {regime.value} regime"
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        return atr
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed indicators
        atr = tr.rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx


# Singleton instance
_market_context_engine: Optional[MarketContextEngine] = None


def get_market_context_engine() -> MarketContextEngine:
    """Get the global MarketContextEngine instance."""
    global _market_context_engine
    if _market_context_engine is None:
        _market_context_engine = MarketContextEngine()
    return _market_context_engine
