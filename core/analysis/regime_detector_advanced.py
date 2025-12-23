"""
Advanced Market Regime Detection
Critical for accuracy: different strategies for different market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeInfo:
    """Market regime information"""
    regime: MarketRegime
    confidence: float  # 0-1
    atr_percentile: float  # Current ATR vs historical
    adx_value: float
    hurst_exponent: Optional[float]
    volatility_cluster: bool
    metadata: Dict


class AdvancedRegimeDetector:
    """
    Professional market regime detection
    
    Uses multiple indicators:
    - ATR for volatility
    - ADX for trend strength
    - Hurst exponent for mean reversion
    - Volatility clustering
    - Price action patterns
    
    Prevents trading wrong strategies in wrong regimes.
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        adx_period: int = 14,
        lookback_period: int = 100
    ):
        """
        Initialize regime detector
        
        Args:
            atr_period: ATR calculation period
            adx_period: ADX calculation period
            lookback_period: Historical comparison window
        """
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.lookback_period = lookback_period
    
    def detect_regime(self, df: pd.DataFrame) -> RegimeInfo:
        """
        Detect current market regime
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            RegimeInfo with detected regime and metadata
        """
        # Calculate indicators
        atr = self._calculate_atr(df)
        adx = self._calculate_adx(df)
        hurst = self._calculate_hurst_exponent(df['close'].tail(self.lookback_period))
        
        # ATR percentile (current vs historical)
        atr_series = pd.Series(atr)
        current_atr = atr_series.iloc[-1]
        atr_percentile = (atr_series.tail(self.lookback_period) < current_atr).mean()
        
        # Current ADX
        adx_series = pd.Series(adx)
        current_adx = adx_series.iloc[-1] if len(adx_series) > 0 else 0
        
        # Volatility clustering detection
        vol_cluster = self._detect_volatility_clustering(atr_series)
        
        # Determine regime
        regime, confidence = self._classify_regime(
            atr_percentile, current_adx, hurst, df
        )
        
        return RegimeInfo(
            regime=regime,
            confidence=confidence,
            atr_percentile=atr_percentile,
            adx_value=current_adx,
            hurst_exponent=hurst,
            volatility_cluster=vol_cluster,
            metadata={
                'atr': current_atr,
                'adx': current_adx,
                'hurst': hurst,
                'vol_cluster': vol_cluster
            }
        )
    
    def _classify_regime(
        self,
        atr_percentile: float,
        adx: float,
        hurst: Optional[float],
        df: pd.DataFrame
    ) -> Tuple[MarketRegime, float]:
        """
        Classify market regime based on indicators
        
        Logic:
        1. High volatility if ATR > 80th percentile
        2. Low volatility if ATR < 20th percentile
        3. Trending if ADX > 25
        4. Ranging if ADX < 20
        5. Direction from price action
        
        Returns:
            Tuple of (regime, confidence)
        """
        confidence = 0.7  # Base confidence
        
        # Extreme volatility regimes
        if atr_percentile > 0.9:
            return MarketRegime.HIGH_VOLATILITY, 0.9
        elif atr_percentile < 0.1:
            return MarketRegime.LOW_VOLATILITY, 0.8
        
        # Trending vs ranging
        if adx > 30:
            # Strong trend - determine direction
            ema_50 = df['close'].ewm(span=50).mean()
            ema_200 = df['close'].ewm(span=200).mean()
            
            if ema_50.iloc[-1] > ema_200.iloc[-1]:
                regime = MarketRegime.TRENDING_UP
                confidence = min(adx / 50, 1.0)  # Higher ADX = higher confidence
            else:
                regime = MarketRegime.TRENDING_DOWN
                confidence = min(adx / 50, 1.0)
                
        elif adx < 20:
            # Weak trend = ranging
            regime = MarketRegime.RANGING
            confidence = 1.0 - (adx / 30)  # Lower ADX = higher confidence in range
            
        else:
            # Transitional regime - use Hurst
            if hurst is not None:
                if hurst < 0.4:
                    regime = MarketRegime.RANGING  # Mean reverting
                    confidence = 0.6
                else:
                    # Determine trend direction
                    if df['close'].iloc[-1] > df['close'].iloc[-50]:
                        regime = MarketRegime.TRENDING_UP
                    else:
                        regime = MarketRegime.TRENDING_DOWN
                    confidence = 0.5
            else:
                # Default to ranging in uncertain conditions
                regime = MarketRegime.RANGING
                confidence = 0.5
        
        return regime, confidence
    
    def should_trade_strategy(
        self,
        strategy_type: str,
        current_regime: MarketRegime
    ) -> bool:
        """
        Determine if strategy should trade in current regime
        
        Rules:
        - Trend following: Only in TRENDING regimes
        - Mean reversion: Only in RANGING or LOW_VOL
        - Breakout: Only in TRENDING or HIGH_VOL
        - Scalping: Only in RANGING or LOW_VOL
        
        Args:
            strategy_type: Type of strategy ('trend', 'mean_reversion', 'breakout', 'scalping')
            current_regime: Current market regime
            
        Returns:
            True if strategy should trade, False otherwise
        """
        trading_matrix = {
            'trend': [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
            'momentum': [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.HIGH_VOLATILITY],
            'quality_momentum': [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
            'mean_reversion': [MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY, MarketRegime.HIGH_VOLATILITY],
            'stat_arb': [MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY],
            'volatility_targeting': [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.HIGH_VOLATILITY],
            'microstructure': [MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY, MarketRegime.HIGH_VOLATILITY],
            'breakout': [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.HIGH_VOLATILITY],
            'scalping': [MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY]
        }
        
        allowed_regimes = trading_matrix.get(strategy_type, [])
        return current_regime in allowed_regimes
    
    def _calculate_atr(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate ATR (Average True Range)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(self.atr_period).mean()
        return atr.values
    
    def _calculate_adx(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate ADX (Average Directional Index)"""
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
        atr = tr.rolling(self.adx_period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(self.adx_period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(self.adx_period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(self.adx_period).mean()
        
        return adx.values
    
    def _calculate_hurst_exponent(self, prices: pd.Series) -> Optional[float]:
        """
        Calculate Hurst exponent
        
        Interpretation:
        - H < 0.5: Mean reverting (good for ranging strategies)
        - H ~ 0.5: Random walk
        - H > 0.5: Trending (good for trend strategies)
        
        Returns:
            Hurst exponent or None if calculation fails
        """
        try:
            lags = range(2, 20)
            tau = [np.std(np.subtract(prices[lag:].values, prices[:-lag].values)) for lag in lags]
            
            # Linear regression
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0
            
            return hurst
        except:
            return None
    
    def _detect_volatility_clustering(self, atr_series: pd.Series) -> bool:
        """
        Detect volatility clustering (GARCH effect)
        
        High volatility tends to follow high volatility.
        
        Returns:
            True if volatility is clustering
        """
        if len(atr_series) < 20:
            return False
        
        # Check if recent volatility is persistently high
        recent_atr = atr_series.tail(10).mean()
        historical_atr = atr_series.tail(50).mean()
        
        return recent_atr > historical_atr * 1.3


if __name__ == '__main__':
    # Example usage
    print("=" * 70)
    print("ADVANCED REGIME DETECTION EXAMPLE")
    print("=" * 70)
    
    # Generate sample data with different regimes
    np.random.seed(42)
    
    # Trending period
    trending_data = np.cumsum(np.random.randn(200) * 0.5) + 100
    
    # Ranging period
    ranging_data = 110 + np.random.randn(200) * 2
    
    # High volatility period
    high_vol_data = 120 + np.cumsum(np.random.randn(200) * 3)
    
    # Combine
    all_data = np.concatenate([trending_data, ranging_data, high_vol_data])
    
    df = pd.DataFrame({
        'close': all_data,
        'high': all_data + np.abs(np.random.randn(600) * 0.5),
        'low': all_data - np.abs(np.random.randn(600) * 0.5),
        'volume': np.random.randint(100, 1000, 600)
    })
    
    # Initialize detector
    detector = AdvancedRegimeDetector(atr_period=14, adx_period=14, lookback_period=100)
    
    # Test on different periods
    periods = {
        'Trending': range(150, 200),
        'Ranging': range(250, 300),
        'High Volatility': range(450, 500)
    }
    
    for period_name, period_range in periods.items():
        period_df = df.iloc[list(period_range)]
        regime_info = detector.detect_regime(period_df)
        
        print(f"\n{period_name} Period:")
        print(f"  Detected Regime: {regime_info.regime.value}")
        print(f"  Confidence: {regime_info.confidence:.2%}")
        print(f"  ADX: {regime_info.adx_value:.2f}")
        print(f"  ATR Percentile: {regime_info.atr_percentile:.2%}")
        print(f"  Hurst: {regime_info.hurst_exponent:.3f}" if regime_info.hurst_exponent else "  Hurst: N/A")
        print(f"  Volatility Clustering: {regime_info.volatility_cluster}")
        
        # Test strategy recommendations
        print(f"\n  Should Trade:")
        for strat in ['trend', 'mean_reversion', 'breakout', 'scalping']:
            should_trade = detector.should_trade_strategy(strat, regime_info.regime)
            status = "✓ YES" if should_trade else "✗ NO"
            print(f"    {strat:15s}: {status}")
    
    print("\n" + "=" * 70)
    print("✓ Regime detection prevents trading wrong strategies")
    print("✓ Each regime has optimal strategy types")
    print("✓ Confidence scoring helps filter uncertain periods")
    print("=" * 70)
