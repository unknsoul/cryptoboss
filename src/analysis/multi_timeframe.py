"""
Multi-Timeframe Feature Fusion
Critical for accuracy: combines HTF trend, MTF momentum, LTF entry timing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TimeframeType(Enum):
    """Timeframe classification"""
    HTF = "higher"  # Higher timeframe (1h, 4h) - trend
    MTF = "medium"  # Medium timeframe (15m) - momentum
    LTF = "lower"   # Lower timeframe (1m, 5m) - entry


@dataclass
class MultiTimeframeSignal:
    """Signal from multi-timeframe analysis"""
    action: Optional[str]  # 'LONG', 'SHORT', or None
    confidence: float  # 0-1
    htf_trend: str  # 'UP', 'DOWN', 'NEUTRAL'
    mtf_momentum: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    ltf_entry_quality: float  # 0-1
    reasons: List[str]
    metadata: Dict


class MultiTimeframeAnalyzer:
    """
    Professional multi-timeframe analysis
    
    Rules:
    1. HTF defines overall trend (1h/4h)
    2. MTF confirms momentum (15m)
    3. LTF times entry (1m/5m)
    
    Only trade when ALL align.
    """
    
    def __init__(
        self,
        htf_period: str = '1h',
        mtf_period: str = '15m',
        ltf_period: str = '5m'
    ):
        """
        Initialize multi-timeframe analyzer
        
        Args:
            htf_period: Higher timeframe for trend
            mtf_period: Medium timeframe for momentum
            ltf_period: Lower timeframe for entry
        """
        self.htf_period = htf_period
        self.mtf_period = mtf_period
        self.ltf_period = ltf_period
    
    def analyze_htf_trend(self, df_htf: pd.DataFrame) -> Tuple[str, Dict]:
        """
        Analyze higher timeframe trend
        
        Uses:
        - EMA 50/200 crossover
        - ADX for trend strength
        - Higher highs/lower lows
        
        Returns:
            Tuple of (trend_direction, metadata)
        """
        # Calculate EMAs
        ema_50 = df_htf['close'].ewm(span=50).mean()
        ema_200 = df_htf['close'].ewm(span=200).mean()
        
        # Calculate ADX for trend strength
        adx = self._calculate_adx(df_htf, period=14)
        
        # Current values
        current_ema_50 = ema_50.iloc[-1]
        current_ema_200 = ema_200.iloc[-1]
        current_adx = adx.iloc[-1] if len(adx) > 0 else 0
        
        # Trend determination
        if current_ema_50 > current_ema_200 and current_adx > 25:
            trend = 'UP'
        elif current_ema_50 < current_ema_200 and current_adx > 25:
            trend = 'DOWN'
        else:
            trend = 'NEUTRAL'
        
        metadata = {
            'ema_50': current_ema_50,
            'ema_200': current_ema_200,
            'adx': current_adx,
            'trend_strength': 'STRONG' if current_adx > 25 else 'WEAK'
        }
        
        return trend, metadata
    
    def analyze_mtf_momentum(self, df_mtf: pd.DataFrame) -> Tuple[str, Dict]:
        """
        Analyze medium timeframe momentum
        
        Uses:
        - RSI
        - MACD
        - Rate of change
        
        Returns:
            Tuple of (momentum_state, metadata)
        """
        # Calculate RSI
        rsi = self._calculate_rsi(df_mtf['close'], period=14)
        
        # Calculate MACD
        macd_line, signal_line = self._calculate_macd(df_mtf['close'])
        
        # Calculate ROC
        roc = df_mtf['close'].pct_change(10)
        
        # Current values
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        current_macd = macd_line.iloc[-1] if len(macd_line) > 0 else 0
        current_signal = signal_line.iloc[-1] if len(signal_line) > 0 else 0
        current_roc = roc.iloc[-1] if len(roc) > 0 else 0
        
        # Momentum determination
        bullish_signals = 0
        if current_rsi > 50:
            bullish_signals += 1
        if current_macd > current_signal:
            bullish_signals += 1
        if current_roc > 0:
            bullish_signals += 1
        
        if bullish_signals >= 2:
            momentum = 'BULLISH'
        elif bullish_signals <= 1:
            momentum = 'BEARISH'
        else:
            momentum = 'NEUTRAL'
        
        metadata = {
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_signal,
            'roc': current_roc,
            'bullish_signals': bullish_signals
        }
        
        return momentum, metadata
    
    def analyze_ltf_entry(self, df_ltf: pd.DataFrame, htf_trend: str) -> Tuple[float, Dict]:
        """
        Analyze lower timeframe for entry quality
        
        Looks for:
        - Pullbacks in HTF direction
        - Support/resistance bounces
        - Volume confirmation
        
        Returns:
            Tuple of (entry_quality 0-1, metadata)
        """
        # Calculate short-term moving average
        sma_20 = df_ltf['close'].rolling(20).mean()
        
        # Price position relative to SMA
        current_price = df_ltf['close'].iloc[-1]
        current_sma = sma_20.iloc[-1]
        
        # Volume analysis
        volume_ma = df_ltf['volume'].rolling(20).mean()
        current_volume = df_ltf['volume'].iloc[-1]
        volume_ratio = current_volume / volume_ma.iloc[-1] if len(volume_ma) > 0 else 1
        
        # Entry quality scoring
        entry_quality = 0.0
        reasons = []
        
        if htf_trend == 'UP':
            # Look for pullbacks to buy (price below SMA but not too far)
            if current_price < current_sma:
                pullback_pct = (current_sma - current_price) / current_sma
                if 0.005 < pullback_pct < 0.02:  # 0.5% to 2% pullback
                    entry_quality += 0.4
                    reasons.append("Good pullback for long entry")
            
            # Volume confirmation
            if volume_ratio > 1.2:
                entry_quality += 0.3
                reasons.append("Volume surge")
            
        elif htf_trend == 'DOWN':
            # Look for rallies to sell short
            if current_price > current_sma:
                rally_pct = (current_price - current_sma) / current_sma
                if 0.005 < rally_pct < 0.02:
                    entry_quality += 0.4
                    reasons.append("Good rally for short entry")
            
            if volume_ratio > 1.2:
                entry_quality += 0.3
                reasons.append("Volume surge")
        
        # Base quality for any setup
        if entry_quality > 0:
            entry_quality = min(entry_quality + 0.3, 1.0)
        
        metadata = {
            'price': current_price,
            'sma_20': current_sma,
            'volume_ratio': volume_ratio,
            'reasons': reasons
        }
        
        return entry_quality, metadata
    
    def generate_signal(
        self,
        df_htf: pd.DataFrame,
        df_mtf: pd.DataFrame,
        df_ltf: pd.DataFrame,
        min_confidence: float = 0.40  # Lowered for more signals
    ) -> MultiTimeframeSignal:
        """
        Generate multi-timeframe signal
        
        Logic:
        1. Analyze HTF trend
        2. Analyze MTF momentum
        3. Analyze LTF entry
        4. Combine all timeframes
        5. Only signal if ALL align
        
        Args:
            df_htf: Higher timeframe data
            df_mtf: Medium timeframe data
            df_ltf: Lower timeframe data
            min_confidence: Minimum confidence to signal (default 0.65)
            
        Returns:
            MultiTimeframeSignal
        """
        # Analyze each timeframe
        htf_trend, htf_meta = self.analyze_htf_trend(df_htf)
        mtf_momentum, mtf_meta = self.analyze_mtf_momentum(df_mtf)
        ltf_quality, ltf_meta = self.analyze_ltf_entry(df_ltf, htf_trend)
        
        # Determine action
        action = None
        confidence = 0.0
        reasons = []
        
        # LONG setup
        if htf_trend == 'UP' and mtf_momentum == 'BULLISH' and ltf_quality > 0.3:
            action = 'LONG'
            confidence = (0.4 + ltf_quality * 0.6)  # HTF/MTF weight + LTF quality
            reasons.append(f"HTF trend: {htf_trend}")
            reasons.append(f"MTF momentum: {mtf_momentum}")
            reasons.append(f"LTF entry quality: {ltf_quality:.2f}")
            reasons.extend(ltf_meta.get('reasons', []))
        
        # SHORT setup
        elif htf_trend == 'DOWN' and mtf_momentum == 'BEARISH' and ltf_quality > 0.3:
            action = 'SHORT'
            confidence = (0.4 + ltf_quality * 0.6)
            reasons.append(f"HTF trend: {htf_trend}")
            reasons.append(f"MTF momentum: {mtf_momentum}")
            reasons.append(f"LTF entry quality: {ltf_quality:.2f}")
            reasons.extend(ltf_meta.get('reasons', []))
        
        # Filter by minimum confidence
        if confidence < min_confidence:
            action = None
            reasons.append(f"Confidence {confidence:.2f} below minimum {min_confidence:.2f}")
        
        return MultiTimeframeSignal(
            action=action,
            confidence=confidence,
            htf_trend=htf_trend,
            mtf_momentum=mtf_momentum,
            ltf_entry_quality=ltf_quality,
            reasons=reasons,
            metadata={
                'htf': htf_meta,
                'mtf': mtf_meta,
                'ltf': ltf_meta
            }
        )
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
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
        atr = tr.rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        return macd_line, signal_line


if __name__ == '__main__':
    # Example usage
    print("=" * 70)
    print("MULTI-TIMEFRAME ANALYSIS EXAMPLE")
    print("=" * 70)
    
    # Generate sample data for different timeframes
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
    
    # Simulate trending market
    trend = np.cumsum(np.random.randn(1000) * 0.1) + 100
    
    # LTF (5m) - 1000 candles
    df_ltf = pd.DataFrame({
        'close': trend + np.random.randn(1000) * 0.2,
        'high': trend + np.abs(np.random.randn(1000) * 0.3),
        'low': trend - np.abs(np.random.randn(1000) * 0.3),
        'volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    
    # MTF (15m) - resample to 15m
    df_mtf = df_ltf.resample('15T').agg({
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'volume': 'sum'
    }).dropna()
    
    # HTF (1h) - resample to 1h
    df_htf = df_ltf.resample('1H').agg({
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'volume': 'sum'
    }).dropna()
    
    # Initialize analyzer
    analyzer = MultiTimeframeAnalyzer(htf_period='1h', mtf_period='15m', ltf_period='5m')
    
    # Generate signal
    signal = analyzer.generate_signal(df_htf, df_mtf, df_ltf, min_confidence=0.40)
    
    print(f"\nMulti-Timeframe Signal:")
    print(f"  Action: {signal.action}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  HTF Trend: {signal.htf_trend}")
    print(f"  MTF Momentum: {signal.mtf_momentum}")
    print(f"  LTF Entry Quality: {signal.ltf_entry_quality:.2f}")
    print(f"\nReasons:")
    for reason in signal.reasons:
        print(f"  - {reason}")
    
    print("\n" + "=" * 70)
    print("✓ Multi-timeframe analysis prevents trading against trend")
    print("✓ Only signals when ALL timeframes align")
    print("✓ Confidence threshold filters low-quality setups")
    print("=" * 70)
