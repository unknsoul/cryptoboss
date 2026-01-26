"""
Dynamic Volatility-Adjusted Labeling
Critical: Labels must align with real execution, not theory
"""

import numpy as np
import pandas as pd
from typing import Tuple


class DynamicLabeler:
    """
    Create realistic labels for ML training
    
    Problems with fixed labeling:
    - Ignores volatility (2% move in low vol ≠ 2% move in high vol)
    - Doesn't account for stop loss/take profit
    - Unrealistic profit targets
    
    Solution:
    - ATR-based dynamic targets
    - Label based on whether SL or TP hits first
    - Volatility-adjusted thresholds
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        tp_atr_multiplier: float = 2.0,
        sl_atr_multiplier: float = 1.0
    ):
        """
        Initialize dynamic labeler
        
        Args:
            atr_period: Period for ATR calculation
            tp_atr_multiplier: Take profit distance in ATR multiples
            sl_atr_multiplier: Stop loss distance in ATR multiples
        """
        self.atr_period = atr_period
        self.tp_multiplier = tp_atr_multiplier
        self.sl_multiplier = sl_atr_multiplier
    
    def create_labels(
        self,
        df: pd.DataFrame,
        lookforward_bars: int = 50
    ) -> pd.Series:
        """
        Create dynamic labels based on SL/TP outcome
        
        For each bar:
        1. Calculate ATR-based TP and SL levels
        2. Look forward to see which hits first
        3. Label:
           - 1 (BUY) if TP hits before SL for long
           - -1 (SELL) if TP hits before SL for short
           - 0 (HOLD) if SL hits first or neither
        
        This aligns training with actual trading outcomes.
        
        Args:
            df: DataFrame with OHLCV data
            lookforward_bars: How many bars ahead to check
            
        Returns:
            Series with labels (-1, 0, 1)
        """
        # Calculate ATR
        atr = self._calculate_atr(df)
        
        labels = []
        
        for i in range(len(df) - lookforward_bars):
            current_close = df['close'].iloc[i]
            current_atr = atr.iloc[i]
            
            if pd.isna(current_atr) or current_atr == 0:
                labels.append(0)
                continue
            
            # Define TP and SL based on ATR
            long_tp = current_close + (current_atr * self.tp_multiplier)
            long_sl = current_close - (current_atr * self.sl_multiplier)
            
            short_tp = current_close - (current_atr * self.tp_multiplier)
            short_sl = current_close + (current_atr * self.sl_multiplier)
            
            # Look forward
            future_highs = df['high'].iloc[i+1:i+1+lookforward_bars]
            future_lows = df['low'].iloc[i+1:i+1+lookforward_bars]
            
            # Check LONG outcome
            long_tp_hit = (future_highs >= long_tp).any()
            long_sl_hit = (future_lows <= long_sl).any()
            
            if long_tp_hit and long_sl_hit:
                # Both hit - which came first?
                tp_bar = (future_highs >= long_tp).idxmax()
                sl_bar = (future_lows <= long_sl).idxmax()
                long_profitable = tp_bar < sl_bar
            elif long_tp_hit:
                long_profitable = True
            else:
                long_profitable = False
            
            # Check SHORT outcome
            short_tp_hit = (future_lows <= short_tp).any()
            short_sl_hit = (future_highs >= short_sl).any()
            
            if short_tp_hit and short_sl_hit:
                tp_bar = (future_lows <= short_tp).idxmax()
                sl_bar = (future_highs >= short_sl).idxmax()
                short_profitable = tp_bar < sl_bar
            elif short_tp_hit:
                short_profitable = True
            else:
                short_profitable = False
            
            # Label decision
            if long_profitable and not short_profitable:
                label = 1  # BUY
            elif short_profitable and not long_profitable:
                label = -1  # SELL
            else:
                label = 0  # HOLD
            
            labels.append(label)
        
        # Pad end with neutral labels
        labels.extend([0] * lookforward_bars)
        
        return pd.Series(labels, index=df.index)
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(self.atr_period).mean()
        return atr


if __name__ == '__main__':
    # Example
    np.random.seed(42)
    
    # Generate sample price data
    trend = np.cumsum(np.random.randn(200) * 0.5) + 100
    df = pd.DataFrame({
        'close': trend,
        'high': trend + np.abs(np.random.randn(200) * 0.3),
        'low': trend - np.abs(np.random.randn(200) * 0.3),
        'volume': np.random.randint(100, 1000, 200)
    })
    
    labeler = DynamicLabeler(atr_period=14, tp_atr_multiplier=2.0, sl_atr_multiplier=1.0)
    labels = labeler.create_labels(df, lookforward_bars=20)
    
    print("Dynamic Label Distribution:")
    print(labels.value_counts(normalize=True))
    print("\n✓ Labels based on realistic SL/TP outcomes")
    print("✓ Volatility-adjusted targets")
    print("✓ Aligns training with actual trading")
