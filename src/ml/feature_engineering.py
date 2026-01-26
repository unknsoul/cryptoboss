"""
Advanced Feature Engineering
Market microstructure and high-quality predictive features
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for improved prediction accuracy
    
    Features added:
    - Market microstructure (volume profile, spread, flow)
    - Time-based patterns (hourly, daily seasonality)
    - Volatility regime classification
    - Momentum quality (not just direction)
    - Cross-asset correlations
    """
    
    def __init__(self):
        self.feature_names = []
    
    def engineer_features(self, df: pd.DataFrame, 
                         orderbook: Optional[Dict] = None,
                         btc_prices: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Create comprehensive feature set
        
        Args:
            df: OHLCV dataframe with columns: open, high, low, close, volume
            orderbook: Optional order book data
            btc_prices: Optional BTC prices for correlation features
        
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Market Microstructure Features
        features = pd.concat([features, self._volume_profile_features(df)], axis=1)
        
        if orderbook:
            features = pd.concat([features, self._spread_features(orderbook)], axis=1)
        
        # 2. Time-Based Features
        features = pd.concat([features, self._time_features(df)], axis=1)
        
        # 3. Advanced Volatility Features
        features = pd.concat([features, self._volatility_features(df)], axis=1)
        
        # 4. Momentum Quality Features
        features = pd.concat([features, self._momentum_quality(df)], axis=1)
        
        # 5. Cross-Asset Features (if BTC data available)
        if btc_prices is not None:
            features = pd.concat([features, self._cross_asset_features(df['close'], btc_prices)], axis=1)
        
        # 6. Existing features (enhanced)
        features = pd.concat([features, self._base_features(df)], axis=1)
        
        self.feature_names = list(features.columns)
        
        return features.dropna()
    
    def _volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume profile analysis
        Identifies support/resistance levels
        """
        features = pd.DataFrame(index=df.index)
        
        window = 100
        
        # Volume-weighted average price (VWAP)
        features['vwap'] = (df['close'] * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
        features['price_to_vwap'] = df['close'] / features['vwap'] - 1
        
        # Volume concentration (where is volume concentrated?)
        features['volume_concentration'] = df['volume'].rolling(window).apply(
            lambda x: x.max() / (x.mean() + 1e-8)
        )
        
        # Volume momentum
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        
        # Money flow (buying vs selling pressure)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0)
        negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0)
        
        features['money_flow_ratio'] = (
            positive_flow.rolling(14).sum() / 
            (negative_flow.rolling(14).sum() + 1e-8)
        )
        
        return features
    
    def _spread_features(self, orderbook: Dict) -> pd.DataFrame:
        """
        Order book spread analysis
        """
        # This would be populated with real-time orderbook data
        # Simplified version here
        features = pd.DataFrame()
        
        if 'bids' in orderbook and 'asks' in orderbook:
            best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
            best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
            
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            features['spread_pct'] = [spread / mid_price if mid_price > 0 else 0]
            features['orderbook_imbalance'] = [self._calculate_ob_imbalance(orderbook)]
        
        return features
    
    def _calculate_ob_imbalance(self, orderbook: Dict, depth: int = 10) -> float:
        """
        Calculate order book imbalance
        +1 = all bids, -1 = all asks
        """
        if not orderbook.get('bids') or not orderbook.get('asks'):
            return 0
        
        bid_volume = sum(bid[1] for bid in orderbook['bids'][:depth])
        ask_volume = sum(ask[1] for ask in orderbook['asks'][:depth])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0
        
        return (bid_volume - ask_volume) / total_volume
    
    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-based cyclical features
        Markets behave differently at different times
        """
        features = pd.DataFrame(index=df.index)
        
        # Assuming df.index is DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour
            day_of_week = df.index.dayofweek
            
            # Cyclical encoding (preserves circular nature of time)
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            
            # Categorical time features
            features['is_asian_hours'] = ((hour >= 0) & (hour < 8)).astype(int)
            features['is_european_hours'] = ((hour >= 8) & (hour < 16)).astype(int)
            features['is_us_hours'] = ((hour >= 16) & (hour < 24)).astype(int)
            features['is_weekend'] = (day_of_week >= 5).astype(int)
        
        return features
    
    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multiple volatility measures
        Different measures capture different aspects
        """
        features = pd.DataFrame(index=df.index)
        
        # Parkinson volatility (uses high-low range)
        features['parkinson_vol'] = np.sqrt(
            1/(4*np.log(2)) * (np.log(df['high']/df['low'])**2).rolling(14).mean()
        )
        
        # Close-to-close volatility
        returns = df['close'].pct_change()
        features['cc_volatility'] = returns.rolling(14).std() * np.sqrt(365 * 24)  # Annualized
        
        # Volatility of volatility
        features['vol_of_vol'] = features['cc_volatility'].rolling(14).std()
        
        # Volatility ratio (current vs historical)
        vol_ma = features['cc_volatility'].rolling(50).mean()
        features['vol_ratio'] = features['cc_volatility'] / (vol_ma + 1e-8)
        
        # Volatility regime classification
        features['vol_regime'] = pd.cut(
            features['vol_ratio'],
            bins=[0, 0.7, 1.3, 2.0, np.inf],
            labels=[0, 1, 2, 3]  # LOW, NORMAL, ELEVATED, EXTREME
        ).astype(float)
        
        return features
    
    def _momentum_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum quality features (strength, not just direction)
        """
        features = pd.DataFrame(index=df.index)
        
        prices = df['close']
        
        # Trend strength (R-squared of linear regression)
        def rolling_r_squared(prices, window=50):
            r_squared_list = []
            for i in range(len(prices)):
                if i < window:
                    r_squared_list.append(np.nan)
                else:
                    y = prices.iloc[i-window:i].values
                    x = np.arange(window)
                    
                    if len(y) > 0 and not np.isnan(y).any():
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        r_squared_list.append(r_value**2)
                    else:
                        r_squared_list.append(np.nan)
            
            return pd.Series(r_squared_list, index=prices.index)
        
        features['trend_strength'] = rolling_r_squared(prices, window=50)
        
        # Momentum streak (consecutive positive/negative returns)
        returns = prices.pct_change()
        sign_changes = (returns.fillna(0) >= 0).astype(int).diff().fillna(0) != 0
        streak_id = sign_changes.cumsum()
        features['momentum_streak'] = returns.groupby(streak_id).cumsum()
        
        # Hurst exponent approximation (trending vs mean-reverting)
        features['hurst_approx'] = self._approximate_hurst(prices, window=100)
        
        return features
    
    def _approximate_hurst(self, prices: pd.Series, window: int = 100) -> pd.Series:
        """
        Approximate Hurst exponent
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        hurst_values = []
        
        for i in range(len(prices)):
            if i < window:
                hurst_values.append(np.nan)
            else:
                series = prices.iloc[i-window:i]
                
                # Simplified Hurst calculation using R/S analysis
                lags = range(2, min(window // 2, 20))
                tau = []
                
                for lag in lags:
                    # Calculate std of differences
                    std = series.diff(lag).dropna().std()
                    tau.append(std)
                
                if len(tau) > 0 and not np.isnan(tau).any():
                    # Hurst = slope of log(tau) vs log(lag)
                    log_lags = np.log(list(lags))
                    log_tau = np.log(tau)
                    
                    slope, _, _, _, _ = stats.linregress(log_lags, log_tau)
                    hurst = slope  # Approximation
                    hurst_values.append(np.clip(hurst, 0, 1))
                else:
                    hurst_values.append(0.5)
        
        return pd.Series(hurst_values, index=prices.index)
    
    def _cross_asset_features(self, asset_prices: pd.Series, btc_prices: pd.Series) -> pd.DataFrame:
        """
        Cross-asset correlation features
        Altcoins often follow BTC with lag
        """
        features = pd.DataFrame(index=asset_prices.index)
        
        # Align indices
        common_idx = asset_prices.index.intersection(btc_prices.index)
        asset_aligned = asset_prices.loc[common_idx]
        btc_aligned = btc_prices.loc[common_idx]
        
        # Rolling correlation
        asset_returns = asset_aligned.pct_change()
        btc_returns = btc_aligned.pct_change()
        
        features['btc_correlation'] = asset_returns.rolling(50).corr(btc_returns)
        
        # Beta to BTC
        covariance = asset_returns.rolling(50).cov(btc_returns)
        btc_variance = btc_returns.rolling(50).var()
        features['btc_beta'] = covariance / (btc_variance + 1e-8)
        
        # BTC momentum (leading indicator)
        features['btc_momentum_5'] = btc_aligned.pct_change(5)
        features['btc_momentum_20'] = btc_aligned.pct_change(20)
        
        return features
    
    def _base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced base features
        """
        features = pd.DataFrame(index=df.index)
        
        prices = df['close']
        
        # Z-score price
        sma_50 = prices.rolling(50).mean()
        std_50 = prices.rolling(50).std()
        features['z_score_price'] = (prices - sma_50) / (std_50 + 1e-8)
        
        # RSI normalized
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features['rsi_norm'] = (rsi - 50) / 50  # Scale to -1..1
        
        # Log returns and lags
        features['log_ret'] = np.log(prices / prices.shift(1))
        for lag in [1, 2, 3, 5]:
            features[f'ret_lag_{lag}'] = features['log_ret'].shift(lag)
        
        return features
    
    def get_feature_importance(self, model, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
        
        return pd.DataFrame()


if __name__ == "__main__":
    # Test feature engineer
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    
    df = pd.DataFrame({
        'open': 45000 + np.random.randn(1000).cumsum() * 100,
        'high': 45100 + np.random.randn(1000).cumsum() * 100,
        'low': 44900 + np.random.randn(1000).cumsum() * 100,
        'close': 45000 + np.random.randn(1000).cumsum() * 100,
        'volume': np.random.uniform(800, 1200, 1000)
    }, index=dates)
    
    engineer = AdvancedFeatureEngineer()
    features = engineer.engineer_features(df)
    
    print("=" * 70)
    print("ADVANCED FEATURE ENGINEERING TEST")
    print("=" * 70)
    print(f"\nTotal features created: {len(features.columns)}")
    print(f"\nFeature names:")
    for i, feat in enumerate(features.columns, 1):
        print(f"  {i}. {feat}")
    
    print(f"\nSample data:")
    print(features.tail())
    
    print("\n✅ Feature engineering test complete")
