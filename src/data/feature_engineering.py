"""
Feature Engineering Pipeline
Systematic feature generation with registry and storage.

Critical Rule: Training features = Live features (byte-for-byte identical)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Systematic feature generation.
    
    Features are defined once in FEATURE_REGISTRY and used consistently
    for both training and live trading.
    
    Features include:
    - Returns (log, rolling, z-score)
    - Volatility (ATR, realized, expanding)
    - Momentum (slopes, acceleration)
    - Volume (relative, imbalance)
    - Regime indicators (trending/ranging)
    """
    
    # Feature registry - single source of truth
    FEATURE_REGISTRY: Dict[str, Callable] = {}
    
    def __init__(self):
        """Initialize feature engine."""
        self._register_features()
        logger.info(f"FeatureEngine initialized with {len(self.FEATURE_REGISTRY)} features")
    
    def _register_features(self):
        """Register all feature definitions."""
        
        # === RETURNS FEATURES ===
        self.FEATURE_REGISTRY['returns_1'] = lambda df: df['close'].pct_change(1)
        self.FEATURE_REGISTRY['returns_4'] = lambda df: df['close'].pct_change(4)
        self.FEATURE_REGISTRY['returns_24'] = lambda df: df['close'].pct_change(24)
        
        # Log returns
        self.FEATURE_REGISTRY['log_returns_1'] = lambda df: np.log(df['close'] / df['close'].shift(1))
        self.FEATURE_REGISTRY['log_returns_4'] = lambda df: np.log(df['close'] / df['close'].shift(4))
        
        # Rolling returns
        self.FEATURE_REGISTRY['returns_roll_mean_10'] = lambda df: df['close'].pct_change().rolling(10).mean()
        self.FEATURE_REGISTRY['returns_roll_std_10'] = lambda df: df['close'].pct_change().rolling(10).std()
        
        # Z-score of returns
        self.FEATURE_REGISTRY['returns_zscore_20'] = lambda df: self._zscore(df['close'].pct_change(), 20)
        
        # === VOLATILITY FEATURES ===
        self.FEATURE_REGISTRY['volatility_atr_14'] = lambda df: self._calculate_atr(df, 14)
        self.FEATURE_REGISTRY['volatility_atr_28'] = lambda df: self._calculate_atr(df, 28)
        
        # Realized volatility
        self.FEATURE_REGISTRY['volatility_realized_24'] = lambda df: df['close'].pct_change().rolling(24).std()
        self.FEATURE_REGISTRY['volatility_realized_168'] = lambda df: df['close'].pct_change().rolling(168).std()  # 1 week
        
        # Volatility ratio
        self.FEATURE_REGISTRY['volatility_ratio'] = lambda df: (
            df['close'].pct_change().rolling(24).std() / 
            df['close'].pct_change().rolling(168).std()
        )
        
        # === MOMENTUM FEATURES ===
        # Price momentum
        self.FEATURE_REGISTRY['momentum_roc_10'] = lambda df: ((df['close'] - df['close'].shift(10)) / df['close'].shift(10))
        self.FEATURE_REGISTRY['momentum_roc_20'] = lambda df: ((df['close'] - df['close'].shift(20)) / df['close'].shift(20))
        
        # Momentum slope (regression slope of price)
        self.FEATURE_REGISTRY['momentum_slope_20'] = lambda df: df['close'].rolling(20).apply(self._linear_slope, raw=False)
        
        # Acceleration (2nd derivative)
        self.FEATURE_REGISTRY['momentum_acceleration'] = lambda df: df['close'].pct_change().diff()
        
        # === VOLUME FEATURES ===
        # Relative volume
        self.FEATURE_REGISTRY['volume_relative_24'] = lambda df: df['volume'] / df['volume'].rolling(24).mean()
        
        # Volume momentum
        self.FEATURE_REGISTRY['volume_momentum_10'] = lambda df: df['volume'].pct_change(10)
        
        # Price-volume correlation
        self.FEATURE_REGISTRY['price_volume_corr_20'] = lambda df: (
            df['close'].pct_change().rolling(20).corr(df['volume'].pct_change())
        )
        
        # === TREND FEATURES ===
        # Moving averages
        self.FEATURE_REGISTRY['ma_ratio_20_50'] = lambda df: df['close'].rolling(20).mean() / df['close'].rolling(50).mean()
        self.FEATURE_REGISTRY['price_to_ma20'] = lambda df: df['close'] / df['close'].rolling(20).mean()
        self.FEATURE_REGISTRY['price_to_ma50'] = lambda df: df['close'] / df['close'].rolling(50).mean()
        
        # ADX (trend strength)
        self.FEATURE_REGISTRY['adx_14'] = lambda df: self._calculate_adx(df, 14)
        
        # === REGIME FEATURES ===
        # Trending vs ranging (based on ADX & volatility)
        self.FEATURE_REGISTRY['regime_trending'] = lambda df: (self._calculate_adx(df, 14) > 25).astype(int)
        
        # High volatility regime
        self.FEATURE_REGISTRY['regime_high_vol'] = lambda df: (
            (df['close'].pct_change().rolling(24).std() > 
             df['close'].pct_change().rolling(168).std() * 1.5).astype(int)
        )
        
        # === STATISTICAL FEATURES ===
        # Skewness of returns
        self.FEATURE_REGISTRY['returns_skew_20'] = lambda df: df['close'].pct_change().rolling(20).skew()
        
        # Kurtosis of returns
        self.FEATURE_REGISTRY['returns_kurt_20'] = lambda df: df['close'].pct_change().rolling(20).kurt()
    
    def generate_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns
            features: List of feature names to generate (None = all)
            
        Returns:
            DataFrame with features
        """
        if features is None:
            features = list(self.FEATURE_REGISTRY.keys())
        
        logger.info(f"Generating {len(features)} features...")
        
        feature_df = df.copy()
        
        for feature_name in features:
            if feature_name not in self.FEATURE_REGISTRY:
                logger.warning(f"Feature '{feature_name}' not in registry, skipping")
                continue
            
            try:
                feature_df[feature_name] = self.FEATURE_REGISTRY[feature_name](df)
            except Exception as e:
                logger.error(f"Failed to generate feature '{feature_name}': {e}")
                feature_df[feature_name] = np.nan
        
        # Drop rows with NaN (due to rolling windows)
        initial_rows = len(feature_df)
        feature_df = feature_df.dropna()
        logger.info(f"Generated features. Dropped {initial_rows - len(feature_df)} rows due to warmup")
        
        return feature_df
    
    def save_features(
        self,
        features: pd.DataFrame,
        path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save features to Parquet for consistency.
        
        Args:
            features: DataFrame with features
            path: Path to save parquet file
            metadata: Optional metadata to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features.to_parquet(path)
        logger.info(f"Saved {len(features)} rows x {len(features.columns)} features to {path}")
        
        # Save metadata
        if metadata:
            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata to {metadata_path}")
    
    def load_features(self, path: str) -> pd.DataFrame:
        """
        Load features from Parquet.
        
        Args:
            path: Path to parquet file
            
        Returns:
            DataFrame with features
        """
        features = pd.read_parquet(path)
        logger.info(f"Loaded {len(features)} rows x {len(features.columns)} features from {path}")
        return features
    
    def verify_consistency(
        self,
        train_features: pd.DataFrame,
        live_df: pd.DataFrame
    ) -> Dict:
        """
        Verify that live features match training features.
        
        Critical for preventing train/live mismatch.
        
        Args:
            train_features: Training feature DataFrame
            live_df: Live OHLCV data
            
        Returns:
            Verification result dict
        """
        # Generate live features
        train_feature_names = [col for col in train_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        live_features = self.generate_features(live_df, features=train_feature_names)
        
        # Compare columns
        train_cols = set(train_feature_names)
        live_cols = set([col for col in live_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
        
        missing = train_cols - live_cols
        extra = live_cols - train_cols
        
        # Compare sample values (last 10 rows)
        common_cols = train_cols & live_cols
        if len(common_cols) > 0 and len(live_features) >= 10:
            sample_mismatch = []
            for col in list(common_cols)[:5]:  # Check first 5 common features
                train_sample = train_features[col].iloc[-10:].values
                live_sample = live_features[col].iloc[-10:].values
                
                if not np.allclose(train_sample, live_sample, rtol=1e-5, equal_nan=True):
                    sample_mismatch.append(col)
        else:
            sample_mismatch = []
        
        is_consistent = (len(missing) == 0 and len(extra) == 0 and len(sample_mismatch) == 0)
        
        result = {
            'consistent': is_consistent,
            'missing_in_live': list(missing),
            'extra_in_live': list(extra),
            'value_mismatch': sample_mismatch
        }
        
        if is_consistent:
            logger.info("✅ Feature consistency verified - train = live")
        else:
            logger.warning(f"⚠️  Feature inconsistency detected: {result}")
        
        return result
    
    # === Helper Methods ===
    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score."""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        return (series - rolling_mean) / rolling_std
    
    @staticmethod
    def _linear_slope(y):
        """Calculate linear regression slope."""
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (trend strength)."""
        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=df.index)
        minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=df.index)
        
        # Calculate ATR
        atr = FeatureEngine._calculate_atr(df, period)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX (smoothed DX)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def get_feature_list(self) -> List[str]:
        """Get list of all available features."""
        return list(self.FEATURE_REGISTRY.keys())
    
    def get_feature_metadata(self) -> Dict:
        """Get metadata about all features."""
        return {
            'total_features': len(self.FEATURE_REGISTRY),
            'feature_names': self.get_feature_list(),
            'categories': {
                'returns': [f for f in self.get_feature_list() if 'returns' in f],
                'volatility': [f for f in self.get_feature_list() if 'volatility' in f],
                'momentum': [f for f in self.get_feature_list() if 'momentum' in f],
                'volume': [f for f in self.get_feature_list() if 'volume' in f],
                'trend': [f for f in self.get_feature_list() if 'ma_' in f or 'adx' in f],
                'regime': [f for f in self.get_feature_list() if 'regime' in f],
                'statistical': [f for f in self.get_feature_list() if 'skew' in f or 'kurt' in f]
            }
        }
