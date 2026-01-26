"""
Real Machine Learning Training Pipeline
Proper feature engineering, labeling, and model training for trade signal filtering
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import joblib
from datetime import datetime

# ML libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier


class RealMLTrainer:
    """
    Production-grade ML training pipeline
    
    Features:
    - Proper time-based train/test split
    - Realistic label generation (forward returns)
    - Feature importance tracking
    - Model evaluation metrics
    - No data leakage
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize ML trainer
        
        Args:
            model_type: 'xgboost', 'lightgbm', or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.training_metrics = {}
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create realistic features from OHLCV data
        
        Features created:
        - Price features: returns, log returns, price ratio
        - Volatility: realized vol, ATR, Bollinger width
        - Momentum: RSI, MACD, ROC
        - Volume: volume ratio, volume change
        - Microstructure: high-low range, close position in range
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['returns_1'] = df['close'].pct_change(1)
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_10'] = df['close'].pct_change(10)
        features['returns_20'] = df['close'].pct_change(20)
        
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility features
        features['realized_vol_20'] = df['close'].pct_change().rolling(20).std()
        features['realized_vol_50'] = df['close'].pct_change().rolling(50).std()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features['atr_14'] = true_range.rolling(14).mean()
        features['atr_ratio'] = features['atr_14'] / df['close']
        
        # Bollinger Band width
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_width'] = (std_20 * 2) / sma_20
        
        # Momentum indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Rate of Change
        features['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_change'] = df['volume'].pct_change()
        
        # Microstructure features
        features['hl_range'] = (df['high'] - df['low']) / df['close']
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Trend features
        features['ema_50'] = df['close'].ewm(span=50).mean()
        features['ema_200'] = df['close'].ewm(span=200).mean()
        features['price_to_ema50'] = df['close'] / features['ema_50']
        features['price_to_ema200'] = df['close'] / features['ema_200']
        features['ema_cross'] = (features['ema_50'] > features['ema_200']).astype(int)
        
        return features
    
    def create_labels(
        self, 
        df: pd.DataFrame, 
        forward_periods: int = 24,
        threshold: float = 0.02
    ) -> pd.Series:
        """
        Create realistic labels from forward returns
        
        Label definition:
        - 1: Price increases by > threshold in next N periods
        - 0: Price changes less than threshold (neutral)
        - -1: Price decreases by > threshold in next N periods
        
        Args:
            df: DataFrame with OHLCV data
            forward_periods: How many periods ahead to look
            threshold: Return threshold for signal (0.02 = 2%)
            
        Returns:
            Series with labels
        """
        # Calculate forward returns
        forward_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
        
        # Create labels
        labels = pd.Series(0, index=df.index)
        labels[forward_returns > threshold] = 1      # Buy signal
        labels[forward_returns < -threshold] = -1    # Sell signal
        # Leave 0 for neutral (no trade)
        
        return labels
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        forward_periods: int = 24,
        threshold: float = 0.02,
        train_ratio: float = 0.7
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare training and test data with NO DATA LEAKAGE
        
        Important: Uses time-based split (NO random shuffling)
        
        Args:
            df: DataFrame with OHLCV data
            forward_periods: Periods ahead for labels
            threshold: Return threshold
            train_ratio: Train/test split ratio
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        print("Creating features...")
        features = self.create_features(df)
        
        print("Creating labels...")
        labels = self.create_labels(df, forward_periods, threshold)
        
        # Remove rows with missing values
        valid_mask = ~(features.isna().any(axis=1) | labels.isna())
        features_clean = features[valid_mask]
        labels_clean = labels[valid_mask]
        
        # TIME-BASED split (critical for time series)
        split_idx = int(len(features_clean) * train_ratio)
        
        X_train = features_clean.iloc[:split_idx]
        y_train = labels_clean.iloc[:split_idx]
        X_test = features_clean.iloc[split_idx:]
        y_test = labels_clean.iloc[split_idx:]
        
        self.feature_names = list(features_clean.columns)
        
        print(f"\nData prepared:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"\nLabel distribution (train):")
        print(y_train.value_counts(normalize=True))
        
        return X_train, y_train, X_test, y_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        Train ML model with proper validation
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print(f"\nTraining {self.model_type} model...")
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softmax',
                num_class=3,  # -1, 0, 1
                random_state=42,
                eval_metric='mlogloss'
            )
            
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multiclass',
                num_class=3,
                random_state=42
            )
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        
        # Train model
        if X_val is not None:
            if self.model_type == 'xgboost':
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=20,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print("✓ Training complete!")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['SELL', 'NEUTRAL', 'BUY']))
        
        # Accuracy per class
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro')
        }
        
        print(f"\nOverall Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Feature importance
        if self.feature_importance is not None:
            print(f"\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10))
        
        self.training_metrics = metrics
        return metrics
    
    def save_model(self, filepath: Path):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'metrics': self.training_metrics,
            'trained_at': datetime.now().isoformat()
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'RealMLTrainer':
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        trainer = cls(model_type=model_data['model_type'])
        trainer.model = model_data['model']
        trainer.feature_names = model_data['feature_names']
        trainer.feature_importance = model_data.get('feature_importance')
        trainer.training_metrics = model_data.get('metrics', {})
        
        print(f"✓ Model loaded from {filepath}")
        print(f"Trained at: {model_data.get('trained_at', 'unknown')}")
        
        return trainer


if __name__ == '__main__':
    # Example usage
    print("=" * 70)
    print("ML TRAINING PIPELINE EXAMPLE")
    print("=" * 70)
    
    # Load data (you would load actual BTC data here)
    print("\nGenerating sample data...")
    dates = pd.date_range('2023-01-01', periods=10000, freq='1H')
    np.random.seed(42)
    
    # Simulated OHLCV data
    close = 100 + np.cumsum(np.random.randn(10000) * 0.5)
    df = pd.DataFrame({
        'open': close + np.random.randn(10000) * 0.2,
        'high': close + np.abs(np.random.randn(10000) * 0.5),
        'low': close - np.abs(np.random.randn(10000) * 0.5),
        'close': close,
        'volume': np.random.randint(100, 1000, 10000)
    }, index=dates)
    
    # Initialize trainer
    trainer = RealMLTrainer(model_type='xgboost')
    
    # Prepare data
    X_train, y_train, X_test, y_test = trainer.prepare_training_data(
        df,
        forward_periods=24,  # 24 hours ahead
        threshold=0.02,      # 2% threshold
        train_ratio=0.7
    )
    
    # Train model
    trainer.train(X_train, y_train)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    trainer.save_model(Path('models/signal_filter_xgboost.pkl'))
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nTo use this model:")
    print("1. Train on real BTC data")
    print("2. Evaluate on out-of-sample data")
    print("3. Test in paper trading")
    print("4. Monitor performance and retrain when accuracy degrades")
