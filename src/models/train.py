"""
ML Training Pipeline
Walk-forward validation and proper train/test splits.

Critical: No data leakage, temporal splits only.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed. Using RandomForest only.")

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""
    periods: int
    avg_accuracy: float
    avg_precision: float
    avg_recall: float
    all_predictions: List[np.ndarray]
    all_actuals: List[np.ndarray]
    feature_importance: Dict[str, float]


class MLPipeline:
    """
    Proper ML training with walk-forward validation.
    
    Rules:
    - Train on past data only
    - No future data leakage
    - Temporal train/test splits
    - Feature scaling on training set only
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize ML pipeline.
        
        Args:
            model_type: 'xgboost' or 'randomforest'
        """
        self.model_type = model_type
        if model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, using RandomForest")
            self.model_type = 'randomforest'
        
        logger.info(f"ML Pipeline initialized (model: {self.model_type})")
    
    def create_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 4,
        threshold_pct: float = 0.5
    ) -> pd.Series:
        """
        Create forward-looking labels.
        
        Label: 1 if price increases > threshold_pct in next `horizon` bars, else 0
        
        Args:
            df: DataFrame with 'close' column
            horizon: Forward-looking periods
            threshold_pct: Minimum % move to be positive
            
        Returns:
            Series with labels (1 or 0)
        """
        future_return = (df['close'].shift(-horizon) - df['close']) / df['close'] * 100
        labels = (future_return > threshold_pct).astype(int)
        
        # Drop last `horizon` rows (no labels)
        labels.iloc[-horizon:] = np.nan
        
        logger.info(f"Created labels: {labels.sum()} positive, {(labels==0).sum()} negative")
        return labels
    
    def train_walk_forward(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        train_window: int = 1000,
        test_window: int = 200,
        step_size: int = 100
    ) -> WalkForwardResult:
        """
        Walk-forward validation (gold standard for time series).
        
        Process:
        [Train 1000] → [Test 200]
                    [Train 1000] → [Test 200]
                                [Train 1000] → [Test 200]
        
        Args:
            features: Feature DataFrame
            labels: Target labels
            train_window: Training window size
            test_window: Test window size
            step_size: Step size for rolling window
            
        Returns:
            WalkForwardResult with aggregated metrics
        """
        feature_cols = [col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        all_predictions = []
        all_actuals = []
        feature_importance_sum = {}
        periods = 0
        
        logger.info(f"Starting walk-forward validation...")
        logger.info(f"Train window: {train_window}, Test window: {test_window}, Step: {step_size}")
        
        start = 0
        while start + train_window + test_window <= len(features):
            # Train period
            train_start = start
            train_end = start + train_window
            
            # Test period
            test_start = train_end
            test_end = test_start + test_window
            
            # Extract data
            X_train = features[feature_cols].iloc[train_start:train_end]
            y_train = labels.iloc[train_start:train_end]
            
            X_test = features[feature_cols].iloc[test_start:test_end]
            y_test = labels.iloc[test_start:test_end]
            
            # Drop NaN
            train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
            test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())
            
            X_train = X_train[train_valid]
            y_train = y_train[train_valid]
            X_test = X_test[test_valid]
            y_test = y_test[test_valid]
            
            if len(X_train) < 100 or len(X_test) < 10:
                start += step_size
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if self.model_type == 'xgboost':
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            
            model.fit(X_train_scaled, y_train)
            
            # Predict
            predictions = model.predict(X_test_scaled)
            
            # Store results
            all_predictions.append(predictions)
            all_actuals.append(y_test.values)
            
            # Accumulate feature importance
            if hasattr(model, 'feature_importances_'):
                for feature, importance in zip(feature_cols, model.feature_importances_):
                    feature_importance_sum[feature] = feature_importance_sum.get(feature, 0) + importance
            
            periods += 1
            logger.debug(f"Period {periods}: Train [{train_start}:{train_end}], Test [{test_start}:{test_end}]")
            
            # Move window forward
            start += step_size
        
        # Calculate aggregated metrics
        all_pred_flat = np.concatenate(all_predictions)
        all_actual_flat = np.concatenate(all_actuals)
        
        accuracy = (all_pred_flat == all_actual_flat).mean()
        
        # Precision & Recall
        true_positives = ((all_pred_flat == 1) & (all_actual_flat == 1)).sum()
        false_positives = ((all_pred_flat == 1) & (all_actual_flat == 0)).sum()
        false_negatives = ((all_pred_flat == 0) & (all_actual_flat == 1)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Average feature importance
        avg_feature_importance = {
            feature: importance / periods
            for feature, importance in feature_importance_sum.items()
        }
        
        # Sort by importance
        avg_feature_importance = dict(sorted(avg_feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"✅ Walk-forward complete: {periods} periods")
        logger.info(f"   Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        return WalkForwardResult(
            periods=periods,
            avg_accuracy=accuracy,
            avg_precision=precision,
            avg_recall=recall,
            all_predictions=all_predictions,
            all_actuals=all_actuals,
            feature_importance=avg_feature_importance
        )
    
    def train_final_model(
        self,
        features: pd.DataFrame,
        labels: pd.Series
    ) -> Tuple[Any, StandardScaler, List[str]]:
        """
        Train final model on all data.
        
        Args:
            features: Full feature DataFrame
            labels: Full labels
            
        Returns:
            (model, scaler, feature_list)
        """
        feature_cols = [col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Drop NaN
        valid = ~(features[feature_cols].isna().any(axis=1) | labels.isna())
        X = features[feature_cols][valid]
        y = labels[valid]
        
        logger.info(f"Training final model on {len(X)} samples...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        if self.model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            )
        else:
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                random_state=42
            )
        
        model.fit(X_scaled, y)
        
        logger.info("✅ Final model trained")
        
        return model, scaler, feature_cols
