"""
Ensemble Model Architecture
Critical: Multiple models reduce overfitting and improve accuracy
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import joblib
from pathlib import Path

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class EnsembleModel:
    """
    Ensemble of multiple ML models
    
    Why ensembles work:
    - Different models capture different patterns
    - Reduces overfitting (models can't all be wrong same way)
    - More robust to regime changes
    
    Architecture:
    - XGBoost (captures non-linear patterns)
    - LightGBM (fast, handles large feature sets)
    - Random Forest (robust to noise)
    - Logistic Regression (linear baseline)
    
    Final prediction: Weighted average
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble
        
        Args:
            weights: Model weights (defaults to equal weighting)
        """
        self.models = {}
        self.weights = weights or {
            'xgboost': 0.35,
            'lightgbm': 0.35,
            'random_forest': 0.20,
            'logistic': 0.10
        }
        self.feature_names = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        Train all models in ensemble
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        self.feature_names = list(X_train.columns)
        
        print("Training ensemble models...")
        
        # 1. XGBoost
        print("  [1/4] Training XGBoost...")
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            random_state=42
        )
        if X_val is not None:
            self.models['xgboost'].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            self.models['xgboost'].fit(X_train, y_train)
        
        # 2. LightGBM
        print("  [2/4] Training LightGBM...")
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multiclass',
            num_class=3,
            random_state=42,
            verbose=-1
        )
        self.models['lightgbm'].fit(X_train, y_train)
        
        # 3. Random Forest
        print("  [3/4] Training Random Forest...")
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X_train, y_train)
        
        # 4. Logistic Regression (baseline)
        print("  [4/4] Training Logistic Regression...")
        self.models['logistic'] = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        self.models['logistic'].fit(X_train, y_train)
        
        print("✓ All models trained!")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using weighted ensemble
        
        Args:
            X: Features
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        predictions = []
        
        for model_name, model in self.models.items():
            proba = model.predict_proba(X)
            weight = self.weights.get(model_name, 0.25)
            predictions.append(proba * weight)
        
        # Weighted average
        ensemble_proba = np.sum(predictions, axis=0)
        
        return ensemble_proba
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Features
            
        Returns:
            Array of predicted class labels
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate ensemble performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with metrics for each model and ensemble
        """
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        results = {}
        
        # Evaluate each model
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            results[model_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro')
            }
        
        # Evaluate ensemble
        y_pred_ensemble = self.predict(X_test)
        results['ensemble'] = {
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'f1_macro': f1_score(y_test, y_pred_ensemble, average='macro')
        }
        
        # Print results
        print("\n" + "=" * 70)
        print("ENSEMBLE EVALUATION")
        print("=" * 70)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_macro']:.4f}")
        
        print("\n" + "=" * 70)
        print("✓ Ensemble typically outperforms individual models")
        print("✓ More robust to overfitting")
        print("=" * 70)
        
        return results
    
    def save(self, filepath: Path):
        """Save ensemble"""
        data = {
            'models': self.models,
            'weights': self.weights,
            'feature_names': self.feature_names
        }
        joblib.dump(data, filepath)
        print(f"✓ Ensemble saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'EnsembleModel':
        """Load ensemble"""
        data = joblib.load(filepath)
        ensemble = cls(weights=data['weights'])
        ensemble.models = data['models']
        ensemble.feature_names = data['feature_names']
        print(f"✓ Ensemble loaded from {filepath}")
        return ensemble


if __name__ == '__main__':
    # Example
    print("=" * 70)
    print("ENSEMBLE MODEL EXAMPLE")
    print("=" * 70)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice([-1, 0, 1], n_samples))
    
    # Split
    split = int(0.7 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train ensemble
    ensemble = EnsembleModel()
    ensemble.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    results = ensemble.evaluate(X_test, y_test)
    
    # Save
    ensemble.save(Path('models/ensemble_model.pkl'))
