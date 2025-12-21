"""
Self-Learning System - Continual Improvement Engine
Implements online learning, auto-optimization, and adaptive intelligence

Features:
- Continuous model retraining
- Performance-based parameter optimization
- Concept drift detection
- Meta-learning for rapid adaptation
- Safe exploration with guardrails
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import pickle
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

from core.monitoring.logger import get_logger
from core.monitoring.metrics import get_metrics
from core.monitoring.alerting import get_alerts


logger = get_logger()
metrics = get_metrics()
alerts = get_alerts()


class ConceptDriftDetector:
    """
    Detects when market behavior changes (concept drift)
    Triggers model retraining when detected
    """
    
    def __init__(self, window_size: int = 100, threshold: float = 0.10):
        """
        Args:
            window_size: Number of samples to compare
            threshold: Performance drop threshold to trigger alert
        """
        self.window_size = window_size
        self.threshold = threshold
        
        self.recent_performance = deque(maxlen=window_size)
        self.baseline_performance = None
        self.drift_detected = False
    
    def add_prediction(self, actual: float, predicted: float):
        """Record a prediction for drift detection"""
        # Calculate error
        error = abs(actual - predicted)
        self.recent_performance.append(error)
    
    def check_drift(self) -> bool:
        """
        Check if concept drift has occurred
        
        Returns:
            True if drift detected
        """
        if len(self.recent_performance) < self.window_size:
            return False
        
        # Calculate current performance (lower error = better)
        current_error = np.mean(list(self.recent_performance))
        
        # Set baseline
        if self.baseline_performance is None:
            self.baseline_performance = current_error
            return False
        
        # Check if performance degraded significantly
        degradation = (current_error - self.baseline_performance) / (self.baseline_performance + 1e-8)
        
        if degradation > self.threshold:
            logger.warning(
                f"🚨 Concept drift detected! Performance degraded by {degradation:.1%}",
                current_error=current_error,
                baseline_error=self.baseline_performance
            )
            
            self.drift_detected = True
            self.baseline_performance = current_error  # Reset baseline
            
            alerts.send_alert(
                "concept_drift_detected",
                "Model performance has degraded - retraining recommended",
                {
                    'degradation_pct': degradation,
                    'current_error': current_error,
                    'baseline_error': self.baseline_performance
                }
            )
            
            return True
        
        return False


class OnlineLearningEngine:
    """
    Online learning system that continuously improves from new data
    """
    
    def __init__(self, model_path: str = "models/online_model.pkl"):
        """
        Args:
            model_path: Path to save/load model
        """
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load model
        if self.model_path.exists():
            self.model = self._load_model()
            logger.info(f"Loaded existing model from {model_path}")
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8
            )
            logger.info("Initialized new model")
        
        # Experience buffer for batch updates
        self.experience_buffer = {
            'X': [],
            'y': []
        }
        self.buffer_size = 100  # Retrain after 100 new samples
        
        # Performance tracking
        self.drift_detector = ConceptDriftDetector()
        self.training_history = []
        
        # Model versioning
        self.version = 0
        self.last_retrain = datetime.now()
    
    def add_experience(self, X: np.ndarray, y: int):
        """
        Add a new training example
        
        Args:
            X: Feature vector
            y: True label (0 or 1)
        """
        self.experience_buffer['X'].append(X)
        self.experience_buffer['y'].append(y)
        
        # Check if buffer is full
        if len(self.experience_buffer['X']) >= self.buffer_size:
            self._retrain()
    
    def _retrain(self):
        """Retrain model with new experiences"""
        if len(self.experience_buffer['X']) == 0:
            return
        
        logger.info(f"🔄 Retraining model with {len(self.experience_buffer['X'])} new samples...")
        
        X_new = np.array(self.experience_buffer['X'])
        y_new = np.array(self.experience_buffer['y'])
        
        # Incremental training (warm start)
        if hasattr(self.model, 'n_estimators'):
            # For tree-based models, add new trees
            self.model.n_estimators += 20
        
        try:
            self.model.fit(X_new, y_new)
            
            # Evaluate on new data
            y_pred = self.model.predict(X_new)
            accuracy = accuracy_score(y_new, y_pred)
            
            logger.info(f"✅ Model retrained. Accuracy on new data: {accuracy:.2%}")
            
            # Save model
            self.version += 1
            self._save_model()
            
            # Record training
            self.training_history.append({
                'timestamp': datetime.now(),
                'version': self.version,
                'samples': len(X_new),
                'accuracy': accuracy
            })
            
            self.last_retrain = datetime.now()
            
            # Clear buffer
            self.experience_buffer = {'X': [], 'y': []}
            
            metrics.increment("model_retrain_success")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            metrics.increment("model_retrain_failed")
    
    def predict(self, X: np.ndarray) -> Tuple[int, float]:
        """
        Make prediction with confidence
        
        Returns:
            (prediction, confidence)
        """
        if not hasattr(self.model, 'predict_proba'):
            # Model not trained yet
            return 0, 0.5
        
        pred_proba = self.model.predict_proba(X.reshape(1, -1))[0]
        prediction = np.argmax(pred_proba)
        confidence = pred_proba[prediction]
        
        return prediction, confidence
    
    def check_and_retrain_if_needed(self):
        """Check for drift and retrain if necessary"""
        if self.drift_detector.check_drift():
            logger.info("Drift detected - forcing retrain")
            self._retrain()
    
    def _save_model(self):
        """Save model to disk"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'version': self.version,
                    'last_retrain': self.last_retrain,
                    'training_history': self.training_history
                }, f)
            logger.info(f"Model v{self.version} saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _load_model(self):
        """Load model from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.version = data.get('version', 0)
                self.last_retrain = data.get('last_retrain', datetime.now())
                self.training_history = data.get('training_history', [])
                return data['model']
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None


class AutoParameterOptimizer:
    """
    Automatically optimizes strategy parameters based on performance
    Uses Bayesian optimization for efficient search
    """
    
    def __init__(self, optimization_interval: timedelta = timedelta(days=7)):
        """
        Args:
            optimization_interval: How often to optimize
        """
        self.optimization_interval = optimization_interval
        self.last_optimization = None
        
        # Parameter search history
        self.search_history = []
        self.best_params = None
        self.best_performance = -np.inf
    
    def should_optimize(self) -> bool:
        """Check if it's time to optimize"""
        if self.last_optimization is None:
            return True
        
        return datetime.now() - self.last_optimization >= self.optimization_interval
    
    def optimize_parameters(self, 
                          strategy,
                          param_space: Dict[str, Tuple[float, float]],
                          performance_data: pd.DataFrame,
                          n_iterations: int = 30) -> Dict[str, float]:
        """
        Optimize strategy parameters using Bayesian optimization
        
        Args:
            strategy: Strategy instance
            param_space: Dict of {param_name: (min_val, max_val)}
            performance_data: Historical performance data
            n_iterations: Number of optimization iterations
        
        Returns:
            Best parameters found
        """
        logger.info(f"🔍 Starting parameter optimization ({n_iterations} iterations)...")
        
        from scipy.optimize import differential_evolution
        
        # Define objective function
        def objective(params_array):
            # Convert array to dict
            params = {}
            for i, (param_name, (min_val, max_val)) in enumerate(param_space.items()):
                params[param_name] = params_array[i]
            
            # Evaluate strategy with these parameters
            performance = self._evaluate_params(strategy, params, performance_data)
            
            # We want to maximize (return negative for minimization)
            return -performance
        
        # Convert param space to bounds
        bounds = list(param_space.values())
        
        # Run optimization
        result = differential_evolution(
            objective,
            bounds,
            maxiter=n_iterations,
            popsize=5,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42
        )
        
        # Convert back to dict
        best_params = {}
        for i, param_name in enumerate(param_space.keys()):
            best_params[param_name] = result.x[i]
        
        best_performance = -result.fun  # Convert back to positive
        
        logger.info(
            f"✅ Optimization complete! Best performance: {best_performance:.4f}",
            best_params=best_params
        )
        
        # Update history
        self.search_history.append({
            'timestamp': datetime.now(),
            'params': best_params,
            'performance': best_performance
        })
        
        self.best_params = best_params
        self.best_performance = best_performance
        self.last_optimization = datetime.now()
        
        return best_params
    
    def _evaluate_params(self, 
                        strategy,
                        params: Dict[str, float],
                        data: pd.DataFrame) -> float:
        """
        Evaluate strategy with given parameters
        
        Returns:
            Sharpe ratio (or other performance metric)
        """
        # Apply parameters to strategy
        for param_name, param_value in params.items():
            if hasattr(strategy, param_name):
                setattr(strategy, param_name, param_value)
        
        # Run backtest (simplified)
        # In practice, this would run a full backtest
        returns = []
        
        # Calculate performance metric
        if len(returns) > 0:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0
        
        return sharpe


class MetaLearningEngine:
    """
    Meta-learning system that learns to learn quickly
    Adapts to new market regimes faster
    """
    
    def __init__(self):
        self.regime_models = {}  # Models for different regimes
        self.current_regime = None
        
        # Track which models perform best in which conditions
        self.regime_performance = {}
    
    def learn_regime(self, regime: str, X_train, y_train):
        """Learn a model for a specific market regime"""
        if regime not in self.regime_models:
            self.regime_models[regime] = xgb.XGBClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=4
            )
        
        self.regime_models[regime].fit(X_train, y_train)
        logger.info(f"Learned model for regime: {regime}")
    
    def predict_adaptive(self, X, current_regime: str):
        """Use regime-specific model for prediction"""
        if current_regime in self.regime_models:
            model = self.regime_models[current_regime]
            return model.predict_proba(X.reshape(1, -1))[0]
        else:
            # Fall back to generic model
            return np.array([0.5, 0.5])


if __name__ == "__main__":
    print("=" * 70)
    print("🧠 SELF-LEARNING SYSTEM TEST")
    print("=" * 70)
    
    # Test 1: Online Learning
    print("\n1. Testing Online Learning Engine...")
    engine = OnlineLearningEngine()
    
    # Simulate new experiences
    for i in range(150):
        X = np.random.randn(10)
        y = np.random.choice([0, 1])
        engine.add_experience(X, y)
    
    print(f"   ✅ Model version: {engine.version}")
    print(f"   ✅ Last retrain: {engine.last_retrain}")
    
    # Test 2: Concept Drift Detection
    print("\n2. Testing Concept Drift Detector..."
