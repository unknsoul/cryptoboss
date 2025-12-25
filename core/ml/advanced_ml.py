"""
Advanced ML Features - Enterprise Features #265, #270, #275, #280
Feature Importance, Hyperparameter Optimization, Ensemble Voting, Online Learning.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Feature #265: Feature Importance Analyzer
    
    Analyzes which features contribute most to model predictions.
    """
    
    def __init__(self):
        """Initialize feature importance analyzer."""
        self.feature_scores: Dict[str, List[float]] = defaultdict(list)
        self.permutation_scores: Dict[str, float] = {}
        
        logger.info("Feature Importance Analyzer initialized")
    
    def record_prediction(self, features: Dict[str, float], outcome: bool):
        """Record a prediction for importance analysis."""
        outcome_val = 1 if outcome else 0
        
        for name, value in features.items():
            # Simple correlation-based scoring
            self.feature_scores[name].append((value, outcome_val))
    
    def calculate_importance(self) -> Dict[str, float]:
        """Calculate feature importance scores."""
        importance = {}
        
        for name, scores in self.feature_scores.items():
            if len(scores) < 20:
                importance[name] = 0
                continue
            
            # Calculate correlation with outcome
            values = [s[0] for s in scores]
            outcomes = [s[1] for s in scores]
            
            mean_v = sum(values) / len(values)
            mean_o = sum(outcomes) / len(outcomes)
            
            cov = sum((v - mean_v) * (o - mean_o) for v, o in zip(values, outcomes)) / len(values)
            std_v = (sum((v - mean_v) ** 2 for v in values) / len(values)) ** 0.5
            std_o = (sum((o - mean_o) ** 2 for o in outcomes) / len(outcomes)) ** 0.5
            
            if std_v > 0 and std_o > 0:
                corr = cov / (std_v * std_o)
                importance[name] = round(abs(corr), 4)
            else:
                importance[name] = 0
        
        # Normalize to sum to 1
        total = sum(importance.values()) or 1
        importance = {k: round(v / total, 4) for k, v in importance.items()}
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        importance = self.calculate_importance()
        return list(importance.items())[:n]
    
    def suggest_feature_removal(self, threshold: float = 0.01) -> List[str]:
        """Suggest features with low importance for removal."""
        importance = self.calculate_importance()
        return [name for name, score in importance.items() if score < threshold]


class HyperparameterOptimizer:
    """
    Feature #270: Hyperparameter Optimizer
    
    Optimizes model hyperparameters using grid/random search.
    """
    
    def __init__(self):
        """Initialize hyperparameter optimizer."""
        self.param_history: List[Dict] = []
        self.best_params: Optional[Dict] = None
        self.best_score: float = 0
        
        logger.info("Hyperparameter Optimizer initialized")
    
    def grid_search(
        self,
        param_grid: Dict[str, List],
        evaluate_fn: Callable[[Dict], float],
        max_iterations: Optional[int] = None
    ) -> Dict:
        """
        Perform grid search over parameter combinations.
        
        Args:
            param_grid: Dict of param_name -> list of values
            evaluate_fn: Function that takes params and returns score
            max_iterations: Max combinations to try
            
        Returns:
            Best parameters and score
        """
        # Generate all combinations
        keys = list(param_grid.keys())
        combinations = [{}]
        
        for key in keys:
            new_combinations = []
            for combo in combinations:
                for value in param_grid[key]:
                    new_combo = combo.copy()
                    new_combo[key] = value
                    new_combinations.append(new_combo)
            combinations = new_combinations
        
        if max_iterations:
            combinations = combinations[:max_iterations]
        
        # Evaluate each
        for params in combinations:
            score = evaluate_fn(params)
            
            self.param_history.append({
                'params': params,
                'score': score,
                'timestamp': datetime.now().isoformat()
            })
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'iterations': len(combinations)
        }
    
    def random_search(
        self,
        param_distributions: Dict[str, Tuple],
        evaluate_fn: Callable[[Dict], float],
        n_iterations: int = 20
    ) -> Dict:
        """
        Perform random search over parameter space.
        
        Args:
            param_distributions: Dict of param_name -> (min, max) or list
            evaluate_fn: Evaluation function
            n_iterations: Number of random samples
        """
        for _ in range(n_iterations):
            params = {}
            for name, dist in param_distributions.items():
                if isinstance(dist, tuple):
                    params[name] = random.uniform(dist[0], dist[1])
                elif isinstance(dist, list):
                    params[name] = random.choice(dist)
            
            score = evaluate_fn(params)
            
            self.param_history.append({
                'params': params,
                'score': score,
                'timestamp': datetime.now().isoformat()
            })
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'iterations': n_iterations
        }
    
    def get_history(self) -> List[Dict]:
        """Get optimization history."""
        return sorted(self.param_history, key=lambda x: x['score'], reverse=True)


class EnsembleVoting:
    """
    Feature #275: Model Ensemble Voting
    
    Combines predictions from multiple models.
    """
    
    def __init__(self):
        """Initialize ensemble voting."""
        self.models: Dict[str, Dict] = {}  # name -> model config
        self.weights: Dict[str, float] = {}
        
        logger.info("Ensemble Voting initialized")
    
    def add_model(self, name: str, predict_fn: Callable, weight: float = 1.0):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name
            predict_fn: Function that takes features and returns prediction
            weight: Voting weight
        """
        self.models[name] = {'predict_fn': predict_fn}
        self.weights[name] = weight
        logger.info(f"Added model '{name}' with weight {weight}")
    
    def set_weight(self, name: str, weight: float):
        """Update model weight."""
        if name in self.weights:
            self.weights[name] = weight
    
    def vote(self, features: Dict) -> Dict:
        """
        Get ensemble vote from all models.
        
        Args:
            features: Input features
            
        Returns:
            Voting result with confidence
        """
        if not self.models:
            return {'prediction': 'HOLD', 'confidence': 0}
        
        votes = {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
        total_weight = sum(self.weights.values())
        
        model_predictions = {}
        
        for name, config in self.models.items():
            try:
                pred = config['predict_fn'](features)
                weight = self.weights.get(name, 1.0)
                
                if isinstance(pred, dict):
                    direction = pred.get('direction', 'HOLD')
                    conf = pred.get('confidence', 0.5)
                else:
                    direction = pred
                    conf = 0.5
                
                votes[direction] += weight * conf
                model_predictions[name] = {'direction': direction, 'confidence': conf}
            except Exception as e:
                logger.error(f"Model {name} prediction failed: {e}")
        
        # Get winning vote
        max_vote = max(votes, key=votes.get)
        max_score = votes[max_vote]
        
        # Calculate confidence
        confidence = max_score / total_weight if total_weight > 0 else 0
        
        return {
            'prediction': max_vote,
            'confidence': round(confidence, 3),
            'votes': {k: round(v, 3) for k, v in votes.items()},
            'model_predictions': model_predictions
        }
    
    def weighted_average(self, predictions: Dict[str, float]) -> float:
        """Calculate weighted average of numeric predictions."""
        total = 0
        weight_sum = 0
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 1.0)
            total += pred * weight
            weight_sum += weight
        
        return total / weight_sum if weight_sum > 0 else 0


class OnlineLearningModule:
    """
    Feature #280: Online Learning Module
    
    Updates model incrementally with new data.
    """
    
    def __init__(self, learning_rate: float = 0.01, window_size: int = 100):
        """
        Initialize online learning module.
        
        Args:
            learning_rate: Learning rate for updates
            window_size: Size of recent data window
        """
        self.learning_rate = learning_rate
        self.window_size = window_size
        
        self.weights: Dict[str, float] = {}
        self.bias: float = 0
        self.data_window: List[Dict] = []
        self.update_count: int = 0
        
        logger.info(f"Online Learning initialized - LR: {learning_rate}, Window: {window_size}")
    
    def initialize_weights(self, feature_names: List[str]):
        """Initialize weights for features."""
        for name in feature_names:
            if name not in self.weights:
                self.weights[name] = random.gauss(0, 0.1)
    
    def predict(self, features: Dict[str, float]) -> float:
        """Make prediction with current weights."""
        score = self.bias
        for name, value in features.items():
            if name in self.weights:
                score += self.weights[name] * value
        
        # Sigmoid activation
        return 1 / (1 + 2.718 ** (-score))
    
    def update(self, features: Dict[str, float], actual: float):
        """
        Update model with new data point.
        
        Args:
            features: Feature values
            actual: Actual outcome (0 or 1)
        """
        # Initialize weights if needed
        self.initialize_weights(list(features.keys()))
        
        # Forward pass
        prediction = self.predict(features)
        
        # Calculate error
        error = actual - prediction
        
        # Gradient descent update
        for name, value in features.items():
            if name in self.weights:
                gradient = error * prediction * (1 - prediction) * value
                self.weights[name] += self.learning_rate * gradient
        
        self.bias += self.learning_rate * error * prediction * (1 - prediction)
        
        # Store in window
        self.data_window.append({
            'features': features,
            'actual': actual,
            'predicted': prediction
        })
        self.data_window = self.data_window[-self.window_size:]
        
        self.update_count += 1
    
    def get_accuracy(self) -> float:
        """Calculate recent prediction accuracy."""
        if len(self.data_window) < 10:
            return 0
        
        correct = sum(
            1 for d in self.data_window
            if (d['predicted'] >= 0.5) == (d['actual'] >= 0.5)
        )
        
        return correct / len(self.data_window)
    
    def get_status(self) -> Dict:
        """Get module status."""
        return {
            'update_count': self.update_count,
            'window_size': len(self.data_window),
            'accuracy': round(self.get_accuracy(), 3),
            'feature_count': len(self.weights),
            'top_features': sorted(
                self.weights.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
        }


# Singletons
_feature_analyzer: Optional[FeatureImportanceAnalyzer] = None
_hyperparam_opt: Optional[HyperparameterOptimizer] = None
_ensemble: Optional[EnsembleVoting] = None
_online_learning: Optional[OnlineLearningModule] = None


def get_feature_analyzer() -> FeatureImportanceAnalyzer:
    global _feature_analyzer
    if _feature_analyzer is None:
        _feature_analyzer = FeatureImportanceAnalyzer()
    return _feature_analyzer


def get_hyperparam_optimizer() -> HyperparameterOptimizer:
    global _hyperparam_opt
    if _hyperparam_opt is None:
        _hyperparam_opt = HyperparameterOptimizer()
    return _hyperparam_opt


def get_ensemble() -> EnsembleVoting:
    global _ensemble
    if _ensemble is None:
        _ensemble = EnsembleVoting()
    return _ensemble


def get_online_learner() -> OnlineLearningModule:
    global _online_learning
    if _online_learning is None:
        _online_learning = OnlineLearningModule()
    return _online_learning


if __name__ == '__main__':
    # Test feature importance
    fi = FeatureImportanceAnalyzer()
    for i in range(50):
        features = {'rsi': random.uniform(20, 80), 'volume': random.uniform(100, 1000)}
        outcome = features['rsi'] < 40 or features['volume'] > 800
        fi.record_prediction(features, outcome)
    
    print(f"Feature importance: {fi.calculate_importance()}")
    
    # Test ensemble
    ens = EnsembleVoting()
    ens.add_model('model1', lambda x: {'direction': 'LONG', 'confidence': 0.7}, weight=1.0)
    ens.add_model('model2', lambda x: {'direction': 'LONG', 'confidence': 0.6}, weight=0.8)
    ens.add_model('model3', lambda x: {'direction': 'SHORT', 'confidence': 0.5}, weight=0.5)
    
    result = ens.vote({'rsi': 45})
    print(f"Ensemble vote: {result}")
    
    # Test online learning
    ol = OnlineLearningModule()
    for i in range(100):
        features = {'x': random.uniform(-1, 1), 'y': random.uniform(-1, 1)}
        actual = 1 if features['x'] + features['y'] > 0 else 0
        ol.update(features, actual)
    
    print(f"Online learner status: {ol.get_status()}")
