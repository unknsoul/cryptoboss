"""
ML & AI Extended Features - Features #261-284
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
import random
import math

logger = logging.getLogger(__name__)

class FeatureNormalizer:
    """Feature #261: Feature Normalizer"""
    def __init__(self):
        self.stats: Dict[str, Dict] = {}
    
    def fit(self, name: str, values: List[float]):
        if values:
            self.stats[name] = {'mean': sum(values)/len(values), 
                               'std': (sum((v-sum(values)/len(values))**2 for v in values)/len(values))**0.5}
    
    def transform(self, name: str, value: float) -> float:
        if name in self.stats and self.stats[name]['std'] > 0:
            return (value - self.stats[name]['mean']) / self.stats[name]['std']
        return value

class FeatureSelector:
    """Feature #262: Feature Selector"""
    def __init__(self):
        self.importance: Dict[str, float] = {}
    
    def update_importance(self, feature: str, score: float):
        self.importance[feature] = score
    
    def get_top_features(self, n: int = 10) -> List[str]:
        return sorted(self.importance.keys(), key=lambda x: self.importance[x], reverse=True)[:n]

class ModelRegistry:
    """Feature #263: Model Registry"""
    def __init__(self):
        self.models: Dict[str, Dict] = {}
    
    def register(self, name: str, model, metadata: Optional[Dict] = None):
        self.models[name] = {'model': model, 'metadata': metadata or {}, 'registered_at': datetime.now().isoformat()}
    
    def get(self, name: str):
        return self.models.get(name, {}).get('model')

class PredictionLogger:
    """Feature #264: Prediction Logger"""
    def __init__(self):
        self.predictions: List[Dict] = []
    
    def log(self, prediction: float, actual: Optional[float] = None, features: Optional[Dict] = None):
        self.predictions.append({'prediction': prediction, 'actual': actual, 'features': features,
                                 'time': datetime.now().isoformat()})
        self.predictions = self.predictions[-10000:]
    
    def get_accuracy(self) -> float:
        correct = [p for p in self.predictions if p['actual'] is not None and 
                   (p['prediction'] > 0.5) == (p['actual'] > 0)]
        return len(correct) / len([p for p in self.predictions if p['actual'] is not None]) if self.predictions else 0

class ModelPerformanceTracker:
    """Feature #266: Model Performance Tracker"""
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record(self, model: str, accuracy: float, loss: float):
        if model not in self.metrics:
            self.metrics[model] = []
        self.metrics[model].append({'accuracy': accuracy, 'loss': loss, 'time': datetime.now().isoformat()})
    
    def get_trend(self, model: str) -> str:
        if model not in self.metrics or len(self.metrics[model]) < 5:
            return 'unknown'
        recent = [m['accuracy'] for m in self.metrics[model][-5:]]
        return 'improving' if recent[-1] > recent[0] else 'degrading'

class ABTestManager:
    """Feature #267: A/B Test Manager"""
    def __init__(self):
        self.tests: Dict[str, Dict] = {}
    
    def create_test(self, name: str, variants: List[str]):
        self.tests[name] = {'variants': variants, 'results': {v: [] for v in variants}}
    
    def assign_variant(self, test: str) -> str:
        if test in self.tests:
            return random.choice(self.tests[test]['variants'])
        return 'control'
    
    def record_outcome(self, test: str, variant: str, success: bool):
        if test in self.tests and variant in self.tests[test]['results']:
            self.tests[test]['results'][variant].append(1 if success else 0)

class CrossValidator:
    """Feature #268: Cross Validation"""
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
    
    def split(self, data: List) -> List[Dict]:
        fold_size = len(data) // self.n_folds
        folds = []
        for i in range(self.n_folds):
            test_start = i * fold_size
            test_end = test_start + fold_size
            test = data[test_start:test_end]
            train = data[:test_start] + data[test_end:]
            folds.append({'train': train, 'test': test})
        return folds

class TransferLearningAdapter:
    """Feature #269: Transfer Learning Adapter"""
    def __init__(self):
        self.base_weights: Dict = {}
        self.adapted_weights: Dict = {}
    
    def load_base(self, weights: Dict):
        self.base_weights = weights
    
    def adapt(self, task: str, learning_rate: float = 0.01):
        self.adapted_weights = {k: v * (1 + random.gauss(0, learning_rate)) 
                                for k, v in self.base_weights.items()}

class ReinforcementLearner:
    """Feature #271: Reinforcement Learning Module"""
    def __init__(self, actions: List[str], epsilon: float = 0.1):
        self.actions = actions
        self.epsilon = epsilon
        self.q_values: Dict[str, float] = {a: 0 for a in actions}
    
    def select_action(self, state: str = '') -> str:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_values.keys(), key=lambda a: self.q_values[a])
    
    def update(self, action: str, reward: float, alpha: float = 0.1):
        self.q_values[action] = self.q_values[action] + alpha * (reward - self.q_values[action])

class PatternMatcher:
    """Feature #272: Pattern Matcher"""
    def __init__(self):
        self.patterns: Dict[str, List[float]] = {}
    
    def add_pattern(self, name: str, pattern: List[float]):
        self.patterns[name] = pattern
    
    def match(self, data: List[float]) -> Optional[str]:
        if len(data) < 3:
            return None
        best_match = None
        best_score = 0
        for name, pattern in self.patterns.items():
            if len(pattern) <= len(data):
                score = self._similarity(data[-len(pattern):], pattern)
                if score > best_score:
                    best_score = score
                    best_match = name
        return best_match if best_score > 0.7 else None
    
    def _similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0
        norm_a = [(v - min(a)) / (max(a) - min(a) + 0.0001) for v in a]
        norm_b = [(v - min(b)) / (max(b) - min(b) + 0.0001) for v in b]
        diff = sum(abs(norm_a[i] - norm_b[i]) for i in range(len(norm_a)))
        return 1 - diff / len(norm_a)

class TimeSeriesPredictor:
    """Feature #274: Time Series Predictor"""
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.history: List[float] = []
    
    def add(self, value: float):
        self.history.append(value)
        self.history = self.history[-100:]
    
    def predict(self) -> float:
        if len(self.history) < self.lookback:
            return self.history[-1] if self.history else 0
        recent = self.history[-self.lookback:]
        trend = (recent[-1] - recent[0]) / self.lookback
        return recent[-1] + trend

class AnomalyDetector:
    """Feature #276: Anomaly Detector"""
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.values: List[float] = []
    
    def add(self, value: float) -> bool:
        self.values.append(value)
        self.values = self.values[-100:]
        if len(self.values) < 10:
            return False
        mean = sum(self.values) / len(self.values)
        std = (sum((v - mean) ** 2 for v in self.values) / len(self.values)) ** 0.5
        return abs(value - mean) > self.threshold * std if std > 0 else False

class ClusterAnalyzer:
    """Feature #277: Cluster Analyzer"""
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.centroids: List[float] = []
    
    def fit(self, data: List[float]):
        if len(data) < self.n_clusters:
            self.centroids = data[:]
            return
        sorted_data = sorted(data)
        step = len(sorted_data) // self.n_clusters
        self.centroids = [sorted_data[i * step] for i in range(self.n_clusters)]
    
    def predict(self, value: float) -> int:
        if not self.centroids:
            return 0
        return min(range(len(self.centroids)), key=lambda i: abs(value - self.centroids[i]))

class TrendClassifier:
    """Feature #278: Trend Classifier"""
    def classify(self, values: List[float]) -> str:
        if len(values) < 5:
            return 'UNKNOWN'
        slope = (values[-1] - values[0]) / len(values)
        volatility = (sum((values[i] - values[i-1])**2 for i in range(1, len(values))) / len(values)) ** 0.5
        if abs(slope) < volatility * 0.1:
            return 'RANGING'
        return 'UPTREND' if slope > 0 else 'DOWNTREND'

class VolatilityPredictor:
    """Feature #279: Volatility Predictor"""
    def __init__(self):
        self.history: List[float] = []
    
    def add(self, volatility: float):
        self.history.append(volatility)
        self.history = self.history[-50:]
    
    def predict(self) -> float:
        if len(self.history) < 5:
            return self.history[-1] if self.history else 0.02
        recent = self.history[-5:]
        return sum(recent) / len(recent) * 1.1  # Slight upward bias

class SignalGenerator:
    """Feature #281: ML Signal Generator"""
    def __init__(self):
        self.weights = {'rsi': 0.3, 'macd': 0.3, 'trend': 0.4}
    
    def generate(self, rsi: float, macd: float, trend: float) -> Dict:
        rsi_signal = -1 if rsi > 70 else 1 if rsi < 30 else 0
        macd_signal = 1 if macd > 0 else -1
        trend_signal = trend
        score = (rsi_signal * self.weights['rsi'] + macd_signal * self.weights['macd'] + 
                 trend_signal * self.weights['trend'])
        return {'signal': 'BUY' if score > 0.3 else 'SELL' if score < -0.3 else 'HOLD', 'score': round(score, 3)}

class ConfidenceEstimator:
    """Feature #282: Confidence Estimator"""
    def estimate(self, indicators: Dict[str, float]) -> float:
        agreement = 0
        total = 0
        for name, value in indicators.items():
            if 'signal' in name.lower():
                total += 1
                if value > 0:
                    agreement += 1
        return agreement / total if total > 0 else 0.5

class RegressionModel:
    """Feature #283: Simple Regression"""
    def __init__(self):
        self.slope = 0
        self.intercept = 0
    
    def fit(self, x: List[float], y: List[float]):
        if len(x) < 2:
            return
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_xx = sum(xi ** 2 for xi in x)
        denom = n * sum_xx - sum_x ** 2
        if denom != 0:
            self.slope = (n * sum_xy - sum_x * sum_y) / denom
            self.intercept = (sum_y - self.slope * sum_x) / n
    
    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept

class ProbabilityEstimator:
    """Feature #284: Probability Estimator"""
    def __init__(self):
        self.outcomes: List[bool] = []
    
    def record(self, success: bool):
        self.outcomes.append(success)
        self.outcomes = self.outcomes[-1000:]
    
    def estimate(self) -> float:
        if not self.outcomes:
            return 0.5
        return sum(self.outcomes) / len(self.outcomes)

# Factories
def get_normalizer(): return FeatureNormalizer()
def get_selector(): return FeatureSelector()
def get_model_registry(): return ModelRegistry()
def get_prediction_logger(): return PredictionLogger()
def get_performance_tracker(): return ModelPerformanceTracker()
def get_ab_test(): return ABTestManager()
def get_cross_validator(): return CrossValidator()
def get_transfer_adapter(): return TransferLearningAdapter()
def get_rl_learner(): return ReinforcementLearner(['BUY', 'SELL', 'HOLD'])
def get_pattern_matcher(): return PatternMatcher()
def get_ts_predictor(): return TimeSeriesPredictor()
def get_anomaly_detector(): return AnomalyDetector()
def get_cluster_analyzer(): return ClusterAnalyzer()
def get_trend_classifier(): return TrendClassifier()
def get_vol_predictor(): return VolatilityPredictor()
def get_signal_generator(): return SignalGenerator()
def get_confidence_estimator(): return ConfidenceEstimator()
def get_regression(): return RegressionModel()
def get_prob_estimator(): return ProbabilityEstimator()
