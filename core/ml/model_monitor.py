"""
Model Drift Detector - Enterprise ML Feature #89
Detects when ML model performance degrades and triggers alerts.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class ModelDriftDetector:
    """
    Detects when ML model predictions drift from expected performance.
    
    Monitors:
    - Prediction accuracy over rolling window
    - Confidence calibration (are high-confidence predictions more accurate?)
    - Feature distribution shifts
    - Win rate decay
    
    Triggers alert when performance drops below thresholds.
    """
    
    def __init__(
        self,
        window_size: int = 50,               # Trades to consider
        min_accuracy: float = 0.55,          # Minimum win rate
        min_calibration: float = 0.6,        # Min calibration score
        drift_threshold: float = 0.15,       # Max allowed feature drift
        alert_consecutive: int = 3           # Consecutive checks to trigger
    ):
        """
        Initialize drift detector.
        
        Args:
            window_size: Number of recent predictions to analyze
            min_accuracy: Minimum required accuracy before alert
            min_calibration: Minimum confidence calibration score
            drift_threshold: Feature distribution shift threshold
            alert_consecutive: Consecutive failures to trigger retraining
        """
        self.window_size = window_size
        self.min_accuracy = min_accuracy
        self.min_calibration = min_calibration
        self.drift_threshold = drift_threshold
        self.alert_consecutive = alert_consecutive
        
        # Tracking
        self.predictions: deque = deque(maxlen=window_size)
        self.feature_baselines: Dict[str, Dict] = {}
        self.consecutive_failures = 0
        self.last_check_time: Optional[datetime] = None
        self.drift_history: List[Dict] = []
        
        logger.info(f"Model Drift Detector initialized - Window: {window_size}, "
                   f"Min accuracy: {min_accuracy:.0%}")
    
    def record_prediction(
        self,
        prediction: str,           # 'BUY', 'SELL', or 'HOLD'
        confidence: float,         # Model confidence 0-1
        actual_outcome: Optional[str] = None,  # Actual result if known
        features: Optional[Dict] = None        # Features used
    ):
        """
        Record a model prediction for drift analysis.
        
        Args:
            prediction: The model's prediction
            confidence: Model confidence score
            actual_outcome: Actual market outcome (set later when known)
            features: Feature values used for prediction
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'actual': actual_outcome,
            'features': features,
            'correct': None
        }
        self.predictions.append(record)
    
    def update_outcome(self, prediction_idx: int, actual: str, pnl: float):
        """Update a prediction with its actual outcome."""
        if 0 <= prediction_idx < len(self.predictions):
            self.predictions[prediction_idx]['actual'] = actual
            self.predictions[prediction_idx]['correct'] = (pnl > 0)
    
    def mark_last_prediction(self, correct: bool, pnl: float):
        """Mark the most recent prediction as correct/incorrect."""
        if self.predictions:
            self.predictions[-1]['correct'] = correct
            self.predictions[-1]['pnl'] = pnl
    
    def check_drift(self) -> Tuple[bool, Dict]:
        """
        Check for model drift.
        
        Returns:
            Tuple of (drift_detected, details)
        """
        self.last_check_time = datetime.now()
        
        results = {
            'timestamp': self.last_check_time.isoformat(),
            'sample_size': len(self.predictions),
            'accuracy': None,
            'calibration': None,
            'drift_detected': False,
            'reasons': []
        }
        
        # Need minimum samples
        relevant = [p for p in self.predictions if p['correct'] is not None]
        if len(relevant) < 20:
            results['status'] = 'insufficient_data'
            return False, results
        
        # 1. Check accuracy
        correct_count = sum(1 for p in relevant if p['correct'])
        accuracy = correct_count / len(relevant)
        results['accuracy'] = round(accuracy, 3)
        
        if accuracy < self.min_accuracy:
            results['reasons'].append(f"Accuracy drop: {accuracy:.1%} < {self.min_accuracy:.1%}")
        
        # 2. Check calibration (high confidence should be more accurate)
        high_conf = [p for p in relevant if p['confidence'] >= 0.7]
        low_conf = [p for p in relevant if p['confidence'] < 0.5]
        
        if high_conf and low_conf:
            high_acc = sum(1 for p in high_conf if p['correct']) / len(high_conf)
            low_acc = sum(1 for p in low_conf if p['correct']) / len(low_conf) if low_conf else 0
            
            calibration = high_acc - low_acc  # Should be positive (high conf = more accurate)
            results['calibration'] = round(calibration, 3)
            
            if high_acc < low_acc:
                results['reasons'].append(f"Calibration inverted: high-conf={high_acc:.1%}, low-conf={low_acc:.1%}")
        
        # 3. Check feature drift (if baselines set)
        if self.feature_baselines and relevant[-1].get('features'):
            drift_score = self._calculate_feature_drift(relevant[-1]['features'])
            results['feature_drift'] = round(drift_score, 3)
            
            if drift_score > self.drift_threshold:
                results['reasons'].append(f"Feature drift: {drift_score:.2f} > {self.drift_threshold}")
        
        # Determine if drift detected
        has_drift = len(results['reasons']) > 0
        results['drift_detected'] = has_drift
        
        # Track consecutive failures
        if has_drift:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        
        results['consecutive_failures'] = self.consecutive_failures
        results['retrain_recommended'] = self.consecutive_failures >= self.alert_consecutive
        
        # Record history
        self.drift_history.append(results)
        self.drift_history = self.drift_history[-20:]  # Keep last 20
        
        if has_drift:
            logger.warning(f"Model drift detected: {results['reasons']}")
        
        return has_drift, results
    
    def set_feature_baseline(self, features: Dict[str, float]):
        """Set baseline feature distributions for drift detection."""
        for key, value in features.items():
            if key not in self.feature_baselines:
                self.feature_baselines[key] = {'values': [], 'mean': None, 'std': None}
            
            self.feature_baselines[key]['values'].append(value)
            
            # Update stats after 20+ samples
            if len(self.feature_baselines[key]['values']) >= 20:
                vals = self.feature_baselines[key]['values'][-100:]  # Last 100
                self.feature_baselines[key]['mean'] = np.mean(vals)
                self.feature_baselines[key]['std'] = np.std(vals) + 0.001  # Avoid div by 0
    
    def _calculate_feature_drift(self, current_features: Dict) -> float:
        """Calculate feature drift score using z-scores."""
        if not self.feature_baselines:
            return 0.0
        
        drift_scores = []
        for key, value in current_features.items():
            if key in self.feature_baselines:
                baseline = self.feature_baselines[key]
                if baseline['mean'] is not None and baseline['std'] is not None:
                    z_score = abs(value - baseline['mean']) / baseline['std']
                    drift_scores.append(min(z_score, 5))  # Cap at 5
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def should_retrain(self) -> Tuple[bool, str]:
        """Check if model retraining is recommended."""
        if self.consecutive_failures >= self.alert_consecutive:
            return True, f"Model drift detected {self.consecutive_failures} consecutive times"
        return False, "Model performing within thresholds"
    
    def get_status(self) -> Dict:
        """Get current drift detector status."""
        relevant = [p for p in self.predictions if p['correct'] is not None]
        accuracy = sum(1 for p in relevant if p['correct']) / len(relevant) if relevant else 0
        
        return {
            'total_predictions': len(self.predictions),
            'evaluated': len(relevant),
            'current_accuracy': round(accuracy, 3),
            'consecutive_failures': self.consecutive_failures,
            'retrain_recommended': self.consecutive_failures >= self.alert_consecutive,
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            'drift_history_count': len(self.drift_history)
        }


# Singleton instance
_drift_detector: Optional[ModelDriftDetector] = None


def get_drift_detector() -> ModelDriftDetector:
    """Get or create the global drift detector instance."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = ModelDriftDetector()
    return _drift_detector


if __name__ == '__main__':
    # Test drift detector
    detector = ModelDriftDetector(window_size=30, min_accuracy=0.55)
    
    # Simulate predictions
    import random
    for i in range(40):
        pred = random.choice(['BUY', 'SELL'])
        conf = random.uniform(0.4, 0.9)
        detector.record_prediction(pred, conf)
        
        # Simulate outcome (60% correct)
        correct = random.random() < 0.6
        detector.mark_last_prediction(correct, 10 if correct else -10)
    
    drift, results = detector.check_drift()
    print(f"Drift detected: {drift}")
    print(f"Results: {results}")
    print(f"Status: {detector.get_status()}")
