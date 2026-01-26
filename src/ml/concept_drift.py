"""
Concept Drift Detector - Enterprise ML Feature #273
Detects when market patterns change and signals shift.
Implements ADWIN (Adaptive Windowing) algorithm for drift detection.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
import math

logger = logging.getLogger(__name__)


class ConceptDriftDetector:
    """
    Detects concept drift in trading data using ADWIN algorithm.
    
    Concept drift occurs when:
    - Market regime changes (trending -> ranging)
    - Volatility patterns shift
    - Correlations break down
    - Strategy alpha decays
    """
    
    def __init__(
        self,
        delta: float = 0.002,           # Confidence parameter (lower = more sensitive)
        min_window: int = 20,           # Minimum samples before checking
        max_window: int = 500           # Maximum window size
    ):
        """
        Initialize ADWIN drift detector.
        
        Args:
            delta: Confidence parameter (0.002 = 99.8% confidence)
            min_window: Minimum samples before drift checking
            max_window: Maximum samples to track
        """
        self.delta = delta
        self.min_window = min_window
        self.max_window = max_window
        
        # Data windows
        self.values: deque = deque(maxlen=max_window)
        self.timestamps: deque = deque(maxlen=max_window)
        
        # Drift tracking
        self.drift_count = 0
        self.last_drift_time: Optional[datetime] = None
        self.drift_history: List[Dict] = []
        
        logger.info(f"Concept Drift Detector initialized - delta={delta}")
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None):
        """Add a new value to track (e.g., win rate, PnL, indicator value)."""
        self.values.append(value)
        self.timestamps.append(timestamp or datetime.now())
    
    def detect_drift(self) -> Tuple[bool, Dict]:
        """
        Detect if concept drift has occurred using ADWIN.
        
        Returns:
            Tuple of (drift_detected, details)
        """
        if len(self.values) < self.min_window:
            return False, {'status': 'insufficient_data', 'size': len(self.values)}
        
        values_list = list(self.values)
        n = len(values_list)
        
        # Try different split points
        for split in range(self.min_window, n - self.min_window + 1):
            w0 = values_list[:split]
            w1 = values_list[split:]
            
            n0, n1 = len(w0), len(w1)
            if n0 == 0 or n1 == 0:
                continue
            
            # Calculate means
            mean0 = sum(w0) / n0
            mean1 = sum(w1) / n1
            
            # Calculate variance (for epsilon)
            var0 = sum((x - mean0) ** 2 for x in w0) / n0
            var1 = sum((x - mean1) ** 2 for x in w1) / n1
            
            # ADWIN cut threshold
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / self.delta))
            
            # Drift detected if means differ significantly
            if abs(mean0 - mean1) > epsilon:
                now = datetime.now()
                self.drift_count += 1
                self.last_drift_time = now
                
                details = {
                    'detected': True,
                    'drift_point': split,
                    'window_size': n,
                    'mean_before': round(mean0, 4),
                    'mean_after': round(mean1, 4),
                    'change': round(mean1 - mean0, 4),
                    'threshold': round(epsilon, 4),
                    'timestamp': now.isoformat()
                }
                
                self.drift_history.append(details)
                self.drift_history = self.drift_history[-20:]
                
                # Remove old values (before drift point)
                for _ in range(split // 2):
                    if len(self.values) > self.min_window:
                        self.values.popleft()
                        self.timestamps.popleft()
                
                logger.warning(f"Concept drift detected! Mean shift: {mean0:.3f} -> {mean1:.3f}")
                return True, details
        
        return False, {'detected': False, 'window_size': n}
    
    def get_status(self) -> Dict:
        """Get current drift detector status."""
        return {
            'window_size': len(self.values),
            'drift_count': self.drift_count,
            'last_drift': self.last_drift_time.isoformat() if self.last_drift_time else None,
            'current_mean': sum(self.values) / len(self.values) if self.values else 0,
            'history_count': len(self.drift_history)
        }


class MultiFactorConfidenceScorer:
    """
    Feature #97: Multi-Factor Confidence Scoring
    Combines multiple signals into a unified confidence score.
    """
    
    def __init__(self):
        """Initialize multi-factor confidence scorer."""
        self.weights = {
            'ml_confidence': 0.30,
            'quality_score': 0.25,
            'regime_alignment': 0.20,
            'multi_timeframe': 0.15,
            'sentiment': 0.10
        }
    
    def calculate(
        self,
        ml_confidence: float = 0.5,
        quality_score: float = 70,
        regime_alignment: bool = True,
        mtf_confirmed: bool = True,
        sentiment_score: float = 0.5
    ) -> float:
        """
        Calculate unified confidence score from multiple factors.
        
        Args:
            ml_confidence: ML model confidence (0-1)
            quality_score: Trade quality score (0-100)
            regime_alignment: Does signal align with regime?
            mtf_confirmed: Multi-timeframe confirmation?
            sentiment_score: Sentiment score (-1 to 1)
            
        Returns:
            Unified confidence score (0-1)
        """
        scores = {
            'ml_confidence': ml_confidence,
            'quality_score': quality_score / 100,  # Normalize to 0-1
            'regime_alignment': 1.0 if regime_alignment else 0.3,
            'multi_timeframe': 1.0 if mtf_confirmed else 0.4,
            'sentiment': (sentiment_score + 1) / 2  # Normalize -1,1 to 0,1
        }
        
        weighted_sum = sum(scores[k] * self.weights[k] for k in scores)
        
        return round(min(max(weighted_sum, 0), 1), 3)
    
    def get_breakdown(
        self,
        ml_confidence: float,
        quality_score: float,
        regime_alignment: bool,
        mtf_confirmed: bool,
        sentiment_score: float
    ) -> Dict:
        """Get detailed breakdown of confidence factors."""
        raw_scores = {
            'ml_confidence': ml_confidence,
            'quality_score': quality_score / 100,
            'regime_alignment': 1.0 if regime_alignment else 0.3,
            'multi_timeframe': 1.0 if mtf_confirmed else 0.4,
            'sentiment': (sentiment_score + 1) / 2
        }
        
        weighted_scores = {k: raw_scores[k] * self.weights[k] for k in raw_scores}
        
        return {
            'raw_scores': raw_scores,
            'weights': self.weights,
            'weighted_scores': weighted_scores,
            'total': sum(weighted_scores.values())
        }


# Singleton instances
_concept_drift: Optional[ConceptDriftDetector] = None
_confidence_scorer: Optional[MultiFactorConfidenceScorer] = None


def get_concept_drift_detector() -> ConceptDriftDetector:
    global _concept_drift
    if _concept_drift is None:
        _concept_drift = ConceptDriftDetector()
    return _concept_drift


def get_confidence_scorer() -> MultiFactorConfidenceScorer:
    global _confidence_scorer
    if _confidence_scorer is None:
        _confidence_scorer = MultiFactorConfidenceScorer()
    return _confidence_scorer


if __name__ == '__main__':
    # Test concept drift
    drift = ConceptDriftDetector()
    
    # Stable period
    for i in range(30):
        drift.add_value(0.6 + (i % 3) * 0.01)
    
    # New regime (shifted values)
    for i in range(30):
        drift.add_value(0.4 + (i % 3) * 0.01)
    
    detected, details = drift.detect_drift()
    print(f"Drift detected: {detected}")
    print(f"Details: {details}")
    
    # Test confidence scorer
    scorer = MultiFactorConfidenceScorer()
    conf = scorer.calculate(
        ml_confidence=0.75,
        quality_score=82,
        regime_alignment=True,
        mtf_confirmed=True,
        sentiment_score=0.3
    )
    print(f"\nMulti-factor confidence: {conf:.1%}")
