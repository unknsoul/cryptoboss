"""
Market Regime Detection - Enterprise Features #140, #141, #142, #143, #144, #145
Regime Classification, Volatility Regime, Trend Detection, Correlation Regime.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
import math

logger = logging.getLogger(__name__)


class MarketRegimeClassifier:
    """
    Feature #140: Market Regime Classifier
    
    Classifies current market into trading regimes.
    """
    
    REGIMES = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOLATILITY', 'LOW_VOLATILITY', 'BREAKOUT']
    
    def __init__(self):
        """Initialize regime classifier."""
        self.current_regime = 'UNKNOWN'
        self.regime_history: List[Dict] = []
        self.probabilities: Dict[str, float] = {}
        
        logger.info("Market Regime Classifier initialized")
    
    def classify(
        self,
        returns: List[float],
        volatility: float,
        trend_strength: float,
        volume_ratio: float
    ) -> Dict:
        """
        Classify current market regime.
        
        Args:
            returns: Recent returns
            volatility: Current volatility
            trend_strength: ADX or trend indicator
            volume_ratio: Volume vs average
        """
        if not returns:
            return {'regime': 'UNKNOWN', 'confidence': 0}
        
        avg_return = sum(returns) / len(returns)
        
        # Calculate regime probabilities
        probs = {
            'TRENDING_UP': 0,
            'TRENDING_DOWN': 0,
            'RANGING': 0,
            'HIGH_VOLATILITY': 0,
            'LOW_VOLATILITY': 0
        }
        
        # Trend detection
        if trend_strength > 25 and avg_return > 0:
            probs['TRENDING_UP'] = min(1.0, trend_strength / 50)
        elif trend_strength > 25 and avg_return < 0:
            probs['TRENDING_DOWN'] = min(1.0, trend_strength / 50)
        else:
            probs['RANGING'] = 1 - (trend_strength / 50)
        
        # Volatility regime
        if volatility > 0.03:
            probs['HIGH_VOLATILITY'] = min(1.0, volatility / 0.05)
        elif volatility < 0.01:
            probs['LOW_VOLATILITY'] = 1 - (volatility * 100)
        
        # Find dominant regime
        dominant = max(probs, key=probs.get)
        confidence = probs[dominant]
        
        self.current_regime = dominant
        self.probabilities = probs
        
        self.regime_history.append({
            'regime': dominant,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        self.regime_history = self.regime_history[-1000:]
        
        return {
            'regime': dominant,
            'confidence': round(confidence, 3),
            'probabilities': {k: round(v, 3) for k, v in probs.items()},
            'volatility': volatility,
            'trend_strength': trend_strength
        }
    
    def get_regime_duration(self) -> int:
        """Get how long current regime has lasted."""
        if not self.regime_history:
            return 0
        
        current = self.current_regime
        duration = 0
        
        for entry in reversed(self.regime_history):
            if entry['regime'] == current:
                duration += 1
            else:
                break
        
        return duration


class VolatilityRegimeDetector:
    """
    Feature #141: Volatility Regime Detector
    
    Detects volatility regime changes.
    """
    
    def __init__(self, lookback: int = 20):
        """Initialize volatility detector."""
        self.lookback = lookback
        self.volatility_history: List[float] = []
        
        logger.info("Volatility Regime Detector initialized")
    
    def update(self, volatility: float):
        """Update volatility history."""
        self.volatility_history.append(volatility)
        self.volatility_history = self.volatility_history[-500:]
    
    def detect_regime(self) -> Dict:
        """Detect current volatility regime."""
        if len(self.volatility_history) < self.lookback:
            return {'regime': 'UNKNOWN'}
        
        recent = self.volatility_history[-self.lookback:]
        current = recent[-1]
        avg = sum(recent) / len(recent)
        
        # Calculate percentile
        sorted_hist = sorted(self.volatility_history)
        percentile = sorted_hist.index(min(sorted_hist, key=lambda x: abs(x - current))) / len(sorted_hist) * 100
        
        if percentile > 80:
            regime = 'EXTREME_HIGH'
        elif percentile > 60:
            regime = 'HIGH'
        elif percentile > 40:
            regime = 'NORMAL'
        elif percentile > 20:
            regime = 'LOW'
        else:
            regime = 'EXTREME_LOW'
        
        return {
            'regime': regime,
            'current_vol': round(current, 4),
            'avg_vol': round(avg, 4),
            'percentile': round(percentile, 1),
            'expanding': current > avg
        }


class TrendStrengthAnalyzer:
    """
    Feature #142: Trend Strength Analyzer
    
    Analyzes trend strength using multiple methods.
    """
    
    def __init__(self):
        """Initialize trend analyzer."""
        self.price_history: List[float] = []
        
        logger.info("Trend Strength Analyzer initialized")
    
    def add_price(self, price: float):
        """Add price to history."""
        self.price_history.append(price)
        self.price_history = self.price_history[-200:]
    
    def calculate_adx(self, period: int = 14) -> float:
        """Calculate ADX-like trend strength."""
        if len(self.price_history) < period + 1:
            return 0
        
        # Simplified directional movement
        plus_dm = []
        minus_dm = []
        tr = []
        
        for i in range(1, len(self.price_history)):
            high_diff = self.price_history[i] - self.price_history[i-1]
            low_diff = self.price_history[i-1] - self.price_history[i]
            
            plus_dm.append(max(0, high_diff))
            minus_dm.append(max(0, low_diff))
            tr.append(abs(high_diff) + abs(low_diff))
        
        # Simple moving average
        avg_plus = sum(plus_dm[-period:]) / period
        avg_minus = sum(minus_dm[-period:]) / period
        avg_tr = sum(tr[-period:]) / period
        
        if avg_tr == 0:
            return 0
        
        plus_di = (avg_plus / avg_tr) * 100
        minus_di = (avg_minus / avg_tr) * 100
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 0.001) * 100
        
        return round(dx, 2)
    
    def analyze(self) -> Dict:
        """Comprehensive trend analysis."""
        if len(self.price_history) < 20:
            return {'trend': 'UNKNOWN', 'strength': 0}
        
        adx = self.calculate_adx()
        
        # Price momentum
        momentum = (self.price_history[-1] - self.price_history[-10]) / self.price_history[-10] * 100
        
        # Trend direction
        if momentum > 1:
            direction = 'UP'
        elif momentum < -1:
            direction = 'DOWN'
        else:
            direction = 'SIDEWAYS'
        
        # Strength classification
        if adx > 40:
            strength = 'STRONG'
        elif adx > 25:
            strength = 'MODERATE'
        elif adx > 15:
            strength = 'WEAK'
        else:
            strength = 'NO_TREND'
        
        return {
            'direction': direction,
            'strength': strength,
            'adx': adx,
            'momentum_pct': round(momentum, 2),
            'is_trending': adx > 20
        }


class CorrelationRegimeTracker:
    """
    Feature #143: Correlation Regime Tracker
    
    Tracks correlation regime changes.
    """
    
    def __init__(self):
        """Initialize correlation tracker."""
        self.asset_returns: Dict[str, List[float]] = {}
        
        logger.info("Correlation Regime Tracker initialized")
    
    def add_return(self, asset: str, return_pct: float):
        """Add return for an asset."""
        if asset not in self.asset_returns:
            self.asset_returns[asset] = []
        self.asset_returns[asset].append(return_pct)
        self.asset_returns[asset] = self.asset_returns[asset][-100:]
    
    def calculate_correlation(self, asset1: str, asset2: str) -> float:
        """Calculate correlation between two assets."""
        if asset1 not in self.asset_returns or asset2 not in self.asset_returns:
            return 0
        
        r1 = self.asset_returns[asset1]
        r2 = self.asset_returns[asset2]
        
        n = min(len(r1), len(r2))
        if n < 10:
            return 0
        
        r1 = r1[-n:]
        r2 = r2[-n:]
        
        mean1 = sum(r1) / n
        mean2 = sum(r2) / n
        
        cov = sum((r1[i] - mean1) * (r2[i] - mean2) for i in range(n)) / n
        std1 = (sum((x - mean1) ** 2 for x in r1) / n) ** 0.5
        std2 = (sum((x - mean2) ** 2 for x in r2) / n) ** 0.5
        
        if std1 == 0 or std2 == 0:
            return 0
        
        return round(cov / (std1 * std2), 3)
    
    def detect_regime(self, assets: List[str]) -> Dict:
        """Detect overall correlation regime."""
        if len(assets) < 2:
            return {'regime': 'UNKNOWN'}
        
        correlations = []
        for i, a1 in enumerate(assets):
            for a2 in assets[i+1:]:
                corr = self.calculate_correlation(a1, a2)
                correlations.append(corr)
        
        if not correlations:
            return {'regime': 'UNKNOWN'}
        
        avg_corr = sum(correlations) / len(correlations)
        
        if avg_corr > 0.7:
            regime = 'HIGH_CORRELATION'
        elif avg_corr > 0.3:
            regime = 'MODERATE_CORRELATION'
        elif avg_corr > -0.3:
            regime = 'LOW_CORRELATION'
        else:
            regime = 'NEGATIVE_CORRELATION'
        
        return {
            'regime': regime,
            'avg_correlation': round(avg_corr, 3),
            'num_pairs': len(correlations)
        }


class RegimeTransitionDetector:
    """
    Feature #144: Regime Transition Detector
    
    Detects when market is transitioning between regimes.
    """
    
    def __init__(self, sensitivity: float = 0.3):
        """Initialize transition detector."""
        self.sensitivity = sensitivity
        self.regime_history: List[str] = []
        self.transition_alerts: List[Dict] = []
        
        logger.info("Regime Transition Detector initialized")
    
    def update(self, regime: str, confidence: float):
        """Update with new regime reading."""
        self.regime_history.append(regime)
        self.regime_history = self.regime_history[-50:]
        
        # Detect transition
        if len(self.regime_history) >= 5:
            recent = self.regime_history[-5:]
            prev = self.regime_history[-10:-5] if len(self.regime_history) >= 10 else self.regime_history[:5]
            
            if recent[-1] != prev[-1] if prev else None:
                self.transition_alerts.append({
                    'from_regime': prev[-1] if prev else 'UNKNOWN',
                    'to_regime': recent[-1],
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                })
                self.transition_alerts = self.transition_alerts[-50:]
    
    def is_transitioning(self) -> bool:
        """Check if market is currently transitioning."""
        if len(self.regime_history) < 10:
            return False
        
        recent = self.regime_history[-10:]
        unique = len(set(recent))
        
        return unique >= 3
    
    def get_transition_probability(self, from_regime: str, to_regime: str) -> float:
        """Get probability of transition between regimes."""
        if len(self.regime_history) < 20:
            return 0.5
        
        transitions = 0
        total = 0
        
        for i in range(1, len(self.regime_history)):
            if self.regime_history[i-1] == from_regime:
                total += 1
                if self.regime_history[i] == to_regime:
                    transitions += 1
        
        return round(transitions / max(total, 1), 3)


class AdaptiveStrategySelector:
    """
    Feature #145: Adaptive Strategy Selector
    
    Selects optimal strategy based on regime.
    """
    
    def __init__(self):
        """Initialize strategy selector."""
        self.strategy_map: Dict[str, str] = {
            'TRENDING_UP': 'TREND_FOLLOWING_LONG',
            'TRENDING_DOWN': 'TREND_FOLLOWING_SHORT',
            'RANGING': 'MEAN_REVERSION',
            'HIGH_VOLATILITY': 'BREAKOUT',
            'LOW_VOLATILITY': 'RANGE_TRADING'
        }
        self.performance: Dict[str, Dict] = {}
        
        logger.info("Adaptive Strategy Selector initialized")
    
    def select_strategy(self, regime: str, confidence: float) -> Dict:
        """Select strategy for current regime."""
        strategy = self.strategy_map.get(regime, 'NEUTRAL')
        
        return {
            'regime': regime,
            'strategy': strategy,
            'confidence': confidence,
            'parameters': self._get_strategy_params(strategy, confidence)
        }
    
    def _get_strategy_params(self, strategy: str, confidence: float) -> Dict:
        """Get parameters for strategy."""
        base_params = {
            'TREND_FOLLOWING_LONG': {'entry_pullback': 0.02, 'trailing_stop': True},
            'TREND_FOLLOWING_SHORT': {'entry_pullback': 0.02, 'trailing_stop': True},
            'MEAN_REVERSION': {'rsi_oversold': 30, 'rsi_overbought': 70},
            'BREAKOUT': {'atr_multiple': 2.0, 'volume_confirm': True},
            'RANGE_TRADING': {'support_buffer': 0.005, 'resistance_buffer': 0.005}
        }
        
        params = base_params.get(strategy, {})
        
        # Adjust for confidence
        if confidence < 0.5:
            params['position_size_adj'] = 0.5
        
        return params
    
    def record_performance(self, strategy: str, pnl: float):
        """Record strategy performance."""
        if strategy not in self.performance:
            self.performance[strategy] = {'trades': 0, 'pnl': 0, 'wins': 0}
        
        self.performance[strategy]['trades'] += 1
        self.performance[strategy]['pnl'] += pnl
        if pnl > 0:
            self.performance[strategy]['wins'] += 1


# Singletons
_regime_classifier: Optional[MarketRegimeClassifier] = None
_vol_detector: Optional[VolatilityRegimeDetector] = None
_trend_analyzer: Optional[TrendStrengthAnalyzer] = None
_corr_tracker: Optional[CorrelationRegimeTracker] = None
_transition_detector: Optional[RegimeTransitionDetector] = None
_strategy_selector: Optional[AdaptiveStrategySelector] = None


def get_regime_classifier() -> MarketRegimeClassifier:
    global _regime_classifier
    if _regime_classifier is None:
        _regime_classifier = MarketRegimeClassifier()
    return _regime_classifier


def get_vol_detector() -> VolatilityRegimeDetector:
    global _vol_detector
    if _vol_detector is None:
        _vol_detector = VolatilityRegimeDetector()
    return _vol_detector


def get_trend_analyzer() -> TrendStrengthAnalyzer:
    global _trend_analyzer
    if _trend_analyzer is None:
        _trend_analyzer = TrendStrengthAnalyzer()
    return _trend_analyzer


def get_corr_tracker() -> CorrelationRegimeTracker:
    global _corr_tracker
    if _corr_tracker is None:
        _corr_tracker = CorrelationRegimeTracker()
    return _corr_tracker


def get_transition_detector() -> RegimeTransitionDetector:
    global _transition_detector
    if _transition_detector is None:
        _transition_detector = RegimeTransitionDetector()
    return _transition_detector


def get_strategy_selector() -> AdaptiveStrategySelector:
    global _strategy_selector
    if _strategy_selector is None:
        _strategy_selector = AdaptiveStrategySelector()
    return _strategy_selector
