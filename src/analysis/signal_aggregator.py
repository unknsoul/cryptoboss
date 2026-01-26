
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalAggregator:
    """
    Advanced signal aggregation using probabilistic logic and confidence weighting.
    Combines input from multiple strategies to form a consensus.
    """
    def __init__(self):
        self.min_confidence_threshold = 0.40  # Lowered for more signals
        
    def aggregate_signals(self, strategy_signals: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Aggregate multiple strategy signals into one final decision.
        
        Args:
            strategy_signals: List of signal dicts from individual strategies
                            Expected format: {'action': 'LONG', 'confidence': 0.8, 'source': 'trend_following'}
        
        Returns:
            Final consensus signal or None if no consensus
        """
        if not strategy_signals:
            return None
            
        long_score = 0.0
        short_score = 0.0
        total_weight = 0.0
        
        reasons = []
        
        for sig in strategy_signals:
            confidence = sig.get('confidence', 0.5)
            action = sig.get('action')
            source = sig.get('source', 'unknown')
            
            # Weight could be dynamic based on strategy historical performance
            weight = 1.0 
            
            if action == 'LONG':
                long_score += confidence * weight
                reasons.append(f"{source}: LONG ({confidence:.0%})")
            elif action == 'SHORT':
                short_score += confidence * weight
                reasons.append(f"{source}: SHORT ({confidence:.0%})")
            
            total_weight += weight
            
        if total_weight == 0:
            return None
            
        # Normalize scores
        long_prob = long_score / total_weight if total_weight > 0 else 0
        short_prob = short_score / total_weight if total_weight > 0 else 0
        
        final_action = None
        final_confidence = 0.0
        
        # Determine winner
        if long_prob > short_prob and long_prob > self.min_confidence_threshold:
            final_action = 'LONG'
            final_confidence = long_prob
        elif short_prob > long_prob and short_prob > self.min_confidence_threshold:
            final_action = 'SHORT'
            final_confidence = short_prob
            
        if final_action:
            return {
                'action': final_action,
                'confidence': final_confidence,
                'reasons': reasons,
                'timestamp': datetime.now().isoformat(),
                'raw_scores': {'long': long_prob, 'short': short_prob}
            }
            
        return None
