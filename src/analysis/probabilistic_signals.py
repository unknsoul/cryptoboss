"""
Probabilistic Signal Generator
Critical: Output probabilities, not binary signals
Trade only high-confidence setups
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SignalAction(Enum):
    """Signal action types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class ProbabilisticSignal:
    """Signal with probability distribution"""
    action: SignalAction
    probabilities: Dict[str, float]  # {'BUY': 0.7, 'SELL': 0.1, 'HOLD': 0.2}
    confidence: float  # 0-1
    reasons: list[str]
    should_trade: bool  # True if exceeds confidence threshold
    metadata: Dict


class ProbabilisticSignalGenerator:
    """
    Generate probabilistic signals instead of binary BUY/SELL
    
    Key principles:
    1. Output probability distribution, not single prediction
    2. Only trade if P(action) > confidence_threshold
    3. Have a NO-TRADE zone for uncertain setups
    4. Combine multiple signal sources with weights
    
    This dramatically improves accuracy by filtering low-quality signals.
    """
    
    def __init__(
        self,
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.60,
        ml_model=None
    ):
        """
        Initialize probabilistic signal generator
        
        Args:
            buy_threshold: Minimum P(BUY) to signal long (default 0.60)
            sell_threshold: Minimum P(SELL) to signal short (default 0.60)
            ml_model: Optional trained ML model that outputs probabilities
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.ml_model = ml_model
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        technical_signals: Optional[Dict] = None,
        regime_info: Optional[Dict] = None
    ) -> ProbabilisticSignal:
        """
        Generate probabilistic signal from multiple sources
        """
        # Collect probabilities and weights
        components = []
        
        # 1. Technical indicators (Base weight 0.5)
        if technical_signals:
            tech_prob = self._assess_technical_signals(technical_signals)
            components.append({'prob': tech_prob, 'weight': 0.5})
            
        # 2. ML model (Base weight 0.3)
        if self.ml_model is not None:
            ml_prob = self._get_ml_probabilities(df)
            components.append({'prob': ml_prob, 'weight': 0.3})
        else:
            # If no ML, redistribute weight to Technicals
            if components:
                components[0]['weight'] += 0.3
        
        # 3. Regime filter (Base weight 0.2)
        if regime_info:
            regime_prob = self._assess_regime(regime_info)
            components.append({'prob': regime_prob, 'weight': 0.2})

        # Calculate weighted sum
        final_prob = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        total_weight = sum(c['weight'] for c in components)
        
        reasons = []
        
        if total_weight > 0:
            for c in components:
                w = c['weight'] / total_weight  # Normalize weights
                for action in final_prob:
                    final_prob[action] += c['prob'].get(action, 0.0) * w
                    
            # Generate reasons
            if technical_signals:
                tech_p = components[0]['prob']
                if tech_p['BUY'] > 0.5: reasons.append(f"Technical bullish ({tech_p['BUY']:.0%})")
                elif tech_p['SELL'] > 0.5: reasons.append(f"Technical bearish ({tech_p['SELL']:.0%})")
                
            if self.ml_model and len(components) > 1:
                ml_p = components[1]['prob']
                if ml_p['BUY'] > 0.5: reasons.append(f"ML bullish ({ml_p['BUY']:.0%})")
                
            if regime_info and not regime_info.get('should_trade', True):
                 reasons.append("Regime filter: unfavorable")

        else:
            final_prob = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 1.0}
        
        # Determine action
        buy_prob = final_prob['BUY']
        sell_prob = final_prob['SELL']
        hold_prob = final_prob['HOLD']
        
        # Decision logic with no-trade zone
        if buy_prob >= self.buy_threshold:
            action = SignalAction.BUY
            confidence = buy_prob
            should_trade = True
            reasons.append(f"HIGH confidence BUY: {buy_prob:.1%} >= {self.buy_threshold:.1%}")
            
        elif sell_prob >= self.sell_threshold:
            action = SignalAction.SELL
            confidence = sell_prob
            should_trade = True
            reasons.append(f"HIGH confidence SELL: {sell_prob:.1%} >= {self.sell_threshold:.1%}")
            
        else:
            action = SignalAction.HOLD
            confidence = hold_prob
            should_trade = False
            reasons.append(f"Wait: Insufficient confidence (B:{buy_prob:.0%} S:{sell_prob:.0%})")
        
        return ProbabilisticSignal(
            action=action,
            probabilities=final_prob,
            confidence=confidence,
            reasons=reasons,
            should_trade=should_trade,
            metadata={
                'buy_threshold': self.buy_threshold,
                'sell_threshold': self.sell_threshold,
                'has_ml': self.ml_model is not None
            }
        )
    
    def _assess_technical_signals(self, signals: Dict) -> Dict[str, float]:
        """
        Assess technical signals and convert to probabilities
        
        Args:
            signals: Dict with technical signals from various indicators
            
        Returns:
            Probability distribution
        """
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # Count signals from different indicators
        for indicator, signal in signals.items():
            total_signals += 1
            if signal == 'BUY' or signal == 'LONG':
                bullish_signals += 1
            elif signal == 'SELL' or signal == 'SHORT':
                bearish_signals += 1
        
        if total_signals == 0:
            return {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 1.0}
        
        buy_prob = bullish_signals / total_signals
        sell_prob = bearish_signals / total_signals
        hold_prob = 1.0 - buy_prob - sell_prob
        
        return {'BUY': buy_prob, 'SELL': sell_prob, 'HOLD': max(0, hold_prob)}
    
    def _get_ml_probabilities(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get probabilities from ML model
        
        Args:
            df: DataFrame with features
            
        Returns:
            Probability distribution
        """
        if self.ml_model is None:
            return {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 1.0}
        
        try:
            # Get features for ML model
            # This would use the same feature engineering as training
            # For now, simplified
            latest_features = df.tail(1)  # Would need proper feature extraction
            
            # Get class probabilities
            proba = self.ml_model.predict_proba(latest_features)[0]
            
            # Assuming model predicts [SELL, HOLD, BUY] classes
            return {
                'SELL': proba[0] if len(proba) > 0 else 0.0,
                'HOLD': proba[1] if len(proba) > 1 else 1.0,
                'BUY': proba[2] if len(proba) > 2 else 0.0
            }
        except:
            return {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 1.0}
    
    def _assess_regime(self, regime_info: Dict) -> Dict[str, float]:
        """
        Adjust probabilities based on market regime
        
        Args:
            regime_info: Dict with regime information
            
        Returns:
            Probability adjustment
        """
        regime = regime_info.get('regime', 'neutral')
        should_trade = regime_info.get('should_trade', True)
        
        if not should_trade:
            # Wrong regime - heavily favor HOLD
            return {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 1.0}
        
        # Regime-based bias
        if regime == 'trending_up':
            return {'BUY': 0.6, 'SELL': 0.1, 'HOLD': 0.3}
        elif regime == 'trending_down':
            return {'BUY': 0.1, 'SELL': 0.6, 'HOLD': 0.3}
        else:  # ranging, high_vol, low_vol
            return {'BUY': 0.3, 'SELL': 0.3, 'HOLD': 0.4}
    
    def _combine_probabilities(
        self,
        prob1: Dict[str, float],
        prob2: Dict[str, float],
        weight: float
    ) -> Dict[str, float]:
        """
        Combine two probability distributions with weighted average
        
        Args:
            prob1: First probability distribution
            prob2: Second probability distribution
            weight: Weight for prob2 (prob1 gets 1-weight)
            
        Returns:
            Combined probability distribution
        """
        combined = {}
        for key in prob1.keys():
            combined[key] = prob1.get(key, 0) * (1 - weight) + prob2.get(key, 0) * weight
        
        return combined


if __name__ == '__main__':
    # Example usage
    print("=" * 70)
    print("PROBABILISTIC SIGNAL GENERATION EXAMPLE")
    print("=" * 70)
    
    # Initialize generator
    generator = ProbabilisticSignalGenerator(
        buy_threshold=0.65,
        sell_threshold=0.65
    )
    
    # Sample data
    df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    # Test cases
    test_cases = [
        {
            'name': 'Strong Buy',
            'technical': {'EMA': 'BUY', 'RSI': 'BUY', 'MACD': 'BUY'},
            'regime': {'regime': 'trending_up', 'should_trade': True}
        },
        {
            'name': 'Weak Buy (filtered)',
            'technical': {'EMA': 'BUY', 'RSI': 'SELL', 'MACD': 'HOLD'},
            'regime': {'regime': 'ranging', 'should_trade': True}
        },
        {
            'name': 'Wrong Regime (no trade)',
            'technical': {'EMA': 'BUY', 'RSI': 'BUY', 'MACD': 'BUY'},
            'regime': {'regime': 'high_volatility', 'should_trade': False}
        }
    ]
    
    for test in test_cases:
        print(f"\nTest Case: {test['name']}")
        print("-" * 70)
        
        signal = generator.generate_signal(
            df,
            technical_signals=test.get('technical'),
            regime_info=test.get('regime')
        )
        
        print(f"Action: {signal.action.value}")
        print(f"Should Trade: {signal.should_trade}")
        print(f"Confidence: {signal.confidence:.2%}")
        print(f"Probabilities:")
        for action, prob in signal.probabilities.items():
            print(f"  {action}: {prob:.2%}")
        print(f"Reasons:")
        for reason in signal.reasons:
            print(f"  - {reason}")
    
    print("\n" + "=" * 70)
    print("✓ Probabilistic signals filter low-confidence setups")
    print("✓ No-trade zone prevents overtrading")
    print("✓ Multiple signal sources combined with weights")
    print("✓ Accuracy improves by trading LESS, not more")
    print("=" * 70)
