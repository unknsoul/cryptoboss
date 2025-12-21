"""
Strategy Ensemble - Multi-Strategy Coordination
Combines multiple strategies with intelligent weighting and allocation

Features:
- Weighted voting system
- Performance-based allocation
- Dynamic strategy weighting
- Ensemble confidence scoring
- Multi-strategy portfolio management
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque

from core.strategies.factory import AdvancedStrategyFactory
from core.monitoring.logger import get_logger
from core.monitoring.metrics import get_metrics


logger = get_logger()
metrics = get_metrics()


class StrategyEnsemble:
    """
    Coordinates multiple trading strategies
    """
    
    def __init__(self,
                 strategy_names: List[str],
                 weighting_method: str = 'performance',
                 lookback_window: int = 50):
        """
        Args:
            strategy_names: List of strategy names to include
            weighting_method: 'equal', 'performance', or 'sharpe'
            lookback_window: Performance lookback for weighting
        """
        self.strategy_names = strategy_names
        self.weighting_method = weighting_method
        self.lookback_window = lookback_window
        
        # Create strategy instances
        self.strategies = {}
        for name in strategy_names:
            try:
                self.strategies[name] = AdvancedStrategyFactory.create(name)
                logger.info(f"Loaded strategy: {name}")
            except Exception as e:
                logger.error(f"Failed to load strategy {name}: {e}")
        
        # Performance tracking
        self.strategy_performance = {name: deque(maxlen=lookback_window) 
                                    for name in self.strategies.keys()}
        
        # Current weights
        self.weights = self._initialize_weights()
        
        logger.info(
            f"Strategy ensemble initialized with {len(self.strategies)} strategies",
            strategies=list(self.strategies.keys()),
            weighting_method=weighting_method
        )
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize equal weights"""
        n = len(self.strategies)
        if n == 0:
            return {}
        
        return {name: 1.0/n for name in self.strategies.keys()}
    
    def get_ensemble_signal(self,
                           highs: np.ndarray,
                           lows: np.ndarray,
                           closes: np.ndarray,
                           volumes: Optional[np.ndarray] = None,
                           **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get combined signal from all strategies
        
        Returns:
            Ensemble signal or None
        """
        signals = {}
        
        # Get signal from each strategy
        for name, strategy in self.strategies.items():
            try:
                signal = strategy.signal(highs, lows, closes, volumes)
                if signal and signal.get('action') != 'HOLD':
                    signals[name] = signal
            except Exception as e:
                logger.error(f"Error getting signal from {name}: {e}")
        
        if not signals:
            return None
        
        # Combine signals
        ensemble_signal = self._combine_signals(signals)
        
        return ensemble_signal
    
    def _combine_signals(self, signals: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
        """
        Combine multiple strategy signals
        
        Args:
            signals: Dict of strategy_name -> signal
        
        Returns:
            Combined ensemble signal
        """
        if not signals:
            return None
        
        # Update weights based on recent performance
        if self.weighting_method != 'equal':
            self.weights = self._calculate_performance_weights()
        
        # Weighted voting
        long_score = 0
        short_score = 0
        hold_score = 0
        
        total_weight = 0
        
        for name, signal in signals.items():
            weight = self.weights.get(name, 0)
            total_weight += weight
            
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0.5)
            
            weighted_confidence = weight * confidence
            
            if action == 'LONG':
                long_score += weighted_confidence
            elif action == 'SHORT':
                short_score += weighted_confidence
            else:
                hold_score += weighted_confidence
        
        # Normalize scores
        if total_weight > 0:
            long_score /= total_weight
            short_score /= total_weight
            hold_score /= total_weight
        
        # Determine ensemble action
        max_score = max(long_score, short_score, hold_score)
        
        if max_score < 0.5:  # Minimum confidence threshold
            return None
        
        if long_score == max_score:
            action = 'LONG'
            confidence = long_score
        elif short_score == max_score:
            action = 'SHORT'
            confidence = short_score
        else:
            return None  # Hold
        
        # Calculate ensemble stop/target (average of all signals)
        stops = [s.get('stop', 0) for s in signals.values() if s.get('stop')]
        targets = [s.get('target', 0) for s in signals.values() if s.get('target')]
        
        avg_stop = np.mean(stops) if stops else 0
        avg_target = np.mean(targets) if targets else 0
        
        # Ensemble metadata
        metadata = {
            'ensemble_type': self.weighting_method,
            'num_strategies': len(signals),
            'strategy_votes': {
                'long': long_score,
                'short': short_score,
                'hold': hold_score
            },
            'contributing_strategies': list(signals.keys()),
            'weights_used': {name: self.weights[name] for name in signals.keys()}
        }
        
        logger.info(
            f"Ensemble signal: {action}",
            confidence=f"{confidence:.2f}",
            num_strategies=len(signals),
            long_score=f"{long_score:.2f}",
            short_score=f"{short_score:.2f}"
        )
        
        metrics.increment(f"ensemble_signal_{action.lower()}")
        
        return {
            'action': action,
            'confidence': confidence,
            'stop': avg_stop,
            'target': avg_target,
            'metadata': metadata
        }
    
    def record_trade_result(self, strategy_name: str, pnl: float):
        """
        Record trade result for a strategy
        
        Args:
            strategy_name: Name of strategy
            pnl: Profit/loss of the trade
        """
        if strategy_name in self.strategy_performance:
            self.strategy_performance[strategy_name].append({
                'timestamp': datetime.now(),
                'pnl': pnl
            })
            
            logger.debug(f"Recorded trade for {strategy_name}: {pnl:.2f}")
    
    def _calculate_performance_weights(self) -> Dict[str, float]:
        """
        Calculate strategy weights based on recent performance
        
        Returns:
            Dict of strategy_name -> weight
        """
        weights = {}
        
        for name in self.strategies.keys():
            performance = self.strategy_performance.get(name, [])
            
            if len(performance) < 5:
                # Not enough data, use equal weight
                weights[name] = 1.0
                continue
            
            pnls = [p['pnl'] for p in performance]
            
            if self.weighting_method == 'performance':
                # Weight by total PnL
                total_pnl = sum(pnls)
                weights[name] = max(0, total_pnl)  # Only positive weights
                
            elif self.weighting_method == 'sharpe':
                # Weight by Sharpe ratio
                mean_pnl = np.mean(pnls)
                std_pnl = np.std(pnls)
                sharpe = mean_pnl / (std_pnl + 1e-8)
                weights[name] = max(0, sharpe)
            
            else:
                weights[name] = 1.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            # All negative, use equal weights
            weights = self._initialize_weights()
        
        return weights
    
    def get_strategy_statistics(self) -> pd.DataFrame:
        """
        Get performance statistics for all strategies
        
        Returns:
            DataFrame with strategy stats
        """
        stats = []
        
        for name in self.strategies.keys():
            performance = self.strategy_performance.get(name, [])
            
            if not performance:
                continue
            
            pnls = [p['pnl'] for p in performance]
            
            stats.append({
                'strategy': name,
                'num_trades': len(pnls),
                'total_pnl': sum(pnls),
                'avg_pnl': np.mean(pnls),
                'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
                'sharpe': np.mean(pnls) / (np.std(pnls) + 1e-8),
                'current_weight': self.weights.get(name, 0)
            })
        
        return pd.DataFrame(stats)
    
    def rebalance_weights(self):
        """Force weight recalculation"""
        self.weights = self._calculate_performance_weights()
        logger.info("Ensemble weights rebalanced", weights=self.weights)


if __name__ == "__main__":
    print("=" * 70)
    print("🎯 STRATEGY ENSEMBLE TEST")
    print("=" * 70)
    
    # Create ensemble
    strategies = [
        'statistical_arbitrage',
        'breakout_momentum',
        'mean_reversion'
    ]
    
    ensemble = StrategyEnsemble(
        strategy_names=strategies,
        weighting_method='performance'
    )
    
    # Test signal generation
    print("\n1. Testing Ensemble Signal Generation:")
    
    # Generate sample data
    np.random.seed(42)
    closes = 45000 + np.random.randn(200).cumsum() * 50
    highs = closes * 1.002
    lows = closes * 0.998
    volumes = np.random.uniform(900, 1100, 200)
    
    signal = ensemble.get_ensemble_signal(highs, lows, closes, volumes)
    
    if signal:
        print(f"   Action: {signal['action']}")
        print(f"   Confidence: {signal['confidence']:.2f}")
        print(f"   Contributing strategies: {signal['metadata']['contributing_strategies']}")
        print(f"   Votes: {signal['metadata']['strategy_votes']}")
    else:
        print("   No ensemble signal generated")
    
    # Test performance tracking
    print("\n2. Testing Performance Tracking:")
    
    # Simulate some trades
    for _ in range(20):
        for strategy in strategies:
            pnl = np.random.normal(10, 50)  # Random P&L
            ensemble.record_trade_result(strategy, pnl)
    
    # Get statistics
    stats = ensemble.get_strategy_statistics()
    print(stats.to_string(index=False))
    
    print("\n3. Rebalanced Weights:")
    ensemble.rebalance_weights()
    for name, weight in ensemble.weights.items():
        print(f"   {name}: {weight:.1%}")
    
    print("\n✅ Strategy ensemble test complete")
