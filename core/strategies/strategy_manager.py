
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Centralized manager for all trading strategies.
    Handles data distribution, signal aggregation, and voting.
    """
    def __init__(self):
        self.strategies = {}
        self.regime = "neutral"
        self.active_strategies = []
        self.min_consensus = 0.6  # 60% consensus required (if using voting)
        
    def register_strategy(self, name: str, strategy_instance: Any):
        """Register a new strategy"""
        self.strategies[name] = strategy_instance
        if name not in self.active_strategies:
            self.active_strategies.append(name)
        logger.info(f"Registered strategy: {name}")

    def enable_strategy(self, name: str):
        """Enable a strategy"""
        if name in self.strategies and name not in self.active_strategies:
            self.active_strategies.append(name)

    def disable_strategy(self, name: str):
        """Disable a strategy"""
        if name in self.active_strategies:
            self.active_strategies.remove(name)

    def update_regime(self, regime: str):
        """Update market regime to enable/disable strategies"""
        self.regime = regime
        logger.info(f"Market regime updated to: {regime}")

    def generate_raw_signals(self, candles: List[Dict]) -> Dict[str, str]:
        """
        Generate raw signals for probabilistic engine.
        Returns: {'momentum': 'BUY', 'mean_reversion': 'SELL'}
        """
        if not self.strategies:
            return {}

        df = pd.DataFrame(candles)
        raw_signals = {}
        
        for name, strategy in self.strategies.items():
            if name not in self.active_strategies:
                continue
                
            try:
                sig = None
                
                # Method 1: generate_signal(df) interface (newer strategies)
                if hasattr(strategy, 'generate_signal'):
                    sig = strategy.generate_signal(df)
                
                # Method 2: signal(arrays) interface (institutional/base strategies)
                if sig is None and hasattr(strategy, 'signal'):
                    highs = df['high'].values
                    lows = df['low'].values
                    closes = df['close'].values
                    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
                    sig = strategy.signal(highs, lows, closes, volumes)
                
                # Extract action from signal
                if sig:
                    action = sig.get('action', 'HOLD')
                    raw_signals[name] = action
                    logger.debug(f"Strategy {name} generated: {action}")
                        
            except Exception as e:
                logger.error(f"Error in strategy {name}: {e}")
                
        return raw_signals

    def generate_signals(self, candles: List[Dict]) -> Dict[str, Any]:
        """
        Generate signals from all active strategies and aggregate them.
        """
        if not self.strategies:
            return None

        # Convert candles to DataFrame for strategies that need it
        df = pd.DataFrame(candles)
        
        signals = []
        votes = {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
        total_weight = 0
        
        detailed_signals = {}
        
        for name, strategy in self.strategies.items():
            if name not in self.active_strategies:
                continue
                
            try:
                # Assuming all strategies have a consistent interface
                # If not, we'll need adapters or standardized base class
                if hasattr(strategy, 'generate_signal'):
                    sig = strategy.generate_signal(df)
                else:
                    continue
                    
                if sig:
                    detailed_signals[name] = sig
                    action = sig.get('action', 'HOLD')
                    confidence = sig.get('confidence', 0.5)
                    
                    # Weighted voting
                    if action in votes:
                        votes[action] += confidence
                        total_weight += 1
                        
                    signals.append({
                        'strategy': name,
                        'action': action,
                        'confidence': confidence,
                        'reasons': sig.get('reasons', [])
                    })
            except Exception as e:
                logger.error(f"Error in strategy {name}: {e}")

        # Consensus Logic
        best_action = 'HOLD'
        best_score = 0
        
        # Normalize scores
        if total_weight > 0:
            # Simple sum of confidence for now
            pass

        # Determine winner
        for action, score in votes.items():
            if action != 'HOLD' and score > best_score:
                best_score = score
                best_action = action
        
        # Check if consensus/confidence is high enough
        # This is a simplified aggregation, will be enhanced in SignalAggregator
        
        aggregated_signal = {
            'action': best_action,
            'confidence': best_score / len(self.active_strategies) if self.active_strategies else 0,
            'votes': votes,
            'individual_signals': detailed_signals,
            'timestamp': datetime.now().isoformat()
        }
        
        return aggregated_signal
