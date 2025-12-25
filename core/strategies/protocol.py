"""
Strategy Protocol - Unified Interface for All Trading Strategies

All strategies MUST implement this interface to work with the StrategyManager.
This enforces consistency across the codebase.
"""
from typing import Protocol, Dict, Any, Optional, List
import numpy as np
import pandas as pd


class IStrategy(Protocol):
    """
    Protocol defining the standard interface for all trading strategies.
    
    All strategies must implement:
    - name: str - Human-readable strategy name
    - strategy_type: str - Category (momentum, mean_reversion, etc.)
    - generate_signal(df) - Signal generation from DataFrame
    """
    
    name: str
    strategy_type: str
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generate a trading signal from market data.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Signal dict with keys:
            - action: 'LONG' | 'SHORT' | None
            - confidence: float (0.0 - 1.0)
            - stop: float (ATR-based stop distance)
            - target: float (ATR-based target distance)
            - reasons: List[str] (human-readable reasons)
            - metadata: Dict (strategy-specific data)
            
            Or None if no signal.
        """
        ...


class BaseStrategyV2:
    """
    Base class that implements common functionality for strategies.
    All new strategies should inherit from this class.
    """
    
    def __init__(self, name: str, strategy_type: str = "generic"):
        self.name = name
        self.strategy_type = strategy_type
        self.parameters: Dict[str, Any] = {}
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Override in subclass"""
        raise NotImplementedError("Subclass must implement generate_signal()")
    
    def validate_data(self, df: pd.DataFrame, min_rows: int = 50) -> bool:
        """Check if data is sufficient for signal generation"""
        if df is None or len(df) < min_rows:
            return False
        required_cols = ['close', 'high', 'low']
        return all(col in df.columns for col in required_cols)
    
    def create_signal(
        self,
        action: str,
        confidence: float,
        stop: float,
        target: float,
        reasons: List[str],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Helper to create a properly formatted signal dict"""
        return {
            'action': action,
            'confidence': min(max(confidence, 0.0), 1.0),
            'stop': stop,
            'target': target,
            'reasons': reasons,
            'metadata': metadata or {'strategy': self.name}
        }
