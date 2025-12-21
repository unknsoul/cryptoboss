"""
Base Strategy Interface
Abstract base class for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    
    All strategies must implement:
    - signal(): Generate trading signals
    - check_exit(): Determine when to exit positions
    """
    
    def __init__(self, name: str, strategy_type: str):
        """
        Initialize base strategy
        
        Args:
            name: Strategy name
            strategy_type: Type (momentum, mean_reversion, breakout, etc.)
        """
        self.name = name
        self.strategy_type = strategy_type
        self.parameters = {}
        
    @abstractmethod
    def signal(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
               volumes: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal
        
        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            volumes: Volume data (optional)
            
        Returns:
            Signal dictionary with keys:
                - action: 'LONG' or 'SHORT'
                - stop: Stop loss distance
                - target: Take profit distance (optional)
                - confidence: Signal confidence 0-1
                - metadata: Additional signal information
            
            Returns None if no signal
        """
        pass
    
    @abstractmethod
    def check_exit(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                   position_side: str, entry_price: float, 
                   entry_index: int, current_index: int) -> bool:
        """
        Check if position should be exited
        
        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            position_side: 'LONG' or 'SHORT'
            entry_price: Entry price
            entry_index: Entry bar index
            current_index: Current bar index
            
        Returns:
            True if should exit, False otherwise
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get strategy metadata
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'name': self.name,
            'type': self.strategy_type,
            'parameters': self.parameters
        }
    
    def set_parameters(self, **kwargs):
        """Update strategy parameters"""
        self.parameters.update(kwargs)
    
    def validate_data(self, *arrays) -> bool:
        """
        Validate input data arrays
        
        Returns:
            True if data is valid, False otherwise
        """
        if not arrays:
            return False
        
        length = len(arrays[0])
        if length == 0:
            return False
        
        # Check all arrays have same length
        for arr in arrays[1:]:
            if len(arr) != length:
                return False
        
        return True
    
    def calculate_position_size(self, signal: Dict[str, Any], 
                                capital: float, risk_per_trade: float) -> float:
        """
        Calculate position size based on risk
        
        Args:
            signal: Signal dictionary with 'stop' key
            capital: Available capital
            risk_per_trade: Risk percentage (0.01 = 1%)
            
        Returns:
            Position size in dollars
        """
        if 'stop' not in signal or signal['stop'] <= 0:
            return capital * risk_per_trade
        
        risk_amount = capital * risk_per_trade
        stop_distance = signal['stop']
        
        # Position size = Risk amount / Stop distance
        position_size = risk_amount / stop_distance
        
        return min(position_size, capital)
