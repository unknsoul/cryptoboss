
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DynamicPositionSizer:
    """
    Calculates position size based on current volatility (ATR),
    account equity, and risk parameters.
    """
    def __init__(self, risk_per_trade: float = 0.015):
        self.risk_per_trade = risk_per_trade  # 1.5% risk per trade default
        self.max_position_size_pct = 0.5      # Max 50% of equity in one trade
        self.min_position_size_usd = 100.0    # Minimum trade size
        
    def calculate_size(self, equity: float, price: float, atr: float, confidence: float = 1.0) -> Dict[str, float]:
        """
        Calculate safe position size.
        
        Strategy:
        1. Fixed fractional risk (e.g., risk 1.5% of equity)
        2. Volatility sizing: Distance to stop loss is proportional to ATR
        3. Confidence scaling: Reduce size for lower confidence signals
        """
        if equity <= 0 or price <= 0:
            return {'size': 0.0, 'risk_amount': 0.0, 'stop_loss_dist': 0.0}
            
        # 1. Determine Stop Loss Distance
        # 2x ATR is a standard initial stop
        stop_loss_dist = atr * 2.0
        if stop_loss_dist == 0:
            stop_loss_dist = price * 0.02 # Fallback 2%
            
        # 2. Amount to Risk in USD
        risk_amount = equity * self.risk_per_trade
        
        # 3. Scale by Confidence (optional)
        # If confidence is 0.7, we might risk only 70% of max, or keep it fixed.
        # Here we'll simple scale linearly from 0.5 to 1.0 confidence
        # 0.5 conf -> 0.5 size, 1.0 conf -> 1.0 size
        confidence_multiplier = max(0.5, min(1.0, confidence))
        risk_amount *= confidence_multiplier
        
        # 4. Calculate Quantity
        # Quantity = Risk Amount / Risk Per Unit (Stop Distance)
        quantity = risk_amount / stop_loss_dist
        
        # 5. Cap by Max Position Size (Notional Value)
        # We don't want to use 5x leverage just because vol is low
        max_notional_usd = equity * self.max_position_size_pct
        current_notional_usd = quantity * price
        
        if current_notional_usd > max_notional_usd:
            quantity = max_notional_usd / price
            
        # 6. Check Minimum
        curr_notional = quantity * price
        if curr_notional < self.min_position_size_usd:
            return {'size': 0.0, 'risk_amount': 0.0, 'stop_loss_dist': stop_loss_dist}
            
        return {
            'size': round(quantity, 6),
            'risk_amount': round(risk_amount, 2),
            'stop_loss_dist': stop_loss_dist,
            'confidence_multiplier': confidence_multiplier
        }
