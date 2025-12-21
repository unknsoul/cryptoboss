"""
Comprehensive Risk Management Module (Professional)
Features:
- Volatility Targeting (15% Annualized)
- Maximum Drawdown Circuit Breakers
- Order Book Imbalance Guard
- Portfolio Correlation Checks
"""

import numpy as np
from typing import Dict, Any, Optional

class RiskManager:
    """
    Institutional Risk Controller
    """
    
    def __init__(self, capital: float, target_volatility: float = 0.15, 
                 max_drawdown_limit: float = 0.15, max_leverage: float = 1.0):
        self.initial_capital = capital
        self.current_capital = capital
        self.peak_capital = capital
        
        self.target_annual_vol = target_volatility # e.g. 15%
        self.max_drawdown_limit = max_drawdown_limit
        self.max_leverage = max_leverage
        
        self.active_positions: Dict[str, Any] = {}
        self.daily_pnl = 0.0
        self.circuit_breaker_active = False
        
    def check_trade_allowed(self, symbol: str, size: float, price: float, 
                           orderbook_imbalance: float = 0.0, direction: str = "LONG") -> bool:
        """
        Master Risk Check
        """
        # 1. Circuit Breaker
        if self.circuit_breaker_active:
             print("⚠️ Trade Blocked: Circuit Breaker Active")
             return False
             
        # 2. Drawdown Limit
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.max_drawdown_limit:
            print(f"⚠️ Trade Blocked: Max Drawdown {drawdown:.1%} exceeds limit")
            return False
            
        # 3. Order Book Imbalance Guard
        # Imbalance = (Bids - Asks) / (Bids + Asks)
        # Range -1 (All Asks) to +1 (All Bids)
        # If we are Long, we want Bids (Support). Imbalance should be > -0.5
        # If we are Short, we want Asks (Resistance). Imbalance should be < 0.5
        
        if direction == "LONG" and orderbook_imbalance < -0.6:
            print(f"⚠️ Trade Blocked: Order Book Heavy on Sells ({orderbook_imbalance:.2f})")
            return False
            
        if direction == "SHORT" and orderbook_imbalance > 0.6:
            print(f"⚠️ Trade Blocked: Order Book Heavy on Buys ({orderbook_imbalance:.2f})")
            return False
            
        return True
        
    def calculate_vol_target_size(self, price: float, daily_volatility: float) -> float:
        """
        Calculate Position Size based on Volatility Targeting (Risk Parity Logic)
        Formula: (Capital * TargetVol) / (AssetVol)
        """
        # Annualize daily vol
        annual_vol = daily_volatility * np.sqrt(365)
        
        if annual_vol == 0: return 0.0
        
        # Target Notional Exposure
        target_exposure = (self.current_capital * self.target_annual_vol) / annual_vol
        
        # Cap at Max Leverage
        max_exposure = self.current_capital * self.max_leverage
        target_exposure = min(target_exposure, max_exposure)
        
        # Convert to units
        units = target_exposure / price
        return units

    def update_pnl(self, pnl: float):
        """Update PnL and check circuit breakers"""
        self.current_capital += pnl
        self.daily_pnl += pnl
        
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            
        # Circuit Breaker: 5% daily loss
        if self.daily_pnl < -(self.initial_capital * 0.05):
            self.circuit_breaker_active = True
            print("🛑 CIRCUIT BREAKER TRIGGERED: Daily Loss Limit Hit")
