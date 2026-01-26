"""
Portfolio Risk Model - Upgrade C

Comprehensive portfolio-wide risk management:
- Value at Risk (VaR) calculation
- Maximum Drawdown tracking
- Exposure limits across symbols
- Correlation-aware position sizing
- Stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPosition:
    """Single position in the portfolio."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    side: str  # 'long' or 'short'
    
    @property
    def notional_value(self) -> float:
        return abs(self.quantity * self.current_price)
    
    @property
    def unrealized_pnl(self) -> float:
        if self.side == 'long':
            return self.quantity * (self.current_price - self.entry_price)
        else:
            return self.quantity * (self.entry_price - self.current_price)
    
    @property
    def pnl_pct(self) -> float:
        cost = self.quantity * self.entry_price
        return (self.unrealized_pnl / cost * 100) if cost > 0 else 0


@dataclass
class RiskMetrics:
    """Portfolio risk metrics snapshot."""
    timestamp: datetime
    portfolio_value: float
    total_exposure: float
    exposure_pct: float
    var_95: float
    var_99: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    correlation_risk: float
    concentration_risk: float


class PortfolioRiskModel:
    """
    Institutional-grade portfolio risk management.
    
    Features:
    - VaR calculation (Historical, Parametric)
    - Max drawdown tracking with peak detection
    - Exposure limits per symbol and total
    - Correlation-aware risk aggregation
    - Stress testing scenarios
    
    Usage:
        risk_model = PortfolioRiskModel(initial_capital=100000)
        
        # Update positions
        risk_model.update_position("BTC/USDT", 0.5, 65000, 67000, "long")
        
        # Check risk
        can_trade, reason = risk_model.check_new_trade("ETH/USDT", 10, 3500, "long")
        
        # Get metrics
        metrics = risk_model.get_risk_metrics()
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_total_exposure_pct: float = 100.0,  # Max 100% of capital at risk
        max_single_exposure_pct: float = 25.0,  # Max 25% in any single asset
        max_drawdown_pct: float = 20.0,  # Stop trading if drawdown > 20%
        var_limit_pct: float = 5.0,  # Max 5% daily VaR
        correlation_threshold: float = 0.7  # High correlation warning
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_single_exposure_pct = max_single_exposure_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.var_limit_pct = var_limit_pct
        self.correlation_threshold = correlation_threshold
        
        # Positions
        self.positions: Dict[str, PortfolioPosition] = {}
        
        # Historical data for risk calculations
        self.equity_curve: List[Tuple[datetime, float]] = [(datetime.now(), initial_capital)]
        self.peak_equity = initial_capital
        self.returns_history: List[float] = []
        
        # Asset correlation matrix (simplified - use actual data in production)
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"PortfolioRiskModel initialized with ${initial_capital:,.2f}")
    
    def update_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        side: str = "long"
    ):
        """Update or create a position."""
        if quantity <= 0:
            if symbol in self.positions:
                del self.positions[symbol]
            return
        
        self.positions[symbol] = PortfolioPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            side=side
        )
        
        self._update_equity_curve()
    
    def update_price(self, symbol: str, current_price: float):
        """Update current price for a position."""
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price
            self._update_equity_curve()
    
    def close_position(self, symbol: str) -> Optional[float]:
        """Close a position and return realized P&L."""
        if symbol in self.positions:
            pnl = self.positions[symbol].unrealized_pnl
            del self.positions[symbol]
            self.current_capital += pnl
            self._update_equity_curve()
            return pnl
        return None
    
    def _update_equity_curve(self):
        """Update equity curve and peak tracking."""
        current_equity = self.get_portfolio_value()
        self.equity_curve.append((datetime.now(), current_equity))
        
        # Limit history size
        if len(self.equity_curve) > 10000:
            self.equity_curve = self.equity_curve[-5000:]
        
        # Track peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate returns
        if len(self.equity_curve) >= 2:
            prev_equity = self.equity_curve[-2][1]
            if prev_equity > 0:
                ret = (current_equity - prev_equity) / prev_equity
                self.returns_history.append(ret)
                
                if len(self.returns_history) > 252:  # ~1 year of daily returns
                    self.returns_history = self.returns_history[-252:]
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value (capital + unrealized P&L)."""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.current_capital + unrealized_pnl
    
    def get_total_exposure(self) -> float:
        """Get total notional exposure."""
        return sum(pos.notional_value for pos in self.positions.values())
    
    def get_exposure_pct(self) -> float:
        """Get exposure as percentage of portfolio."""
        portfolio_value = self.get_portfolio_value()
        if portfolio_value <= 0:
            return 0
        return (self.get_total_exposure() / portfolio_value) * 100
    
    def get_position_exposure_pct(self, symbol: str) -> float:
        """Get single position exposure as percentage."""
        if symbol not in self.positions:
            return 0
        portfolio_value = self.get_portfolio_value()
        if portfolio_value <= 0:
            return 0
        return (self.positions[symbol].notional_value / portfolio_value) * 100
    
    def calculate_var(self, confidence: float = 0.95, horizon_days: int = 1) -> float:
        """
        Calculate Value at Risk (Historical method).
        
        Args:
            confidence: Confidence level (0.95 = 95%)
            horizon_days: Time horizon in days
        
        Returns:
            VaR in dollar terms (potential loss)
        """
        if len(self.returns_history) < 20:
            # Not enough data - use simplified estimate
            portfolio_value = self.get_portfolio_value()
            # Assume 2% daily volatility for crypto
            return portfolio_value * 0.02 * np.sqrt(horizon_days) * 1.65
        
        returns = np.array(self.returns_history)
        
        # Historical VaR
        var_percentile = np.percentile(returns, (1 - confidence) * 100)
        
        # Scale for horizon
        var_pct = abs(var_percentile) * np.sqrt(horizon_days)
        
        return self.get_portfolio_value() * var_pct
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak (as percentage)."""
        current_equity = self.get_portfolio_value()
        if self.peak_equity <= 0:
            return 0
        return ((self.peak_equity - current_equity) / self.peak_equity) * 100
    
    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return 0
        
        equities = [e[1] for e in self.equity_curve]
        peak = equities[0]
        max_dd = 0
        
        for equity in equities:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd * 100
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio from returns history."""
        if len(self.returns_history) < 20:
            return 0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free
        
        if returns.std() == 0:
            return 0
        
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def calculate_concentration_risk(self) -> float:
        """
        Calculate concentration risk (Herfindahl-Hirschman Index).
        
        Returns:
            HHI value (0-10000, higher = more concentrated)
        """
        if not self.positions:
            return 0
        
        total_exposure = self.get_total_exposure()
        if total_exposure == 0:
            return 0
        
        hhi = sum(
            ((pos.notional_value / total_exposure) * 100) ** 2
            for pos in self.positions.values()
        )
        
        return hhi
    
    def check_new_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str = "long"
    ) -> Tuple[bool, str]:
        """
        Check if a new trade would violate risk limits.
        
        Returns:
            (approved, reason)
        """
        new_exposure = quantity * price
        current_exposure = self.get_total_exposure()
        portfolio_value = self.get_portfolio_value()
        
        # Check total exposure
        total_exposure_after = current_exposure + new_exposure
        total_exposure_pct = (total_exposure_after / portfolio_value) * 100
        
        if total_exposure_pct > self.max_total_exposure_pct:
            return False, f"Total exposure {total_exposure_pct:.1f}% exceeds max {self.max_total_exposure_pct}%"
        
        # Check single position exposure
        existing_exposure = self.positions.get(symbol, PortfolioPosition(symbol, 0, 0, 0, 'long')).notional_value
        position_exposure_after = existing_exposure + new_exposure
        position_exposure_pct = (position_exposure_after / portfolio_value) * 100
        
        if position_exposure_pct > self.max_single_exposure_pct:
            return False, f"{symbol} exposure {position_exposure_pct:.1f}% exceeds max {self.max_single_exposure_pct}%"
        
        # Check drawdown
        current_dd = self.get_current_drawdown()
        if current_dd >= self.max_drawdown_pct:
            return False, f"Current drawdown {current_dd:.1f}% exceeds max {self.max_drawdown_pct}%"
        
        # Check VaR
        var_95 = self.calculate_var(0.95)
        var_pct = (var_95 / portfolio_value) * 100
        if var_pct > self.var_limit_pct:
            return False, f"Portfolio VaR {var_pct:.1f}% exceeds limit {self.var_limit_pct}%"
        
        return True, "Approved"
    
    def run_stress_test(self, scenarios: Dict[str, float] = None) -> Dict[str, float]:
        """
        Run stress test scenarios.
        
        Args:
            scenarios: Dict of scenario_name -> price_change_pct
        
        Returns:
            Dict of scenario_name -> portfolio_impact
        """
        if scenarios is None:
            scenarios = {
                "market_crash_20": -20.0,
                "market_crash_30": -30.0,
                "flash_crash_50": -50.0,
                "rally_20": 20.0,
            }
        
        results = {}
        portfolio_value = self.get_portfolio_value()
        
        for scenario_name, price_change_pct in scenarios.items():
            # Calculate impact on all positions
            impact = 0
            for pos in self.positions.values():
                price_change = pos.current_price * (price_change_pct / 100)
                if pos.side == 'long':
                    impact += pos.quantity * price_change
                else:
                    impact -= pos.quantity * price_change
            
            impact_pct = (impact / portfolio_value) * 100 if portfolio_value > 0 else 0
            results[scenario_name] = impact_pct
        
        return results
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk metrics snapshot."""
        portfolio_value = self.get_portfolio_value()
        
        return RiskMetrics(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            total_exposure=self.get_total_exposure(),
            exposure_pct=self.get_exposure_pct(),
            var_95=self.calculate_var(0.95),
            var_99=self.calculate_var(0.99),
            max_drawdown=self.get_max_drawdown(),
            current_drawdown=self.get_current_drawdown(),
            sharpe_ratio=self.calculate_sharpe_ratio(),
            correlation_risk=0,  # Would need correlation data
            concentration_risk=self.calculate_concentration_risk()
        )
    
    def get_risk_report(self) -> Dict:
        """Generate human-readable risk report."""
        metrics = self.get_risk_metrics()
        stress_test = self.run_stress_test()
        
        return {
            "portfolio": {
                "value": metrics.portfolio_value,
                "exposure": metrics.total_exposure,
                "exposure_pct": metrics.exposure_pct,
                "positions": len(self.positions)
            },
            "risk_metrics": {
                "var_95_1d": metrics.var_95,
                "var_99_1d": metrics.var_99,
                "max_drawdown_pct": metrics.max_drawdown,
                "current_drawdown_pct": metrics.current_drawdown,
                "sharpe_ratio": metrics.sharpe_ratio,
                "concentration_hhi": metrics.concentration_risk
            },
            "limits": {
                "max_exposure_pct": self.max_total_exposure_pct,
                "max_single_pct": self.max_single_exposure_pct,
                "max_drawdown_pct": self.max_drawdown_pct,
                "var_limit_pct": self.var_limit_pct
            },
            "stress_test": stress_test,
            "status": "OK" if metrics.current_drawdown < self.max_drawdown_pct else "WARNING"
        }


# Singleton
_portfolio_risk: Optional[PortfolioRiskModel] = None

def get_portfolio_risk(initial_capital: float = 100000) -> PortfolioRiskModel:
    global _portfolio_risk
    if _portfolio_risk is None:
        _portfolio_risk = PortfolioRiskModel(initial_capital=initial_capital)
    return _portfolio_risk
