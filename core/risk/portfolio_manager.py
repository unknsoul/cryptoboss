"""
Portfolio Correlation Heatmap
Tracks correlation between positions to identify concentration risk

Important for:
- Avoiding correlated positions (amplifies risk)
- Diversification analysis
- Risk concentration detection
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class PortfolioCorrelationTracker:
    """
    Track correlations between trading instruments
    
    Features:
    - Real-time correlation calculation
    - Concentration risk alerts
    - Correlation heatmap data
    - Historical correlation tracking
    """
    
    def __init__(self, window: int = 30, alert_threshold: float = 0.7):
        """
        Initialize correlation tracker
        
        Args:
            window: Rolling window for correlation (days)
            alert_threshold: Alert when correlation > this value
        """
        self.window = window
        self.alert_threshold = alert_threshold
        self.returns = defaultdict(list)  # symbol -> list of returns
        self.positions = {}  # Current positions
        
    def update_position(self, symbol: str, size: float, entry_price: float):
        """Record an open position"""
        self.positions[symbol] = {
            'size': size,
            'entry_price': entry_price,
            'timestamp': datetime.now()
        }
    
    def close_position(self, symbol: str):
        """Remove closed position"""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def record_return(self, symbol: str, daily_return: float):
        """Record daily return for a symbol"""
        self.returns[symbol].append({
            'return': daily_return,
            'timestamp': datetime.now()
        })
        
        # Keep only recent data
        if len(self.returns[symbol]) > self.window * 2:
            self.returns[symbol] = self.returns[symbol][-self.window * 2:]
    
    def calculate_correlation(self, symbol1: str, symbol2: str,
                             window: Optional[int] = None) -> Optional[float]:
        """
        Calculate correlation between two symbols
        
        Args:
            symbol1, symbol2: Trading symbols
            window: Lookback period (uses self.window if None)
            
        Returns:
            Correlation coefficient (-1 to 1) or None
        """
        w = window or self.window
        
        if symbol1 not in self.returns or symbol2 not in self.returns:
            return None
        
        returns1 = [r['return'] for r in self.returns[symbol1][-w:]]
        returns2 = [r['return'] for r in self.returns[symbol2][-w:]]
        
        if len(returns1) < 10 or len(returns2) < 10:
            return None
        
        # Align lengths
        min_len = min(len(returns1), len(returns2))
        returns1 = returns1[-min_len:]
        returns2 = returns2[-min_len:]
        
        # Calculate correlation
        corr = np.corrcoef(returns1, returns2)[0, 1]
        
        return corr
    
    def get_correlation_matrix(self) -> Dict:
        """
        Get correlation matrix for all tracked symbols
        
        Returns:
            Dict with correlation data
        """
        symbols = list(self.returns.keys())
        
        if len(symbols) < 2:
            return {'status': 'insufficient_symbols', 'symbols': symbols}
        
        matrix = {}
        high_correlations = []
        
        for i, sym1 in enumerate(symbols):
            matrix[sym1] = {}
            for j, sym2 in enumerate(symbols):
                if i == j:
                    matrix[sym1][sym2] = 1.0
                elif j < i:
                    # Use previously calculated value
                    matrix[sym1][sym2] = matrix[sym2][sym1]
                else:
                    corr = self.calculate_correlation(sym1, sym2)
                    matrix[sym1][sym2] = round(corr, 3) if corr is not None else None
                    
                    # Check for high correlation
                    if corr and abs(corr) > self.alert_threshold and sym1 != sym2:
                        high_correlations.append({
                            'symbol1': sym1,
                            'symbol2': sym2,
                            'correlation': round(corr, 3)
                        })
        
        return {
            'matrix': matrix,
            'symbols': symbols,
            'high_correlations': high_correlations,
            'threshold': self.alert_threshold,
            'status': 'ok'
        }
    
    def check_position_correlation(self) -> Dict:
        """
        Check correlation risk for current positions
        
        Returns:
            Risk analysis of current portfolio
        """
        if len(self.positions) < 2:
            return {'status': 'single_position', 'risk': 'low'}
        
        position_symbols = list(self.positions.keys())
        correlations = []
        
        for i, sym1 in enumerate(position_symbols):
            for sym2 in position_symbols[i+1:]:
                corr = self.calculate_correlation(sym1, sym2)
                if corr is not None:
                    correlations.append({
                        'symbol1': sym1,
                        'symbol2': sym2,
                        'correlation': corr,
                        'risk_level': self._assess_correlation_risk(corr)
                    })
        
        if not correlations:
            return {'status': 'insufficient_data'}
        
        # Calculate average correlation
        avg_corr = np.mean([abs(c['correlation']) for c in correlations])
        
        # Overall risk assessment
        if avg_corr > 0.7:
            risk = 'high'
            msg = "⚠️ High correlation between positions - concentrated risk"
        elif avg_corr > 0.5:
            risk = 'medium'
            msg = "Moderate correlation - some concentration"
        else:
            risk = 'low'
            msg = "✓ Well diversified positions"
        
        return {
            'correlations': correlations,
            'avg_correlation': round(avg_corr, 3),
            'risk_level': risk,
            'message': msg,
            'num_positions': len(self.positions),
            'status': 'ok'
        }
    
    def _assess_correlation_risk(self, corr: float) -> str:
        """Assess risk level based on correlation"""
        abs_corr = abs(corr)
        if abs_corr > 0.8:
            return 'very_high'
        elif abs_corr > 0.6:
            return 'high'
        elif abs_corr > 0.4:
            return 'moderate'
        else:
            return 'low'


class ExposureManager:
    """
    Time-based exposure limits
    
    Limits total capital allocated per:
    - Hour
    - Day
    - Week
    - Per position
    """
    
    def __init__(self, 
                 hourly_limit_pct: float = 0.1,
                 daily_limit_pct: float = 0.5,
                 weekly_limit_pct: float = 1.0,
                 position_limit_pct: float = 0.2):
        """
        Initialize exposure manager
        
        Args:
            hourly_limit_pct: Max % of capital per hour (e.g., 0.1 = 10%)
            daily_limit_pct: Max % per day
            weekly_limit_pct: Max % per week
            position_limit_pct: Max % per single position
        """
        self.limits = {
            'hourly': hourly_limit_pct,
            'daily': daily_limit_pct,
            'weekly': weekly_limit_pct,
            'position': position_limit_pct
        }
        
        self.exposure_history = []
        
    def record_exposure(self, amount: float, equity: float):
        """Record a new exposure (trade)"""
        self.exposure_history.append({
            'amount': amount,
            'equity': equity,
            'timestamp': datetime.now()
        })
    
    def get_exposure_window(self, hours: int) -> float:
        """Get total exposure in last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        total = sum(
            exp['amount'] 
            for exp in self.exposure_history 
            if exp['timestamp'] > cutoff
        )
        
        return total
    
    def can_take_position(self, size: float, price: float, equity: float) -> Tuple[bool, str]:
        """
        Check if position is allowed under exposure limits
        
        Returns:
            (allowed, reason)
        """
        position_value = size * price
        
        # Check position limit
        position_pct = position_value / equity
        if position_pct > self.limits['position']:
            return False, f"Position limit exceeded: {position_pct:.1%} > {self.limits['position']:.1%}"
        
        # Check hourly limit
        hourly_exposure = self.get_exposure_window(1)
        hourly_pct = (hourly_exposure + position_value) / equity
        if hourly_pct > self.limits['hourly']:
            return False, f"Hourly limit exceeded: {hourly_pct:.1%} > {self.limits['hourly']:.1%}"
        
        # Check daily limit
        daily_exposure = self.get_exposure_window(24)
        daily_pct = (daily_exposure + position_value) / equity
        if daily_pct > self.limits['daily']:
            return False, f"Daily limit exceeded: {daily_pct:.1%} > {self.limits['daily']:.1%}"
        
        # Check weekly limit
        weekly_exposure = self.get_exposure_window(24 * 7)
        weekly_pct = (weekly_exposure + position_value) / equity
        if weekly_pct > self.limits['weekly']:
            return False, f"Weekly limit exceeded: {weekly_pct:.1%} > {self.limits['weekly']:.1%}"
        
        return True, "OK"
    
    def get_statistics(self) -> Dict:
        """Get exposure statistics"""
        equity = self.exposure_history[-1]['equity'] if self.exposure_history else 10000
        
        return {
            'hourly_exposure': self.get_exposure_window(1),
            'hourly_pct': round(self.get_exposure_window(1) / equity * 100, 2),
            'daily_exposure': self.get_exposure_window(24),
            'daily_pct': round(self.get_exposure_window(24) / equity * 100, 2),
            'weekly_exposure': self.get_exposure_window(24 * 7),
            'weekly_pct': round(self.get_exposure_window(24 * 7) / equity * 100, 2),
            'limits': {k: f"{v*100:.0f}%" for k, v in self.limits.items()},
            'total_exposures': len(self.exposure_history)
        }


# Singletons
_correlation_tracker: Optional[PortfolioCorrelationTracker] = None
_exposure_manager: Optional[ExposureManager] = None


def get_correlation_tracker() -> PortfolioCorrelationTracker:
    global _correlation_tracker
    if _correlation_tracker is None:
        _correlation_tracker = PortfolioCorrelationTracker()
    return _correlation_tracker


def get_exposure_manager() -> ExposureManager:
    global _exposure_manager
    if _exposure_manager is None:
        _exposure_manager = ExposureManager()
    return _exposure_manager


if __name__ == '__main__':
    print("=" * 70)
    print("PORTFOLIO RISK MANAGEMENT - TEST")
    print("=" * 70)
    
    # Test correlation tracker
    print("\n📊 Testing Correlation Tracker...")
    tracker = PortfolioCorrelationTracker()
    
    # Simulate correlated returns
    import random
    random.seed(42)
    
    for i in range(50):
        base_return = random.gauss(0, 0.02)
        tracker.record_return('BTC', base_return)
        tracker.record_return('ETH', base_return + random.gauss(0, 0.01))  # Highly correlated
        tracker.record_return('SOL', random.gauss(0, 0.03))  # Uncorrelated
    
    tracker.update_position('BTC', 0.5, 50000)
    tracker.update_position('ETH', 2.0, 3000)
    
    corr_matrix = tracker.get_correlation_matrix()
    print(f"Symbols tracked: {corr_matrix['symbols']}")
    print(f"High correlations: {len(corr_matrix['high_correlations'])}")
    
    position_risk = tracker.check_position_correlation()
    print(f"Portfolio risk: {position_risk.get('risk_level', 'N/A')}")
    print(f"Message: {position_risk.get('message', 'N/A')}")
    
    # Test exposure manager
    print("\n💰 Testing Exposure Manager...")
    mgr = ExposureManager(hourly_limit_pct=0.1, daily_limit_pct=0.5)
    
    equity = 10000
    
    # Simulate trades
    for i in range(5):
        can_trade, reason = mgr.can_take_position(0.05, 50000, equity)
        print(f"Trade {i+1}: {can_trade} - {reason}")
        if can_trade:
            mgr.record_exposure(0.05 * 50000, equity)
    
    stats = mgr.get_statistics()
    print(f"\nExposure Stats:")
    for key, value in stats.items():
        if key != 'limits':
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ Portfolio risk management working!")
    print("=" * 70)
