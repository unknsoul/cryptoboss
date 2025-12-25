"""
Backtesting & Performance - Features #297-310
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
import math

logger = logging.getLogger(__name__)

class BacktestEngine:
    """Feature #297: Backtest Engine"""
    def __init__(self, initial_capital: float = 10000):
        self.initial = initial_capital
        self.equity = initial_capital
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]
    
    def execute_trade(self, entry: float, exit: float, size: float, side: str):
        pnl = (exit - entry) * size if side == 'LONG' else (entry - exit) * size
        self.equity += pnl
        self.trades.append({'entry': entry, 'exit': exit, 'size': size, 'side': side, 'pnl': pnl})
        self.equity_curve.append(self.equity)
    
    def get_results(self) -> Dict:
        if not self.trades:
            return {}
        wins = [t for t in self.trades if t['pnl'] > 0]
        return {'total_trades': len(self.trades), 'win_rate': len(wins)/len(self.trades), 
                'final_equity': self.equity, 'total_return': (self.equity - self.initial) / self.initial * 100}

class PerformanceMetrics:
    """Feature #298: Performance Metrics"""
    def calculate(self, equity_curve: List[float]) -> Dict:
        if len(equity_curve) < 2:
            return {}
        returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] for i in range(1, len(equity_curve))]
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe = (avg_return / std_return * (252 ** 0.5)) if std_return > 0 else 0
        
        max_dd = 0
        peak = equity_curve[0]
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
        
        return {'sharpe': round(sharpe, 2), 'max_drawdown': round(max_dd * 100, 2), 'volatility': round(std_return * 100, 2)}

class SortinoCalculator:
    """Feature #299: Sortino Ratio"""
    def calculate(self, returns: List[float], target: float = 0) -> float:
        if not returns:
            return 0
        avg = sum(returns) / len(returns)
        downside = [r for r in returns if r < target]
        if not downside:
            return float('inf') if avg > 0 else 0
        downside_dev = (sum((r - target) ** 2 for r in downside) / len(downside)) ** 0.5
        return round((avg - target) / downside_dev * (252 ** 0.5), 2) if downside_dev > 0 else 0

class CalmarCalculator:
    """Feature #300: Calmar Ratio"""
    def calculate(self, annual_return: float, max_drawdown: float) -> float:
        return round(annual_return / max_drawdown, 2) if max_drawdown > 0 else 0

class MAECalculator:
    """Feature #301: Maximum Adverse Excursion"""
    def calculate(self, trades: List[Dict]) -> Dict:
        if not trades:
            return {}
        maes = [t.get('max_adverse', 0) for t in trades]
        return {'avg_mae': round(sum(maes) / len(maes), 2), 'max_mae': max(maes)}

class MFECalculator:
    """Feature #302: Maximum Favorable Excursion"""
    def calculate(self, trades: List[Dict]) -> Dict:
        if not trades:
            return {}
        mfes = [t.get('max_favorable', 0) for t in trades]
        return {'avg_mfe': round(sum(mfes) / len(mfes), 2), 'max_mfe': max(mfes)}

class ExcursionEfficiency:
    """Feature #303: Excursion Efficiency"""
    def calculate(self, trade: Dict) -> float:
        mfe = trade.get('max_favorable', 0)
        actual = trade.get('pnl', 0)
        return round(actual / mfe, 2) if mfe > 0 else 0

class TradeDistribution:
    """Feature #304: Trade Distribution Analyzer"""
    def analyze(self, trades: List[Dict]) -> Dict:
        if not trades:
            return {}
        pnls = [t.get('pnl', 0) for t in trades]
        return {'mean': round(sum(pnls) / len(pnls), 2), 'min': min(pnls), 'max': max(pnls),
                'positive_pct': round(len([p for p in pnls if p > 0]) / len(pnls) * 100, 1)}

class ConsecutiveWinLoss:
    """Feature #306: Consecutive Win/Loss Analyzer"""
    def analyze(self, trades: List[Dict]) -> Dict:
        if not trades:
            return {}
        max_wins = max_losses = current_wins = current_losses = 0
        for t in trades:
            if t.get('pnl', 0) > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        return {'max_consecutive_wins': max_wins, 'max_consecutive_losses': max_losses}

class TimeAnalyzer:
    """Feature #307: Time-Based Analysis"""
    def analyze(self, trades: List[Dict]) -> Dict:
        if not trades:
            return {}
        by_hour = {}
        for t in trades:
            hour = datetime.fromisoformat(t.get('time', datetime.now().isoformat())).hour
            if hour not in by_hour:
                by_hour[hour] = []
            by_hour[hour].append(t.get('pnl', 0))
        best_hour = max(by_hour.keys(), key=lambda h: sum(by_hour[h])) if by_hour else 0
        return {'best_hour': best_hour, 'hours_analyzed': len(by_hour)}

class HoldingTimeAnalyzer:
    """Feature #308: Holding Time Analysis"""
    def analyze(self, trades: List[Dict]) -> Dict:
        durations = [t.get('duration_minutes', 0) for t in trades if 'duration_minutes' in t]
        if not durations:
            return {}
        return {'avg_duration': round(sum(durations) / len(durations), 1), 'min': min(durations), 'max': max(durations)}

class ReturnDistribution:
    """Feature #309: Return Distribution"""
    def analyze(self, returns: List[float]) -> Dict:
        if len(returns) < 10:
            return {}
        sorted_r = sorted(returns)
        n = len(returns)
        return {'p5': round(sorted_r[int(n * 0.05)], 4), 'p25': round(sorted_r[int(n * 0.25)], 4),
                'median': round(sorted_r[n // 2], 4), 'p75': round(sorted_r[int(n * 0.75)], 4),
                'p95': round(sorted_r[int(n * 0.95)], 4)}

class RiskAdjustedReturns:
    """Feature #310: Risk-Adjusted Returns"""
    def calculate(self, total_return: float, max_drawdown: float, volatility: float) -> Dict:
        return_per_dd = total_return / max_drawdown if max_drawdown > 0 else 0
        return_per_vol = total_return / volatility if volatility > 0 else 0
        return {'return_per_drawdown': round(return_per_dd, 2), 'return_per_volatility': round(return_per_vol, 2)}

# Factories
def get_backtest_engine(capital: float = 10000): return BacktestEngine(capital)
def get_performance_metrics(): return PerformanceMetrics()
def get_sortino(): return SortinoCalculator()
def get_calmar(): return CalmarCalculator()
def get_mae(): return MAECalculator()
def get_mfe(): return MFECalculator()
def get_excursion_efficiency(): return ExcursionEfficiency()
def get_trade_distribution(): return TradeDistribution()
def get_consecutive_analyzer(): return ConsecutiveWinLoss()
def get_time_analyzer(): return TimeAnalyzer()
def get_holding_analyzer(): return HoldingTimeAnalyzer()
def get_return_distribution(): return ReturnDistribution()
def get_risk_adjusted(): return RiskAdjustedReturns()
