"""
Capital & Money Management - Features #126-140
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
import math

logger = logging.getLogger(__name__)

class CapitalAllocator:
    """Feature #126: Capital Allocator"""
    def __init__(self, total_capital: float):
        self.total = total_capital
        self.allocated: Dict[str, float] = {}
    
    def allocate(self, strategy: str, percentage: float) -> float:
        amount = self.total * (percentage / 100)
        self.allocated[strategy] = amount
        return amount
    
    def get_available(self) -> float:
        return self.total - sum(self.allocated.values())

class RiskBudgeter:
    """Feature #127: Risk Budgeter"""
    def __init__(self, daily_risk_pct: float = 2.0):
        self.daily_risk = daily_risk_pct
        self.used_risk = 0
    
    def can_trade(self, risk_pct: float) -> bool:
        return self.used_risk + risk_pct <= self.daily_risk
    
    def use_risk(self, risk_pct: float):
        self.used_risk += risk_pct
    
    def reset(self):
        self.used_risk = 0
    
    def remaining(self) -> float:
        return self.daily_risk - self.used_risk

class DrawdownProtector:
    """Feature #128: Drawdown Protection"""
    def __init__(self, max_drawdown: float = 0.1, cooldown_hours: int = 24):
        self.max_dd = max_drawdown
        self.cooldown = cooldown_hours
        self.peak = 0
        self.triggered_at: Optional[datetime] = None
    
    def update(self, equity: float) -> Dict:
        self.peak = max(self.peak, equity)
        dd = (self.peak - equity) / self.peak if self.peak > 0 else 0
        
        if dd >= self.max_dd and not self.triggered_at:
            self.triggered_at = datetime.now()
        
        is_blocked = False
        if self.triggered_at:
            hours = (datetime.now() - self.triggered_at).total_seconds() / 3600
            is_blocked = hours < self.cooldown
            if not is_blocked:
                self.triggered_at = None
        
        return {'drawdown': round(dd * 100, 2), 'blocked': is_blocked}

class ProfitLocker:
    """Feature #129: Profit Lock Mechanism"""
    def __init__(self, lock_threshold_pct: float = 5.0, lock_amount_pct: float = 50):
        self.threshold = lock_threshold_pct
        self.lock_pct = lock_amount_pct
        self.locked_profit = 0
        self.base_equity = 0
    
    def set_base(self, equity: float):
        self.base_equity = equity
    
    def check(self, equity: float) -> float:
        if self.base_equity == 0:
            return 0
        profit_pct = (equity - self.base_equity) / self.base_equity * 100
        if profit_pct >= self.threshold:
            lock_amount = (profit_pct - self.threshold) * self.lock_pct / 100
            self.locked_profit = self.base_equity * lock_amount / 100
        return self.locked_profit

class EquityTracker:
    """Feature #130: Equity Tracker"""
    def __init__(self):
        self.history: List[Dict] = []
    
    def record(self, equity: float):
        self.history.append({'equity': equity, 'time': datetime.now().isoformat()})
        self.history = self.history[-5000:]
    
    def get_returns(self, periods: int = 20) -> List[float]:
        if len(self.history) < 2:
            return []
        recent = self.history[-periods-1:]
        return [(recent[i]['equity'] - recent[i-1]['equity']) / recent[i-1]['equity'] 
                for i in range(1, len(recent))]

class CompoundGrowthCalculator:
    """Feature #131: Compound Growth Calculator"""
    def calculate(self, initial: float, target: float, trades_per_day: float, win_rate: float, avg_win: float, avg_loss: float) -> Dict:
        expected_per_trade = win_rate * avg_win - (1 - win_rate) * avg_loss
        if expected_per_trade <= 0:
            return {'days_to_target': -1, 'error': 'Negative expectancy'}
        
        growth_rate = expected_per_trade / 100  # As decimal
        days = math.log(target / initial) / (math.log(1 + growth_rate) * trades_per_day)
        
        return {'days_to_target': round(days, 1), 'growth_per_trade': round(expected_per_trade, 2)}

class WithdrawalManager:
    """Feature #132: Withdrawal Manager"""
    def __init__(self, max_withdrawal_pct: float = 50):
        self.max_pct = max_withdrawal_pct
        self.withdrawals: List[Dict] = []
    
    def can_withdraw(self, amount: float, equity: float) -> Dict:
        max_amount = equity * (self.max_pct / 100)
        return {'allowed': amount <= max_amount, 'max_allowed': max_amount}
    
    def record_withdrawal(self, amount: float):
        self.withdrawals.append({'amount': amount, 'time': datetime.now().isoformat()})

class ReserveManager:
    """Feature #133: Reserve Manager"""
    def __init__(self, reserve_pct: float = 20):
        self.reserve_pct = reserve_pct
    
    def calculate_tradable(self, total_equity: float) -> Dict:
        reserve = total_equity * (self.reserve_pct / 100)
        tradable = total_equity - reserve
        return {'reserve': reserve, 'tradable': tradable, 'total': total_equity}

class MarginCalculator:
    """Feature #134: Margin Calculator"""
    def calculate(self, position_value: float, leverage: float, maintenance_rate: float = 0.5) -> Dict:
        initial = position_value / leverage
        maintenance = initial * maintenance_rate
        return {'initial_margin': round(initial, 2), 'maintenance_margin': round(maintenance, 2)}
    
    def check_margin_call(self, equity: float, maintenance: float) -> bool:
        return equity < maintenance

class LiquidationCalculator:
    """Feature #135: Liquidation Calculator"""
    def calculate(self, entry_price: float, leverage: float, side: str) -> float:
        margin_pct = 1 / leverage
        if side == 'LONG':
            return entry_price * (1 - margin_pct + 0.01)  # 1% buffer
        else:
            return entry_price * (1 + margin_pct - 0.01)

class FundingCostCalculator:
    """Feature #136: Funding Cost Calculator"""
    def calculate(self, position_value: float, funding_rate: float, hours_held: float) -> float:
        funding_periods = hours_held / 8  # Every 8 hours
        return position_value * funding_rate * funding_periods

class PLCalculator:
    """Feature #137: P&L Calculator"""
    def calculate(self, entry: float, exit: float, size: float, side: str, fees: float = 0) -> Dict:
        if side == 'LONG':
            gross_pnl = (exit - entry) * size
        else:
            gross_pnl = (entry - exit) * size
        net_pnl = gross_pnl - fees
        pnl_pct = (net_pnl / (entry * size)) * 100 if entry * size != 0 else 0
        return {'gross_pnl': round(gross_pnl, 2), 'fees': fees, 'net_pnl': round(net_pnl, 2), 'pnl_pct': round(pnl_pct, 2)}

class ExpectancyCalculator:
    """Feature #138: Expectancy Calculator"""
    def calculate(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        return round(win_rate * avg_win - (1 - win_rate) * avg_loss, 2)

class ProfitFactorCalculator:
    """Feature #139: Profit Factor Calculator"""
    def calculate(self, trades: List[Dict]) -> float:
        gains = sum(t['pnl'] for t in trades if t.get('pnl', 0) > 0)
        losses = abs(sum(t['pnl'] for t in trades if t.get('pnl', 0) < 0))
        return round(gains / losses, 2) if losses > 0 else float('inf') if gains > 0 else 0

class RiskOfRuinCalculator:
    """Feature #140: Risk of Ruin Calculator"""
    def calculate(self, win_rate: float, risk_per_trade: float, num_trades: int = 1000) -> float:
        if win_rate >= 1 or win_rate <= 0:
            return 0 if win_rate >= 1 else 1
        loss_rate = 1 - win_rate
        ruin_prob = ((loss_rate / win_rate) ** (1 / risk_per_trade * 100)) if risk_per_trade > 0 else 0
        return round(min(1, ruin_prob), 4)

# Factories
def get_capital_allocator(cap: float): return CapitalAllocator(cap)
def get_risk_budgeter(): return RiskBudgeter()
def get_dd_protector(): return DrawdownProtector()
def get_profit_locker(): return ProfitLocker()
def get_equity_tracker(): return EquityTracker()
def get_growth_calculator(): return CompoundGrowthCalculator()
def get_withdrawal_manager(): return WithdrawalManager()
def get_reserve_manager(): return ReserveManager()
def get_margin_calculator(): return MarginCalculator()
def get_liquidation_calculator(): return LiquidationCalculator()
def get_funding_calculator(): return FundingCostCalculator()
def get_pnl_calculator(): return PLCalculator()
def get_expectancy_calculator(): return ExpectancyCalculator()
def get_profit_factor(): return ProfitFactorCalculator()
def get_risk_of_ruin(): return RiskOfRuinCalculator()
