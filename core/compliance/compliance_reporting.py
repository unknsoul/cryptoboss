"""
Compliance & Reporting - Features #220-228, #183-188
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class TradingLimitEnforcer:
    """Feature #220: Trading Limit Enforcer"""
    def __init__(self, daily_limit: float = 100000, max_position: float = 10000):
        self.daily_limit = daily_limit
        self.max_position = max_position
        self.daily_volume = 0
        self.last_reset = datetime.now().date()
    
    def check(self, size: float, price: float) -> Dict:
        if datetime.now().date() != self.last_reset:
            self.daily_volume = 0
            self.last_reset = datetime.now().date()
        
        notional = size * price
        can_trade = self.daily_volume + notional <= self.daily_limit and notional <= self.max_position
        return {'allowed': can_trade, 'daily_used': self.daily_volume, 'remaining': self.daily_limit - self.daily_volume}
    
    def record(self, size: float, price: float):
        self.daily_volume += size * price

class RiskLimitMonitor:
    """Feature #221: Risk Limit Monitor"""
    def __init__(self, max_drawdown: float = 0.1, max_daily_loss: float = 0.05):
        self.max_dd = max_drawdown
        self.max_daily = max_daily_loss
        self.peak_equity = 0
        self.daily_start = 0
    
    def check(self, equity: float) -> Dict:
        self.peak_equity = max(self.peak_equity, equity)
        if self.daily_start == 0:
            self.daily_start = equity
        
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        daily_loss = (self.daily_start - equity) / self.daily_start if self.daily_start > 0 else 0
        
        return {'drawdown': round(dd, 4), 'daily_loss': round(daily_loss, 4), 
                'stop_trading': dd >= self.max_dd or daily_loss >= self.max_daily}

class AuditLogger:
    """Feature #222: Compliance Audit Logger"""
    def __init__(self):
        self.logs: List[Dict] = []
    
    def log(self, action: str, details: Dict, user: str = 'system'):
        self.logs.append({'action': action, 'details': details, 'user': user, 'timestamp': datetime.now().isoformat()})
        self.logs = self.logs[-10000:]
    
    def get_logs(self, action: Optional[str] = None) -> List[Dict]:
        if action:
            return [l for l in self.logs if l['action'] == action]
        return self.logs[-100:]

class PositionLimitChecker:
    """Feature #223: Position Limit Checker"""
    def __init__(self, max_positions: int = 5, max_per_asset: float = 0.2):
        self.max_positions = max_positions
        self.max_per_asset = max_per_asset
    
    def check(self, current_positions: int, asset_allocation: float) -> Dict:
        return {'pos_ok': current_positions < self.max_positions, 
                'alloc_ok': asset_allocation <= self.max_per_asset,
                'allowed': current_positions < self.max_positions and asset_allocation <= self.max_per_asset}

class ExposureLimitMonitor:
    """Feature #224: Exposure Limit Monitor"""
    def __init__(self, max_gross: float = 2.0, max_net: float = 1.0):
        self.max_gross = max_gross
        self.max_net = max_net
    
    def check(self, long_exp: float, short_exp: float, equity: float) -> Dict:
        gross = (long_exp + short_exp) / equity if equity > 0 else 0
        net = abs(long_exp - short_exp) / equity if equity > 0 else 0
        return {'gross': round(gross, 2), 'net': round(net, 2), 
                'within_limits': gross <= self.max_gross and net <= self.max_net}

class RegulatoryReporter:
    """Feature #225: Regulatory Reporter"""
    def __init__(self):
        self.trades: List[Dict] = []
    
    def record_trade(self, trade: Dict):
        self.trades.append({**trade, 'reported_at': datetime.now().isoformat()})
    
    def generate_report(self, start: datetime, end: datetime) -> Dict:
        relevant = [t for t in self.trades 
                   if start.isoformat() <= t.get('timestamp', '') <= end.isoformat()]
        return {'period': f"{start.date()} to {end.date()}", 'trade_count': len(relevant), 
                'total_volume': sum(t.get('notional', 0) for t in relevant)}

class ComplianceAlertSystem:
    """Feature #226: Compliance Alert System"""
    def __init__(self):
        self.alerts: List[Dict] = []
        self.rules: Dict[str, Dict] = {}
    
    def add_rule(self, name: str, threshold: float, metric: str):
        self.rules[name] = {'threshold': threshold, 'metric': metric}
    
    def check_rules(self, metrics: Dict) -> List[Dict]:
        triggered = []
        for name, rule in self.rules.items():
            value = metrics.get(rule['metric'], 0)
            if value > rule['threshold']:
                alert = {'rule': name, 'value': value, 'threshold': rule['threshold'], 'time': datetime.now().isoformat()}
                triggered.append(alert)
                self.alerts.append(alert)
        return triggered

class TradeReconciler:
    """Feature #227: Trade Reconciliation"""
    def __init__(self):
        self.internal: List[Dict] = []
        self.external: List[Dict] = []
    
    def add_internal(self, trade: Dict):
        self.internal.append(trade)
    
    def add_external(self, trade: Dict):
        self.external.append(trade)
    
    def reconcile(self) -> Dict:
        matched = len([i for i in self.internal if any(e['id'] == i['id'] for e in self.external)])
        return {'internal': len(self.internal), 'external': len(self.external), 
                'matched': matched, 'unmatched': len(self.internal) - matched}

class WashTradeDetector:
    """Feature #228: Wash Trade Detector"""
    def __init__(self, time_window: int = 60):
        self.window = time_window
        self.recent_trades: List[Dict] = []
    
    def check(self, trade: Dict) -> bool:
        now = datetime.now()
        self.recent_trades = [t for t in self.recent_trades 
                             if (now - datetime.fromisoformat(t['time'])).seconds < self.window]
        
        is_wash = any(t['symbol'] == trade['symbol'] and t['side'] != trade['side'] 
                      and abs(t['price'] - trade['price']) / trade['price'] < 0.001
                      for t in self.recent_trades)
        
        self.recent_trades.append({**trade, 'time': now.isoformat()})
        return is_wash

class DailyReporter:
    """Feature #183: Daily Report Generator"""
    def __init__(self):
        self.daily_data: Dict[str, Dict] = {}
    
    def record(self, date: str, metrics: Dict):
        self.daily_data[date] = metrics
    
    def generate(self, date: str) -> Dict:
        return self.daily_data.get(date, {'error': 'No data'})

class WeeklyReporter:
    """Feature #184: Weekly Report Generator"""
    def generate(self, daily_reports: List[Dict]) -> Dict:
        if not daily_reports:
            return {}
        return {'days': len(daily_reports), 
                'total_pnl': sum(d.get('pnl', 0) for d in daily_reports),
                'avg_pnl': sum(d.get('pnl', 0) for d in daily_reports) / len(daily_reports)}

class MonthlyReporter:
    """Feature #185: Monthly Report Generator"""
    def generate(self, weekly_reports: List[Dict]) -> Dict:
        if not weekly_reports:
            return {}
        return {'weeks': len(weekly_reports),
                'total_pnl': sum(w.get('total_pnl', 0) for w in weekly_reports)}

class PerformanceReporter:
    """Feature #186: Performance Report Generator"""
    def generate(self, trades: List[Dict], equity_curve: List[float]) -> Dict:
        if not trades or not equity_curve:
            return {}
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        return {'total_trades': len(trades), 'win_rate': round(len(wins)/len(trades)*100, 1),
                'total_return': round((equity_curve[-1] - equity_curve[0])/equity_curve[0]*100, 2)}

class RiskReporter:
    """Feature #187: Risk Report Generator"""
    def generate(self, equity_curve: List[float], var_95: float, max_dd: float) -> Dict:
        return {'var_95': round(var_95, 2), 'max_drawdown': round(max_dd, 2),
                'current_equity': equity_curve[-1] if equity_curve else 0}

class CustomReportBuilder:
    """Feature #188: Custom Report Builder"""
    def __init__(self):
        self.sections: List[Dict] = []
    
    def add_section(self, title: str, content: Dict):
        self.sections.append({'title': title, 'content': content})
    
    def build(self) -> Dict:
        return {'generated_at': datetime.now().isoformat(), 'sections': self.sections}

# Factories
def get_limit_enforcer(): return TradingLimitEnforcer()
def get_risk_monitor(): return RiskLimitMonitor()
def get_audit_logger(): return AuditLogger()
def get_position_checker(): return PositionLimitChecker()
def get_exposure_monitor(): return ExposureLimitMonitor()
def get_regulatory_reporter(): return RegulatoryReporter()
def get_compliance_alerts(): return ComplianceAlertSystem()
def get_reconciler(): return TradeReconciler()
def get_wash_detector(): return WashTradeDetector()
def get_daily_reporter(): return DailyReporter()
def get_weekly_reporter(): return WeeklyReporter()
def get_monthly_reporter(): return MonthlyReporter()
def get_performance_reporter(): return PerformanceReporter()
def get_risk_reporter(): return RiskReporter()
def get_custom_builder(): return CustomReportBuilder()
