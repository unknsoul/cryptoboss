"""
Execution Analytics & Trade Optimization - Features #332-338, #142-148
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class ExecutionQualityAnalyzer:
    """Feature #332: Execution Quality Analysis"""
    def __init__(self):
        self.executions: List[Dict] = []
    
    def record(self, expected_price: float, actual_price: float, size: float, side: str):
        slippage = (actual_price - expected_price) / expected_price * 100
        if side == 'SELL':
            slippage = -slippage
        self.executions.append({'slippage': slippage, 'size': size, 'time': datetime.now()})
        self.executions = self.executions[-1000:]
    
    def get_stats(self) -> Dict:
        if not self.executions:
            return {'avg_slippage': 0}
        slips = [e['slippage'] for e in self.executions]
        return {'avg_slippage': round(sum(slips)/len(slips), 4), 'worst': round(max(slips), 4), 'count': len(slips)}

class FillRateTracker:
    """Feature #333: Fill Rate Tracker"""
    def __init__(self):
        self.orders = {'filled': 0, 'partial': 0, 'unfilled': 0}
    
    def record(self, status: str):
        if status in self.orders:
            self.orders[status] += 1
    
    def get_rate(self) -> float:
        total = sum(self.orders.values())
        return round(self.orders['filled'] / total * 100, 1) if total > 0 else 0

class LatencyTracker:
    """Feature #334: Order Latency Tracker"""
    def __init__(self):
        self.latencies: List[float] = []
    
    def record(self, latency_ms: float):
        self.latencies.append(latency_ms)
        self.latencies = self.latencies[-1000:]
    
    def get_stats(self) -> Dict:
        if not self.latencies:
            return {'avg': 0}
        return {'avg': round(sum(self.latencies)/len(self.latencies), 1), 'max': max(self.latencies)}

class ExecutionCostCalculator:
    """Feature #335: Execution Cost Calculator"""
    def calculate(self, size: float, price: float, slippage: float, fee_rate: float) -> Dict:
        slippage_cost = size * price * slippage / 100
        fee_cost = size * price * fee_rate
        total = slippage_cost + fee_cost
        return {'slippage_cost': round(slippage_cost, 2), 'fee_cost': round(fee_cost, 2), 'total': round(total, 2)}

class VWAPCalculator:
    """Feature #336: VWAP Calculator"""
    def __init__(self):
        self.trades: List[Dict] = []
    
    def add(self, price: float, volume: float):
        self.trades.append({'price': price, 'volume': volume})
        self.trades = self.trades[-1000:]
    
    def calculate(self) -> float:
        if not self.trades:
            return 0
        pv_sum = sum(t['price'] * t['volume'] for t in self.trades)
        v_sum = sum(t['volume'] for t in self.trades)
        return round(pv_sum / v_sum, 2) if v_sum > 0 else 0

class TWAPExecutor:
    """Feature #337: TWAP Execution"""
    def generate_schedule(self, total_size: float, duration_minutes: int, intervals: int) -> List[Dict]:
        size_per = total_size / intervals
        interval_min = duration_minutes / intervals
        return [{'size': round(size_per, 6), 'time_offset_min': i * interval_min} for i in range(intervals)]

class ExecutionBenchmark:
    """Feature #338: Execution Benchmark"""
    def __init__(self):
        self.benchmarks: Dict[str, List[float]] = defaultdict(list)
    
    def record(self, benchmark: str, value: float):
        self.benchmarks[benchmark].append(value)
    
    def compare(self, benchmark: str) -> Dict:
        values = self.benchmarks.get(benchmark, [])
        if not values:
            return {'avg': 0}
        return {'avg': round(sum(values)/len(values), 4), 'best': min(values), 'worst': max(values)}

class EntryOptimizer:
    """Feature #142: Entry Point Optimizer"""
    def optimize(self, price: float, atr: float, trend: str) -> Dict:
        if trend == 'UP':
            entry = price - atr * 0.3  # Wait for pullback
        elif trend == 'DOWN':
            entry = price + atr * 0.3
        else:
            entry = price
        return {'optimal_entry': round(entry, 2), 'current': price, 'adjustment': round(entry - price, 2)}

class ExitOptimizer:
    """Feature #143: Exit Point Optimizer"""
    def optimize(self, entry: float, atr: float, confidence: float) -> Dict:
        sl_mult = 1.5 if confidence > 0.7 else 2.0
        tp_mult = 3.0 if confidence > 0.7 else 2.0
        return {'stop_loss': round(entry - atr * sl_mult, 2), 'take_profit': round(entry + atr * tp_mult, 2)}

class TimingOptimizer:
    """Feature #144: Timing Optimizer"""
    def __init__(self):
        self.hourly_returns: Dict[int, List[float]] = defaultdict(list)
    
    def record(self, hour: int, return_pct: float):
        self.hourly_returns[hour].append(return_pct)
    
    def get_best_hours(self) -> List[int]:
        avgs = {h: sum(r)/len(r) for h, r in self.hourly_returns.items() if r}
        return sorted(avgs.keys(), key=lambda h: avgs[h], reverse=True)[:5]

class SizeOptimizer:
    """Feature #145: Position Size Optimizer"""
    def optimize(self, equity: float, volatility: float, confidence: float) -> float:
        base_risk = 0.02  # 2% base risk
        vol_adj = 1 - min(0.5, volatility * 10)
        conf_adj = 0.5 + confidence * 0.5
        risk = base_risk * vol_adj * conf_adj
        return round(equity * risk, 2)

class RiskRewardOptimizer:
    """Feature #146: Risk/Reward Optimizer"""
    def calculate(self, win_rate: float, target_rr: float = 2.0) -> Dict:
        breakeven_rr = (1 - win_rate) / win_rate if win_rate > 0 else 999
        optimal_rr = breakeven_rr * 1.2
        return {'breakeven_rr': round(breakeven_rr, 2), 'optimal_rr': round(optimal_rr, 2), 'target_rr': target_rr}

class TradeScoreOptimizer:
    """Feature #147: Trade Setup Scorer"""
    def score(self, trend_aligned: bool, momentum_ok: bool, volume_ok: bool, regime_ok: bool) -> int:
        score = 0
        if trend_aligned: score += 25
        if momentum_ok: score += 25
        if volume_ok: score += 25
        if regime_ok: score += 25
        return score

class AdaptiveParameterTuner:
    """Feature #148: Adaptive Parameter Tuner"""
    def __init__(self):
        self.params: Dict[str, float] = {'sl_mult': 1.5, 'tp_mult': 2.5, 'risk_pct': 2.0}
    
    def adjust(self, win_rate: float, avg_rr: float):
        if win_rate < 0.4:
            self.params['sl_mult'] *= 1.1
        elif win_rate > 0.6:
            self.params['sl_mult'] *= 0.95
        if avg_rr < 1.5:
            self.params['tp_mult'] *= 1.1
        return self.params

# Factories
def get_execution_analyzer(): return ExecutionQualityAnalyzer()
def get_fill_tracker(): return FillRateTracker()
def get_latency_tracker(): return LatencyTracker()
def get_cost_calculator(): return ExecutionCostCalculator()
def get_vwap(): return VWAPCalculator()
def get_twap(): return TWAPExecutor()
def get_benchmark(): return ExecutionBenchmark()
def get_entry_optimizer(): return EntryOptimizer()
def get_exit_optimizer(): return ExitOptimizer()
def get_timing_optimizer(): return TimingOptimizer()
def get_size_optimizer(): return SizeOptimizer()
def get_rr_optimizer(): return RiskRewardOptimizer()
def get_trade_scorer(): return TradeScoreOptimizer()
def get_param_tuner(): return AdaptiveParameterTuner()
