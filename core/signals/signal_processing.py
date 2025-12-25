"""
Signal Processing & Filters - Features #75-80, #160-168
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class SignalSmoother:
    """Feature #75: EMA Signal Smoothing"""
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.ema = None
    
    def smooth(self, value: float) -> float:
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        return round(self.ema, 4)

class NoiseFilter:
    """Feature #76: Noise Reduction"""
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.last_valid = None
    
    def filter(self, value: float) -> float:
        if self.last_valid is None:
            self.last_valid = value
            return value
        change = abs(value - self.last_valid) / self.last_valid if self.last_valid != 0 else 0
        if change >= self.threshold:
            self.last_valid = value
        return self.last_valid

class SignalConfirmation:
    """Feature #77: Multi-Confirmation"""
    def __init__(self, required: int = 3):
        self.required = required
        self.signals: List[str] = []
    
    def add(self, direction: str) -> bool:
        self.signals.append(direction)
        self.signals = self.signals[-10:]
        recent = self.signals[-self.required:]
        return len(recent) >= self.required and all(s == direction for s in recent)

class SignalStrength:
    """Feature #78: Signal Strength Calculator"""
    def calculate(self, rsi: float, macd: float, trend: float) -> Dict:
        score = 0
        if rsi < 30: score += 1
        elif rsi > 70: score -= 1
        if macd > 0: score += 1
        else: score -= 1
        if trend > 0: score += 1
        else: score -= 1
        return {'direction': 'LONG' if score > 0 else 'SHORT', 'strength': abs(score) / 3}

class SignalCooldown:
    """Feature #79: Signal Cooldown"""
    def __init__(self, seconds: int = 300):
        self.cooldown = seconds
        self.last: Dict[str, datetime] = {}
    
    def can_signal(self, sig_type: str) -> bool:
        if sig_type not in self.last:
            return True
        return (datetime.now() - self.last[sig_type]).total_seconds() >= self.cooldown
    
    def record(self, sig_type: str):
        self.last[sig_type] = datetime.now()

class SignalQueue:
    """Feature #80: Priority Queue"""
    def __init__(self, max_size: int = 10):
        self.queue: List[Dict] = []
        self.max_size = max_size
    
    def add(self, signal: Dict, priority: int = 5):
        self.queue.append({**signal, 'priority': priority})
        self.queue.sort(key=lambda x: x['priority'])
        self.queue = self.queue[:self.max_size]
    
    def pop(self) -> Optional[Dict]:
        return self.queue.pop(0) if self.queue else None

class TimeFilter:
    """Feature #160: Time-Based Filter"""
    BEST_HOURS = [9, 10, 11, 14, 15, 16]
    def check(self) -> bool:
        now = datetime.utcnow()
        return now.hour in self.BEST_HOURS and now.weekday() < 5

class VolatilityFilter:
    """Feature #161: Volatility Filter"""
    def __init__(self, min_v: float = 0.005, max_v: float = 0.05):
        self.min_v, self.max_v = min_v, max_v
    def check(self, vol: float) -> bool:
        return self.min_v <= vol <= self.max_v

class SpreadFilter:
    """Feature #162: Spread Filter"""
    def __init__(self, max_spread: float = 0.1):
        self.max_spread = max_spread
    def check(self, spread: float) -> bool:
        return spread <= self.max_spread

class VolumeFilter:
    """Feature #163: Volume Filter"""
    def __init__(self, min_ratio: float = 0.5):
        self.min_ratio = min_ratio
    def check(self, current: float, avg: float) -> bool:
        return current / avg >= self.min_ratio if avg > 0 else False

class TrendFilter:
    """Feature #164: Trend Alignment"""
    def check(self, short: str, medium: str, long: str) -> bool:
        return short == medium == long

class MomentumFilter:
    """Feature #165: Momentum Filter"""
    def __init__(self, min_mom: float = 0.5):
        self.min_mom = min_mom
    def check(self, momentum: float) -> bool:
        return abs(momentum) >= self.min_mom

class LiquidityFilter:
    """Feature #166: Liquidity Filter"""
    def __init__(self, min_liq: float = 10000):
        self.min_liq = min_liq
    def check(self, liquidity: float) -> bool:
        return liquidity >= self.min_liq

class CorrelationFilter:
    """Feature #167: Correlation Filter"""
    def __init__(self, max_corr: float = 0.8):
        self.max_corr = max_corr
    def check(self, correlations: List[float]) -> bool:
        return max(abs(c) for c in correlations) < self.max_corr if correlations else True

class CompositeFilter:
    """Feature #168: Composite Filter"""
    def __init__(self):
        self.time = TimeFilter()
        self.vol = VolatilityFilter()
        self.spread = SpreadFilter()
    
    def check_all(self, volatility: float, spread: float) -> Dict:
        results = {
            'time': self.time.check(),
            'volatility': self.vol.check(volatility),
            'spread': self.spread.check(spread)
        }
        results['allow'] = all(results.values())
        return results

# Factory functions
def get_smoother(): return SignalSmoother()
def get_noise_filter(): return NoiseFilter()
def get_confirmation(): return SignalConfirmation()
def get_strength(): return SignalStrength()
def get_cooldown(): return SignalCooldown()
def get_queue(): return SignalQueue()
def get_composite_filter(): return CompositeFilter()
