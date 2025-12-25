"""
Advanced Indicators & Technical Analysis - Features #46-60
"""
import logging
from typing import Dict, List, Optional
import math

logger = logging.getLogger(__name__)

class ATRCalculator:
    """Feature #46: ATR Calculator"""
    def calculate(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        if len(highs) < period + 1:
            return 0
        trs = []
        for i in range(1, len(highs)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return sum(trs[-period:]) / period

class BollingerBands:
    """Feature #47: Bollinger Bands"""
    def calculate(self, closes: List[float], period: int = 20, std_dev: float = 2.0) -> Dict:
        if len(closes) < period:
            return {}
        sma = sum(closes[-period:]) / period
        variance = sum((c - sma) ** 2 for c in closes[-period:]) / period
        std = variance ** 0.5
        return {'upper': sma + std_dev * std, 'middle': sma, 'lower': sma - std_dev * std}

class RSICalculator:
    """Feature #48: RSI Calculator"""
    def calculate(self, closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [c if c > 0 else 0 for c in changes[-period:]]
        losses = [-c if c < 0 else 0 for c in changes[-period:]]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

class MACDCalculator:
    """Feature #49: MACD Calculator"""
    def _ema(self, data: List[float], period: int) -> float:
        if len(data) < period:
            return data[-1] if data else 0
        alpha = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def calculate(self, closes: List[float]) -> Dict:
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd = ema12 - ema26
        return {'macd': round(macd, 2), 'signal': round(self._ema(closes[-9:], 9) if len(closes) >= 9 else 0, 2)}

class StochasticOscillator:
    """Feature #50: Stochastic Oscillator"""
    def calculate(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict:
        if len(closes) < period:
            return {'k': 50, 'd': 50}
        highest = max(highs[-period:])
        lowest = min(lows[-period:])
        if highest == lowest:
            return {'k': 50, 'd': 50}
        k = (closes[-1] - lowest) / (highest - lowest) * 100
        return {'k': round(k, 2), 'd': round(k, 2)}

class ADXCalculator:
    """Feature #51: ADX Calculator"""
    def calculate(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        if len(highs) < period + 1:
            return 0
        plus_dm, minus_dm = [], []
        for i in range(1, len(highs)):
            up = highs[i] - highs[i-1]
            down = lows[i-1] - lows[i]
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)
        avg_plus = sum(plus_dm[-period:]) / period
        avg_minus = sum(minus_dm[-period:]) / period
        if avg_plus + avg_minus == 0:
            return 0
        dx = abs(avg_plus - avg_minus) / (avg_plus + avg_minus) * 100
        return round(dx, 2)

class OBVCalculator:
    """Feature #52: On-Balance Volume"""
    def calculate(self, closes: List[float], volumes: List[float]) -> float:
        if len(closes) < 2:
            return 0
        obv = 0
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv += volumes[i]
            elif closes[i] < closes[i-1]:
                obv -= volumes[i]
        return obv

class VWAPCalculator:
    """Feature #53: VWAP Calculator"""
    def calculate(self, highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> float:
        if not volumes or sum(volumes) == 0:
            return closes[-1] if closes else 0
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        pv_sum = sum(tp * v for tp, v in zip(typical_prices, volumes))
        return round(pv_sum / sum(volumes), 2)

class IchimokuCloud:
    """Feature #54: Ichimoku Cloud"""
    def calculate(self, highs: List[float], lows: List[float]) -> Dict:
        def midpoint(h, l, n):
            return (max(h[-n:]) + min(l[-n:])) / 2 if len(h) >= n else 0
        
        tenkan = midpoint(highs, lows, 9)
        kijun = midpoint(highs, lows, 26)
        senkou_a = (tenkan + kijun) / 2
        senkou_b = midpoint(highs, lows, 52)
        
        return {'tenkan': round(tenkan, 2), 'kijun': round(kijun, 2), 
                'senkou_a': round(senkou_a, 2), 'senkou_b': round(senkou_b, 2)}

class FibonacciLevels:
    """Feature #55: Fibonacci Retracements"""
    LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    def calculate(self, swing_high: float, swing_low: float) -> Dict:
        diff = swing_high - swing_low
        return {f'level_{int(l*100)}': round(swing_high - diff * l, 2) for l in self.LEVELS}

class PivotPoints:
    """Feature #56: Pivot Points"""
    def calculate(self, high: float, low: float, close: float) -> Dict:
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        return {'pivot': round(pivot, 2), 'r1': round(r1, 2), 'r2': round(r2, 2), 
                's1': round(s1, 2), 's2': round(s2, 2)}

class ParabolicSAR:
    """Feature #57: Parabolic SAR"""
    def calculate(self, highs: List[float], lows: List[float], af: float = 0.02, max_af: float = 0.2) -> float:
        if len(highs) < 3:
            return lows[-1] if lows else 0
        # Simplified calculation
        sar = lows[-3]
        ep = highs[-1]
        sar = sar + af * (ep - sar)
        return round(sar, 2)

class KeltnerChannels:
    """Feature #58: Keltner Channels"""
    def calculate(self, closes: List[float], atr: float, period: int = 20, multiplier: float = 2.0) -> Dict:
        if len(closes) < period:
            return {}
        ema = sum(closes[-period:]) / period
        return {'upper': round(ema + multiplier * atr, 2), 'middle': round(ema, 2), 
                'lower': round(ema - multiplier * atr, 2)}

class CMFCalculator:
    """Feature #59: Chaikin Money Flow"""
    def calculate(self, highs: List[float], lows: List[float], closes: List[float], volumes: List[float], period: int = 20) -> float:
        if len(closes) < period:
            return 0
        mfv = []
        for i in range(-period, 0):
            hl_range = highs[i] - lows[i]
            if hl_range == 0:
                mfv.append(0)
            else:
                mf_mult = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl_range
                mfv.append(mf_mult * volumes[i])
        return round(sum(mfv) / sum(volumes[-period:]), 4) if sum(volumes[-period:]) > 0 else 0

class WilliamsR:
    """Feature #60: Williams %R"""
    def calculate(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        if len(closes) < period:
            return -50
        highest = max(highs[-period:])
        lowest = min(lows[-period:])
        if highest == lowest:
            return -50
        wr = (highest - closes[-1]) / (highest - lowest) * -100
        return round(wr, 2)

# Factories
def get_atr(): return ATRCalculator()
def get_bollinger(): return BollingerBands()
def get_rsi(): return RSICalculator()
def get_macd(): return MACDCalculator()
def get_stochastic(): return StochasticOscillator()
def get_adx(): return ADXCalculator()
def get_obv(): return OBVCalculator()
def get_vwap(): return VWAPCalculator()
def get_ichimoku(): return IchimokuCloud()
def get_fibonacci(): return FibonacciLevels()
def get_pivot(): return PivotPoints()
def get_psar(): return ParabolicSAR()
def get_keltner(): return KeltnerChannels()
def get_cmf(): return CMFCalculator()
def get_williams(): return WilliamsR()
