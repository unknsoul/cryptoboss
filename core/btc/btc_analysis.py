"""
BTC Master Analysis Suite
Professional BTC-specific market analysis features

Features:
- CME Gap tracking and trading
- BTC Dominance analysis
- Funding rate signals
- Open Interest tracking
- Liquidation level detection
- Whale activity monitoring
- Order block detection
- Fair Value Gap (FVG) identification
"""

import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CMEGapTracker:
    """
    Track CME Bitcoin futures gaps
    
    CME gaps occur every weekend when futures close
    90%+ of gaps eventually get filled - major trading edge
    """
    
    def __init__(self):
        self.cme_close_price = None
        self.cme_open_price = None
        self.current_gap = None
        self.gap_history = []
        
    def update_cme_prices(self, friday_close: float, sunday_open: float):
        """Update CME gap prices"""
        self.cme_close_price = friday_close
        self.cme_open_price = sunday_open
        
        if friday_close and sunday_open:
            gap_size = sunday_open - friday_close
            gap_pct = (gap_size / friday_close) * 100
            
            self.current_gap = {
                'close_price': friday_close,
                'open_price': sunday_open,
                'gap_size': gap_size,
                'gap_pct': gap_pct,
                'direction': 'UP' if gap_size > 0 else 'DOWN',
                'filled': False,
                'created_at': datetime.now()
            }
            
            self.gap_history.append(self.current_gap)
            logger.info(f"CME Gap detected: {gap_pct:+.2f}% (${gap_size:+,.0f})")
    
    def check_gap_fill(self, current_price: float) -> Optional[Dict]:
        """Check if current gap is being filled"""
        if not self.current_gap or self.current_gap['filled']:
            return None
        
        gap = self.current_gap
        
        if gap['direction'] == 'UP':
            # Gap up - need price to come back down to close price
            if current_price <= gap['close_price']:
                gap['filled'] = True
                gap['fill_time'] = datetime.now()
                logger.info(f"✓ CME Gap filled! Took {gap['fill_time'] - gap['created_at']}")
                return {'signal': 'GAP_FILLED', 'direction': 'DOWN'}
        else:
            # Gap down - need price to come back up to close price
            if current_price >= gap['close_price']:
                gap['filled'] = True
                gap['fill_time'] = datetime.now()
                logger.info(f"✓ CME Gap filled!")
                return {'signal': 'GAP_FILLED', 'direction': 'UP'}
        
        return None
    
    def get_gap_trade_signal(self, current_price: float) -> Optional[Dict]:
        """Generate trade signal based on CME gap"""
        if not self.current_gap or self.current_gap['filled']:
            return None
        
        gap = self.current_gap
        distance_to_fill = abs(current_price - gap['close_price'])
        distance_pct = (distance_to_fill / current_price) * 100
        
        # If gap is >0.5% and unfilled, trade towards gap fill
        if distance_pct > 0.5:
            signal_direction = 'SHORT' if gap['direction'] == 'UP' else 'LONG'
            
            return {
                'signal': signal_direction,
                'reason': f'CME gap fill trade ({gap["gap_pct"]:+.2f}%)',
                'target': gap['close_price'],
                'confidence': min(0.7, 0.5 + distance_pct * 0.1)
            }
        
        return None


class BTCDominanceTracker:
    """
    Track Bitcoin dominance (BTC.D)
    
    Rising BTC.D = BTC outperforming alts (bullish for BTC)
    Falling BTC.D = Altseason (bearish for BTC relative)
    """
    
    def __init__(self, lookback: int = 24):
        self.dominance_history = deque(maxlen=lookback * 12)  # 5min candles
        self.current_dominance = 50.0
        
    def update(self, btc_dominance: float):
        """Update dominance value"""
        self.current_dominance = btc_dominance
        self.dominance_history.append({
            'timestamp': datetime.now(),
            'value': btc_dominance
        })
    
    def get_trend(self) -> Dict:
        """Get dominance trend"""
        if len(self.dominance_history) < 12:
            return {'trend': 'NEUTRAL', 'strength': 0}
        
        recent = list(self.dominance_history)[-12:]  # Last hour
        older = list(self.dominance_history)[-24:-12] if len(self.dominance_history) >= 24 else recent
        
        recent_avg = np.mean([d['value'] for d in recent])
        older_avg = np.mean([d['value'] for d in older])
        
        change = recent_avg - older_avg
        
        if change > 0.3:
            return {'trend': 'RISING', 'strength': min(change / 0.5, 1.0), 'btc_bias': 'BULLISH'}
        elif change < -0.3:
            return {'trend': 'FALLING', 'strength': min(abs(change) / 0.5, 1.0), 'btc_bias': 'BEARISH'}
        else:
            return {'trend': 'NEUTRAL', 'strength': 0, 'btc_bias': 'NEUTRAL'}
    
    def get_signal_adjustment(self) -> float:
        """Get signal confidence adjustment based on dominance"""
        trend = self.get_trend()
        
        if trend['btc_bias'] == 'BULLISH':
            return 0.1  # Boost long confidence
        elif trend['btc_bias'] == 'BEARISH':
            return -0.1  # Reduce long confidence
        
        return 0


class FundingRateAnalyzer:
    """
    Analyze perpetual futures funding rates
    
    High positive = Overleveraged longs (short opportunity)
    High negative = Overleveraged shorts (long opportunity)
    """
    
    def __init__(self, extreme_threshold: float = 0.1):
        self.current_rate = 0
        self.rate_history = deque(maxlen=100)
        self.extreme_threshold = extreme_threshold  # 0.1% is significant
        
    def update(self, funding_rate: float):
        """Update funding rate (in %)"""
        self.current_rate = funding_rate
        self.rate_history.append({
            'timestamp': datetime.now(),
            'rate': funding_rate
        })
    
    def get_signal(self) -> Optional[Dict]:
        """Get trading signal based on funding"""
        rate = self.current_rate
        
        if rate > self.extreme_threshold:
            return {
                'signal': 'FUNDING_SHORT',
                'reason': f'Extreme positive funding ({rate:.3f}%)',
                'strength': min(rate / 0.2, 1.0),
                'bias': 'SHORT_BIAS'
            }
        elif rate < -self.extreme_threshold:
            return {
                'signal': 'FUNDING_LONG',
                'reason': f'Extreme negative funding ({rate:.3f}%)',
                'strength': min(abs(rate) / 0.2, 1.0),
                'bias': 'LONG_BIAS'
            }
        
        return None
    
    def get_confidence_adjustment(self, side: str) -> float:
        """Adjust confidence based on funding alignment"""
        rate = self.current_rate
        
        if side == 'LONG':
            if rate < -0.05:
                return 0.15  # Boost - shorts paying you
            elif rate > 0.1:
                return -0.15  # Reduce - crowded long
        else:  # SHORT
            if rate > 0.05:
                return 0.15  # Boost - longs paying you
            elif rate < -0.1:
                return -0.15  # Reduce - crowded short
        
        return 0


class OpenInterestTracker:
    """
    Track Open Interest changes
    
    Rising OI + Rising price = Strong trend (follow)
    Rising OI + Falling price = Accumulating shorts
    Falling OI = Liquidations/closing (reversal possible)
    """
    
    def __init__(self):
        self.oi_history = deque(maxlen=288)  # 24 hours of 5min data
        self.current_oi = 0
        
    def update(self, open_interest: float):
        """Update OI value"""
        self.current_oi = open_interest
        self.oi_history.append({
            'timestamp': datetime.now(),
            'oi': open_interest
        })
    
    def analyze(self, price_change_pct: float) -> Dict:
        """Analyze OI with price action"""
        if len(self.oi_history) < 12:
            return {'signal': 'NEUTRAL', 'strength': 0}
        
        recent = list(self.oi_history)[-12:]
        oi_change = (recent[-1]['oi'] - recent[0]['oi']) / recent[0]['oi'] * 100
        
        # Rising OI + Rising price = Strong bullish
        if oi_change > 1 and price_change_pct > 0.5:
            return {
                'signal': 'STRONG_BULLISH',
                'reason': 'Rising OI with rising price',
                'strength': min(oi_change / 3, 1.0),
                'bias': 'LONG'
            }
        
        # Rising OI + Falling price = Bearish pressure
        elif oi_change > 1 and price_change_pct < -0.5:
            return {
                'signal': 'STRONG_BEARISH',
                'reason': 'Rising OI with falling price (shorts accumulating)',
                'strength': min(oi_change / 3, 1.0),
                'bias': 'SHORT'
            }
        
        # Falling OI = Liquidations, possible reversal
        elif oi_change < -2:
            return {
                'signal': 'LIQUIDATION',
                'reason': f'Falling OI ({oi_change:.1f}%) - positions closing',
                'strength': min(abs(oi_change) / 5, 1.0),
                'bias': 'REVERSAL_POSSIBLE'
            }
        
        return {'signal': 'NEUTRAL', 'strength': 0}


class OrderBlockDetector:
    """
    Detect institutional order blocks
    
    Order blocks are zones where institutions placed large orders
    Price often returns to these zones
    """
    
    def __init__(self):
        self.bullish_obs = []  # Bullish order blocks
        self.bearish_obs = []  # Bearish order blocks
        
    def detect_order_blocks(self, candles: List[Dict]) -> List[Dict]:
        """
        Detect order blocks from candle data
        
        Bullish OB: Last down candle before strong up move
        Bearish OB: Last up candle before strong down move
        """
        if len(candles) < 10:
            return []
        
        new_obs = []
        
        for i in range(3, len(candles) - 3):
            c = candles[i]
            prev_candles = candles[i-3:i]
            next_candles = candles[i+1:i+4]
            
            # Check for bullish OB: down candle followed by strong up move
            if c['close'] < c['open']:  # Down candle
                # Check if next candles make strong up move
                next_highs = [nc['high'] for nc in next_candles]
                max_next_high = max(next_highs) if next_highs else c['high']
                
                move_up = (max_next_high - c['high']) / c['high'] * 100
                
                if move_up > 0.5:  # 0.5% move up
                    ob = {
                        'type': 'BULLISH',
                        'top': c['open'],
                        'bottom': c['low'],
                        'strength': min(move_up / 1.5, 1.0),
                        'created_at': c.get('timestamp', datetime.now())
                    }
                    self.bullish_obs.append(ob)
                    new_obs.append(ob)
            
            # Check for bearish OB: up candle followed by strong down move
            elif c['close'] > c['open']:  # Up candle
                next_lows = [nc['low'] for nc in next_candles]
                min_next_low = min(next_lows) if next_lows else c['low']
                
                move_down = (c['low'] - min_next_low) / c['low'] * 100
                
                if move_down > 0.5:
                    ob = {
                        'type': 'BEARISH',
                        'top': c['high'],
                        'bottom': c['close'],
                        'strength': min(move_down / 1.5, 1.0),
                        'created_at': c.get('timestamp', datetime.now())
                    }
                    self.bearish_obs.append(ob)
                    new_obs.append(ob)
        
        # Keep only recent OBs (last 50)
        self.bullish_obs = self.bullish_obs[-50:]
        self.bearish_obs = self.bearish_obs[-50:]
        
        return new_obs
    
    def check_price_at_ob(self, current_price: float) -> Optional[Dict]:
        """Check if price is at an order block"""
        # Check bullish OBs (price touching = long opportunity)
        for ob in reversed(self.bullish_obs[-10:]):
            if ob['bottom'] <= current_price <= ob['top']:
                return {
                    'signal': 'LONG',
                    'type': 'BULLISH_OB',
                    'reason': f'Price at bullish order block',
                    'stop_loss': ob['bottom'] * 0.997,
                    'confidence': 0.6 + ob['strength'] * 0.2
                }
        
        # Check bearish OBs (price touching = short opportunity)
        for ob in reversed(self.bearish_obs[-10:]):
            if ob['bottom'] <= current_price <= ob['top']:
                return {
                    'signal': 'SHORT',
                    'type': 'BEARISH_OB',
                    'reason': f'Price at bearish order block',
                    'stop_loss': ob['top'] * 1.003,
                    'confidence': 0.6 + ob['strength'] * 0.2
                }
        
        return None


class FairValueGapDetector:
    """
    Detect Fair Value Gaps (FVGs)
    
    FVGs are imbalances where price moved so fast that
    orders couldn't be filled - often gets revisited
    """
    
    def __init__(self):
        self.bullish_fvgs = []
        self.bearish_fvgs = []
        
    def detect_fvgs(self, candles: List[Dict]) -> List[Dict]:
        """
        Detect FVGs from candle data
        
        Bullish FVG: Low of candle[i+1] > High of candle[i-1]
        Bearish FVG: High of candle[i+1] < Low of candle[i-1]
        """
        if len(candles) < 3:
            return []
        
        new_fvgs = []
        
        for i in range(1, len(candles) - 1):
            prev = candles[i - 1]
            curr = candles[i]
            next_c = candles[i + 1]
            
            # Bullish FVG
            if next_c['low'] > prev['high']:
                gap_size = next_c['low'] - prev['high']
                gap_pct = gap_size / curr['close'] * 100
                
                if gap_pct > 0.1:  # Significant gap
                    fvg = {
                        'type': 'BULLISH',
                        'top': next_c['low'],
                        'bottom': prev['high'],
                        'size_pct': gap_pct,
                        'filled': False
                    }
                    self.bullish_fvgs.append(fvg)
                    new_fvgs.append(fvg)
            
            # Bearish FVG
            if next_c['high'] < prev['low']:
                gap_size = prev['low'] - next_c['high']
                gap_pct = gap_size / curr['close'] * 100
                
                if gap_pct > 0.1:
                    fvg = {
                        'type': 'BEARISH',
                        'top': prev['low'],
                        'bottom': next_c['high'],
                        'size_pct': gap_pct,
                        'filled': False
                    }
                    self.bearish_fvgs.append(fvg)
                    new_fvgs.append(fvg)
        
        return new_fvgs
    
    def check_fvg_fill(self, current_price: float) -> Optional[Dict]:
        """Check if price is filling an FVG"""
        # Check unfilled bullish FVGs
        for fvg in self.bullish_fvgs[-20:]:
            if not fvg['filled'] and fvg['bottom'] <= current_price <= fvg['top']:
                return {
                    'signal': 'FVG_FILL_LONG',
                    'reason': 'Price filling bullish FVG',
                    'confidence': 0.65
                }
        
        # Check unfilled bearish FVGs
        for fvg in self.bearish_fvgs[-20:]:
            if not fvg['filled'] and fvg['bottom'] <= current_price <= fvg['top']:
                return {
                    'signal': 'FVG_FILL_SHORT',
                    'reason': 'Price filling bearish FVG',
                    'confidence': 0.65
                }
        
        return None


class LiquidationLevelTracker:
    """
    Track potential liquidation levels
    
    Large liquidation clusters act as price magnets
    """
    
    def __init__(self):
        self.liquidation_levels = []
        
    def estimate_liquidation_levels(self, current_price: float, funding_rate: float) -> List[Dict]:
        """
        Estimate where liquidations might cluster
        
        Based on typical leverage levels (10x, 20x, 50x, 100x)
        """
        levels = []
        
        # Common leverage offsets from current price
        leverage_offsets = {
            100: 0.01,   # 100x = 1% move for liquidation
            50: 0.02,    # 50x = 2% move
            25: 0.04,    # 25x = 4% move
            10: 0.10,    # 10x = 10% move
        }
        
        for leverage, offset in leverage_offsets.items():
            # Long liquidations (below current price)
            long_liq = current_price * (1 - offset)
            levels.append({
                'price': long_liq,
                'type': 'LONG_LIQ',
                'leverage': leverage,
                'direction': 'below'
            })
            
            # Short liquidations (above current price)
            short_liq = current_price * (1 + offset)
            levels.append({
                'price': short_liq,
                'type': 'SHORT_LIQ',
                'leverage': leverage,
                'direction': 'above'
            })
        
        self.liquidation_levels = levels
        return levels
    
    def get_nearest_liquidation(self, current_price: float) -> Optional[Dict]:
        """Get nearest significant liquidation level"""
        if not self.liquidation_levels:
            return None
        
        nearest = min(
            self.liquidation_levels,
            key=lambda x: abs(x['price'] - current_price)
        )
        
        distance_pct = abs(nearest['price'] - current_price) / current_price * 100
        
        return {
            **nearest,
            'distance_pct': distance_pct,
            'is_close': distance_pct < 2
        }


class SessionAnalyzer:
    """
    Analyze trading sessions for BTC
    
    Different sessions have different characteristics
    """
    
    def __init__(self):
        pass
    
    def get_current_session(self) -> Dict:
        """Get current trading session"""
        now = datetime.utcnow()
        hour = now.hour
        
        # Session definitions (UTC)
        if 0 <= hour < 8:
            return {
                'session': 'ASIA',
                'volatility': 'LOW',
                'strategy_bias': 'RANGE',
                'size_multiplier': 0.7
            }
        elif 8 <= hour < 13:
            return {
                'session': 'LONDON',
                'volatility': 'MEDIUM_HIGH',
                'strategy_bias': 'BREAKOUT',
                'size_multiplier': 1.0
            }
        elif 13 <= hour < 21:
            return {
                'session': 'NEW_YORK',
                'volatility': 'HIGH',
                'strategy_bias': 'TREND',
                'size_multiplier': 1.2
            }
        else:
            return {
                'session': 'LATE_NY',
                'volatility': 'LOW',
                'strategy_bias': 'AVOID',
                'size_multiplier': 0.5
            }
    
    def get_day_bias(self) -> Dict:
        """Get day-of-week trading bias"""
        day = datetime.utcnow().weekday()
        
        biases = {
            0: {'day': 'Monday', 'bias': 'DIRECTION_SETTING', 'confidence': 0.8},
            1: {'day': 'Tuesday', 'bias': 'FOLLOW_MONDAY', 'confidence': 1.0},
            2: {'day': 'Wednesday', 'bias': 'MID_WEEK_REVERSAL', 'confidence': 0.9},
            3: {'day': 'Thursday', 'bias': 'TREND_CONTINUATION', 'confidence': 1.0},
            4: {'day': 'Friday', 'bias': 'CME_CLOSE_CAUTION', 'confidence': 0.6},
            5: {'day': 'Saturday', 'bias': 'LOW_LIQUIDITY', 'confidence': 0.5},
            6: {'day': 'Sunday', 'bias': 'CME_GAP_WATCH', 'confidence': 0.4}
        }
        
        return biases.get(day, {'bias': 'NEUTRAL', 'confidence': 0.7})


# Singleton instances
_cme_tracker = None
_dominance_tracker = None
_funding_analyzer = None
_oi_tracker = None
_ob_detector = None
_fvg_detector = None
_liq_tracker = None
_session_analyzer = None


def get_cme_tracker() -> CMEGapTracker:
    global _cme_tracker
    if _cme_tracker is None:
        _cme_tracker = CMEGapTracker()
    return _cme_tracker


def get_dominance_tracker() -> BTCDominanceTracker:
    global _dominance_tracker
    if _dominance_tracker is None:
        _dominance_tracker = BTCDominanceTracker()
    return _dominance_tracker


def get_funding_analyzer() -> FundingRateAnalyzer:
    global _funding_analyzer
    if _funding_analyzer is None:
        _funding_analyzer = FundingRateAnalyzer()
    return _funding_analyzer


def get_oi_tracker() -> OpenInterestTracker:
    global _oi_tracker
    if _oi_tracker is None:
        _oi_tracker = OpenInterestTracker()
    return _oi_tracker


def get_ob_detector() -> OrderBlockDetector:
    global _ob_detector
    if _ob_detector is None:
        _ob_detector = OrderBlockDetector()
    return _ob_detector


def get_fvg_detector() -> FairValueGapDetector:
    global _fvg_detector
    if _fvg_detector is None:
        _fvg_detector = FairValueGapDetector()
    return _fvg_detector


def get_liq_tracker() -> LiquidationLevelTracker:
    global _liq_tracker
    if _liq_tracker is None:
        _liq_tracker = LiquidationLevelTracker()
    return _liq_tracker


def get_session_analyzer() -> SessionAnalyzer:
    global _session_analyzer
    if _session_analyzer is None:
        _session_analyzer = SessionAnalyzer()
    return _session_analyzer


if __name__ == '__main__':
    print("=" * 70)
    print("BTC MASTER ANALYSIS SUITE - TEST")
    print("=" * 70)
    
    # Test CME Gap
    print("\n📊 Testing CME Gap Tracker...")
    cme = CMEGapTracker()
    cme.update_cme_prices(friday_close=50000, sunday_open=50500)
    signal = cme.get_gap_trade_signal(50400)
    print(f"  Gap: {cme.current_gap['gap_pct']:+.2f}%")
    if signal:
        print(f"  Signal: {signal['signal']} (target: ${signal['target']:,.0f})")
    
    # Test Funding
    print("\n💰 Testing Funding Rate Analyzer...")
    funding = FundingRateAnalyzer()
    funding.update(0.15)  # High positive
    signal = funding.get_signal()
    print(f"  Rate: {funding.current_rate:.3f}%")
    if signal:
        print(f"  Signal: {signal['signal']} ({signal['reason']})")
    
    # Test Session
    print("\n⏰ Testing Session Analyzer...")
    session = SessionAnalyzer()
    current = session.get_current_session()
    day = session.get_day_bias()
    print(f"  Session: {current['session']} (vol: {current['volatility']})")
    print(f"  Day: {day['day']} ({day['bias']})")
    
    # Test Order Blocks
    print("\n📦 Testing Order Block Detector...")
    ob = OrderBlockDetector()
    # Simulate candles
    candles = [
        {'open': 50000, 'high': 50100, 'low': 49900, 'close': 49950},
        {'open': 49950, 'high': 50000, 'low': 49850, 'close': 49900},  # Down candle
        {'open': 49900, 'high': 50500, 'low': 49850, 'close': 50400},  # Strong up
        {'open': 50400, 'high': 50600, 'low': 50300, 'close': 50550},
        {'open': 50550, 'high': 50700, 'low': 50500, 'close': 50650},
    ]
    new_obs = ob.detect_order_blocks(candles)
    print(f"  Detected {len(new_obs)} order blocks")
    
    print("\n" + "=" * 70)
    print("✅ BTC Master Analysis Suite working!")
    print("=" * 70)
