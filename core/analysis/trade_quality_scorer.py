"""
Trade Quality Scoring System
Professional-grade trade quality assessment for improved accuracy

Scoring Factors (Total 100 points):
- Trend Alignment (25): HTF trend matches direction
- Structure Level (20): Near key support/resistance
- Momentum (15): RSI/MACD confirmation
- Volume (15): Above average volume
- Risk:Reward (15): Minimum 1.5:1 required
- Session (10): London/NY overlap bonus

Minimum score to trade: 70
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import logging

# Import pattern detector
try:
    from core.analysis.patterns import detect_pattern, get_pattern_confirmation
except ImportError:
    detect_pattern = None
    get_pattern_confirmation = None

logger = logging.getLogger(__name__)


class TradeQualityScorer:
    """
    Professional trade quality assessment system.
    Scores every potential trade 0-100 and enforces minimum quality standards.
    """
    
    # Minimum score required to take a trade
    # Lowered to 68 to allow more opportunities while maintaining quality
    MIN_SCORE_TO_TRADE = 68
    
    # Scoring weights (must sum to 100)
    # UPGRADED: More weight on trend + structure, less on volume + session
    WEIGHTS = {
        'trend_alignment': 30,   # Up from 25 - trend is critical
        'structure_level': 25,   # Up from 20 - need good entry levels
        'momentum': 15,          # Unchanged
        'volume': 10,            # Down from 15 - less important
        'risk_reward': 15,       # Unchanged
        'session': 5             # Down from 10 - less important
    }
    
    def __init__(self, min_score: int = 70):
        self.min_score = min_score
        self.trade_scores = []  # History of scores
        logger.info(f"TradeQualityScorer initialized (min score: {min_score})")
    
    def score_trade(
        self,
        signal: Dict,
        df: pd.DataFrame,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        current_price: float = 0
    ) -> Tuple[int, Dict[str, int], List[str]]:
        """
        Score a potential trade for quality.
        
        Returns:
            (total_score, breakdown, reasons)
        """
        scores = {}
        reasons = []
        
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0.5)
        atr = signal.get('atr', current_price * 0.01)
        
        # 1. Trend Alignment Score (25 points)
        trend_score, trend_reason = self._score_trend_alignment(action, df, df_1h, df_4h)
        scores['trend_alignment'] = trend_score
        reasons.append(trend_reason)
        
        # 2. Structure Level Score (20 points)
        structure_score, structure_reason = self._score_structure_level(action, df, current_price, atr)
        scores['structure_level'] = structure_score
        reasons.append(structure_reason)
        
        # 3. Momentum Score (15 points)
        momentum_score, momentum_reason = self._score_momentum(action, df)
        scores['momentum'] = momentum_score
        reasons.append(momentum_reason)
        
        # 4. Volume Score (15 points)
        volume_score, volume_reason = self._score_volume(df)
        scores['volume'] = volume_score
        reasons.append(volume_reason)
        
        # 5. Risk:Reward Score (15 points)
        rr_score, rr_reason = self._score_risk_reward(signal, atr)
        scores['risk_reward'] = rr_score
        reasons.append(rr_reason)
        
        # 6. Session Score (5 points)
        session_score, session_reason = self._score_session()
        scores['session'] = session_score
        reasons.append(session_reason)
        
        # 7. BONUS: Candlestick Pattern Score (up to +10 bonus points)
        if detect_pattern is not None:
            try:
                pattern_found, pattern_name, pattern_pts = detect_pattern(df, action)
                if pattern_found and pattern_pts > 0:
                    scores['pattern'] = pattern_pts
                    reasons.append(f"Pattern: {pattern_name} (+{pattern_pts})")
                else:
                    scores['pattern'] = 0
                    reasons.append("Pattern: None")
            except Exception as e:
                scores['pattern'] = 0
                reasons.append(f"Pattern: Error")
        else:
            scores['pattern'] = 0
        
        # 8. VOLATILITY GUARD - Reject if market is too chaotic or too dead
        volatility_ok = True
        if atr > 0 and len(df) >= 20:
            # Calculate average ATR over last 20 periods
            atr_values = []
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            for i in range(1, min(20, len(df))):
                tr = max(
                    highs[-i] - lows[-i],
                    abs(highs[-i] - closes[-i-1]) if i < len(closes) - 1 else 0,
                    abs(lows[-i] - closes[-i-1]) if i < len(closes) - 1 else 0
                )
                atr_values.append(tr)
            
            avg_atr = np.mean(atr_values) if atr_values else atr
            atr_ratio = atr / avg_atr if avg_atr > 0 else 1.0
            
            # Reject if ATR is > 2.5x average (extreme volatility)
            if atr_ratio > 2.5:
                volatility_ok = False
                reasons.append(f"VOLATILITY GUARD: ATR {atr_ratio:.1f}x avg - TOO HIGH")
            # Reject if ATR is < 0.3x average (dead market)
            elif atr_ratio < 0.3:
                volatility_ok = False
                reasons.append(f"VOLATILITY GUARD: ATR {atr_ratio:.1f}x avg - TOO LOW")
            else:
                reasons.append(f"Volatility: {atr_ratio:.1f}x avg (OK)")
        
        # If volatility guard failed, return 0 score
        if not volatility_ok:
            return 0, scores, reasons
        
        # Calculate total (base 100 + bonus pattern points)
        total_score = sum(scores.values())
        
        # Store for history
        self.trade_scores.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'score': total_score,
            'breakdown': scores.copy()
        })
        
        # Keep only last 100 scores
        if len(self.trade_scores) > 100:
            self.trade_scores = self.trade_scores[-100:]
        
        return total_score, scores, reasons
    
    def _score_trend_alignment(
        self,
        action: str,
        df: pd.DataFrame,
        df_1h: Optional[pd.DataFrame],
        df_4h: Optional[pd.DataFrame]
    ) -> Tuple[int, str]:
        """Score based on trend alignment with higher timeframes"""
        max_points = self.WEIGHTS['trend_alignment']
        score = 0
        
        # Check 5m trend
        trend_5m = self._detect_trend(df) if len(df) > 30 else 'range'
        
        # Check 1H trend
        trend_1h = 'range'
        if df_1h is not None and len(df_1h) > 20:
            trend_1h = self._detect_trend(df_1h)
        
        # Check 4H trend  
        trend_4h = 'range'
        if df_4h is not None and len(df_4h) > 10:
            trend_4h = self._detect_trend(df_4h)
        
        # Score alignment
        if action == 'LONG':
            if trend_4h == 'up':
                score += 12
            elif trend_4h == 'range':
                score += 6
            
            if trend_1h == 'up':
                score += 8
            elif trend_1h == 'range':
                score += 4
            
            if trend_5m == 'up':
                score += 5
            elif trend_5m == 'range':
                score += 2
                
        elif action == 'SHORT':
            if trend_4h == 'down':
                score += 12
            elif trend_4h == 'range':
                score += 6
            
            if trend_1h == 'down':
                score += 8
            elif trend_1h == 'range':
                score += 4
            
            if trend_5m == 'down':
                score += 5
            elif trend_5m == 'range':
                score += 2
        
        score = min(score, max_points)
        
        if score >= 20:
            return score, f"Trend: Strong alignment ({score}/{max_points})"
        elif score >= 12:
            return score, f"Trend: Partial alignment ({score}/{max_points})"
        else:
            return score, f"Trend: Weak/Counter-trend ({score}/{max_points})"
    
    def _score_structure_level(
        self,
        action: str,
        df: pd.DataFrame,
        current_price: float,
        atr: float
    ) -> Tuple[int, str]:
        """Score based on proximity to key support/resistance levels"""
        max_points = self.WEIGHTS['structure_level']
        
        if len(df) < 50:
            return max_points // 2, "Structure: Insufficient data"
        
        highs = df['high'].values
        lows = df['low'].values
        
        # Find recent swing highs and lows
        swing_high = np.max(highs[-20:])
        swing_low = np.min(lows[-20:])
        
        # Calculate proximity to levels
        distance_to_high = abs(current_price - swing_high)
        distance_to_low = abs(current_price - swing_low)
        
        score = 0
        
        if action == 'LONG':
            # Good: near support (swing low)
            if distance_to_low < atr * 0.5:
                score = max_points
                reason = "Structure: At support level"
            elif distance_to_low < atr * 1.0:
                score = max_points * 0.7
                reason = "Structure: Near support"
            elif distance_to_low < atr * 2.0:
                score = max_points * 0.4
                reason = "Structure: Away from support"
            else:
                score = max_points * 0.2
                reason = "Structure: No clear level"
                
        elif action == 'SHORT':
            # Good: near resistance (swing high)
            if distance_to_high < atr * 0.5:
                score = max_points
                reason = "Structure: At resistance level"
            elif distance_to_high < atr * 1.0:
                score = max_points * 0.7
                reason = "Structure: Near resistance"
            elif distance_to_high < atr * 2.0:
                score = max_points * 0.4
                reason = "Structure: Away from resistance"
            else:
                score = max_points * 0.2
                reason = "Structure: No clear level"
        else:
            score = 0
            reason = "Structure: No action"
        
        return int(score), reason
    
    def _score_momentum(self, action: str, df: pd.DataFrame) -> Tuple[int, str]:
        """Score based on momentum indicators (RSI, MACD)"""
        max_points = self.WEIGHTS['momentum']
        
        if len(df) < 30:
            return max_points // 2, "Momentum: Insufficient data"
        
        closes = df['close'].values
        
        # Calculate RSI
        rsi = self._calculate_rsi(closes, 14)
        current_rsi = rsi[-1] if len(rsi) > 0 else 50
        
        score = 0
        
        if action == 'LONG':
            # Good: RSI 40-60 (not overbought)
            if 40 <= current_rsi <= 60:
                score = max_points
                reason = f"Momentum: RSI neutral ({current_rsi:.0f})"
            elif 30 <= current_rsi < 40:
                score = max_points * 0.8
                reason = f"Momentum: RSI oversold ({current_rsi:.0f})"
            elif current_rsi < 30:
                score = max_points * 0.5  # Too oversold, risky
                reason = f"Momentum: RSI extreme ({current_rsi:.0f})"
            else:
                score = max_points * 0.3
                reason = f"Momentum: RSI overbought ({current_rsi:.0f})"
                
        elif action == 'SHORT':
            if 40 <= current_rsi <= 60:
                score = max_points
                reason = f"Momentum: RSI neutral ({current_rsi:.0f})"
            elif 60 < current_rsi <= 70:
                score = max_points * 0.8
                reason = f"Momentum: RSI overbought ({current_rsi:.0f})"
            elif current_rsi > 70:
                score = max_points * 0.5
                reason = f"Momentum: RSI extreme ({current_rsi:.0f})"
            else:
                score = max_points * 0.3
                reason = f"Momentum: RSI oversold ({current_rsi:.0f})"
        else:
            score = 0
            reason = "Momentum: No action"
        
        return int(score), reason
    
    def _score_volume(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Score based on volume confirmation - UPGRADED: proportional scoring"""
        max_points = self.WEIGHTS['volume']
        
        if 'volume' not in df.columns or len(df) < 20:
            return 0, "Volume: No data"  # Changed from max/2 to 0
        
        volumes = df['volume'].values
        avg_volume = np.mean(volumes[-20:-1])
        current_volume = volumes[-1]
        
        if avg_volume == 0:
            return 0, "Volume: No data"
        
        volume_ratio = current_volume / avg_volume
        
        # UPGRADED: Proportional scoring with cap at 2x
        # Low volume is severely penalized (0.5x or less = 0 points)
        if volume_ratio < 0.5:
            score = 0
            reason = f"Volume: Very low ({volume_ratio:.1f}x) - REJECT"
        elif volume_ratio < 0.8:
            score = int(max_points * 0.2)
            reason = f"Volume: Below avg ({volume_ratio:.1f}x)"
        elif volume_ratio < 1.2:
            score = int(max_points * 0.5)
            reason = f"Volume: Normal ({volume_ratio:.1f}x)"
        elif volume_ratio < 1.5:
            score = int(max_points * 0.8)
            reason = f"Volume: Above avg ({volume_ratio:.1f}x)"
        else:
            # Cap the multiplier at 2x for max score
            ratio_capped = min(volume_ratio, 2.0)
            score = int(max_points * (ratio_capped / 1.5))
            score = min(score, max_points)  # Don't exceed max
            reason = f"Volume: High spike ({volume_ratio:.1f}x)"
        
        return score, reason
    
    def _score_risk_reward(self, signal: Dict, atr: float) -> Tuple[int, str]:
        """Score based on risk:reward ratio"""
        max_points = self.WEIGHTS['risk_reward']
        
        sl_mult = signal.get('sl_multiplier', 1.5)
        tp_mult = signal.get('tp_multiplier', 2.5)
        
        if sl_mult == 0:
            return 0, "R:R: Invalid (no SL)"
        
        rr_ratio = tp_mult / sl_mult
        
        if rr_ratio >= 2.5:
            score = max_points
            reason = f"R:R: Excellent ({rr_ratio:.1f}:1)"
        elif rr_ratio >= 2.0:
            score = int(max_points * 0.9)
            reason = f"R:R: Very good ({rr_ratio:.1f}:1)"
        elif rr_ratio >= 1.5:
            score = int(max_points * 0.7)
            reason = f"R:R: Good ({rr_ratio:.1f}:1)"
        elif rr_ratio >= 1.0:
            score = int(max_points * 0.4)
            reason = f"R:R: Marginal ({rr_ratio:.1f}:1)"
        else:
            score = 0
            reason = f"R:R: Poor ({rr_ratio:.1f}:1)"
        
        return score, reason
    
    def _score_session(self) -> Tuple[int, str]:
        """Score based on current trading session"""
        max_points = self.WEIGHTS['session']
        
        current_hour = datetime.utcnow().hour
        
        # Session definitions (UTC)
        # London: 7-12, NY: 12-17, Overlap: 12-17
        if 12 <= current_hour < 17:
            return max_points, "Session: London/NY Overlap (best)"
        elif 7 <= current_hour < 12:
            return int(max_points * 0.8), "Session: London (good)"
        elif 17 <= current_hour < 22:
            return int(max_points * 0.6), "Session: NY (moderate)"
        else:
            return int(max_points * 0.3), "Session: Off-hours (quiet)"
    
    def _detect_trend(self, df: pd.DataFrame) -> str:
        """Detect trend using EMA crossover"""
        if len(df) < 30:
            return 'range'
        
        closes = df['close'].values
        ema_fast = self._ema(closes, 8)
        ema_slow = self._ema(closes, 21)
        
        if ema_fast[-1] > ema_slow[-1] and closes[-1] > ema_fast[-1]:
            return 'up'
        elif ema_fast[-1] < ema_slow[-1] and closes[-1] < ema_fast[-1]:
            return 'down'
        else:
            return 'range'
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        rsi = np.zeros(len(prices))
        
        if len(gains) < period:
            return rsi
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def should_take_trade(self, score: int) -> Tuple[bool, str]:
        """Determine if trade meets minimum quality requirements"""
        if score >= 85:
            return True, "A+ Trade: Excellent setup"
        elif score >= 75:
            return True, "A Trade: Good setup"
        elif score >= self.min_score:
            return True, "B Trade: Acceptable setup"
        elif score >= 60:
            return False, "C Trade: Below threshold"
        else:
            return False, "D Trade: Poor quality"
    
    def get_average_score(self) -> float:
        """Get average score of recent trades"""
        if not self.trade_scores:
            return 0.0
        return sum(t['score'] for t in self.trade_scores) / len(self.trade_scores)


# Export for use
__all__ = ['TradeQualityScorer']


if __name__ == '__main__':
    print("=" * 60)
    print("TRADE QUALITY SCORER")
    print("=" * 60)
    
    # Test with sample data
    scorer = TradeQualityScorer(min_score=70)
    
    # Create sample signal
    signal = {
        'action': 'LONG',
        'confidence': 0.75,
        'atr': 500,
        'sl_multiplier': 1.5,
        'tp_multiplier': 3.0
    }
    
    # Create sample dataframe
    np.random.seed(42)
    prices = 90000 + np.cumsum(np.random.randn(100) * 100)
    df = pd.DataFrame({
        'close': prices,
        'high': prices + 50,
        'low': prices - 50,
        'volume': np.random.rand(100) * 1000 + 500
    })
    
    # Score the trade
    score, breakdown, reasons = scorer.score_trade(signal, df, current_price=prices[-1])
    
    print(f"\nTotal Score: {score}/100")
    print(f"\nBreakdown:")
    for factor, points in breakdown.items():
        print(f"  {factor}: {points}")
    
    print(f"\nReasons:")
    for reason in reasons:
        print(f"  - {reason}")
    
    should_trade, grade = scorer.should_take_trade(score)
    print(f"\n{grade}")
    print(f"Should take trade: {should_trade}")
