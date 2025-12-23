"""
Advanced Trading Features Module
Implements high-priority features for improved accuracy and reliability
"""

import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Trading session types"""
    ASIAN = "asian"
    LONDON = "london"
    NY = "new_york"
    OVERLAP = "overlap"
    OFF_HOURS = "off_hours"


@dataclass
class SessionConfig:
    """Configuration for trading sessions"""
    session: TradingSession
    start_hour_utc: int
    end_hour_utc: int
    priority: float  # Higher = better session to trade


# ============ MULTI-TIMEFRAME CONFIRMATION ============

class MultiTimeframeConfirmation:
    """
    Confirms signals using higher timeframe trend alignment.
    Key insight: Trade WITH the higher timeframe trend, not against it.
    """
    
    def __init__(self, require_4h_alignment: bool = True, require_1h_alignment: bool = True):
        self.require_4h_alignment = require_4h_alignment
        self.require_1h_alignment = require_1h_alignment
    
    def detect_trend(self, df: pd.DataFrame, lookback: int = 20) -> str:
        """
        Detect trend using EMA crossover and price position
        Returns: 'up', 'down', or 'range'
        """
        if len(df) < lookback + 10:
            return 'range'
        
        closes = df['close'].values
        
        # Calculate EMAs
        ema_fast = self._ema(closes, 8)
        ema_slow = self._ema(closes, 21)
        
        # Current values
        fast_now = ema_fast[-1]
        slow_now = ema_slow[-1]
        price_now = closes[-1]
        
        # Previous values for momentum
        fast_prev = ema_fast[-5]
        slow_prev = ema_slow[-5]
        
        # Strong trend conditions
        if fast_now > slow_now and price_now > fast_now and fast_now > fast_prev:
            return 'up'
        elif fast_now < slow_now and price_now < fast_now and fast_now < fast_prev:
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
    
    def confirm_signal(
        self, 
        signal_action: str,  # 'LONG' or 'SHORT'
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, str]:
        """
        Confirm a signal against higher timeframes.
        
        Strategy: 4H is the major filter, 1H is advisory.
        This allows counter-trend scalps but blocks major trend reversals.
        
        Returns:
            (is_confirmed, reason)
        """
        reasons = []
        confirmed = True
        rejections = 0  # Count rejections
        
        # Check 1H alignment (Advisory - reduces confidence but doesn't block)
        if df_1h is not None and len(df_1h) > 30:
            trend_1h = self.detect_trend(df_1h)
            
            if signal_action == 'LONG':
                if trend_1h == 'down':
                    # Advisory warning - don't block
                    reasons.append("1H trend DOWN - counter-trend trade")
                elif trend_1h == 'up':
                    reasons.append("1H aligned UP ✓")
                else:
                    reasons.append("1H ranging")
            
            elif signal_action == 'SHORT':
                if trend_1h == 'up':
                    reasons.append("1H trend UP - counter-trend trade")
                elif trend_1h == 'down':
                    reasons.append("1H aligned DOWN ✓")
                else:
                    reasons.append("1H ranging")
        
        # Check 4H alignment (Major filter - this CAN block trades)
        if df_4h is not None and len(df_4h) > 20 and self.require_4h_alignment:
            trend_4h = self.detect_trend(df_4h)
            
            if signal_action == 'LONG' and trend_4h == 'down':
                rejections += 1
                reasons.append("4H DOWN - LONG blocked")
            elif signal_action == 'SHORT' and trend_4h == 'up':
                rejections += 1
                reasons.append("4H UP - SHORT blocked")
            elif trend_4h in ['up', 'down']:
                reasons.append(f"4H {trend_4h.upper()} ✓")
            else:
                reasons.append("4H ranging - OK")
        
        # Only block if 4H is against us
        if rejections > 0:
            confirmed = False
        
        return confirmed, "; ".join(reasons) if reasons else "No HTF data"


# ============ SESSION-BASED TRADING ============

class SessionTrading:
    """
    Trades only during optimal trading sessions.
    Avoids low-liquidity periods that lead to choppy price action.
    """
    
    # Session definitions (UTC)
    SESSIONS = [
        SessionConfig(TradingSession.ASIAN, 0, 7, 0.6),      # Lower priority
        SessionConfig(TradingSession.LONDON, 7, 12, 0.9),    # High priority
        SessionConfig(TradingSession.OVERLAP, 12, 17, 1.0),  # Best session
        SessionConfig(TradingSession.NY, 17, 22, 0.8),       # Good
        SessionConfig(TradingSession.OFF_HOURS, 22, 24, 0.4), # Avoid if possible
    ]
    
    def __init__(self, min_session_priority: float = 0.5):
        self.min_session_priority = min_session_priority
    
    def get_current_session(self) -> SessionConfig:
        """Get current trading session based on UTC time"""
        current_hour = datetime.utcnow().hour
        
        for session in self.SESSIONS:
            if session.start_hour_utc <= current_hour < session.end_hour_utc:
                return session
        
        # Default to off-hours for edge cases
        return SessionConfig(TradingSession.OFF_HOURS, 22, 24, 0.4)
    
    def should_trade(self) -> Tuple[bool, str]:
        """
        Check if current session is suitable for trading.
        
        Returns:
            (should_trade, reason)
        """
        session = self.get_current_session()
        
        if session.priority >= self.min_session_priority:
            return True, f"Session: {session.session.value} (priority: {session.priority})"
        else:
            return False, f"Low priority session: {session.session.value} (priority: {session.priority} < {self.min_session_priority})"
    
    def get_session_multiplier(self) -> float:
        """
        Get position size multiplier based on session.
        Trade smaller in lower-quality sessions.
        """
        session = self.get_current_session()
        return session.priority


# ============ DYNAMIC VOLATILITY STOPS ============

class DynamicVolatilityStops:
    """
    Adjusts stop loss and take profit based on current volatility regime.
    Wider stops in high vol, tighter in low vol.
    """
    
    def __init__(
        self,
        low_vol_threshold: float = 0.01,   # 1% daily range
        high_vol_threshold: float = 0.03   # 3% daily range
    ):
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
    
    def calculate_volatility(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate current volatility as percentage of price"""
        if len(df) < period:
            return 0.02  # Default to 2%
        
        highs = df['high'].values[-period:]
        lows = df['low'].values[-period:]
        closes = df['close'].values[-period:]
        
        # True Range
        # Alignment: 
        # range1 = high[1:] - low[1:] (current high - current low)
        # range2 = abs(high[1:] - close[:-1]) (current high - prev close)
        # range3 = abs(low[1:] - close[:-1]) (current low - prev close)
        # All result in N-1 elements
        
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        
        atr = np.mean(tr)
        avg_price = np.mean(closes)
        
        return atr / avg_price if avg_price > 0 else 0.02
    
    def get_regime(self, df: pd.DataFrame) -> str:
        """Detect volatility regime"""
        vol = self.calculate_volatility(df)
        
        if vol < self.low_vol_threshold:
            return 'low'
        elif vol > self.high_vol_threshold:
            return 'high'
        else:
            return 'normal'
    
    def get_stop_params(self, df: pd.DataFrame, atr: float) -> Dict[str, float]:
        """
        Get stop loss and take profit multipliers based on volatility regime.
        
        Returns:
            {'sl_multiplier': float, 'tp_multiplier': float}
        """
        regime = self.get_regime(df)
        
        params = {
            # LOW volatility: Tight stops, reasonable targets (2.5:1 R:R)
            'low': {'sl_multiplier': 0.8, 'tp_multiplier': 2.0},
            # NORMAL volatility: Moderate stops, good targets (2.5:1 R:R)
            'normal': {'sl_multiplier': 1.0, 'tp_multiplier': 2.5},
            # HIGH volatility: Wider stops but much wider targets (2.5:1 R:R)
            'high': {'sl_multiplier': 1.5, 'tp_multiplier': 3.75}
        }
        
        return params.get(regime, params['normal'])


# ============ PARTIAL PROFIT TAKING ============

class PartialProfitManager:
    """
    Manages partial profit taking at key milestones.
    Scale out of position to lock in profits while letting winners run.
    """
    
    def __init__(
        self,
        first_target_rr: float = 1.0,   # Take first profit at 1:1
        second_target_rr: float = 2.0,  # Take second profit at 2:1
        first_exit_pct: float = 0.33,   # Exit 33% at first target
        second_exit_pct: float = 0.33   # Exit another 33% at second target
    ):
        self.first_target_rr = first_target_rr
        self.second_target_rr = second_target_rr
        self.first_exit_pct = first_exit_pct
        self.second_exit_pct = second_exit_pct
    
    def check_partial_exit(
        self,
        position: Dict,
        current_price: float
    ) -> Optional[Dict]:
        """
        Check if a partial exit should be taken.
        
        Returns:
            None or {'exit_pct': float, 'reason': str, 'move_stop_to': float}
        """
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        is_long = position['side'] == 'LONG'
        
        # Calculate risk
        risk = abs(entry_price - stop_loss)
        
        # Calculate current profit in risk units
        if is_long:
            profit = current_price - entry_price
        else:
            profit = entry_price - current_price
        
        rr_achieved = profit / risk if risk > 0 else 0
        
        # Check if already took profits
        first_taken = position.get('first_profit_taken', False)
        second_taken = position.get('second_profit_taken', False)
        
        # First target hit
        if not first_taken and rr_achieved >= self.first_target_rr:
            # Move stop to breakeven
            new_stop = entry_price if is_long else entry_price
            
            return {
                'exit_pct': self.first_exit_pct,
                'reason': f'First target hit ({self.first_target_rr}:1 R:R)',
                'move_stop_to': new_stop,
                'mark': 'first_profit_taken'
            }
        
        # Second target hit
        if first_taken and not second_taken and rr_achieved >= self.second_target_rr:
            # Trail stop more aggressively
            if is_long:
                new_stop = entry_price + risk * 0.5  # Lock in 0.5R
            else:
                new_stop = entry_price - risk * 0.5
            
            return {
                'exit_pct': self.second_exit_pct,
                'reason': f'Second target hit ({self.second_target_rr}:1 R:R)',
                'move_stop_to': new_stop,
                'mark': 'second_profit_taken'
            }
        
        return None


# ============ PORTFOLIO RISK MANAGEMENT ============

class PortfolioRiskManager:
    """
    Portfolio-level risk controls to prevent catastrophic losses.
    """
    
    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,      # 3% max daily loss
        max_weekly_drawdown_pct: float = 0.07, # 7% max weekly drawdown
        max_concurrent_positions: int = 3,
        max_correlation_exposure: float = 0.8  # Don't hold highly correlated positions
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_drawdown_pct = max_weekly_drawdown_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.max_correlation_exposure = max_correlation_exposure
        
        # Track daily/weekly P&L
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.last_daily_reset = datetime.now().date()
        self.last_weekly_reset = datetime.now().isocalendar()[1]
        self.trades_today = 0
    
    def reset_if_needed(self):
        """Reset daily/weekly counters if needed"""
        today = datetime.now().date()
        week = datetime.now().isocalendar()[1]
        
        if today != self.last_daily_reset:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.last_daily_reset = today
        
        if week != self.last_weekly_reset:
            self.weekly_pnl = 0.0
            self.last_weekly_reset = week
    
    def record_trade(self, pnl: float):
        """Record a completed trade"""
        self.reset_if_needed()
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.trades_today += 1
    
    def can_open_trade(self, equity: float, current_positions: int = 0) -> Tuple[bool, str]:
        """
        Check if allowed to open a new trade based on risk limits.
        
        Returns:
            (can_trade, reason)
        """
        self.reset_if_needed()
        
        # Check position limit
        if current_positions >= self.max_concurrent_positions:
            return False, f"Max positions reached ({self.max_concurrent_positions})"
        
        # Check daily loss limit
        daily_loss_limit = equity * self.max_daily_loss_pct
        if self.daily_pnl < -daily_loss_limit:
            return False, f"Daily loss limit reached (${abs(self.daily_pnl):.2f} > ${daily_loss_limit:.2f})"
        
        # Check weekly drawdown
        weekly_loss_limit = equity * self.max_weekly_drawdown_pct
        if self.weekly_pnl < -weekly_loss_limit:
            return False, f"Weekly loss limit reached (${abs(self.weekly_pnl):.2f} > ${weekly_loss_limit:.2f})"
        
        return True, "Risk limits OK"
    
    def adjust_size_for_risk(self, base_size: float, equity: float) -> float:
        """
        Reduce position size if approaching risk limits.
        """
        self.reset_if_needed()
        
        # Calculate how much of daily limit is used
        daily_limit = equity * self.max_daily_loss_pct
        daily_usage = abs(self.daily_pnl) / daily_limit if daily_limit > 0 else 0
        
        if daily_usage > 0.5:
            # Reduce size as approaching limit
            reduction = 1.0 - (daily_usage - 0.5)  # 50% to 100% usage -> 100% to 0% size
            return base_size * max(0.25, reduction)  # Minimum 25% of base size
        
        return base_size


# ============ ORDER BOOK ANALYSIS ============

class OrderBookAnalyzer:
    """
    Analyze order book imbalance for entry confirmation.
    """
    
    def __init__(self, depth_levels: int = 10, imbalance_threshold: float = 0.3):
        self.depth_levels = depth_levels
        self.imbalance_threshold = imbalance_threshold
    
    def analyze_imbalance(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Dict:
        """
        Analyze order book imbalance.
        
        Args:
            bids: List of (price, volume) tuples
            asks: List of (price, volume) tuples
        
        Returns:
            {'imbalance': float, 'signal': str, 'confidence': float}
        """
        if not bids or not asks:
            return {'imbalance': 0, 'signal': 'neutral', 'confidence': 0}
        
        # Sum volume at top N levels
        bid_volume = sum(b[1] for b in bids[:self.depth_levels])
        ask_volume = sum(a[1] for a in asks[:self.depth_levels])
        
        total = bid_volume + ask_volume
        if total == 0:
            return {'imbalance': 0, 'signal': 'neutral', 'confidence': 0}
        
        imbalance = (bid_volume - ask_volume) / total
        
        if imbalance > self.imbalance_threshold:
            return {
                'imbalance': imbalance,
                'signal': 'bullish',
                'confidence': min(abs(imbalance), 1.0)
            }
        elif imbalance < -self.imbalance_threshold:
            return {
                'imbalance': imbalance,
                'signal': 'bearish',
                'confidence': min(abs(imbalance), 1.0)
            }
        else:
            return {
                'imbalance': imbalance,
                'signal': 'neutral',
                'confidence': 0.5
            }


# ============ UNIFIED FEATURE MANAGER ============

class AdvancedTradingFeatures:
    """
    Unified manager for all advanced trading features.
    Coordinates between multi-timeframe, session, volatility, and risk management.
    """
    
    def __init__(
        self,
        enable_mtf: bool = True,
        enable_session: bool = True,
        enable_dynamic_stops: bool = True,
        enable_partial_profits: bool = True,
        enable_portfolio_risk: bool = True
    ):
        self.mtf = MultiTimeframeConfirmation() if enable_mtf else None
        self.session = SessionTrading(min_session_priority=0.5) if enable_session else None
        self.volatility = DynamicVolatilityStops() if enable_dynamic_stops else None
        self.partial = PartialProfitManager() if enable_partial_profits else None
        self.risk_manager = PortfolioRiskManager() if enable_portfolio_risk else None
        
        logger.info("Advanced Trading Features initialized:")
        logger.info(f"  Multi-Timeframe: {'[ON]' if enable_mtf else '[OFF]'}")
        logger.info(f"  Session Trading: {'[ON]' if enable_session else '[OFF]'}")
        logger.info(f"  Dynamic Stops: {'[ON]' if enable_dynamic_stops else '[OFF]'}")
        logger.info(f"  Partial Profits: {'[ON]' if enable_partial_profits else '[OFF]'}")
        logger.info(f"  Portfolio Risk: {'[ON]' if enable_portfolio_risk else '[OFF]'}")
    
    def validate_signal(
        self,
        action: str,
        df_5m: pd.DataFrame,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        equity: float = 10000,
        current_positions: int = 0,
        scalper_mode: bool = True  # Allow counter-trend trades in scalper mode
    ) -> Tuple[bool, List[str]]:
        """
        Run all validation checks on a trading signal.
        
        In scalper_mode, MTF is advisory-only (doesn't block trades).
        
        Returns:
            (is_valid, list_of_reasons)
        """
        reasons = []
        is_valid = True
        
        # 1. Multi-Timeframe Confirmation
        if self.mtf:
            # Only check if we have sufficient HTF data
            has_1h_data = df_1h is not None and len(df_1h) > 30
            has_4h_data = df_4h is not None and len(df_4h) > 20
            
            if has_1h_data or has_4h_data:
                confirmed, mtf_reason = self.mtf.confirm_signal(action, df_1h, df_4h)
                reasons.append(f"MTF: {mtf_reason}")
                
                # In scalper mode, MTF is advisory only
                if not confirmed and not scalper_mode:
                    is_valid = False
                elif not confirmed and scalper_mode:
                    reasons.append("SCALPER: Counter-trend trade allowed")
            else:
                reasons.append("MTF: No HTF data - skipping check")
        
        # 2. Session Check (always advisory)
        if self.session:
            can_trade, session_reason = self.session.should_trade()
            reasons.append(f"Session: {session_reason}")
        
        # 3. Portfolio Risk Check (always enforced)
        if self.risk_manager:
            can_open, risk_reason = self.risk_manager.can_open_trade(equity, current_positions)
            reasons.append(f"Risk: {risk_reason}")
            if not can_open:
                is_valid = False
        
        return is_valid, reasons
    
    def get_stop_params(self, df: pd.DataFrame, base_atr: float) -> Dict[str, float]:
        """Get volatility-adjusted stop parameters"""
        if self.volatility:
            return self.volatility.get_stop_params(df, base_atr)
        return {'sl_multiplier': 1.5, 'tp_multiplier': 2.5}
    
    def check_partial_exit(self, position: Dict, current_price: float) -> Optional[Dict]:
        """Check if partial profit should be taken"""
        if self.partial:
            return self.partial.check_partial_exit(position, current_price)
        return None
    
    def record_trade_result(self, pnl: float):
        """Record trade result for risk tracking"""
        if self.risk_manager:
            self.risk_manager.record_trade(pnl)
    
    def get_size_adjustment(self, base_size: float, equity: float) -> float:
        """Get adjusted position size based on risk and session"""
        size = base_size
        
        if self.risk_manager:
            size = self.risk_manager.adjust_size_for_risk(size, equity)
        
        if self.session:
            session_mult = self.session.get_session_multiplier()
            size *= session_mult
        
        return size
