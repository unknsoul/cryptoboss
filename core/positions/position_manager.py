"""
Position Management - Enterprise Features #115, #118, #121, #125
Position Aggregation, Cross-Margin, Leverage Optimization, Position Sizing.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class MarginMode(Enum):
    ISOLATED = "isolated"
    CROSS = "cross"


class PositionAggregator:
    """
    Feature #115: Position Aggregator
    
    Aggregates positions across accounts and exchanges.
    """
    
    def __init__(self):
        """Initialize position aggregator."""
        self.positions: Dict[str, Dict[str, Dict]] = {}  # exchange -> symbol -> position
        
        logger.info("Position Aggregator initialized")
    
    def add_position(
        self,
        exchange: str,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        leverage: float = 1.0
    ):
        """Add or update a position."""
        if exchange not in self.positions:
            self.positions[exchange] = {}
        
        self.positions[exchange][symbol] = {
            'symbol': symbol,
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'leverage': leverage,
            'notional': size * entry_price,
            'margin': size * entry_price / leverage,
            'updated_at': datetime.now().isoformat()
        }
    
    def get_position(self, exchange: str, symbol: str) -> Optional[Dict]:
        """Get specific position."""
        return self.positions.get(exchange, {}).get(symbol)
    
    def get_aggregate_by_symbol(self, symbol: str) -> Dict:
        """Get aggregated position across all exchanges."""
        total_long = 0
        total_short = 0
        total_notional = 0
        positions = []
        
        for exchange, symbols in self.positions.items():
            if symbol in symbols:
                pos = symbols[symbol]
                positions.append({'exchange': exchange, **pos})
                
                if pos['side'] == 'LONG':
                    total_long += pos['size']
                else:
                    total_short += pos['size']
                total_notional += pos['notional']
        
        net_size = total_long - total_short
        
        return {
            'symbol': symbol,
            'net_size': round(net_size, 6),
            'net_side': 'LONG' if net_size > 0 else 'SHORT' if net_size < 0 else 'FLAT',
            'total_notional': round(total_notional, 2),
            'total_long': total_long,
            'total_short': total_short,
            'positions': positions
        }
    
    def get_total_exposure(self) -> Dict:
        """Get total portfolio exposure."""
        total_long = 0
        total_short = 0
        by_exchange = {}
        
        for exchange, symbols in self.positions.items():
            ex_long = 0
            ex_short = 0
            
            for symbol, pos in symbols.items():
                if pos['side'] == 'LONG':
                    ex_long += pos['notional']
                    total_long += pos['notional']
                else:
                    ex_short += pos['notional']
                    total_short += pos['notional']
            
            by_exchange[exchange] = {
                'long': round(ex_long, 2),
                'short': round(ex_short, 2),
                'net': round(ex_long - ex_short, 2)
            }
        
        return {
            'total_long': round(total_long, 2),
            'total_short': round(total_short, 2),
            'net_exposure': round(total_long - total_short, 2),
            'gross_exposure': round(total_long + total_short, 2),
            'by_exchange': by_exchange
        }


class CrossMarginManager:
    """
    Feature #118: Cross-Margin Manager
    
    Manages cross-margin positions and calculations.
    """
    
    def __init__(self, total_margin: float = 10000):
        """
        Initialize cross-margin manager.
        
        Args:
            total_margin: Total margin available
        """
        self.total_margin = total_margin
        self.positions: List[Dict] = []
        self.mode = MarginMode.CROSS
        
        logger.info(f"Cross-Margin Manager initialized - ${total_margin:,.2f}")
    
    def set_margin(self, amount: float):
        """Set total margin amount."""
        self.total_margin = amount
    
    def add_position(self, symbol: str, size: float, entry: float, leverage: float):
        """Add a cross-margin position."""
        notional = size * entry
        margin_used = notional / leverage
        
        self.positions.append({
            'symbol': symbol,
            'size': size,
            'entry_price': entry,
            'leverage': leverage,
            'notional': notional,
            'margin_used': margin_used,
            'current_pnl': 0
        })
    
    def update_pnl(self, symbol: str, current_price: float):
        """Update position P&L."""
        for pos in self.positions:
            if pos['symbol'] == symbol:
                pnl = (current_price - pos['entry_price']) * pos['size']
                pos['current_pnl'] = pnl
                pos['current_price'] = current_price
    
    def get_margin_status(self) -> Dict:
        """Get current margin status."""
        total_used = sum(p['margin_used'] for p in self.positions)
        total_pnl = sum(p['current_pnl'] for p in self.positions)
        effective_margin = self.total_margin + total_pnl
        
        margin_ratio = effective_margin / total_used if total_used > 0 else float('inf')
        
        return {
            'total_margin': round(self.total_margin, 2),
            'margin_used': round(total_used, 2),
            'margin_available': round(self.total_margin - total_used + total_pnl, 2),
            'unrealized_pnl': round(total_pnl, 2),
            'effective_margin': round(effective_margin, 2),
            'margin_ratio': round(margin_ratio * 100, 1),
            'margin_call_risk': margin_ratio < 0.3,
            'liquidation_risk': margin_ratio < 0.15
        }
    
    def can_add_position(self, notional: float, leverage: float) -> bool:
        """Check if new position can be added."""
        margin_needed = notional / leverage
        status = self.get_margin_status()
        return margin_needed <= status['margin_available']


class LeverageOptimizer:
    """
    Feature #121: Leverage Optimizer
    
    Optimizes leverage based on risk parameters.
    """
    
    def __init__(
        self,
        max_leverage: float = 10.0,
        target_risk_pct: float = 2.0
    ):
        """
        Initialize leverage optimizer.
        
        Args:
            max_leverage: Maximum allowed leverage
            target_risk_pct: Target risk per trade as % of equity
        """
        self.max_leverage = max_leverage
        self.target_risk_pct = target_risk_pct
        
        logger.info(f"Leverage Optimizer initialized - Max: {max_leverage}x")
    
    def calculate_optimal_leverage(
        self,
        volatility: float,
        win_rate: float,
        risk_reward: float
    ) -> Dict:
        """
        Calculate optimal leverage based on Kelly criterion.
        
        Args:
            volatility: Asset volatility (decimal)
            win_rate: Historical win rate (decimal)
            risk_reward: Average risk/reward ratio
            
        Returns:
            Optimal leverage recommendation
        """
        # Kelly fraction
        q = 1 - win_rate
        kelly = (win_rate * risk_reward - q) / risk_reward if risk_reward > 0 else 0
        
        # Volatility-adjusted leverage
        vol_adjusted = 1 / (volatility * 2)
        
        # Combine with safety margin
        raw_leverage = min(kelly * 2, vol_adjusted) * 0.5  # Half-Kelly for safety
        optimal = min(max(raw_leverage, 1.0), self.max_leverage)
        
        return {
            'optimal_leverage': round(optimal, 2),
            'kelly_fraction': round(kelly, 3),
            'volatility_adjusted': round(vol_adjusted, 2),
            'max_allowed': self.max_leverage,
            'recommendation': self._get_recommendation(optimal)
        }
    
    def _get_recommendation(self, leverage: float) -> str:
        if leverage <= 2:
            return 'Conservative - Low leverage recommended'
        elif leverage <= 5:
            return 'Moderate - Standard leverage'
        elif leverage <= 10:
            return 'Aggressive - High leverage, manage risk carefully'
        else:
            return 'Extreme - Maximum leverage, high risk'
    
    def adjust_for_market_conditions(
        self,
        base_leverage: float,
        volatility_ratio: float,
        trend_strength: float
    ) -> float:
        """Adjust leverage for current market conditions."""
        # Reduce in high volatility
        vol_adj = 1 - min(0.5, max(0, volatility_ratio - 1) * 0.3)
        
        # Slight increase in strong trends
        trend_adj = 1 + min(0.2, trend_strength * 0.1)
        
        adjusted = base_leverage * vol_adj * trend_adj
        return round(min(adjusted, self.max_leverage), 2)


class PositionSizingEngine:
    """
    Feature #125: Position Sizing Engine
    
    Calculates optimal position sizes using multiple methods.
    """
    
    def __init__(self, default_risk_pct: float = 2.0):
        """
        Initialize position sizing engine.
        
        Args:
            default_risk_pct: Default risk per trade as % of equity
        """
        self.default_risk_pct = default_risk_pct
        
        logger.info(f"Position Sizing Engine initialized - Risk: {default_risk_pct}%")
    
    def fixed_fractional(
        self,
        equity: float,
        risk_pct: Optional[float] = None,
        entry_price: float = 1.0,
        stop_loss: float = 1.0
    ) -> Dict:
        """
        Fixed fractional position sizing.
        
        Args:
            equity: Account equity
            risk_pct: Risk percentage
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Position size calculation
        """
        risk = risk_pct or self.default_risk_pct
        risk_amount = equity * (risk / 100)
        
        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            return {'size': 0, 'error': 'Invalid stop loss'}
        
        size = risk_amount / price_risk
        notional = size * entry_price
        
        return {
            'method': 'fixed_fractional',
            'size': round(size, 6),
            'notional': round(notional, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_pct': risk,
            'max_loss': round(risk_amount, 2)
        }
    
    def volatility_based(
        self,
        equity: float,
        atr: float,
        atr_multiple: float = 2.0,
        entry_price: float = 1.0
    ) -> Dict:
        """
        Volatility-based position sizing using ATR.
        
        Args:
            equity: Account equity
            atr: Average True Range
            atr_multiple: ATR multiplier for stop
            entry_price: Entry price
        """
        risk_amount = equity * (self.default_risk_pct / 100)
        stop_distance = atr * atr_multiple
        
        if stop_distance == 0:
            return {'size': 0, 'error': 'Invalid ATR'}
        
        size = risk_amount / stop_distance
        notional = size * entry_price
        
        return {
            'method': 'volatility_based',
            'size': round(size, 6),
            'notional': round(notional, 2),
            'atr': atr,
            'stop_distance': round(stop_distance, 2),
            'risk_amount': round(risk_amount, 2)
        }
    
    def kelly_based(
        self,
        equity: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        entry_price: float = 1.0
    ) -> Dict:
        """
        Kelly criterion position sizing.
        
        Args:
            equity: Account equity
            win_rate: Win rate (decimal)
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive number)
        """
        if avg_loss == 0:
            return {'size': 0, 'error': 'Invalid avg_loss'}
        
        rr = avg_win / avg_loss
        kelly = (win_rate * rr - (1 - win_rate)) / rr
        
        # Half-Kelly for safety
        half_kelly = max(0, kelly * 0.5)
        
        position_value = equity * half_kelly
        size = position_value / entry_price
        
        return {
            'method': 'kelly',
            'size': round(size, 6),
            'notional': round(position_value, 2),
            'kelly_fraction': round(kelly, 3),
            'half_kelly': round(half_kelly, 3),
            'risk_pct': round(half_kelly * 100, 1)
        }
    
    def optimal_f(
        self,
        equity: float,
        trade_history: List[float],
        entry_price: float = 1.0
    ) -> Dict:
        """
        Optimal f position sizing.
        
        Args:
            equity: Account equity
            trade_history: List of trade returns
            entry_price: Entry price
        """
        if not trade_history or max(trade_history) <= 0:
            return {'size': 0, 'error': 'Insufficient trade history'}
        
        biggest_loss = abs(min(trade_history))
        if biggest_loss == 0:
            biggest_loss = 1  # Prevent division by zero
        
        # Find optimal f by testing
        best_f = 0
        best_twp = 0
        
        for f in [i / 100 for i in range(1, 51)]:  # 0.01 to 0.50
            twp = 1.0
            for trade in trade_history:
                hpr = 1 + f * (-trade / biggest_loss)
                if hpr <= 0:
                    twp = 0
                    break
                twp *= hpr
            
            if twp > best_twp:
                best_twp = twp
                best_f = f
        
        position_value = equity * best_f
        size = position_value / entry_price
        
        return {
            'method': 'optimal_f',
            'size': round(size, 6),
            'notional': round(position_value, 2),
            'optimal_f': round(best_f, 3),
            'terminal_wealth': round(best_twp, 4)
        }


# Singletons
_aggregator: Optional[PositionAggregator] = None
_cross_margin: Optional[CrossMarginManager] = None
_leverage_opt: Optional[LeverageOptimizer] = None
_sizing: Optional[PositionSizingEngine] = None


def get_position_aggregator() -> PositionAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = PositionAggregator()
    return _aggregator


def get_cross_margin() -> CrossMarginManager:
    global _cross_margin
    if _cross_margin is None:
        _cross_margin = CrossMarginManager()
    return _cross_margin


def get_leverage_optimizer() -> LeverageOptimizer:
    global _leverage_opt
    if _leverage_opt is None:
        _leverage_opt = LeverageOptimizer()
    return _leverage_opt


def get_position_sizer() -> PositionSizingEngine:
    global _sizing
    if _sizing is None:
        _sizing = PositionSizingEngine()
    return _sizing


if __name__ == '__main__':
    # Test position aggregator
    agg = PositionAggregator()
    agg.add_position('binance', 'BTCUSDT', 'LONG', 0.5, 50000, leverage=5)
    agg.add_position('coinbase', 'BTCUSDT', 'LONG', 0.3, 50100, leverage=3)
    print(f"Aggregate: {agg.get_aggregate_by_symbol('BTCUSDT')}")
    
    # Test leverage optimizer
    lev = LeverageOptimizer()
    result = lev.calculate_optimal_leverage(volatility=0.02, win_rate=0.55, risk_reward=2.0)
    print(f"Optimal leverage: {result}")
    
    # Test position sizing
    sizer = PositionSizingEngine()
    size = sizer.fixed_fractional(equity=10000, entry_price=50000, stop_loss=49000)
    print(f"Position size: {size}")
