"""
Professional Position Sizing & Risk/Reward Optimizer
Fixes the problem of small wins and large losses

Key Features:
- Minimum 2:1 risk/reward ratio enforcement
- Dynamic position sizing based on risk
- Proper stop loss and take profit calculation
- Win/loss ratio optimization
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RiskRewardOptimizer:
    """
    Professional risk/reward management
    
    Ensures trades have favorable risk/reward ratios
    """
    
    def __init__(self, 
                 min_risk_reward_ratio: float = 2.5,
                 max_risk_per_trade_pct: float = 0.5,
                 trailing_stop_activation: float = 0.75):
        """
        Initialize risk/reward optimizer
        
        Args:
            min_risk_reward_ratio: Minimum reward/risk (2.0 = $2 profit for $1 risk)
            max_risk_per_trade_pct: Max % of equity to risk per trade
            trailing_stop_activation: Activate trailing stop at X times initial risk
        """
        self.min_rr_ratio = min_risk_reward_ratio
        self.max_risk_pct = max_risk_per_trade_pct
        self.trailing_activation = trailing_stop_activation
        
        logger.info(f"✓ Risk/Reward Optimizer initialized: Min R:R = {min_risk_reward_ratio}:1")
    
    def calculate_position_size(self, 
                                equity: float,
                                entry_price: float,
                                stop_loss_price: float,
                                max_position_value: float = None) -> Dict:
        """
        Calculate optimal position size based on risk
        
        This is CRITICAL for proper risk management
        
        Args:
            equity: Current account equity
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            max_position_value: Optional max position value
            
        Returns:
            Dict with position size and details
        """
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            return {'status': 'error', 'message': 'Invalid stop loss'}
        
        # Max dollar risk
        max_dollar_risk = equity * (self.max_risk_pct / 100)
        
        # Position size = Max Risk / Risk Per Unit
        position_size = max_dollar_risk / risk_per_unit
        
        # Cap by max position value if specified
        if max_position_value:
            max_size = max_position_value / entry_price
            position_size = min(position_size, max_size)
        
        # Calculate actual dollar risk
        position_value = position_size * entry_price
        dollar_risk = position_size * risk_per_unit
        risk_pct = (dollar_risk / equity) * 100
        
        return {
            'position_size': round(position_size, 6),
            'position_value': round(position_value, 2),
            'dollar_risk': round(dollar_risk, 2),
            'risk_pct': round(risk_pct, 2),
            'risk_per_unit': round(risk_per_unit, 2),
            'status': 'ok'
        }
    
    def calculate_stop_and_target(self,
                                  entry_price: float,
                                  side: str,
                                  atr: float,
                                  rr_ratio: float = None) -> Dict:
        """
        Calculate professional stop loss and take profit levels
        
        Uses ATR (Average True Range) for volatility-based stops
        
        Args:
            entry_price: Entry price
            side: 'LONG' or 'SHORT'
            atr: Average True Range (volatility measure)
            rr_ratio: Risk/reward ratio (uses min if None)
            
        Returns:
            Dict with stop and target levels
        """
        rr = rr_ratio or self.min_rr_ratio
        
        # Stop loss: 1.0 * ATR from entry (TIGHTER stops to reduce loss size)
        stop_multiplier = 1.0
        
        if side == 'LONG':
            stop_loss = entry_price - (atr * stop_multiplier)
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * rr)
        else:  # SHORT
            stop_loss = entry_price + (atr * stop_multiplier)
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * rr)
        
        return {
            'entry': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk': round(risk, 2),
            'reward': round(risk * rr, 2),
            'rr_ratio': rr,
            'status': 'ok'
        }
    
    def validate_trade(self, trade_plan: Dict) -> Tuple[bool, str]:
        """
        Validate trade meets minimum risk/reward requirements
        
        Args:
            trade_plan: Dict with entry, stop_loss, take_profit
            
        Returns:
            (is_valid, reason)
        """
        entry = trade_plan.get('entry', 0)
        stop = trade_plan.get('stop_loss', 0)
        target = trade_plan.get('take_profit', 0)
        
        if entry == 0 or stop == 0 or target == 0:
            return False, "Missing price levels"
        
        # Calculate actual R:R
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if risk == 0:
            return False, "Zero risk (invalid stop)"
        
        actual_rr = reward / risk
        
        if actual_rr < self.min_rr_ratio:
            return False, f"R:R {actual_rr:.2f}:1 < minimum {self.min_rr_ratio}:1"
        
        return True, f"✓ R:R = {actual_rr:.2f}:1"
    
    def calculate_trailing_stop(self,
                                entry_price: float,
                                current_price: float,
                                initial_stop: float,
                                side: str,
                                atr: float) -> Dict:
        """
        Calculate trailing stop based on profit
        
        Locks in profits as price moves favorably
        
        Args:
            entry_price: Original entry
            current_price: Current market price
            initial_stop: Initial stop loss
            side: 'LONG' or 'SHORT'
            atr: Current ATR
            
        Returns:
            Dict with new stop level
        """
        initial_risk = abs(entry_price - initial_stop)
        current_profit = 0
        
        if side == 'LONG':
            current_profit = current_price - entry_price
            
            # Activate trailing stop if profit > 1.5x initial risk
            if current_profit >= (initial_risk * self.trailing_activation):
                # Trail stop at ATR below current price
                new_stop = current_price - atr
                # But never lower than breakeven
                new_stop = max(new_stop, entry_price)
                
                return {
                    'trailing_active': True,
                    'new_stop': round(new_stop, 2),
                    'locked_profit': round(new_stop - entry_price, 2),
                    'status': 'trailing'
                }
        
        else:  # SHORT
            current_profit = entry_price - current_price
            
            if current_profit >= (initial_risk * self.trailing_activation):
                new_stop = current_price + atr
                new_stop = min(new_stop, entry_price)
                
                return {
                    'trailing_active': True,
                    'new_stop': round(new_stop, 2),
                    'locked_profit': round(entry_price - new_stop, 2),
                    'status': 'trailing'
                }
        
        return {
            'trailing_active': False,
            'new_stop': initial_stop,
            'status': 'initial_stop'
        }


class WinLossOptimizer:
    """
    Analyze and optimize win/loss patterns
    
    Identifies why losses are larger than wins
    """
    
    def __init__(self):
        """Initialize win/loss optimizer"""
        self.trades = []
        
    def add_trade(self, pnl: float, exit_reason: str):
        """Record a trade"""
        self.trades.append({
            'pnl': pnl,
            'exit_reason': exit_reason,
            'is_winner': pnl > 0
        })
    
    def analyze_patterns(self) -> Dict:
        """
        Analyze win/loss patterns to identify issues
        
        Returns:
            Analysis with recommendations
        """
        if len(self.trades) < 10:
            return {'status': 'insufficient_data'}
        
        winners = [t for t in self.trades if t['is_winner']]
        losers = [t for t in self.trades if not t['is_winner']]
        
        if not winners or not losers:
            return {'status': 'no_variation'}
        
        avg_win = np.mean([t['pnl'] for t in winners])
        avg_loss = abs(np.mean([t['pnl'] for t in losers]))
        
        win_rate = len(winners) / len(self.trades)
        
        # Profit factor
        total_wins = sum(t['pnl'] for t in winners)
        total_losses = abs(sum(t['pnl'] for t in losers))
        profit_factor = total_wins / total_losses if total_losses > 0 else 999
        
        # Analyze problems
        problems = []
        recommendations = []
        
        # Problem 1: Average loss > Average win
        if avg_loss > avg_win:
            problems.append(f"⚠️ Avg loss (${avg_loss:.2f}) > Avg win (${avg_win:.2f})")
            recommendations.append("→ Widen take profits or tighten stop losses")
            recommendations.append("→ Use trailing stops to capture more profit")
        
        # Problem 2: Low win rate
        if win_rate < 0.4:
            problems.append(f"⚠️ Low win rate: {win_rate*100:.1f}%")
            recommendations.append("→ Improve entry timing/signals")
            recommendations.append("→ Use stronger trend confirmation")
        
        # Problem 3: Profit factor < 1.5
        if profit_factor < 1.5:
            problems.append(f"⚠️ Low profit factor: {profit_factor:.2f}")
            recommendations.append("→ Aim for minimum 2:1 risk/reward ratio")
        
        return {
            'total_trades': len(self.trades),
            'win_rate': round(win_rate * 100, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'win_loss_ratio': round(avg_win / avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'problems': problems,
            'recommendations': recommendations,
            'status': 'ok'
        }


# Singletons
_rr_optimizer: Optional[RiskRewardOptimizer] = None
_wl_optimizer: Optional[WinLossOptimizer] = None


def get_rr_optimizer() -> RiskRewardOptimizer:
    global _rr_optimizer
    if _rr_optimizer is None:
        _rr_optimizer = RiskRewardOptimizer(
            min_risk_reward_ratio=2.5,  # Minimum 2.5:1 reward/risk (IMPROVED)
            max_risk_per_trade_pct=0.5,  # Risk max 0.5% per trade (TIGHTER)
            trailing_stop_activation=0.75  # Activate trailing at 0.75x (EARLIER)
        )
    return _rr_optimizer


def get_wl_optimizer() -> WinLossOptimizer:
    global _wl_optimizer
    if _wl_optimizer is None:
        _wl_optimizer = WinLossOptimizer()
    return _wl_optimizer


if __name__ == '__main__':
    print("=" * 70)
    print("RISK/REWARD OPTIMIZATION - TEST")
    print("=" * 70)
    
    # Test risk/reward optimizer
    print("\n💰 Testing Position Sizing...")
    rr_opt = RiskRewardOptimizer(min_risk_reward_ratio=2.0)
    
    # Example: BTC trade
    equity = 10000
    entry = 50000
    stop = 49500  # $500 risk
    
    pos_size = rr_opt.calculate_position_size(equity, entry, stop)
    print(f"  Equity: ${equity:,.0f}")
    print(f"  Entry: ${entry:,.0f}, Stop: ${stop:,.0f}")
    print(f"  Position size: {pos_size['position_size']} BTC")
    print(f"  Position value: ${pos_size['position_value']:,.2f}")
    print(f"  Dollar risk: ${pos_size['dollar_risk']:.2f} ({pos_size['risk_pct']:.1f}%)")
    
    # Test stop/target calculation
    print("\n🎯 Testing Stop & Target Calculation...")
    atr = 200  # $200 ATR
    levels = rr_opt.calculate_stop_and_target(entry, 'LONG', atr, rr_ratio=2.5)
    print(f"  Entry: ${levels['entry']:,.0f}")
    print(f"  Stop Loss: ${levels['stop_loss']:,.0f}")
    print(f"  Take Profit: ${levels['take_profit']:,.0f}")
    print(f"  Risk: ${levels['risk']:.0f}")
    print(f"  Reward: ${levels['reward']:.0f}")
    print(f"  R:R Ratio: {levels['rr_ratio']}:1")
    
    # Validate trade
    is_valid, reason = rr_opt.validate_trade(levels)
    print(f"  Valid: {is_valid} - {reason}")
    
    # Test trailing stop
    print("\n📈 Testing Trailing Stop...")
    current_price = 51000  # Price moved up $1000
    trailing = rr_opt.calculate_trailing_stop(entry, current_price, levels['stop_loss'], 'LONG', atr)
    print(f"  Current: ${current_price:,.0f}")
    print(f"  Trailing active: {trailing['trailing_active']}")
    if trailing['trailing_active']:
        print(f"  New stop: ${trailing['new_stop']:,.0f}")
        print(f"  Locked profit: ${trailing['locked_profit']:.0f}")
    
    # Test win/loss analyzer
    print("\n📊 Testing Win/Loss Analyzer...")
    wl_opt = WinLossOptimizer()
    
    # Simulate the PROBLEM: small wins, large losses
    import random
    for i in range(30):
        if random.random() < 0.6:  # 60% winners
            pnl = random.uniform(10, 50)  # Small wins
            wl_opt.add_trade(pnl, 'take_profit')
        else:
            pnl = random.uniform(-100, -200)  # Large losses
            wl_opt.add_trade(pnl, 'stop_loss')
    
    analysis = wl_opt.analyze_patterns()
    print(f"\n  Total trades: {analysis['total_trades']}")
    print(f"  Win rate: {analysis['win_rate']}%")
    print(f"  Avg win: ${analysis['avg_win']:.2f}")
    print(f"  Avg loss: ${analysis['avg_loss']:.2f}")
    print(f"  Win/Loss ratio: {analysis['win_loss_ratio']:.2f}")
    print(f"  Profit factor: {analysis['profit_factor']:.2f}")
    
    print(f"\n  Problems detected:")
    for problem in analysis['problems']:
        print(f"    {problem}")
    
    print(f"\n  Recommendations:")
    for rec in analysis['recommendations']:
        print(f"    {rec}")
    
    print("\n" + "=" * 70)
    print("✅ Risk/Reward optimization working!")
    print("=" * 70)
