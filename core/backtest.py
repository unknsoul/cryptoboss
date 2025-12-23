"""
Professional Backtest Engine with Advanced Features
- Execution realism (slippage + fees)
- Multi-strategy support
- Enhanced metrics
- Risk controls
"""

import numpy as np
from typing import List, Dict, Optional, Any


class EnhancedBacktest:
    """
    Professional backtest engine with institutional-grade features:
    - Realistic execution (slippage, fees)
    - Multi-strategy position management
    - Advanced risk controls
    - Comprehensive performance metrics
    """
    
    def __init__(self, 
                 capital=10000,
                 risk_per_trade=0.02,
                 fee=0.001,
                 slippage=0.0005,
                 max_drawdown_limit=0.20,
                 daily_loss_limit=0.05,
                 cooldown_after_losses=3,
                 # Tier 1 improvements
                 use_partial_profits=True,
                 partial_profit_1r=0.25,
                 partial_profit_2r=0.50,
                 use_breakeven_stop=True,
                 max_hold_hours=48):
        
        self.initial_capital = capital
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.fee = fee
        self.slippage = slippage
        
        # Risk controls
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_loss_limit = daily_loss_limit
        self.cooldown_after_losses = cooldown_after_losses
        
        # Tracking
        self.position = None
        self.equity_history = []
        self.trades = []
        self.peak_equity = capital
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        self.trading_halted = False
        
        # Daily tracking
        self.daily_start_capital = capital
        self.current_day = None
        
        # Tier 1 improvements - Partial profits and time-based exits
        self.use_partial_profits = use_partial_profits
        self.partial_profit_1r = partial_profit_1r  # Take 25% at 1R
        self.partial_profit_2r = partial_profit_2r  # Take 50% more at 2R
        self.use_breakeven_stop = use_breakeven_stop
        self.max_hold_hours = max_hold_hours
    
    def _check_risk_limits(self, current_equity, timestamp=None):
        """Check if risk limits are breached"""
        
        # Max drawdown check
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        current_dd = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        if current_dd >= self.max_drawdown_limit:
            self.trading_halted = True
            print(f"⚠️  TRADING HALTED: Max drawdown {current_dd:.2%} exceeded limit {self.max_drawdown_limit:.2%}")
            return False
        
        # Daily loss limit check (if timestamp provided)
        if timestamp is not None and self.current_day is not None:
            day = timestamp.date() if hasattr(timestamp, 'date') else timestamp
            if day != self.current_day:
                # New day
                self.current_day = day
                self.daily_start_capital = current_equity
            else:
                # Check daily loss
                daily_loss = (self.daily_start_capital - current_equity) / self.daily_start_capital if self.daily_start_capital > 0 else 0
                if daily_loss >= self.daily_loss_limit:
                    print(f"⚠️  Daily loss limit reached: {daily_loss:.2%}")
                    return False
        
        return True
    
    def _calculate_position_size(self, signal, current_price):
        """
        Calculate position size based on risk (Tier 1: Enhanced with adaptive sizing)
        - Reduces size after consecutive losses
        - Reduces size during drawdown
        - Prevents overtrading in unfavorable conditions
        """
        stop_distance = signal['stop']
        
        if stop_distance <= 0:
            return 0
        
        # Risk amount in dollars
        risk_amount = self.capital * self.risk_per_trade
        
        # Position size = Risk / Stop Distance
        position_size = risk_amount / stop_distance
        
        # Tier 1 Improvement: Adaptive scaling
        scale_factor = 1.0
        
        # Reduce after consecutive losses (50% reduction)
        if self.consecutive_losses >= 2:
            scale_factor *= 0.5
        
        # Reduce during significant drawdown (50% reduction)
        current_dd = (self.peak_equity - self.capital) / self.peak_equity if self.peak_equity > 0 else 0
        if current_dd > 0.10:  # More than 10% drawdown
            scale_factor *= 0.5
        
        # Apply adaptive scaling
        position_size *= scale_factor
        
        # Ensure we can afford it
        cost = position_size * current_price * (1 + self.fee + self.slippage)
        if cost > self.capital:
            position_size = self.capital / (current_price * (1 + self.fee + self.slippage))
        
        return position_size
    
    def _apply_slippage(self, price, side):
        """Apply slippage to execution price"""
        if side == 'LONG':
            # Buy at higher price
            return price * (1 + self.slippage)
        else:
            # Sell short at lower price
            return price * (1 - self.slippage)
    
    def run(self, highs, lows, closes, strategy, timestamps=None, volumes=None):
        """
        Run backtest with professional features
        
        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            strategy: Strategy instance with signal() and check_exit() methods
            timestamps: Optional timestamps for daily tracking
            volumes: Optional volume data
        
        Returns:
            equity_history: List of equity values per bar
        """
        self.equity_history = []
        self.trades = []
        self.position = None
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        self.trading_halted = False
        
        num_bars = len(closes)
        
        for i in range(num_bars):
            current_price = closes[i]
            current_timestamp = timestamps[i] if timestamps is not None else None
            
            # Calculate current equity
            if self.position is None:
                current_equity = self.capital
            else:
                # Mark-to-market
                if self.position['side'] == 'LONG':
                    unrealized_pnl = (current_price - self.position['entry_price']) * self.position['size']
                else:  # SHORT
                    unrealized_pnl = (self.position['entry_price'] - current_price) * self.position['size']
                
                current_equity = self.capital + unrealized_pnl
            
            self.equity_history.append(current_equity)
            
            # Check risk limits
            if not self._check_risk_limits(current_equity, current_timestamp):
                # Skip trading but continue tracking equity
                continue
            
            # Manage existing position
            if self.position is not None:
                # Check partial profit taking (Tier 1 improvement)
                if self.use_partial_profits:
                    self._check_partial_profits(current_price, i)
                
                # Check max duration (Tier 1 improvement)
                if self._check_max_duration(i):
                    self._close_position(current_price, i, 'max_duration')
                    continue
                # Update trailing stop
                if self.position['side'] == 'LONG':
                    # Calculate new stop (can only move up for longs)
                    new_stop = current_price - self.position['stop_distance']
                    if new_stop > self.position['stop_price']:
                        self.position['stop_price'] = new_stop
                    
                    # Check if stopped out
                    if lows[i] <= self.position['stop_price']:
                        # Use stop price for exit
                        exit_price = self.position['stop_price']
                        self._close_position(exit_price, i, 'stop_loss')
                        continue
                
                else:  # SHORT
                    # Calculate new stop (can only move down for shorts)
                    new_stop = current_price + self.position['stop_distance']
                    if new_stop < self.position['stop_price']:
                        self.position['stop_price'] = new_stop
                    
                    # Check if stopped out
                    if highs[i] >= self.position['stop_price']:
                        exit_price = self.position['stop_price']
                        self._close_position(exit_price, i, 'stop_loss')
                        continue
                
                # Check strategy exit
                should_exit = strategy.check_exit(
                    highs[:i+1], lows[:i+1], closes[:i+1],
                    self.position['side'],
                    self.position['entry_price'],
                    self.position['entry_bar'],
                    i
                )
                
                if should_exit:
                    self._close_position(current_price, i, 'strategy_exit')
                    continue
            
            # Check for new entry
            if self.position is None:
                # Cooldown check
                if self.cooldown_remaining > 0:
                    self.cooldown_remaining -= 1
                    continue
                
                # Get signal
                signal = strategy.signal(
                    highs[:i+1], 
                    lows[:i+1], 
                    closes[:i+1],
                    volumes[:i+1] if volumes is not None else None
                )
                
                if signal is not None:
                    self._open_position(current_price, i, signal)
        
        # Close any remaining position
        if self.position is not None:
            self._close_position(closes[-1], len(closes) - 1, 'end_of_data')
        
        return self.equity_history
    
    def _open_position(self, price, bar_index, signal):
        """Open a new position"""
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, price)
        
        if position_size <= 0:
            return
        
        # Apply slippage
        execution_price = self._apply_slippage(price, signal['action'])
        
        stop_distance = signal['stop']
        
        if signal['action'] == 'LONG':
            # Calculate cost with fees and slippage
            cost = position_size * execution_price * (1 + self.fee)
            
            if cost > self.capital:
                return  # Can't afford
            
            self.capital -= cost
            
            self.position = {
                'side': 'LONG',
                'size': position_size,
                'entry_price': execution_price,
                'stop_price': execution_price - stop_distance,
                'stop_distance': stop_distance,
                'entry_bar': bar_index,
                'metadata': signal.get('metadata', {}),
                # Tier 1 improvements - Partial profit tracking
                'original_size': position_size,
                'took_profit_1r': False,
                'took_profit_2r': False,
                'moved_to_breakeven': False
            }
        
        elif signal['action'] == 'SHORT':
            self.position = {
                'side': 'SHORT',
                'size': position_size,
                'entry_price': execution_price,
                'stop_price': execution_price + stop_distance,
                'stop_distance': stop_distance,
                'entry_bar': bar_index,
                'metadata': signal.get('metadata', {}),
                # Tier 1 improvements - Partial profit tracking
                'original_size': position_size,
                'took_profit_1r': False,
                'took_profit_2r': False,
                'moved_to_breakeven': False
            }
    
    def _close_position(self, price, bar_index, exit_reason):
        """Close current position and record trade"""
        
        if self.position is None:
            return
        
        # Apply slippage to exit
        if self.position['side'] == 'LONG':
            execution_price = self._apply_slippage(price, 'SHORT')  # Selling
        else:
            execution_price = self._apply_slippage(price, 'LONG')  # Buying to cover
        
        side = self.position['side']
        size = self.position['size']
        entry_price = self.position['entry_price']
        entry_bar = self.position['entry_bar']
        
        duration = bar_index - entry_bar
        
        if side == 'LONG':
            # Sell position
            proceeds = size * execution_price * (1 - self.fee)
            self.capital += proceeds
            
            # PnL
            cost_basis = size * entry_price
            pnl = proceeds - cost_basis
            return_pct = (execution_price / entry_price - 1) * 100
        
        else:  # SHORT
            # Buy to cover
            cover_cost = size * execution_price * (1 + self.fee)
            pnl = (entry_price - execution_price) * size - cover_cost * self.fee
            self.capital += pnl
            
            return_pct = (entry_price / execution_price - 1) * 100
        
        # Record trade
        trade = {
            'side': side,
            'entry': entry_price,
            'exit': execution_price,
            'size': size,
            'pnl': pnl,
            'return_pct': return_pct,
            'duration': duration,
            'exit_reason': exit_reason,
            'metadata': self.position.get('metadata', {})
        }
        
        self.trades.append(trade)
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cooldown_after_losses:
                self.cooldown_remaining = 5  # Wait 5 bars
                print(f"⚠️  Cooldown activated after {self.consecutive_losses} consecutive losses")
        else:
            self.consecutive_losses = 0
        
        # Clear position
        self.position = None
    
    def _check_partial_profits(self, current_price, bar_index):
        """
        Tier 1 Improvement: Check and execute partial profit taking
        - Take 25% at 1R (when profit = 1x stop distance)
        - Take 50% at 2R (when profit = 2x stop distance)
        - Move stop to breakeven after 1R
        """
        if self.position is None:
            return
        
        entry_price = self.position['entry_price']
        stop_distance = self.position['stop_distance']
        side = self.position['side']
        
        if side == 'LONG':
            profit_distance = current_price - entry_price
            
            # Check 1R profit
            if not self.position['took_profit_1r'] and profit_distance >= stop_distance:
                # Take 25% profit
                partial_size = self.position['original_size'] * self.partial_profit_1r
                self._partial_close(partial_size, current_price, bar_index, 'partial_profit_1r')
                self.position['took_profit_1r'] = True
                
                # Move stop to breakeven
                if self.use_breakeven_stop and not self.position['moved_to_breakeven']:
                    self.position['stop_price'] = entry_price
                    self.position['moved_to_breakeven'] = True
            
            # Check 2R profit
            if not self.position['took_profit_2r'] and profit_distance >= stop_distance * 2:
                # Take 50% of remaining (which is 37.5% of original)
                remaining_size = self.position['size']
                partial_size = remaining_size * self.partial_profit_2r
                self._partial_close(partial_size, current_price, bar_index, 'partial_profit_2r')
                self.position['took_profit_2r'] = True
        
        else:  # SHORT
            profit_distance = entry_price - current_price
            
            # Check 1R profit
            if not self.position['took_profit_1r'] and profit_distance >= stop_distance:
                partial_size = self.position['original_size'] * self.partial_profit_1r
                self._partial_close(partial_size, current_price, bar_index, 'partial_profit_1r')
                self.position['took_profit_1r'] = True
                
                # Move stop to breakeven
                if self.use_breakeven_stop and not self.position['moved_to_breakeven']:
                    self.position['stop_price'] = entry_price
                    self.position['moved_to_breakeven'] = True
            
            # Check 2R profit
            if not self.position['took_profit_2r'] and profit_distance >= stop_distance * 2:
                remaining_size = self.position['size']
                partial_size = remaining_size * self.partial_profit_2r
                self._partial_close(partial_size, current_price, bar_index, 'partial_profit_2r')
                self.position['took_profit_2r'] = True
    
    def _partial_close(self, partial_size, price, bar_index, reason):
        """
        Close a partial position and add capital
        """
        if self.position is None or partial_size <= 0:
            return
        
        # Don't close more than we have
        partial_size = min(partial_size, self.position['size'])
        
        side = self.position['side']
        entry_price = self.position['entry_price']
        
        # Apply slippage to exit
        if side == 'LONG':
            execution_price = self._apply_slippage(price, 'SHORT')
            proceeds = partial_size * execution_price * (1 - self.fee)
            self.capital += proceeds
            pnl = proceeds - (partial_size * entry_price)
        else:  # SHORT
            execution_price = self._apply_slippage(price, 'LONG')
            cover_cost = partial_size * execution_price * (1 + self.fee)
            pnl = (entry_price - execution_price) * partial_size - cover_cost * self.fee
            self.capital += pnl
        
        # Update position size
        self.position['size'] -= partial_size
        
        # Record partial trade
        trade = {
            'side': side,
            'entry': entry_price,
            'exit': execution_price,
            'size': partial_size,
            'pnl': pnl,
            'return_pct': ((execution_price / entry_price - 1) * 100) if side == 'LONG' else ((entry_price / execution_price - 1) * 100),
            'duration': bar_index - self.position['entry_bar'],
            'exit_reason': reason,
            'metadata': {'partial': True, **self.position.get('metadata', {})}
        }
        self.trades.append(trade)
    
    def _check_max_duration(self, current_bar):
        """
        Tier 1 Improvement: Check if position has been held too long
        Returns True if position should be closed due to max duration
        """
        if self.position is None or self.max_hold_hours <= 0:
            return False
        
        duration = current_bar - self.position['entry_bar']
        return duration >= self.max_hold_hours
    
    def get_metrics(self):
        """Calculate comprehensive performance metrics"""
        
        if len(self.equity_history) == 0:
            return {}
        
        equity = np.array(self.equity_history)
        
        # Basic metrics
        final_equity = equity[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Returns
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        # Sharpe ratio (annualized for hourly data)
        sharpe = 0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        
        # Sortino ratio
        sortino = 0
        if len(returns) > 0:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252 * 24)
        
        # Max drawdown
        max_dd = 0
        peak = equity[0]
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        # Trade metrics
        if self.trades:
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            
            # Profit factor
            total_wins = sum(t['pnl'] for t in winning_trades)
            total_losses = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            # Duration stats
            avg_duration = np.mean([t['duration'] for t in self.trades])
            
            # Exit reason breakdown
            exit_reasons = {}
            for trade in self.trades:
                reason = trade['exit_reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            expectancy = 0
            profit_factor = 0
            avg_duration = 0
            exit_reasons = {}
        
        return {
            'final_equity': final_equity,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'avg_duration_hours': avg_duration,
            'exit_reasons': exit_reasons,
            'trading_halted': self.trading_halted
        }


# Backward compatibility
class ProfessionalBacktest(EnhancedBacktest):
    """Alias for backward compatibility"""
    pass
