"""
Paper Trading Engine
Simulates order execution and position tracking in real-time
"""

import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper trading engine with realistic execution simulation
    
    Features:
    - Position tracking
    - Partial profit taking (25% at 1R, 50% at 2R)
    - Breakeven stops
    - Time-based exits
    - Adaptive position sizing
    - Trade history and PnL tracking
    """
    
    def __init__(self, initial_capital=10000, config=None):
        """
        Initialize paper trader
        
        Args:
            initial_capital: Starting capital
            config: Configuration dict with risk parameters
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.equity = initial_capital
        
        # Configuration (same as EnhancedBacktest)
        config = config or {}
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        self.fee = config.get('fee', 0.001)
        self.slippage = config.get('slippage', 0.0005)
        
        # Tier 1 improvements
        self.use_partial_profits = config.get('use_partial_profits', True)
        self.partial_profit_1r = config.get('partial_profit_1r', 0.25)
        self.partial_profit_2r = config.get('partial_profit_2r', 0.50)
        self.use_breakeven_stop = config.get('use_breakeven_stop', True)
        self.max_hold_bars = config.get('max_hold_hours', 48)
        
        # Risk controls
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.20)
        self.cooldown_after_losses = config.get('cooldown_after_losses', 3)
        
        # State tracking
        self.position = None
        self.trades = []
        self.peak_equity = initial_capital
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        self.current_bar = 0
        
        logger.info(f"Paper trader initialized with ${initial_capital:,.2f}")
    
    def has_position(self):
        """Check if position is open"""
        return self.position is not None
    
    def execute_signal(self, signal, current_price, timestamp=None):
        """
        Execute trading signal (open position)
        
        Args:
            signal: Signal dict from strategy
            current_price: Current market price
            timestamp: Current time (optional)
        
        Returns:
            bool: True if executed, False if skipped
        """
        # Check if already in position
        if self.has_position():
            logger.debug("Already in position, skipping signal")
            return False
        
        # Check cooldown
        if self.cooldown_remaining > 0:
            logger.info(f"⏸️ In cooldown ({self.cooldown_remaining} bars remaining)")
            self.cooldown_remaining -= 1
            return False
        
        # Calculate position size (with adaptive scaling)
        position_size = self._calculate_position_size(signal, current_price)
        if position_size <= 0:
            logger.warning("Position size is zero, skipping")
            return False
        
        # Apply slippage
        execution_price = self._apply_slippage(current_price, signal['action'])
        
        # Calculate cost
        cost = position_size * execution_price * (1 + self.fee)
        if cost > self.capital:
            logger.warning(f"Insufficient capital: ${cost:.2f} > ${self.capital:.2f}")
            return False
        
        # Deduct capital
        self.capital -= cost
        
        # Create position
        stop_distance = signal['stop']
        self.position = {
            'side': signal['action'],
            'entry_price': execution_price,
            'size': position_size,
            'stop_price': execution_price - stop_distance if signal['action'] == 'LONG' else execution_price + stop_distance,
            'stop_distance': stop_distance,
            'entry_bar': self.current_bar,
            'entry_time': timestamp or datetime.now(),
            'metadata': signal.get('metadata', {}),
            # Partial profit tracking
            'original_size': position_size,
            'took_profit_1r': False,
            'took_profit_2r': False,
            'moved_to_breakeven': False
        }
        
        logger.info(f"🟢 OPENED {signal['action']} @ ${execution_price:.2f} | "
                   f"Size: {position_size:.4f} | Stop: ${self.position['stop_price']:.2f}")
        
        return True
    
    def update_position(self, current_high, current_low, current_close):
        """
        Update position (check stops, partial profits, time-based exits)
        
        Args:
            current_high: Bar high
            current_low: Bar low
            current_close: Bar close
        """
        if not self.has_position():
            return
        
        self.current_bar += 1
        
        # Check partial profits first
        if self.use_partial_profits:
            self._check_partial_profits(current_close)
        
        # Check max duration
        duration = self.current_bar - self.position['entry_bar']
        if self.max_hold_bars > 0 and duration >= self.max_hold_bars:
            logger.info(f"⏱️ Max duration reached ({duration} bars)")
            self.close_position(current_close, 'max_duration')
            return
        
        # Check stop loss
        if self.position['side'] == 'LONG':
            if current_low <= self.position['stop_price']:
                logger.info(f"🛑 Stop hit @ ${self.position['stop_price']:.2f}")
                self.close_position(self.position['stop_price'], 'stop_loss')
                return
        else:  # SHORT
            if current_high >= self.position['stop_price']:
                logger.info(f"🛑 Stop hit @ ${self.position['stop_price']:.2f}")
                self.close_position(self.position['stop_price'], 'stop_loss')
                return
        
        # Update trailing stop (only moves in profit direction)
        if self.position['side'] == 'LONG':
            new_stop = current_close - self.position['stop_distance']
            if new_stop > self.position['stop_price']:
                self.position['stop_price'] = new_stop
                logger.debug(f"📈 Trailing stop moved to ${new_stop:.2f}")
        else:  # SHORT
            new_stop = current_close + self.position['stop_distance']
            if new_stop < self.position['stop_price']:
                self.position['stop_price'] = new_stop
                logger.debug(f"📉 Trailing stop moved to ${new_stop:.2f}")
        
        # Update equity (mark-to-market)
        self._update_equity(current_close)
    
    def close_position(self, exit_price, reason='manual'):
        """
        Close position entirely
        
        Args:
            exit_price: Exit price
            reason: Exit reason
        """
        if not self.has_position():
            return
        
        # Apply slippage
        if self.position['side'] == 'LONG':
            execution_price = self._apply_slippage(exit_price, 'SHORT')
        else:
            execution_price = self._apply_slippage(exit_price, 'LONG')
        
        # Calculate PnL
        size = self.position['size']
        entry_price = self.position['entry_price']
        
        if self.position['side'] == 'LONG':
            proceeds = size * execution_price * (1 - self.fee)
            self.capital += proceeds
            pnl = proceeds - (size * entry_price)
            return_pct = (execution_price / entry_price - 1) * 100
        else:  # SHORT
            cost = size * execution_price * (1 + self.fee)
            pnl = (entry_price - execution_price) * size - cost * self.fee
            self.capital += pnl
            return_pct = (entry_price / execution_price - 1) * 100
        
        # Record trade
        duration = self.current_bar - self.position['entry_bar']
        trade = {
            'side': self.position['side'],
            'entry': entry_price,
            'exit': execution_price,
            'size': size,
            'pnl': pnl,
            'return_pct': return_pct,
            'duration': duration,
            'exit_reason': reason,
            'entry_time': self.position['entry_time'],
            'exit_time': datetime.now(),
            'metadata': self.position['metadata']
        }
        self.trades.append(trade)
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cooldown_after_losses:
                self.cooldown_remaining = 5
                logger.warning(f"⚠️ Cooldown activated ({self.consecutive_losses} losses)")
        else:
            self.consecutive_losses = 0
        
        # Log result
        pnl_icon = "✅" if pnl > 0 else "❌"
        logger.info(f"{pnl_icon} CLOSED {self.position['side']} @ ${execution_price:.2f} | "
                   f"PnL: ${pnl:+.2f} ({return_pct:+.2f}%) | Reason: {reason}")
        
        # Clear position
        self.position = None
        self._update_equity(execution_price)
    
    def _check_partial_profits(self, current_price):
        """Check and execute partial profit taking"""
        if not self.position:
            return
        
        entry_price = self.position['entry_price']
        stop_distance = self.position['stop_distance']
        side = self.position['side']
        
        if side == 'LONG':
            profit_distance = current_price - entry_price
            
            # 1R profit
            if not self.position['took_profit_1r'] and profit_distance >= stop_distance:
                partial_size = self.position['original_size'] * self.partial_profit_1r
                self._partial_close(partial_size, current_price, 'partial_profit_1r')
                self.position['took_profit_1r'] = True
                
                # Breakeven stop
                if self.use_breakeven_stop and not self.position['moved_to_breakeven']:
                    self.position['stop_price'] = entry_price
                    self.position['moved_to_breakeven'] = True
                    logger.info(f"🔒 Stop moved to breakeven @ ${entry_price:.2f}")
            
            # 2R profit
            if not self.position['took_profit_2r'] and profit_distance >= stop_distance * 2:
                remaining_size = self.position['size']
                partial_size = remaining_size * self.partial_profit_2r
                self._partial_close(partial_size, current_price, 'partial_profit_2r')
                self.position['took_profit_2r'] = True
        
        else:  # SHORT
            profit_distance = entry_price - current_price
            
            # 1R profit
            if not self.position['took_profit_1r'] and profit_distance >= stop_distance:
                partial_size = self.position['original_size'] * self.partial_profit_1r
                self._partial_close(partial_size, current_price, 'partial_profit_1r')
                self.position['took_profit_1r'] = True
                
                # Breakeven stop
                if self.use_breakeven_stop and not self.position['moved_to_breakeven']:
                    self.position['stop_price'] = entry_price
                    self.position['moved_to_breakeven'] = True
                    logger.info(f"🔒 Stop moved to breakeven @ ${entry_price:.2f}")
            
            # 2R profit
            if not self.position['took_profit_2r'] and profit_distance >= stop_distance * 2:
                remaining_size = self.position['size']
                partial_size = remaining_size * self.partial_profit_2r
                self._partial_close(partial_size, current_price, 'partial_profit_2r')
                self.position['took_profit_2r'] = True
    
    def _partial_close(self, partial_size, price, reason):
        """Close partial position"""
        if not self.position or partial_size <= 0:
            return
        
        partial_size = min(partial_size, self.position['size'])
        
        side = self.position['side']
        entry_price = self.position['entry_price']
        
        # Apply slippage
        if side == 'LONG':
            execution_price = self._apply_slippage(price, 'SHORT')
            proceeds = partial_size * execution_price * (1 - self.fee)
            self.capital += proceeds
            pnl = proceeds - (partial_size * entry_price)
        else:
            execution_price = self._apply_slippage(price, 'LONG')
            cost = partial_size * execution_price * (1 + self.fee)
            pnl = (entry_price - execution_price) * partial_size - cost * self.fee
            self.capital += pnl
        
        # Update position size
        self.position['size'] -= partial_size
        
        logger.info(f"💰 Partial close ({reason}): {partial_size:.4f} @ ${execution_price:.2f} | PnL: ${pnl:+.2f}")
    
    def _calculate_position_size(self, signal, current_price):
        """Calculate position size with adaptive scaling"""
        stop_distance = signal['stop']
        if stop_distance <= 0:
            return 0
        
        # Base position size
        risk_amount = self.capital * self.risk_per_trade
        position_size = risk_amount / stop_distance
        
        # Adaptive scaling
        scale_factor = 1.0
        
        # Reduce after consecutive losses
        if self.consecutive_losses >= 2:
            scale_factor *= 0.5
        
        # Reduce during drawdown
        current_dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        if current_dd > 0.10:
            scale_factor *= 0.5
        
        position_size *= scale_factor
        
        # Ensure we can afford it
        cost = position_size * current_price * (1 + self.fee + self.slippage)
        if cost > self.capital:
            position_size = self.capital / (current_price * (1 + self.fee + self.slippage))
        
        return position_size
    
    def _apply_slippage(self, price, side):
        """Apply slippage to execution price"""
        if side in ['LONG', 'BUY']:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)
    
    def _update_equity(self, current_price):
        """Update equity (capital + unrealized PnL)"""
        if self.has_position():
            if self.position['side'] == 'LONG':
                unrealized_pnl = (current_price - self.position['entry_price']) * self.position['size']
            else:
                unrealized_pnl = (self.position['entry_price'] - current_price) * self.position['size']
            
            self.equity = self.capital + unrealized_pnl
        else:
            self.equity = self.capital
        
        # Update peak
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
    
    def get_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'expectancy': 0,
                'profit_factor': 0
            }
        
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades)
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        
        return {
            'total_return': total_return,
            'equity': self.equity,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'profit_factor': profit_factor
        }
    
    def print_status(self):
        """Print current status"""
        metrics = self.get_metrics()
        
        print(f"\n💼 Paper Trading Status")
        print("=" * 60)
        print(f"Equity: ${self.equity:,.2f} | "
              f"Return: {metrics['total_return']:+.2%} | "
              f"Trades: {metrics['num_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1%} | "
              f"Expectancy: ${metrics['expectancy']:.2f} | "
              f"PF: {metrics['profit_factor']:.2f}")
        
        if self.has_position():
            unrealized_pnl = self.equity - self.capital
            print(f"\n📍 Open Position:")
            print(f"  {self.position['side']} @ ${self.position['entry_price']:.2f}")
            print(f"  Size: {self.position['size']:.4f} | Stop: ${self.position['stop_price']:.2f}")
            print(f"  Unrealized PnL: ${unrealized_pnl:+.2f}")
        
        print("=" * 60)
