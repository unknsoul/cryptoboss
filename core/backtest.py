"""
Professional Backtest Engine
Enhanced risk management with dynamic position sizing and regime tracking
"""

import numpy as np


class ProfessionalBacktest:
    """
    Advanced backtest engine with:
    - Volatility-based dynamic position sizing
    - ADX-based exit signals
    - Enhanced performance metrics
    - Trade duration tracking
    - Regime performance analysis
    """
    
    def __init__(self, capital=10000, base_risk=0.02, fee=0.001):
        self.initial_capital = capital
        self.capital = capital
        self.base_risk = base_risk  # Base risk percentage
        self.fee = fee
        
        # Position tracking
        self.position = 0.0
        self.entry_price = None
        self.stop_price = None
        self.side = None
        self.stop_distance = None
        self.entry_bar = None
        
        # Results tracking
        self.equity_curve = []
        self.trades = []
        self.regime_stats = {'STRONG_TREND': [], 'WEAK_TREND': [], 'RANGING': []}
    
    def _calculate_position_size(self, signal):
        """
        Calculate position size with volatility adjustment
        
        Higher volatility = smaller position size
        Lower volatility = larger position size
        """
        # Base risk amount
        base_risk_amount = self.capital * self.base_risk
        
        # Adjust risk based on volatility percentile
        volatility_pct = signal.get('volatility_pct', 50)
        
        if volatility_pct > 75:  # High volatility (top 25%)
            risk_multiplier = 0.7  # Reduce risk by 30%
        elif volatility_pct > 50:  # Above average volatility
            risk_multiplier = 0.85  # Reduce risk by 15%
        elif volatility_pct < 25:  # Low volatility (bottom 25%)
            risk_multiplier = 1.2  # Increase risk by 20%
        else:  # Average volatility
            risk_multiplier = 1.0
        
        adjusted_risk = base_risk_amount * risk_multiplier
        
        # Calculate position size based on stop distance
        stop_distance = signal['stop']
        if stop_distance == 0:
            return 0
        
        position_size = adjusted_risk / stop_distance
        
        return position_size
    
    def run(self, highs, lows, closes, strategy, volumes=None):
        """
        Run backtest with professional features
        """
        self.equity_history = []
        self.trades = []
        self.position = None
        
        num_bars = len(closes)
        
        for i in range(num_bars):
            # Update equity
            if self.position is None:
                current_equity = self.capital
            else:
                if self.position['side'] == 'LONG':
                    unrealized_pnl = (closes[i] - self.position['entry_price']) * self.position['size']
                else:  # SHORT
                    unrealized_pnl = (self.position['entry_price'] - closes[i]) * self.position['size']
                
                current_equity = self.capital + unrealized_pnl
            
            self.equity_history.append(current_equity)
            
            # Check if we have a position
            if self.position is not None:
                # Check for exit
                should_exit = strategy.check_exit(
                    highs, lows, closes,
                    self.position['side'],
                    self.position['entry_price'],
                    self.position['entry_index'],
                    i
                )
                
                if should_exit:
                    self._close_position(closes[i], i, 'strategy_exit')
                    continue
                
                # Check stop loss and take profit
                if self.position['side'] == 'LONG':
                    if closes[i] <= self.position['stop_price']:
                        self._close_position(self.position['stop_price'], i, 'stop_loss')
                        continue
                    if 'target_price' in self.position and closes[i] >= self.position['target_price']:
                        self._close_position(self.position['target_price'], i, 'take_profit')
                        continue
                else:  # SHORT
                    if closes[i] >= self.position['stop_price']:
                        self._close_position(self.position['stop_price'], i, 'stop_loss')
                        continue
                    if 'target_price' in self.position and closes[i] <= self.position['target_price']:
                        self._close_position(self.position['target_price'], i, 'take_profit')
                        continue
            
            # No position, check for entry signal
            if self.position is None:
                # Get signal from strategy, pass volumes if available
                signal = strategy.signal(highs[:i+1], lows[:i+1], closes[:i+1], 
                                       volumes[:i+1] if volumes is not None else None)
                
                if signal is not None:
                    self._open_position(closes[i], i, signal)
        
        # Close any remaining position at end
        if self.position is not None:
            self._close_position(closes[-1], len(closes) - 1, "END_OF_DATA")
        
        return self.equity_history
    
    def _open_position(self, price, bar_index, signal):
        """Open a new position with volatility-adjusted sizing"""
        
        # Calculate dynamic position size
        position_size = self._calculate_position_size(signal)
        
        if position_size == 0:
            return
        
        stop_distance = signal['stop']
        
        if signal['action'] == "LONG":
            # Calculate cost including fees
            cost = position_size * price * (1 + self.fee)
            
            # Only enter if we have enough capital
            if cost <= self.capital:
                self.capital -= cost
                self.position = {
                    'side': 'LONG',
                    'size': position_size,
                    'entry_price': price,
                    'stop_price': price - stop_distance,
                    'stop_distance': stop_distance,
                    'target_price': signal['target'] if 'target' in signal else None,
                    'entry_index': bar_index
                }
        
        elif signal['action'] == "SHORT":
            # For shorts, we don't deduct capital upfront (simplified margin)
            # Typically requires margin = value * margin_req
            # Here we just track short position value
            self.position = {
                'side': 'SHORT',
                'size': position_size,
                'entry_price': price,
                'stop_price': price + stop_distance,
                'stop_distance': stop_distance,
                'target_price': signal['target'] if 'target' in signal else None,  # target is lower for short
                'entry_index': bar_index
            }
    
    def _close_position(self, price, bar_index, exit_reason):
        """Close current position and record trade"""
        
        if self.position is None:
            return
        
        side = self.position['side']
        size = self.position['size']
        entry_price = self.position['entry_price']
        entry_index = self.position['entry_index']
        
        trade_duration = bar_index - entry_index
        
        if side == "LONG":
            # Sell position
            proceeds = size * price * (1 - self.fee)
            self.capital += proceeds
            
            # Calculate PnL
            cost = size * entry_price
            pnl = proceeds - cost
            return_pct = (price / entry_price - 1) * 100
            
            self.trades.append({
                'side': 'LONG',
                'entry': entry_price,
                'exit': price,
                'pnl': pnl,
                'return_pct': return_pct,
                'exit_reason': exit_reason,
                'duration': trade_duration
            })
        
        elif side == "SHORT":
            # Close short position
            # Profit = (Entry - Exit) * Size
            pnl = (entry_price - price) * size
            # Fee on closing value
            close_cost = size * price * self.fee
            net_pnl = pnl - close_cost
            self.capital += net_pnl
            
            return_pct = (entry_price / price - 1) * 100
            
            self.trades.append({
                'side': 'SHORT',
                'entry': entry_price,
                'exit': price,
                'pnl': net_pnl,
                'return_pct': return_pct,
                'exit_reason': exit_reason,
                'duration': trade_duration
            })
        
        # Reset position
        self.position = None
    
    def get_metrics(self):
        """Calculate comprehensive performance metrics"""
        
        if len(self.equity_history) == 0:
            return {}
        
        equity = np.array(self.equity_history)
        
        # Returns
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        # Sharpe ratio (annualized for hourly data)
        sharpe = 0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        
        # Sortino ratio (only penalize downside volatility)
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
        
        # Calmar ratio (annualized return / max drawdown)
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        years = len(equity) / (252 * 24)  # Assuming hourly data
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        # Win metrics
        if self.trades:
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Average trade duration
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
            profit_factor = 0
            avg_duration = 0
            exit_reasons = {}
        
        return {
            'final_equity': equity[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_duration_hours': avg_duration,
            'exit_reasons': exit_reasons
        }
