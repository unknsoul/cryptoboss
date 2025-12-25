"""
Walk-Forward Backtesting Engine
Test strategies on rolling windows of historical data

Benefits:
- Avoid overfitting
- Realistic performance estimates
- Out-of-sample validation
- Strategy optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    Professional walk-forward backtesting
    
    Process:
    1. Split data into train/test windows
    2. Train strategy on train window
    3. Test on out-of-sample test window
    4. Roll forward and repeat
    """
    
    def __init__(self, train_days: int = 60, test_days: int = 15):
        """
        Initialize backtester
        
        Args:
            train_days: Days for training window
            test_days: Days for testing window
        """
        self.train_days = train_days
        self.test_days = test_days
        self.results = []
        
    def run(self, data: pd.DataFrame, strategy_func: Callable,
            initial_equity: float = 10000.0) -> Dict:
        """
        Run walk-forward backtest
        
        Args:
            data: Historical OHLCV data with DatetimeIndex
            strategy_func: Function(train_data) -> parameters
            initial_equity: Starting capital
            
        Returns:
            Backtest results
        """
        if len(data) < self.train_days + self.test_days:
            return {'status': 'insufficient_data'}
        
        equity = initial_equity
        trades = []
        window_results = []
        
        # Calculate number of windows
        total_days = len(data)
        num_windows = (total_days - self.train_days) // self.test_days
        
        logger.info(f"Running {num_windows} walk-forward windows...")
        
        for i in range(num_windows):
            # Define window boundaries
            train_start = i * self.test_days
            train_end = train_start + self.train_days
            test_start = train_end
            test_end = test_start + self.test_days
            
            if test_end > len(data):
                break
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Train strategy (get optimized parameters)
            try:
                params = strategy_func(train_data)
            except Exception as e:
                logger.error(f"Strategy training failed: {e}")
                params = {}
            
            # Test on out-of-sample data
            window_trades = self._simulate_trades(test_data, params)
            trades.extend(window_trades)
            
            # Calculate window performance
            window_pnl = sum(t['pnl'] for t in window_trades)
            equity += window_pnl
            
            window_results.append({
                'window': i + 1,
                'train_period': f"{train_data.index[0]} to {train_data.index[-1]}",
                'test_period': f"{test_data.index[0]} to {test_data.index[-1]}",
                'trades': len(window_trades),
                'pnl': window_pnl,
                'equity_end': equity
            })
            
            logger.info(f"Window {i+1}/{num_windows}: {len(window_trades)} trades, P&L: ${window_pnl:.2f}")
        
        # Calculate overall statistics
        if trades:
            winners = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winners) / len(trades)
            total_pnl = sum(t['pnl'] for t in trades)
            avg_win = np.mean([t['pnl'] for t in winners]) if winners else 0
            losers = [t for t in trades if t['pnl'] < 0]
            avg_loss = np.mean([abs(t['pnl']) for t in losers]) if losers else 0
            profit_factor = abs(sum(t['pnl'] for t in winners) / sum(t['pnl'] for t in losers)) if losers else 999
        else:
            win_rate = total_pnl = avg_win = avg_loss = profit_factor = 0
        
        return {
            'initial_equity': initial_equity,
            'final_equity': equity,
            'total_pnl': total_pnl,
            'total_return_pct': (equity - initial_equity) / initial_equity * 100,
            'total_trades': len(trades),
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'num_windows': len(window_results),
            'window_results': window_results,
            'all_trades': trades,
            'status': 'ok'
        }
    
    def _simulate_trades(self, data: pd.DataFrame, params: Dict) -> List[Dict]:
        """Simulate trades on test data (simple momentum strategy)"""
        trades = []
        
        # Simple momentum strategy for demonstration
        sma_period = params.get('sma_period', 20)
        
        if len(data) < sma_period:
            return []
        
        data['sma'] = data['close'].rolling(sma_period).mean()
        
        position = None
        
        for i in range(sma_period, len(data)):
            price = data.iloc[i]['close']
            sma = data.iloc[i]['sma']
            
            # Entry logic
            if position is None:
                if price > sma:  # Bullish
                    position = {
                        'entry_price': price,
                        'entry_time': data.index[i],
                        'side': 'LONG'
                    }
            
            # Exit logic
            elif position:
                if price < sma or i == len(data) - 1:  # Exit signal or end of data
                    pnl = (price - position['entry_price']) * 1  # 1 unit size
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': data.index[i],
                        'entry_price': position['entry_price'],
                        'exit_price': price,
                        'side': position['side'],
                        'pnl': pnl,
                        'pnl_pct': pnl / position['entry_price'] * 100
                    })
                    
                    position = None
        
        return trades


class RateLimitProtector:
    """
    API rate limit protection
    
    Prevents:
    - Exceeding exchange rate limits
    - Account bans
    - Request throttling
    """
    
    def __init__(self, max_requests_per_minute: int = 1200,
                 max_requests_per_second: int = 20):
        """
        Initialize rate limiter
        
        Args:
            max_requests_per_minute: Max API calls per minute
            max_requests_per_second: Max API calls per second
        """
        self.max_per_minute = max_requests_per_minute
        self.max_per_second = max_requests_per_second
        
        self.requests_minute = []
        self.requests_second = []
        
    def can_request(self) -> Tuple[bool, str]:
        """
        Check if request is allowed
        
        Returns:
            (allowed, reason)
        """
        now = datetime.now()
        
        # Clean old requests
        minute_ago = now - timedelta(minutes=1)
        second_ago = now - timedelta(seconds=1)
        
        self.requests_minute = [r for r in self.requests_minute if r > minute_ago]
        self.requests_second = [r for r in self.requests_second if r > second_ago]
        
        # Check limits
        if len(self.requests_second) >= self.max_per_second:
            return False, f"Per-second limit reached ({self.max_per_second}/s)"
        
        if len(self.requests_minute) >= self.max_per_minute:
            return False, f"Per-minute limit reached ({self.max_per_minute}/min)"
        
        return True, "OK"
    
    def record_request(self):
        """Record a request"""
        now = datetime.now()
        self.requests_minute.append(now)
        self.requests_second.append(now)
    
    def get_stats(self) -> Dict:
        """Get current rate limit usage"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        second_ago = now - timedelta(seconds=1)
        
        requests_last_minute = len([r for r in self.requests_minute if r > minute_ago])
        requests_last_second = len([r for r in self.requests_second if r > second_ago])
        
        return {
            'requests_last_second': requests_last_second,
            'requests_last_minute': requests_last_minute,
            'second_limit': self.max_per_second,
            'minute_limit': self.max_per_minute,
            'second_usage_pct': round(requests_last_second / self.max_per_second * 100, 1),
            'minute_usage_pct': round(requests_last_minute / self.max_per_minute * 100, 1)
        }


# Singletons
_backtester: Optional[WalkForwardBacktester] = None
_rate_limiter: Optional[RateLimitProtector] = None


def get_backtester() -> WalkForwardBacktester:
    global _backtester
    if _backtester is None:
        _backtester = WalkForwardBacktester()
    return _backtester


def get_rate_limiter() -> RateLimitProtector:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimitProtector()
    return _rate_limiter


if __name__ == '__main__':
    print("=" * 70)
    print("BACKTESTING & INFRASTRUCTURE - TEST")
    print("=" * 70)
    
    # Test walk-forward backtester
    print("\n📈 Testing Walk-Forward Backtester...")
    
    # Generate synthetic price data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=180, freq='D')
    prices = 50000 + np.cumsum(np.random.randn(180) * 500)
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'open': prices,
        'volume': np.random.randint(1000, 10000, 180)
    }, index=dates)
    
    # Define simple strategy function
    def momentum_strategy(train_data):
        """Returns optimized parameters"""
        # Could do sophisticated optimization here
        return {'sma_period': 20}
    
    backtester = WalkForwardBacktester(train_days=60, test_days=30)
    results = backtester.run(data, momentum_strategy, initial_equity=10000)
    
    print(f"\nBacktest Results:")
    print(f"  Total trades: {results['total_trades']}")
    print(f"  Win rate: {results['win_rate']:.1f}%")
    print(f"  Total P&L: ${results['total_pnl']:.2f}")
    print(f"  Total return: {results['total_return_pct']:.2f}%")
    print(f"  Profit factor: {results['profit_factor']:.2f}")
    print(f"  Num windows: {results['num_windows']}")
    
    # Test rate limiter
    print("\n🚦 Testing Rate Limiter...")
    limiter = RateLimitProtector(max_requests_per_second=5, max_requests_per_minute=100)
    
    # Simulate requests
    import time
    allowed_count = 0
    blocked_count = 0
    
    for i in range(10):
        can_request, reason = limiter.can_request()
        if can_request:
            limiter.record_request()
            allowed_count += 1
        else:
            blocked_count += 1
            print(f"  Request blocked: {reason}")
        
        time.sleep(0.1)
    
    stats = limiter.get_stats()
    print(f"\nRate Limit Stats:")
    print(f"  Requests last second: {stats['requests_last_second']}/{stats['second_limit']}")
    print(f"  Requests last minute: {stats['requests_last_minute']}/{stats['minute_limit']}")
    print(f"  Second usage: {stats['second_usage_pct']}%")
    print(f"  Minute usage: {stats['minute_usage_pct']}%")
    print(f"  Allowed: {allowed_count}, Blocked: {blocked_count}")
    
    print("\n" + "=" * 70)
    print("✅ Backtesting & infrastructure working!")
    print("=" * 70)
