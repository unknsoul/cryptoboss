"""
Enhanced Drawdown Tracker
Advanced drawdown analysis with recovery tracking and alerts

Tracks:
- Current drawdown
- Maximum drawdown
- Drawdown duration
- Recovery time
- Underwater periods
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


class EnhancedDrawdownTracker:
    """
    Professional drawdown tracking and analysis
    
    Features:
    - Real-time drawdown calculation
    - Recovery time tracking
    - Underwater period analysis
    - Drawdown alerts
    - Historical drawdown series
    """
    
    def __init__(self, alert_thresholds: List[float] = None):
        """
        Initialize drawdown tracker
        
        Args:
            alert_thresholds: DD levels to trigger alerts (e.g., [0.05, 0.10, 0.15])
        """
        self.alert_thresholds = alert_thresholds or [0.05, 0.10, 0.15, 0.20]
        self.equity_history = []
        self.peak_equity = 0
        self.max_drawdown = 0
        self.current_drawdown_start = None
        self.longest_drawdown_days = 0
        self.drawdown_periods = []
        self.alerts_sent = set()
        
    def update(self, equity: float) -> Dict:
        """
        Update with new equity value
        
        Returns:
            Drawdown status dict
        """
        timestamp = datetime.now()
        
        self.equity_history.append({
            'equity': equity,
            'timestamp': timestamp
        })
        
        # Update peak
        if equity > self.peak_equity:
            # New peak - end current drawdown period if any
            if self.current_drawdown_start:
                dd_period = {
                    'start': self.current_drawdown_start,
                    'end': timestamp,
                    'duration_days': (timestamp - self.current_drawdown_start).days,
                    'max_dd': self.max_drawdown
                }
                self.drawdown_periods.append(dd_period)
                
                # Update longest drawdown
                if dd_period['duration_days'] > self.longest_drawdown_days:
                    self.longest_drawdown_days = dd_period['duration_days']
                
                self.current_drawdown_start = None
                self.alerts_sent.clear()  # Reset alerts
            
            self.peak_equity = equity
        
        # Calculate current drawdown
        current_dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # Track drawdown start
        if current_dd > 0 and not self.current_drawdown_start:
            self.current_drawdown_start = timestamp
        
        # Update max drawdown
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        # Check alert thresholds
        alert = None
        for threshold in sorted(self.alert_thresholds):
            if current_dd >= threshold and threshold not in self.alerts_sent:
                alert = {
                    'level': threshold,
                    'current_dd': current_dd,
                    'message': f"⚠️ Drawdown Alert: {current_dd*100:.1f}% (threshold: {threshold*100:.0f}%)"
                }
                self.alerts_sent.add(threshold)
                logger.warning(alert['message'])
                break
        
        # Calculate time in drawdown
        days_in_dd = 0
        if self.current_drawdown_start:
            days_in_dd = (timestamp - self.current_drawdown_start).days
        
        return {
            'equity': equity,
            'peak_equity': self.peak_equity,
            'current_drawdown_pct': round(current_dd * 100, 2),
            'max_drawdown_pct': round(self.max_drawdown * 100, 2),
            'days_in_drawdown': days_in_dd,
            'is_underwater': current_dd > 0,
            'alert': alert,
            'status': 'ok'
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive drawdown statistics"""
        if not self.equity_history:
            return {'status': 'no_data'}
        
        # Calculate recovery rate
        completed_dds = [dd for dd in self.drawdown_periods if dd.get('end')]
        
        if completed_dds:
            avg_recovery_days = np.mean([dd['duration_days'] for dd in completed_dds])
            avg_dd_depth = np.mean([dd['max_dd'] for dd in completed_dds])
        else:
            avg_recovery_days = 0
            avg_dd_depth = 0
        
        return {
            'peak_equity': round(self.peak_equity, 2),
            'max_drawdown_pct': round(self.max_drawdown * 100, 2),
            'total_drawdown_periods': len(self.drawdown_periods),
            'longest_drawdown_days': self.longest_drawdown_days,
            'avg_recovery_days': round(avg_recovery_days, 1),
            'avg_drawdown_depth_pct': round(avg_dd_depth * 100, 2),
            'currently_underwater': self.current_drawdown_start is not None,
            'status': 'ok'
        }
    
    def get_drawdown_series(self, window: Optional[int] = None) -> List[Dict]:
        """
        Get drawdown time series
        
        Returns:
            List of {timestamp, drawdown_pct}
        """
        history = self.equity_history[-window:] if window else self.equity_history
        
        if not history:
            return []
        
        series = []
        running_peak = 0
        
        for point in history:
            equity = point['equity']
            if equity > running_peak:
                running_peak = equity
            
            dd = (running_peak - equity) / running_peak if running_peak > 0 else 0
            
            series.append({
                'timestamp': point['timestamp'].isoformat(),
                'equity': equity,
                'peak': running_peak,
                'drawdown_pct': round(dd * 100, 2)
            })
        
        return series


class TradeDurationAnalyzer:
    """
    Analyze trade hold times and profitability by duration
    
    Helps identify:
    - Optimal hold times
    - When to exit trades
    - Duration vs profitability patterns
    """
    
    def __init__(self):
        """Initialize trade duration analyzer"""
        self.trades = []
        
    def record_trade(self, entry_time: datetime, exit_time: datetime,
                     pnl: float, side: str):
        """Record a completed trade"""
        duration = (exit_time - entry_time).total_seconds()
        duration_hours = duration / 3600
        
        self.trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'duration_seconds': duration,
            'duration_hours': duration_hours,
            'pnl': pnl,
            'side': side,
            'is_winner': pnl > 0
        })
    
    def get_duration_stats(self) -> Dict:
        """Get duration statistics"""
        if not self.trades:
            return {'status': 'no_data'}
        
        durations = [t['duration_hours'] for t in self.trades]
        winners = [t for t in self.trades if t['is_winner']]
        losers = [t for t in self.trades if not t['is_winner']]
        
        stats = {
            'avg_duration_hours': round(np.mean(durations), 2),
            'median_duration_hours': round(np.median(durations), 2),
            'min_duration_hours': round(np.min(durations), 2),
            'max_duration_hours': round(np.max(durations), 2)
        }
        
        if winners:
            winner_durations = [t['duration_hours'] for t in winners]
            stats['avg_winner_duration_hours'] = round(np.mean(winner_durations), 2)
        
        if losers:
            loser_durations = [t['duration_hours'] for t in losers]
            stats['avg_loser_duration_hours'] = round(np.mean(loser_durations), 2)
        
        stats['status'] = 'ok'
        return stats
    
    def get_profitability_by_duration(self, buckets: int = 5) -> Dict:
        """
        Bucket trades by duration and show profitability
        
        Args:
            buckets: Number of duration buckets
            
        Returns:
            Profitability analysis by duration bucket
        """
        if not self.trades:
            return {'status': 'no_data'}
        
        durations = [t['duration_hours'] for t in self.trades]
        max_duration = max(durations)
        bucket_size = max_duration / buckets
        
        bucket_data = []
        
        for i in range(buckets):
            bucket_min = i * bucket_size
            bucket_max = (i + 1) * bucket_size
            
            bucket_trades = [
                t for t in self.trades 
                if bucket_min <= t['duration_hours'] < bucket_max
            ]
            
            if bucket_trades:
                avg_pnl = np.mean([t['pnl'] for t in bucket_trades])
                win_rate = len([t for t in bucket_trades if t['is_winner']]) / len(bucket_trades)
                
                bucket_data.append({
                    'duration_range': f"{bucket_min:.1f}-{bucket_max:.1f}h",
                    'trade_count': len(bucket_trades),
                    'avg_pnl': round(avg_pnl, 2),
                    'win_rate': round(win_rate * 100, 1)
                })
        
        return {
            'buckets': bucket_data,
            'total_trades': len(self.trades),
            'status': 'ok'
        }


# Singletons
_drawdown_tracker: Optional[EnhancedDrawdownTracker] = None
_duration_analyzer: Optional[TradeDurationAnalyzer] = None


def get_drawdown_tracker() -> EnhancedDrawdownTracker:
    global _drawdown_tracker
    if _drawdown_tracker is None:
        _drawdown_tracker = EnhancedDrawdownTracker()
    return _drawdown_tracker


def get_duration_analyzer() -> TradeDurationAnalyzer:
    global _duration_analyzer
    if _duration_analyzer is None:
        _duration_analyzer = TradeDurationAnalyzer()
    return _duration_analyzer


if __name__ == '__main__':
    print("=" * 70)
    print("ENHANCED ANALYTICS - TEST")
    print("=" * 70)
    
    # Test drawdown tracker
    print("\n📉 Testing Drawdown Tracker...")
    dd_tracker = EnhancedDrawdownTracker(alert_thresholds=[0.05, 0.10])
    
    # Simulate equity curve with drawdowns
    import random
    random.seed(42)
    
    equity = 10000
    for i in range(100):
        # Random walk with slight upward drift
        change = random.gauss(10, 100)
        equity = max(1000, equity + change)
        
        status = dd_tracker.update(equity)
        
        if status['alert']:
            print(f"  Day {i}: {status['alert']['message']}")
    
    stats = dd_tracker.get_statistics()
    print(f"\nDrawdown Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test duration analyzer
    print("\n⏱️  Testing Duration Analyzer...")
    duration_analyzer = TradeDurationAnalyzer()
    
    # Simulate trades
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(50):
        entry = base_time + timedelta(hours=i*12)
        hold_hours = random.uniform(0.5, 24)
        exit_time = entry + timedelta(hours=hold_hours)
        
        # Shorter trades tend to be better
        pnl = random.gauss(50, 100) - (hold_hours * 2)
        
        duration_analyzer.record_trade(entry, exit_time, pnl, 'LONG')
    
    duration_stats = duration_analyzer.get_duration_stats()
    print(f"\nDuration Statistics:")
    for key, value in duration_stats.items():
        print(f"  {key}: {value}")
    
    profitability = duration_analyzer.get_profitability_by_duration(buckets=4)
    print(f"\nProfitability by Duration:")
    for bucket in profitability['buckets']:
        print(f"  {bucket['duration_range']}: {bucket['trade_count']} trades, "
              f"Avg P&L: ${bucket['avg_pnl']:.2f}, WR: {bucket['win_rate']}%")
    
    print("\n" + "=" * 70)
    print("✅ Enhanced analytics working!")
    print("=" * 70)
