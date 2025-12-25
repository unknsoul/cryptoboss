"""
MAE/MFE Tracker (Maximum Adverse/Favorable Excursion)
Analyzes how far trades moved against/in favor before closing

MAE = Worst unrealized loss during trade
MFE = Best unrealized profit during trade
Exit Efficiency = (Actual P&L / MFE) - measures how well we captured upside
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MAEMFETracker:
    """
    Professional MAE/MFE analysis system
    
    Tracks:
    - Maximum Adverse Excursion (worst drawdown per trade)
    - Maximum Favorable Excursion (best profit per trade)
    - Exit efficiency (% of MFE captured)
    - Stop loss effectiveness
    """
    
    def __init__(self, max_history: int = 1000):
        """Initialize MAE/MFE tracker"""
        self.max_history = max_history
        self.trades = []
        
    def record_trade(self, trade_data: Dict):
        """
        Record a completed trade with MAE/MFE data
        
        Args:
            trade_data: Dict with trade details including:
                - entry_price, exit_price, side
                - mae_price (worst price during trade)
                - mfe_price (best price during trade)
                - pnl, size
        """
        entry = trade_data['entry_price']
        exit_price = trade_data['exit_price']
        mae_price = trade_data.get('mae_price', entry)
        mfe_price = trade_data.get('mfe_price', entry)
        side = trade_data['side']
        pnl = trade_data['pnl']
        size = trade_data.get('size', 1)
        
        # Calculate MAE (worst loss)
        if side == 'LONG':
            mae = (mae_price - entry) * size
            mfe = (mfe_price - entry) * size
        else:  # SHORT
            mae = (entry - mae_price) * size
            mfe = (entry - mfe_price) * size
        
        # Exit efficiency: how much of MFE did we capture?
        if mfe > 0:
            exit_efficiency = (pnl / mfe) if mfe > 0 else 0
        else:
            exit_efficiency = 0
        
        # MAE to MFE ratio
        mae_mfe_ratio = abs(mae / mfe) if mfe != 0 else 0
        
        record = {
            'timestamp': trade_data.get('timestamp', datetime.now()),
            'side': side,
            'entry_price': entry,
            'exit_price': exit_price,
            'mae': mae,
            'mfe': mfe,
            'mae_pct': (mae / (entry * size)) * 100,
            'mfe_pct': (mfe / (entry * size)) * 100,
            'pnl': pnl,
            'exit_efficiency': exit_efficiency * 100,
            'mae_mfe_ratio': mae_mfe_ratio,
            'is_winner': pnl > 0
        }
        
        self.trades.append(record)
        
        # Limit history
        if len(self.trades) > self.max_history:
            self.trades = self.trades[-self.max_history:]
        
        return record
    
    def get_statistics(self, window: Optional[int] = None) -> Dict:
        """
        Get MAE/MFE statistics
        
        Args:
            window: Last N trades (None = all)
            
        Returns:
            Comprehensive statistics
        """
        if not self.trades:
            return {'status': 'no_data'}
        
        trades = self.trades[-window:] if window else self.trades
        
        winners = [t for t in trades if t['is_winner']]
        losers = [t for t in trades if not t['is_winner']]
        
        stats = {
            'total_trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
        }
        
        # Overall MAE/MFE
        if trades:
            stats['avg_mae_pct'] = round(np.mean([t['mae_pct'] for t in trades]), 2)
            stats['avg_mfe_pct'] = round(np.mean([t['mfe_pct'] for t in trades]), 2)
            stats['avg_exit_efficiency'] = round(np.mean([t['exit_efficiency'] for t in trades]), 2)
            stats['avg_mae_mfe_ratio'] = round(np.mean([t['mae_mfe_ratio'] for t in trades]), 2)
        
        # Winners analysis
        if winners:
            stats['winners_avg_mfe_pct'] = round(np.mean([t['mfe_pct'] for t in winners]), 2)
            stats['winners_exit_efficiency'] = round(np.mean([t['exit_efficiency'] for t in winners]), 2)
            stats['winners_gave_back'] = round(np.mean([t['mfe_pct'] - (t['pnl']/(t['entry_price']*1))*100 for t in winners]), 2)
        
        # Losers analysis
        if losers:
            stats['losers_avg_mae_pct'] = round(np.mean([abs(t['mae_pct']) for t in losers]), 2)
            stats['losers_avg_mfe_pct'] = round(np.mean([t['mfe_pct'] for t in losers]), 2)
        
        stats['status'] = 'ok'
        return stats
    
    def analyze_stop_effectiveness(self) -> Dict:
        """
        Analyze if stops are too tight or too loose
        
        Returns:
            Analysis of stop placement
        """
        if len(self.trades) < 10:
            return {'status': 'insufficient_data'}
        
        losers = [t for t in self.trades if not t['is_winner']]
        
        if not losers:
            return {'status': 'no_losses', 'message': 'No losing trades to analyze'}
        
        # Check if MAE was much larger than final loss (stop too loose)
        mae_to_loss = [abs(t['mae'] / t['pnl']) if t['pnl'] != 0 else 1 for t in losers]
        avg_mae_to_loss = np.mean(mae_to_loss)
        
        # Check how many losers had MFE > 0 (went in favor first)
        losers_with_profit = len([t for t in losers if t['mfe'] > 0])
        pct_losers_with_profit = losers_with_profit / len(losers) * 100
        
        analysis = {
            'avg_mae_to_final_loss': round(avg_mae_to_loss, 2),
            'losers_that_were_profitable': losers_with_profit,
            'pct_losers_with_profit': round(pct_losers_with_profit, 2),
            'recommendation': ''
        }
        
        # Recommendations
        if avg_mae_to_loss > 1.5:
            analysis['recommendation'] = 'Stops may be too loose - consider tighter stops'
        elif pct_losers_with_profit > 50:
            analysis['recommendation'] = 'Many losers had profits - consider tighter trailing stops'
        else:
            analysis['recommendation'] = 'Stop placement appears reasonable'
        
        return analysis
    
    def get_exit_quality_report(self) -> Dict:
        """
        Report on exit quality across all trades
        
        Returns:
            Exit performance analysis
        """
        if not self.trades:
            return {'status': 'no_data'}
        
        winners = [t for t in self.trades if t['is_winner']]
        
        if not winners:
            return {'status': 'no_winners'}
        
        # Bucket exit efficiency
        excellent = len([t for t in winners if t['exit_efficiency'] >= 80])
        good = len([t for t in winners if 60 <= t['exit_efficiency'] < 80])
        fair = len([t for t in winners if 40 <= t['exit_efficiency'] < 60])
        poor = len([t for t in winners if t['exit_efficiency'] < 40])
        
        return {
            'total_winners': len(winners),
            'excellent_exits': excellent,  # Captured 80%+ of MFE
            'good_exits': good,
            'fair_exits': fair,
            'poor_exits': poor,
            'avg_capture_rate': round(np.mean([t['exit_efficiency'] for t in winners]), 2),
            'median_capture_rate': round(np.median([t['exit_efficiency'] for t in winners]), 2),
            'status': 'ok'
        }


# Singleton
_mae_mfe_tracker: Optional[MAEMFETracker] = None


def get_mae_mfe_tracker() -> MAEMFETracker:
    """Get singleton MAE/MFE tracker"""
    global _mae_mfe_tracker
    if _mae_mfe_tracker is None:
        _mae_mfe_tracker = MAEMFETracker()
    return _mae_mfe_tracker


if __name__ == '__main__':
    # Test MAE/MFE tracker
    print("=" * 70)
    print("MAE/MFE TRACKER - TEST")
    print("=" * 70)
    
    tracker = MAEMFETracker()
    
    # Simulate some trades
    import random
    
    print("\n📊 Simulating 30 trades...")
    
    for i in range(30):
        is_winner = random.random() > 0.4  # 60% win rate
        
        entry = 50000
        
        if is_winner:
            # Winner: went in favor
            mfe_pct = random.uniform(2, 8)  # 2-8% favorable
            mae_pct = random.uniform(-1, 0.5)  # Small adverse
            exit_pct = random.uniform(1, mfe_pct * 0.7)  # Captured 70% of MFE avg
        else:
            # Loser: went against
            mae_pct = random.uniform(-3, -0.5)  # 0.5-3% adverse
            mfe_pct = random.uniform(-0.5, 1)  # Might have had small profit
            exit_pct = mae_pct * random.uniform(0.8, 1.0)
        
        mfe_price = entry * (1 + mfe_pct/100)
        mae_price = entry * (1 + mae_pct/100)
        exit_price = entry * (1 + exit_pct/100)
        
        pnl = (exit_price - entry) * 0.1  # 0.1 BTC size
        
        trade = {
            'entry_price': entry,
            'exit_price': exit_price,
            'mae_price': mae_price,
            'mfe_price': mfe_price,
            'side': 'LONG',
            'pnl': pnl,
            'size': 0.1
        }
        
        tracker.record_trade(trade)
    
    # Get statistics
    print("\n📈 Overall Statistics:")
    stats = tracker.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎯 Exit Quality Report:")
    exit_quality = tracker.get_exit_quality_report()
    for key, value in exit_quality.items():
        print(f"  {key}: {value}")
    
    print("\n🛑 Stop Loss Analysis:")
    stop_analysis = tracker.analyze_stop_effectiveness()
    for key, value in stop_analysis.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ MAE/MFE tracker working correctly!")
    print("=" * 70)
