"""
Analytics & Reporting - Enterprise Features #180, #185, #190, #195
Trade Journal, Equity Analysis, Drawdown Analysis, Win Streaks.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class TradeJournalSystem:
    """
    Feature #180: Trade Journal System
    
    Comprehensive trade journaling with notes and analysis.
    """
    
    def __init__(self, storage_path: str = 'data/trade_journal.json'):
        """
        Initialize trade journal.
        
        Args:
            storage_path: Path to journal storage
        """
        self.storage_path = storage_path
        self.entries: List[Dict] = []
        self.tags: Dict[str, List[str]] = defaultdict(list)
        
        logger.info("Trade Journal System initialized")
    
    def add_entry(
        self,
        trade: Dict,
        notes: str = '',
        tags: Optional[List[str]] = None,
        emotions: str = 'neutral',
        setup_type: str = 'unknown',
        screenshot_path: Optional[str] = None
    ) -> str:
        """
        Add a trade journal entry.
        
        Args:
            trade: Trade data
            notes: Trading notes
            tags: Category tags
            emotions: Emotional state
            setup_type: Trade setup type
            screenshot_path: Path to chart screenshot
            
        Returns:
            Entry ID
        """
        entry_id = f"J{len(self.entries)+1:05d}"
        
        entry = {
            'id': entry_id,
            'timestamp': datetime.now().isoformat(),
            'trade': trade,
            'notes': notes,
            'tags': tags or [],
            'emotions': emotions,
            'setup_type': setup_type,
            'screenshot': screenshot_path,
            'lessons_learned': '',
            'mistakes': [],
            'what_went_well': []
        }
        
        self.entries.append(entry)
        
        # Index by tags
        for tag in entry['tags']:
            self.tags[tag].append(entry_id)
        
        return entry_id
    
    def update_entry(self, entry_id: str, updates: Dict) -> bool:
        """Update an existing journal entry."""
        for entry in self.entries:
            if entry['id'] == entry_id:
                entry.update(updates)
                return True
        return False
    
    def add_lesson(self, entry_id: str, lesson: str):
        """Add a lesson learned to an entry."""
        for entry in self.entries:
            if entry['id'] == entry_id:
                entry['lessons_learned'] = lesson
                return True
        return False
    
    def get_entries_by_tag(self, tag: str) -> List[Dict]:
        """Get all entries with a specific tag."""
        entry_ids = self.tags.get(tag, [])
        return [e for e in self.entries if e['id'] in entry_ids]
    
    def get_entries_by_date(self, date: datetime) -> List[Dict]:
        """Get entries for a specific date."""
        date_str = date.strftime('%Y-%m-%d')
        return [e for e in self.entries if e['timestamp'].startswith(date_str)]
    
    def analyze_patterns(self) -> Dict:
        """Analyze journaled trade patterns."""
        if not self.entries:
            return {'total_entries': 0}
        
        by_setup = defaultdict(list)
        by_emotion = defaultdict(list)
        
        for entry in self.entries:
            pnl = entry['trade'].get('pnl', 0)
            by_setup[entry['setup_type']].append(pnl)
            by_emotion[entry['emotions']].append(pnl)
        
        setup_analysis = {
            setup: {
                'count': len(pnls),
                'total_pnl': sum(pnls),
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0
            }
            for setup, pnls in by_setup.items()
        }
        
        emotion_analysis = {
            emotion: {
                'count': len(pnls),
                'avg_pnl': sum(pnls) / len(pnls) if pnls else 0
            }
            for emotion, pnls in by_emotion.items()
        }
        
        return {
            'total_entries': len(self.entries),
            'by_setup': setup_analysis,
            'by_emotion': emotion_analysis,
            'most_common_tags': sorted(self.tags.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        }


class EquityCurveAnalyzer:
    """
    Feature #185: Equity Curve Analyzer
    
    Analyzes equity curve characteristics and health.
    """
    
    def __init__(self):
        """Initialize equity analyzer."""
        self.equity_history: List[Dict] = []
        
        logger.info("Equity Curve Analyzer initialized")
    
    def record_equity(self, equity: float, timestamp: Optional[datetime] = None):
        """Record equity point."""
        self.equity_history.append({
            'timestamp': (timestamp or datetime.now()).isoformat(),
            'equity': equity
        })
        self.equity_history = self.equity_history[-5000:]  # Keep 5000 points
    
    def analyze(self) -> Dict:
        """Analyze the equity curve."""
        if len(self.equity_history) < 2:
            return {'insufficient_data': True}
        
        equities = [e['equity'] for e in self.equity_history]
        
        # Calculate returns
        returns = [(equities[i] - equities[i-1]) / equities[i-1] 
                   for i in range(1, len(equities)) if equities[i-1] != 0]
        
        # Calculate metrics
        total_return = (equities[-1] - equities[0]) / equities[0] if equities[0] != 0 else 0
        
        # Smoothness (R-squared of linear fit)
        n = len(equities)
        x_mean = (n - 1) / 2
        y_mean = sum(equities) / n
        
        ss_xx = sum((i - x_mean) ** 2 for i in range(n))
        ss_xy = sum((i - x_mean) * (equities[i] - y_mean) for i in range(n))
        
        slope = ss_xy / ss_xx if ss_xx != 0 else 0
        intercept = y_mean - slope * x_mean
        
        predicted = [slope * i + intercept for i in range(n)]
        ss_res = sum((equities[i] - predicted[i]) ** 2 for i in range(n))
        ss_tot = sum((e - y_mean) ** 2 for e in equities)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Trend strength
        positive_periods = sum(1 for r in returns if r > 0)
        trend_consistency = positive_periods / len(returns) if returns else 0.5
        
        return {
            'total_return': round(total_return * 100, 2),
            'smoothness': round(r_squared, 3),
            'trend_consistency': round(trend_consistency, 3),
            'data_points': len(equities),
            'start_equity': equities[0],
            'end_equity': equities[-1],
            'peak': max(equities),
            'trough': min(equities),
            'is_healthy': r_squared > 0.7 and total_return > 0
        }
    
    def detect_regime_changes(self) -> List[Dict]:
        """Detect significant equity curve regime changes."""
        if len(self.equity_history) < 50:
            return []
        
        equities = [e['equity'] for e in self.equity_history]
        changes = []
        window = 20
        
        for i in range(window, len(equities) - window):
            before = sum(equities[i-window:i]) / window
            after = sum(equities[i:i+window]) / window
            
            change_pct = (after - before) / before if before != 0 else 0
            
            if abs(change_pct) > 0.1:  # 10% change
                changes.append({
                    'index': i,
                    'timestamp': self.equity_history[i]['timestamp'],
                    'change_pct': round(change_pct * 100, 2),
                    'direction': 'positive' if change_pct > 0 else 'negative'
                })
        
        return changes


class DrawdownAnalyzer:
    """
    Feature #190: Drawdown Analyzer
    
    Detailed drawdown analysis and tracking.
    """
    
    def __init__(self):
        """Initialize drawdown analyzer."""
        self.equity_history: List[float] = []
        self.drawdowns: List[Dict] = []
        
        logger.info("Drawdown Analyzer initialized")
    
    def update(self, equity: float):
        """Update with new equity value."""
        self.equity_history.append(equity)
        self._analyze_current_dd()
    
    def _analyze_current_dd(self):
        """Analyze current drawdown state."""
        if len(self.equity_history) < 2:
            return
        
        peak = max(self.equity_history)
        current = self.equity_history[-1]
        current_dd = (peak - current) / peak if peak > 0 else 0
        
        # Track drawdown periods
        if current_dd > 0.01:  # In drawdown (>1%)
            if not self.drawdowns or self.drawdowns[-1].get('recovered'):
                self.drawdowns.append({
                    'start_index': len(self.equity_history) - 1,
                    'peak_equity': peak,
                    'max_dd': current_dd,
                    'recovered': False
                })
            else:
                dd = self.drawdowns[-1]
                dd['max_dd'] = max(dd['max_dd'], current_dd)
                dd['trough_equity'] = current
        elif self.drawdowns and not self.drawdowns[-1].get('recovered'):
            # Recovery
            dd = self.drawdowns[-1]
            dd['recovered'] = True
            dd['recovery_index'] = len(self.equity_history) - 1
            dd['duration'] = dd['recovery_index'] - dd['start_index']
    
    def get_current_drawdown(self) -> Dict:
        """Get current drawdown status."""
        if not self.equity_history:
            return {'in_drawdown': False}
        
        peak = max(self.equity_history)
        current = self.equity_history[-1]
        dd = (peak - current) / peak if peak > 0 else 0
        
        return {
            'current_dd_pct': round(dd * 100, 2),
            'peak': peak,
            'current': current,
            'in_drawdown': dd > 0.01
        }
    
    def get_max_drawdown(self) -> float:
        """Get maximum drawdown."""
        if len(self.equity_history) < 2:
            return 0
        
        peak = self.equity_history[0]
        max_dd = 0
        
        for equity in self.equity_history:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return round(max_dd * 100, 2)
    
    def get_drawdown_stats(self) -> Dict:
        """Get comprehensive drawdown statistics."""
        all_dds = [d['max_dd'] for d in self.drawdowns]
        durations = [d.get('duration', 0) for d in self.drawdowns if d.get('recovered')]
        
        return {
            'max_drawdown': self.get_max_drawdown(),
            'current': self.get_current_drawdown(),
            'total_dd_events': len(self.drawdowns),
            'avg_dd_pct': round(sum(all_dds) / len(all_dds) * 100, 2) if all_dds else 0,
            'avg_recovery_periods': round(sum(durations) / len(durations), 1) if durations else 0,
            'longest_dd_duration': max(durations) if durations else 0
        }


class WinStreakTracker:
    """
    Feature #195: Win Streak Tracker
    
    Tracks winning and losing streaks.
    """
    
    def __init__(self):
        """Initialize streak tracker."""
        self.trade_results: List[bool] = []  # True = win, False = loss
        self.current_streak: int = 0
        self.streak_type: Optional[str] = None  # 'win' or 'loss'
        
        self.max_win_streak: int = 0
        self.max_loss_streak: int = 0
        self.streak_history: List[Dict] = []
        
        logger.info("Win Streak Tracker initialized")
    
    def record_trade(self, is_win: bool):
        """Record a trade result."""
        self.trade_results.append(is_win)
        
        if not self.trade_results[:-1]:
            self.current_streak = 1
            self.streak_type = 'win' if is_win else 'loss'
        elif is_win and self.streak_type == 'win':
            self.current_streak += 1
        elif not is_win and self.streak_type == 'loss':
            self.current_streak += 1
        else:
            # Streak ended
            self.streak_history.append({
                'type': self.streak_type,
                'length': self.current_streak,
                'ended_at': datetime.now().isoformat()
            })
            self.current_streak = 1
            self.streak_type = 'win' if is_win else 'loss'
        
        # Update max streaks
        if self.streak_type == 'win':
            self.max_win_streak = max(self.max_win_streak, self.current_streak)
        else:
            self.max_loss_streak = max(self.max_loss_streak, self.current_streak)
    
    def get_current_streak(self) -> Dict:
        """Get current streak status."""
        return {
            'type': self.streak_type or 'none',
            'length': self.current_streak,
            'is_winning': self.streak_type == 'win',
            'max_win_streak': self.max_win_streak,
            'max_loss_streak': self.max_loss_streak
        }
    
    def get_streak_analysis(self) -> Dict:
        """Analyze streak patterns."""
        if len(self.trade_results) < 10:
            return {'insufficient_data': True}
        
        win_streaks = [s['length'] for s in self.streak_history if s['type'] == 'win']
        loss_streaks = [s['length'] for s in self.streak_history if s['type'] == 'loss']
        
        return {
            'total_trades': len(self.trade_results),
            'current': self.get_current_streak(),
            'avg_win_streak': round(sum(win_streaks) / len(win_streaks), 1) if win_streaks else 0,
            'avg_loss_streak': round(sum(loss_streaks) / len(loss_streaks), 1) if loss_streaks else 0,
            'longest_win': max(win_streaks) if win_streaks else 0,
            'longest_loss': max(loss_streaks) if loss_streaks else 0,
            'streak_history': self.streak_history[-10:]
        }
    
    def should_reduce_size(self) -> bool:
        """Check if should reduce position size based on streak."""
        return self.streak_type == 'loss' and self.current_streak >= 3


# Singletons
_journal: Optional[TradeJournalSystem] = None
_equity_analyzer: Optional[EquityCurveAnalyzer] = None
_dd_analyzer: Optional[DrawdownAnalyzer] = None
_streak_tracker: Optional[WinStreakTracker] = None


def get_trade_journal() -> TradeJournalSystem:
    global _journal
    if _journal is None:
        _journal = TradeJournalSystem()
    return _journal


def get_equity_analyzer() -> EquityCurveAnalyzer:
    global _equity_analyzer
    if _equity_analyzer is None:
        _equity_analyzer = EquityCurveAnalyzer()
    return _equity_analyzer


def get_dd_analyzer() -> DrawdownAnalyzer:
    global _dd_analyzer
    if _dd_analyzer is None:
        _dd_analyzer = DrawdownAnalyzer()
    return _dd_analyzer


def get_streak_tracker() -> WinStreakTracker:
    global _streak_tracker
    if _streak_tracker is None:
        _streak_tracker = WinStreakTracker()
    return _streak_tracker


if __name__ == '__main__':
    # Test journal
    journal = TradeJournalSystem()
    entry_id = journal.add_entry(
        {'pnl': 50, 'side': 'LONG'},
        notes='Good setup',
        tags=['trend', 'breakout'],
        setup_type='breakout'
    )
    print(f"Journal entry: {entry_id}")
    
    # Test equity analyzer
    eq = EquityCurveAnalyzer()
    for i in range(100):
        eq.record_equity(10000 + i * 10 + (i % 5) * 5)
    print(f"Equity analysis: {eq.analyze()}")
    
    # Test streak tracker
    streak = WinStreakTracker()
    for win in [True, True, True, False, False, True, True, True, True, True]:
        streak.record_trade(win)
    print(f"Streak: {streak.get_current_streak()}")
