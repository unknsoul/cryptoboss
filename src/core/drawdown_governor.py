"""
Drawdown Governor - Multi-Timeframe Drawdown Control

Controls trading activity based on drawdown levels across multiple timeframes.
Implements automatic risk reduction when approaching drawdown limits.

v11.0 - Production-Grade Platform Upgrade
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DrawdownSeverity(str, Enum):
    """Drawdown severity levels."""
    NORMAL = "normal"          # < 30% of limit
    ELEVATED = "elevated"      # 30-50% of limit
    WARNING = "warning"        # 50-70% of limit
    CRITICAL = "critical"      # 70-90% of limit
    BREACHED = "breached"      # >= 90% of limit


class DrawdownTimeframe(str, Enum):
    """Drawdown tracking timeframes."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    TOTAL = "total"


@dataclass
class DrawdownState:
    """Current state of drawdown for a timeframe."""
    timeframe: DrawdownTimeframe
    current_drawdown_pct: float
    peak_equity: float
    current_equity: float
    limit_pct: float
    severity: DrawdownSeverity
    last_updated: datetime
    
    @property
    def utilization(self) -> float:
        """Drawdown utilization as ratio of limit (0.0 - 1.0+)."""
        if self.limit_pct <= 0:
            return 0.0
        return self.current_drawdown_pct / self.limit_pct
    
    @property
    def remaining_pct(self) -> float:
        """Remaining drawdown allowance percentage."""
        return max(0, self.limit_pct - self.current_drawdown_pct)
    
    def to_dict(self) -> Dict:
        return {
            'timeframe': self.timeframe.value,
            'current_drawdown_pct': self.current_drawdown_pct,
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity,
            'limit_pct': self.limit_pct,
            'severity': self.severity.value,
            'utilization': self.utilization,
            'remaining_pct': self.remaining_pct,
            'last_updated': self.last_updated.isoformat(),
        }


@dataclass
class DrawdownEvent:
    """Logged drawdown event."""
    timestamp: datetime
    event_type: str  # 'severity_change', 'limit_breach', 'reset', 'defensive_mode'
    timeframe: DrawdownTimeframe
    old_severity: Optional[DrawdownSeverity]
    new_severity: DrawdownSeverity
    drawdown_pct: float
    details: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'timeframe': self.timeframe.value,
            'old_severity': self.old_severity.value if self.old_severity else None,
            'new_severity': self.new_severity.value,
            'drawdown_pct': self.drawdown_pct,
            'details': self.details,
        }


class DrawdownGovernor:
    """
    Multi-Timeframe Drawdown Governor.
    
    Controls trading activity based on drawdown across:
    - Daily: Resets at midnight UTC
    - Weekly: Resets on Sunday midnight UTC
    - Monthly: Resets on 1st of month
    - Total: Never resets (lifetime)
    
    Features:
    - Automatic severity classification
    - Position size multipliers based on drawdown
    - Defensive mode activation
    - Drawdown events logging
    
    Usage:
        governor = DrawdownGovernor(
            initial_equity=10000,
            daily_limit_pct=5.0,
            weekly_limit_pct=10.0,
            monthly_limit_pct=15.0,
            total_limit_pct=25.0
        )
        
        # Check if trading allowed
        allowed, reason = governor.check_trading_allowed()
        
        # Get size multiplier
        multiplier = governor.get_size_multiplier()
        
        # Update equity after trade
        governor.update_equity(new_equity)
    """
    
    # Severity thresholds (as ratio of limit)
    SEVERITY_THRESHOLDS = {
        DrawdownSeverity.NORMAL: 0.0,
        DrawdownSeverity.ELEVATED: 0.30,
        DrawdownSeverity.WARNING: 0.50,
        DrawdownSeverity.CRITICAL: 0.70,
        DrawdownSeverity.BREACHED: 0.90,
    }
    
    # Size multipliers by severity
    SIZE_MULTIPLIERS = {
        DrawdownSeverity.NORMAL: 1.0,
        DrawdownSeverity.ELEVATED: 0.75,
        DrawdownSeverity.WARNING: 0.50,
        DrawdownSeverity.CRITICAL: 0.25,
        DrawdownSeverity.BREACHED: 0.0,
    }
    
    def __init__(
        self,
        initial_equity: float = 10000.0,
        daily_limit_pct: float = 5.0,
        weekly_limit_pct: float = 10.0,
        monthly_limit_pct: float = 15.0,
        total_limit_pct: float = 25.0,
        log_dir: str = "logs/drawdown",
        enable_defensive_mode: bool = True,
        defensive_threshold: float = 0.70,  # Enter defensive at 70% of limit
    ):
        """
        Initialize DrawdownGovernor.
        
        Args:
            initial_equity: Starting portfolio equity
            daily_limit_pct: Maximum daily drawdown percentage
            weekly_limit_pct: Maximum weekly drawdown percentage
            monthly_limit_pct: Maximum monthly drawdown percentage
            total_limit_pct: Maximum total drawdown percentage
            log_dir: Directory for drawdown logs
            enable_defensive_mode: Auto-reduce risk when approaching limits
            defensive_threshold: Utilization ratio to enter defensive mode
        """
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.log_dir = log_dir
        self.enable_defensive_mode = enable_defensive_mode
        self.defensive_threshold = defensive_threshold
        self.in_defensive_mode = False
        
        # Initialize limits
        self._limits = {
            DrawdownTimeframe.DAILY: daily_limit_pct,
            DrawdownTimeframe.WEEKLY: weekly_limit_pct,
            DrawdownTimeframe.MONTHLY: monthly_limit_pct,
            DrawdownTimeframe.TOTAL: total_limit_pct,
        }
        
        # Initialize peaks (high water marks)
        self._peaks = {
            DrawdownTimeframe.DAILY: initial_equity,
            DrawdownTimeframe.WEEKLY: initial_equity,
            DrawdownTimeframe.MONTHLY: initial_equity,
            DrawdownTimeframe.TOTAL: initial_equity,
        }
        
        # Initialize period start times
        now = datetime.utcnow()
        self._period_starts = {
            DrawdownTimeframe.DAILY: now.replace(hour=0, minute=0, second=0, microsecond=0),
            DrawdownTimeframe.WEEKLY: now - timedelta(days=now.weekday()),
            DrawdownTimeframe.MONTHLY: now.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            DrawdownTimeframe.TOTAL: now,
        }
        
        # Event log
        self._events: List[DrawdownEvent] = []
        self._max_events = 10000
        
        # Severity tracking
        self._current_severities = {tf: DrawdownSeverity.NORMAL for tf in DrawdownTimeframe}
        
        # Ensure log directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"DrawdownGovernor initialized: initial_equity={initial_equity}, "
            f"limits: daily={daily_limit_pct}%, weekly={weekly_limit_pct}%, "
            f"monthly={monthly_limit_pct}%, total={total_limit_pct}%"
        )
    
    def update_equity(self, new_equity: float) -> None:
        """
        Update current equity and recalculate drawdown states.
        
        Args:
            new_equity: New portfolio equity value
        """
        self._check_period_resets()
        
        self.current_equity = new_equity
        
        # Update peaks if new high
        for timeframe in DrawdownTimeframe:
            if new_equity > self._peaks[timeframe]:
                self._peaks[timeframe] = new_equity
        
        # Check severity changes
        for timeframe in DrawdownTimeframe:
            state = self._get_drawdown_state(timeframe)
            old_severity = self._current_severities[timeframe]
            
            if state.severity != old_severity:
                self._log_event(DrawdownEvent(
                    timestamp=datetime.utcnow(),
                    event_type='severity_change',
                    timeframe=timeframe,
                    old_severity=old_severity,
                    new_severity=state.severity,
                    drawdown_pct=state.current_drawdown_pct,
                    details=f"Drawdown severity changed from {old_severity.value} to {state.severity.value}"
                ))
                self._current_severities[timeframe] = state.severity
                
                # Check for breach
                if state.severity == DrawdownSeverity.BREACHED:
                    logger.warning(f"DRAWDOWN LIMIT BREACHED: {timeframe.value} at {state.current_drawdown_pct:.2f}%")
        
        # Check defensive mode
        self._check_defensive_mode()
    
    def get_drawdown_state(self, timeframe: DrawdownTimeframe) -> DrawdownState:
        """Get current drawdown state for a timeframe."""
        self._check_period_resets()
        return self._get_drawdown_state(timeframe)
    
    def get_all_states(self) -> Dict[DrawdownTimeframe, DrawdownState]:
        """Get drawdown states for all timeframes."""
        self._check_period_resets()
        return {tf: self._get_drawdown_state(tf) for tf in DrawdownTimeframe}
    
    def check_trading_allowed(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on drawdown limits.
        
        Returns:
            (allowed, reason)
        """
        self._check_period_resets()
        
        for timeframe in DrawdownTimeframe:
            state = self._get_drawdown_state(timeframe)
            if state.severity == DrawdownSeverity.BREACHED:
                return False, f"{timeframe.value.capitalize()} drawdown limit breached ({state.current_drawdown_pct:.2f}% >= {state.limit_pct}%)"
        
        return True, "Drawdown within limits"
    
    def check_daily_limit(self) -> Tuple[bool, float]:
        """
        Check if daily drawdown limit has been hit.
        
        Returns:
            (limit_ok, current_drawdown_pct)
        """
        state = self.get_drawdown_state(DrawdownTimeframe.DAILY)
        return state.severity != DrawdownSeverity.BREACHED, state.current_drawdown_pct
    
    def check_weekly_limit(self) -> Tuple[bool, float]:
        """
        Check if weekly drawdown limit has been hit.
        
        Returns:
            (limit_ok, current_drawdown_pct)
        """
        state = self.get_drawdown_state(DrawdownTimeframe.WEEKLY)
        return state.severity != DrawdownSeverity.BREACHED, state.current_drawdown_pct
    
    def get_size_multiplier(self) -> float:
        """
        Get position size multiplier based on worst drawdown severity.
        
        Returns:
            Multiplier between 0.0 and 1.0
        """
        self._check_period_resets()
        
        # Find worst severity across all timeframes
        worst_severity = DrawdownSeverity.NORMAL
        for timeframe in DrawdownTimeframe:
            state = self._get_drawdown_state(timeframe)
            if list(DrawdownSeverity).index(state.severity) > list(DrawdownSeverity).index(worst_severity):
                worst_severity = state.severity
        
        base_multiplier = self.SIZE_MULTIPLIERS[worst_severity]
        
        # Further reduce if in defensive mode
        if self.in_defensive_mode:
            base_multiplier *= 0.5
        
        return base_multiplier
    
    def enter_defensive_mode(self, reason: str = "Manual trigger") -> None:
        """Manually enter defensive mode."""
        if not self.in_defensive_mode:
            self.in_defensive_mode = True
            self._log_event(DrawdownEvent(
                timestamp=datetime.utcnow(),
                event_type='defensive_mode',
                timeframe=DrawdownTimeframe.TOTAL,
                old_severity=None,
                new_severity=self._current_severities[DrawdownTimeframe.TOTAL],
                drawdown_pct=self._get_drawdown_state(DrawdownTimeframe.TOTAL).current_drawdown_pct,
                details=f"Entered defensive mode: {reason}"
            ))
            logger.warning(f"DEFENSIVE MODE ACTIVATED: {reason}")
    
    def exit_defensive_mode(self, reason: str = "Manual exit") -> None:
        """Exit defensive mode."""
        if self.in_defensive_mode:
            self.in_defensive_mode = False
            self._log_event(DrawdownEvent(
                timestamp=datetime.utcnow(),
                event_type='defensive_mode',
                timeframe=DrawdownTimeframe.TOTAL,
                old_severity=None,
                new_severity=self._current_severities[DrawdownTimeframe.TOTAL],
                drawdown_pct=self._get_drawdown_state(DrawdownTimeframe.TOTAL).current_drawdown_pct,
                details=f"Exited defensive mode: {reason}"
            ))
            logger.info(f"Defensive mode deactivated: {reason}")
    
    def get_status(self) -> Dict:
        """Get comprehensive status."""
        states = self.get_all_states()
        return {
            'current_equity': self.current_equity,
            'initial_equity': self.initial_equity,
            'in_defensive_mode': self.in_defensive_mode,
            'size_multiplier': self.get_size_multiplier(),
            'trading_allowed': self.check_trading_allowed()[0],
            'states': {tf.value: state.to_dict() for tf, state in states.items()},
            'worst_severity': max(
                (s.severity for s in states.values()),
                key=lambda x: list(DrawdownSeverity).index(x)
            ).value,
        }
    
    def get_events(self, limit: int = 100, event_type: str = None) -> List[Dict]:
        """Get recent drawdown events."""
        events = self._events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return [e.to_dict() for e in events[-limit:]]
    
    def _get_drawdown_state(self, timeframe: DrawdownTimeframe) -> DrawdownState:
        """Internal method to calculate drawdown state."""
        peak = self._peaks[timeframe]
        limit = self._limits[timeframe]
        
        if peak <= 0:
            drawdown_pct = 0.0
        else:
            drawdown_pct = ((peak - self.current_equity) / peak) * 100
        
        drawdown_pct = max(0, drawdown_pct)  # Can't have negative drawdown
        
        # Determine severity
        severity = DrawdownSeverity.NORMAL
        utilization = drawdown_pct / limit if limit > 0 else 0
        
        for sev in reversed(list(DrawdownSeverity)):
            if utilization >= self.SEVERITY_THRESHOLDS[sev]:
                severity = sev
                break
        
        return DrawdownState(
            timeframe=timeframe,
            current_drawdown_pct=drawdown_pct,
            peak_equity=peak,
            current_equity=self.current_equity,
            limit_pct=limit,
            severity=severity,
            last_updated=datetime.utcnow()
        )
    
    def _check_period_resets(self) -> None:
        """Check and reset periods if needed."""
        now = datetime.utcnow()
        
        # Daily reset
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if today_start > self._period_starts[DrawdownTimeframe.DAILY]:
            self._reset_period(DrawdownTimeframe.DAILY, today_start)
        
        # Weekly reset (Sunday)
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        if week_start > self._period_starts[DrawdownTimeframe.WEEKLY]:
            self._reset_period(DrawdownTimeframe.WEEKLY, week_start)
        
        # Monthly reset
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if month_start > self._period_starts[DrawdownTimeframe.MONTHLY]:
            self._reset_period(DrawdownTimeframe.MONTHLY, month_start)
    
    def _reset_period(self, timeframe: DrawdownTimeframe, new_start: datetime) -> None:
        """Reset a period's peak to current equity."""
        old_peak = self._peaks[timeframe]
        self._peaks[timeframe] = self.current_equity
        self._period_starts[timeframe] = new_start
        self._current_severities[timeframe] = DrawdownSeverity.NORMAL
        
        self._log_event(DrawdownEvent(
            timestamp=datetime.utcnow(),
            event_type='reset',
            timeframe=timeframe,
            old_severity=None,
            new_severity=DrawdownSeverity.NORMAL,
            drawdown_pct=0.0,
            details=f"Period reset. Old peak: {old_peak:.2f}, New peak: {self.current_equity:.2f}"
        ))
        
        logger.info(f"Drawdown period reset: {timeframe.value}")
    
    def _check_defensive_mode(self) -> None:
        """Auto-enter/exit defensive mode based on drawdown."""
        if not self.enable_defensive_mode:
            return
        
        # Find max utilization
        max_utilization = 0.0
        for timeframe in DrawdownTimeframe:
            state = self._get_drawdown_state(timeframe)
            max_utilization = max(max_utilization, state.utilization)
        
        # Enter defensive mode if approaching limits
        if max_utilization >= self.defensive_threshold and not self.in_defensive_mode:
            self.enter_defensive_mode(f"Drawdown utilization at {max_utilization:.1%}")
        
        # Exit defensive mode if recovered
        elif max_utilization < self.defensive_threshold * 0.5 and self.in_defensive_mode:
            self.exit_defensive_mode(f"Drawdown recovered to {max_utilization:.1%}")
    
    def _log_event(self, event: DrawdownEvent) -> None:
        """Log a drawdown event."""
        self._events.append(event)
        
        # Trim if needed
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        
        # Persist to file
        try:
            log_file = Path(self.log_dir) / f"drawdown_events_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist drawdown event: {e}")


# Singleton instance
_drawdown_governor: Optional[DrawdownGovernor] = None


def get_drawdown_governor(initial_equity: float = 10000.0) -> DrawdownGovernor:
    """Get the global DrawdownGovernor instance."""
    global _drawdown_governor
    if _drawdown_governor is None:
        _drawdown_governor = DrawdownGovernor(initial_equity=initial_equity)
    return _drawdown_governor


def reset_drawdown_governor() -> None:
    """Reset the global DrawdownGovernor instance."""
    global _drawdown_governor
    _drawdown_governor = None
