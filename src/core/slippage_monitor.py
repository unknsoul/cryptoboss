"""
Slippage Monitor - Execution Quality Tracking

Monitors and analyzes slippage between expected and actual fill prices.
Provides metrics for execution quality and alerts on abnormal slippage.

v11.0 - Production-Grade Platform Upgrade
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import statistics
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SlippageDirection(str, Enum):
    """Direction of slippage."""
    FAVORABLE = "favorable"      # Got better price than expected
    UNFAVORABLE = "unfavorable"  # Got worse price than expected
    NEUTRAL = "neutral"          # No slippage


class SlippageSeverity(str, Enum):
    """Severity of slippage."""
    MINIMAL = "minimal"      # < 1 bps
    LOW = "low"              # 1-5 bps
    MODERATE = "moderate"    # 5-15 bps
    HIGH = "high"            # 15-30 bps
    EXTREME = "extreme"      # > 30 bps


@dataclass
class SlippageRecord:
    """Record of slippage for a single execution."""
    timestamp: datetime
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    expected_price: float
    fill_price: float
    size: float
    slippage_bps: float
    direction: SlippageDirection
    severity: SlippageSeverity
    latency_ms: Optional[float] = None
    exchange: str = "binance"
    
    @property
    def slippage_usd(self) -> float:
        """Calculate absolute slippage in USD."""
        return abs(self.fill_price - self.expected_price) * self.size
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'expected_price': self.expected_price,
            'fill_price': self.fill_price,
            'size': self.size,
            'slippage_bps': self.slippage_bps,
            'slippage_usd': self.slippage_usd,
            'direction': self.direction.value,
            'severity': self.severity.value,
            'latency_ms': self.latency_ms,
            'exchange': self.exchange,
        }


@dataclass
class SlippageStats:
    """Aggregated slippage statistics."""
    period: str
    total_executions: int
    total_slippage_bps: float
    avg_slippage_bps: float
    median_slippage_bps: float
    p95_slippage_bps: float
    max_slippage_bps: float
    favorable_count: int
    unfavorable_count: int
    total_slippage_usd: float
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_symbol: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'period': self.period,
            'total_executions': self.total_executions,
            'total_slippage_bps': self.total_slippage_bps,
            'avg_slippage_bps': self.avg_slippage_bps,
            'median_slippage_bps': self.median_slippage_bps,
            'p95_slippage_bps': self.p95_slippage_bps,
            'max_slippage_bps': self.max_slippage_bps,
            'favorable_count': self.favorable_count,
            'unfavorable_count': self.unfavorable_count,
            'favorable_rate': self.favorable_count / self.total_executions if self.total_executions > 0 else 0,
            'total_slippage_usd': self.total_slippage_usd,
            'by_severity': self.by_severity,
            'by_symbol': self.by_symbol,
        }


class SlippageMonitor:
    """
    Slippage Monitor - Track and analyze execution quality.
    
    Features:
    - Record expected vs actual fill prices
    - Calculate slippage in basis points
    - Aggregate statistics over time periods
    - Alert on abnormal slippage
    - Persist records for analysis
    
    Usage:
        monitor = SlippageMonitor()
        
        # Record execution
        monitor.record_execution(
            order_id="abc123",
            symbol="BTC/USDT",
            side="buy",
            expected_price=45000.0,
            fill_price=45010.0,
            size=0.5
        )
        
        # Check if slippage is acceptable
        is_ok = monitor.is_slippage_acceptable()
        
        # Get statistics
        stats = monitor.get_stats(hours=24)
    """
    
    # Severity thresholds in basis points
    SEVERITY_THRESHOLDS = {
        SlippageSeverity.MINIMAL: 1.0,
        SlippageSeverity.LOW: 5.0,
        SlippageSeverity.MODERATE: 15.0,
        SlippageSeverity.HIGH: 30.0,
        SlippageSeverity.EXTREME: float('inf'),
    }
    
    # Acceptable slippage threshold (bps)
    DEFAULT_ACCEPTABLE_THRESHOLD = 20.0
    
    def __init__(
        self,
        acceptable_threshold_bps: float = 20.0,
        alert_threshold_bps: float = 30.0,
        max_records: int = 50000,
        log_dir: str = "logs/slippage",
        on_alert: callable = None
    ):
        """
        Initialize SlippageMonitor.
        
        Args:
            acceptable_threshold_bps: Threshold for "acceptable" slippage
            alert_threshold_bps: Threshold to trigger alerts
            max_records: Maximum records to keep in memory
            log_dir: Directory for slippage logs
            on_alert: Callback for slippage alerts
        """
        self.acceptable_threshold_bps = acceptable_threshold_bps
        self.alert_threshold_bps = alert_threshold_bps
        self.max_records = max_records
        self.log_dir = log_dir
        self.on_alert = on_alert
        
        self._records: List[SlippageRecord] = []
        self._by_symbol: Dict[str, List[SlippageRecord]] = {}
        
        # Running statistics
        self._total_executions = 0
        self._total_slippage_bps = 0.0
        self._total_slippage_usd = 0.0
        self._favorable_count = 0
        self._unfavorable_count = 0
        
        # Ensure log directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SlippageMonitor initialized: threshold={acceptable_threshold_bps}bps, alert={alert_threshold_bps}bps")
    
    def record_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        expected_price: float,
        fill_price: float,
        size: float,
        latency_ms: float = None,
        exchange: str = "binance"
    ) -> SlippageRecord:
        """
        Record an execution and calculate slippage.
        
        Args:
            order_id: Unique order identifier
            symbol: Trading pair
            side: 'buy' or 'sell'
            expected_price: Expected fill price
            fill_price: Actual fill price
            size: Position size
            latency_ms: Order latency in milliseconds
            exchange: Exchange name
            
        Returns:
            SlippageRecord with calculated slippage
        """
        # Calculate slippage in basis points
        if expected_price > 0:
            # For buys: positive slippage = unfavorable, negative = favorable
            # For sells: positive slippage = favorable, negative = unfavorable
            raw_slippage = ((fill_price - expected_price) / expected_price) * 10000
            
            if side == 'sell':
                raw_slippage = -raw_slippage
        else:
            raw_slippage = 0.0
        
        # Determine direction
        if abs(raw_slippage) < 0.1:
            direction = SlippageDirection.NEUTRAL
        elif raw_slippage > 0:
            direction = SlippageDirection.UNFAVORABLE
        else:
            direction = SlippageDirection.FAVORABLE
        
        # Determine severity (based on absolute value)
        abs_slippage = abs(raw_slippage)
        severity = SlippageSeverity.EXTREME
        for sev, threshold in sorted(self.SEVERITY_THRESHOLDS.items(), key=lambda x: x[1]):
            if abs_slippage <= threshold:
                severity = sev
                break
        
        # Create record
        record = SlippageRecord(
            timestamp=datetime.utcnow(),
            order_id=order_id,
            symbol=symbol,
            side=side,
            expected_price=expected_price,
            fill_price=fill_price,
            size=size,
            slippage_bps=raw_slippage,
            direction=direction,
            severity=severity,
            latency_ms=latency_ms,
            exchange=exchange
        )
        
        # Store record
        self._records.append(record)
        
        if symbol not in self._by_symbol:
            self._by_symbol[symbol] = []
        self._by_symbol[symbol].append(record)
        
        # Update running stats
        self._total_executions += 1
        self._total_slippage_bps += raw_slippage
        self._total_slippage_usd += record.slippage_usd
        
        if direction == SlippageDirection.FAVORABLE:
            self._favorable_count += 1
        elif direction == SlippageDirection.UNFAVORABLE:
            self._unfavorable_count += 1
        
        # Check for alert
        if abs_slippage >= self.alert_threshold_bps:
            self._trigger_alert(record)
        
        # Trim if needed
        self._trim()
        
        # Persist
        self._persist_record(record)
        
        logger.debug(f"Slippage recorded: {symbol} {side} | {raw_slippage:.2f}bps ({severity.value})")
        
        return record
    
    def is_slippage_acceptable(self, hours: int = 24) -> bool:
        """
        Check if recent slippage is within acceptable range.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            True if average slippage is acceptable
        """
        stats = self.get_stats(hours=hours)
        return abs(stats.avg_slippage_bps) <= self.acceptable_threshold_bps
    
    def get_stats(self, hours: int = 24, symbol: str = None) -> SlippageStats:
        """
        Get aggregated slippage statistics.
        
        Args:
            hours: Number of hours to analyze
            symbol: Optional symbol filter
            
        Returns:
            SlippageStats with aggregated metrics
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        if symbol:
            records = [r for r in self._by_symbol.get(symbol, []) if r.timestamp >= cutoff]
        else:
            records = [r for r in self._records if r.timestamp >= cutoff]
        
        if not records:
            return SlippageStats(
                period=f"last_{hours}h",
                total_executions=0,
                total_slippage_bps=0,
                avg_slippage_bps=0,
                median_slippage_bps=0,
                p95_slippage_bps=0,
                max_slippage_bps=0,
                favorable_count=0,
                unfavorable_count=0,
                total_slippage_usd=0,
                by_severity={},
                by_symbol={}
            )
        
        slippages = [r.slippage_bps for r in records]
        abs_slippages = [abs(s) for s in slippages]
        sorted_abs = sorted(abs_slippages)
        
        # Calculate percentiles
        n = len(sorted_abs)
        p95_idx = int(n * 0.95)
        
        # Aggregate by severity
        by_severity = {}
        for sev in SlippageSeverity:
            by_severity[sev.value] = sum(1 for r in records if r.severity == sev)
        
        # Aggregate by symbol
        by_symbol = {}
        for r in records:
            if r.symbol not in by_symbol:
                by_symbol[r.symbol] = []
            by_symbol[r.symbol].append(r.slippage_bps)
        by_symbol = {s: statistics.mean(v) if v else 0 for s, v in by_symbol.items()}
        
        return SlippageStats(
            period=f"last_{hours}h",
            total_executions=len(records),
            total_slippage_bps=sum(slippages),
            avg_slippage_bps=statistics.mean(slippages),
            median_slippage_bps=statistics.median(slippages),
            p95_slippage_bps=sorted_abs[p95_idx] if p95_idx < n else sorted_abs[-1],
            max_slippage_bps=max(abs_slippages),
            favorable_count=sum(1 for r in records if r.direction == SlippageDirection.FAVORABLE),
            unfavorable_count=sum(1 for r in records if r.direction == SlippageDirection.UNFAVORABLE),
            total_slippage_usd=sum(r.slippage_usd for r in records),
            by_severity=by_severity,
            by_symbol=by_symbol
        )
    
    def get_recent_records(self, limit: int = 100, symbol: str = None) -> List[Dict]:
        """Get recent slippage records."""
        if symbol:
            records = self._by_symbol.get(symbol, [])
        else:
            records = self._records
        return [r.to_dict() for r in records[-limit:]]
    
    def get_status(self) -> Dict:
        """Get monitor status."""
        stats_24h = self.get_stats(hours=24)
        stats_1h = self.get_stats(hours=1)
        
        return {
            'total_records': len(self._records),
            'acceptable_threshold_bps': self.acceptable_threshold_bps,
            'is_acceptable': self.is_slippage_acceptable(),
            'stats_1h': stats_1h.to_dict(),
            'stats_24h': stats_24h.to_dict(),
            'lifetime': {
                'total_executions': self._total_executions,
                'avg_slippage_bps': self._total_slippage_bps / self._total_executions if self._total_executions > 0 else 0,
                'total_slippage_usd': self._total_slippage_usd,
                'favorable_rate': self._favorable_count / self._total_executions if self._total_executions > 0 else 0,
            }
        }
    
    def _trigger_alert(self, record: SlippageRecord) -> None:
        """Trigger slippage alert."""
        logger.warning(
            f"SLIPPAGE ALERT: {record.symbol} {record.side} | "
            f"{record.slippage_bps:.2f}bps (threshold: {self.alert_threshold_bps}bps) | "
            f"Expected: {record.expected_price}, Got: {record.fill_price}"
        )
        
        if self.on_alert:
            try:
                self.on_alert(record)
            except Exception as e:
                logger.error(f"Slippage alert callback failed: {e}")
    
    def _trim(self) -> None:
        """Trim old records if over limit."""
        if len(self._records) > self.max_records:
            # Remove oldest 10%
            cutoff = int(self.max_records * 0.1)
            removed = self._records[:cutoff]
            self._records = self._records[cutoff:]
            
            # Update symbol index
            removed_ids = {r.order_id for r in removed}
            for symbol in self._by_symbol:
                self._by_symbol[symbol] = [
                    r for r in self._by_symbol[symbol] 
                    if r.order_id not in removed_ids
                ]
    
    def _persist_record(self, record: SlippageRecord) -> None:
        """Persist record to file."""
        try:
            log_file = Path(self.log_dir) / f"slippage_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(record.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist slippage record: {e}")


# Singleton instance
_slippage_monitor: Optional[SlippageMonitor] = None


def get_slippage_monitor() -> SlippageMonitor:
    """Get the global SlippageMonitor instance."""
    global _slippage_monitor
    if _slippage_monitor is None:
        _slippage_monitor = SlippageMonitor()
    return _slippage_monitor
