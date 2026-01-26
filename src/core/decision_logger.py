"""
Decision Logger - Complete Decision Auditability

Logs every decision and non-decision with full context.
This enables post-mortem analysis and ensures no unexplained trades.

Logs every:
- Context decision
- Bias decision
- Permission block
- Entry proposal
- Entry execution/rejection
- Exit reason
- Risk event

Architecture: Transparent decision making for professional trading
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of decisions to log."""
    CONTEXT_DECISION = "context_decision"
    BIAS_DECISION = "bias_decision"
    PERMISSION_CHECK = "permission_check"
    PERMISSION_BLOCK = "permission_block"
    ENTRY_PROPOSAL = "entry_proposal"
    ENTRY_EXECUTED = "entry_executed"
    ENTRY_REJECTED = "entry_rejected"
    EXIT_EXECUTED = "exit_executed"
    RISK_EVENT = "risk_event"
    NO_TRADE_PERIOD = "no_trade_period"


@dataclass
class DecisionLog:
    """Single decision log entry."""
    timestamp: datetime
    decision_type: DecisionType
    symbol: str
    
    # Decision outcome
    approved: bool
    reason: str
    
    # Context
    context: Dict
    
    # Metadata
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "decision_type": self.decision_type.value,
            "symbol": self.symbol,
            "approved": self.approved,
            "reason": self.reason,
            "context": self.context,
            "metadata": self.metadata
        }


class DecisionLogger:
    """
    Decision Logger - Audit trail for all trading decisions.
    
    Every candle/tick should have a logged decision.
    No trades should occur without a complete decision chain.
    
    Usage:
        logger = DecisionLogger(log_dir="logs/decisions")
        
        # Log context decision
        logger.log_context_decision(
            symbol="BTC/USDT",
            context=market_context,
            approved=True,
            reason="Trending market with acceptable liquidity"
        )
        
        # Log permission block
        logger.log_permission_block(
            symbol="BTC/USDT",
            reason="Spread 0.15% exceeds maximum 0.10%",
            context={"spread_bps": 15, "limit_bps": 10}
        )
        
        # Query decisions
        no_trade_periods = logger.get_no_trade_periods(hours=24)
        rejected_entries = logger.get_rejected_entries(hours=24)
    """
    
    def __init__(
        self,
        log_dir: str = "logs/decisions",
        max_memory_logs: int = 10000,
        auto_flush_interval: int = 100
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_logs = max_memory_logs
        self.auto_flush_interval = auto_flush_interval
        
        # In-memory log buffer
        self.log_buffer: List[DecisionLog] = []
        self.flush_counter = 0
        
        # Current log file
        self.current_log_file = self._get_log_file_path()
        
        logger.info(f"DecisionLogger initialized: {self.log_dir}")
    
    def log_context_decision(
        self,
        symbol: str,
        context: Any,  # MarketContext
        approved: bool,
        reason: str,
        metadata: Dict = None
    ):
        """Log a market context decision."""
        self._log(
            DecisionType.CONTEXT_DECISION,
            symbol=symbol,
            approved=approved,
            reason=reason,
            context={
                "regime": context.regime.value if hasattr(context, 'regime') else "unknown",
                "trading_allowed": context.trading_allowed if hasattr(context, 'trading_allowed') else False,
                "confidence": context.confidence if hasattr(context, 'confidence') else 0,
                "trend_1h": context.trend_1h if hasattr(context, 'trend_1h') else None,
                "trend_4h": context.trend_4h if hasattr(context, 'trend_4h') else None,
                "volatility_regime": context.volatility_regime if hasattr(context, 'volatility_regime') else None
            },
            metadata=metadata or {}
        )
    
    def log_bias_decision(
        self,
        symbol: str,
        bias: Any,  # BiasState
        reason: str,
        metadata: Dict = None
    ):
        """Log a bias determination."""
        self._log(
            DecisionType.BIAS_DECISION,
            symbol=symbol,
            approved=bias.bias.value != "neutral" if hasattr(bias, 'bias') else False,
            reason=reason,
            context={
                "bias": bias.bias.value if hasattr(bias, 'bias') else "unknown",
                "conviction": bias.conviction if hasattr(bias, 'conviction') else 0,
                "higher_tf_trend": bias.higher_tf_trend if hasattr(bias, 'higher_tf_trend') else None,
                "momentum_direction": bias.momentum_direction if hasattr(bias, 'momentum_direction') else None
            },
            metadata=metadata or {}
        )
    
    def log_permission_check(
        self,
        symbol: str,
        permission: Any,  # PermissionResult
        metadata: Dict = None
    ):
        """Log a permission check result."""
        decision_type = (
            DecisionType.PERMISSION_BLOCK if not permission.approved
            else DecisionType.PERMISSION_CHECK
        )
        
        self._log(
            decision_type,
            symbol=symbol,
            approved=permission.approved,
            reason=permission.reason,
            context={
                "checks_passed": permission.checks_passed if hasattr(permission, 'checks_passed') else {},
                "denial_category": permission.denial_category.value if hasattr(permission, 'denial_category') and permission.denial_category else None
            },
            metadata=metadata or {}
        )
    
    def log_entry_proposal(
        self,
        symbol: str,
        strategy_id: str,
        direction: str,
        entry_price: float,
        reasoning: str,
        confidence: float,
        metadata: Dict = None
    ):
        """Log a strategy entry proposal."""
        self._log(
            DecisionType.ENTRY_PROPOSAL,
            symbol=symbol,
            approved=True,  # Proposal itself is logged, execution is separate
            reason=reasoning,
            context={
                "strategy_id": strategy_id,
                "direction": direction,
                "entry_price": entry_price,
                "confidence": confidence
            },
            metadata=metadata or {}
        )
    
    def log_entry_executed(
        self,
        symbol: str,
        strategy_id: str,
        direction: str,
        entry_price: float,
        size: float,
        reason: str,
        metadata: Dict = None
    ):
        """Log an executed entry."""
        self._log(
            DecisionType.ENTRY_EXECUTED,
            symbol=symbol,
            approved=True,
            reason=reason,
            context={
                "strategy_id": strategy_id,
                "direction": direction,
                "entry_price": entry_price,
                "size": size
            },
            metadata=metadata or {}
        )
    
    def log_entry_rejected(
        self,
        symbol: str,
        strategy_id: str,
        direction: str,
        reason: str,
        metadata: Dict = None
    ):
        """Log a rejected entry proposal."""
        self._log(
            DecisionType.ENTRY_REJECTED,
            symbol=symbol,
            approved=False,
            reason=reason,
            context={
                "strategy_id": strategy_id,
                "direction": direction
            },
            metadata=metadata or {}
        )
    
    def log_exit_executed(
        self,
        symbol: str,
        strategy_id: str,
        exit_price: float,
        pnl: float,
        reason: str,
        metadata: Dict = None
    ):
        """Log an executed exit."""
        self._log(
            DecisionType.EXIT_EXECUTED,
            symbol=symbol,
            approved=True,
            reason=reason,
            context={
                "strategy_id": strategy_id,
                "exit_price": exit_price,
                "pnl": pnl
            },
            metadata=metadata or {}
        )
    
    def log_risk_event(
        self,
        symbol: str,
        event_type: str,
        reason: str,
        metadata: Dict = None
    ):
        """Log a risk event (circuit breaker, kill switch, etc.)."""
        self._log(
            DecisionType.RISK_EVENT,
            symbol=symbol,
            approved=False,
            reason=reason,
            context={
                "event_type": event_type
            },
            metadata=metadata or {}
        )
    
    def log_no_trade_period(
        self,
        symbol: str,
        reason: str,
        metadata: Dict = None
    ):
        """Log a period where no trading occurred."""
        self._log(
            DecisionType.NO_TRADE_PERIOD,
            symbol=symbol,
            approved=False,
            reason=reason,
            context={},
            metadata=metadata or {}
        )
    
    def _log(
        self,
        decision_type: DecisionType,
        symbol: str,
        approved: bool,
        reason: str,
        context: Dict,
        metadata: Dict
    ):
        """Internal logging method."""
        log_entry = DecisionLog(
            timestamp=datetime.now(),
            decision_type=decision_type,
            symbol=symbol,
            approved=approved,
            reason=reason,
            context=context,
            metadata=metadata
        )
        
        # Add to buffer
        self.log_buffer.append(log_entry)
        self.flush_counter += 1
        
        # Auto-flush if needed
        if self.flush_counter >= self.auto_flush_interval:
            self.flush()
        
        # Trim buffer if too large
        if len(self.log_buffer) > self.max_memory_logs:
            self.log_buffer = self.log_buffer[-self.max_memory_logs:]
    
    def flush(self):
        """Flush in-memory logs to disk."""
        if not self.log_buffer:
            return
        
        try:
            with open(self.current_log_file, 'a') as f:
                for log_entry in self.log_buffer:
                    f.write(json.dumps(log_entry.to_dict()) + '\n')
            
            self.flush_counter = 0
            logger.debug(f"Flushed {len(self.log_buffer)} decision logs to disk")
            
        except Exception as e:
            logger.error(f"Failed to flush decision logs: {e}")
    
    def get_no_trade_periods(self, hours: int = 24) -> List[Dict]:
        """
        Get all NO_TRADE periods in last N hours.
        
        Returns: List of no-trade period logs
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        no_trade_logs = [
            log.to_dict() for log in self.log_buffer
            if log.decision_type == DecisionType.NO_TRADE_PERIOD
            and log.timestamp >= cutoff
        ]
        
        return no_trade_logs
    
    def get_rejected_entries(self, hours: int = 24) -> List[Dict]:
        """
        Get all rejected entry proposals in last N hours.
        
        Returns: List of rejected entry logs
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        rejected_logs = [
            log.to_dict() for log in self.log_buffer
            if log.decision_type == DecisionType.ENTRY_REJECTED
            and log.timestamp >= cutoff
        ]
        
        return rejected_logs
    
    def get_permission_blocks(self, hours: int = 24) -> List[Dict]:
        """
        Get all permission blocks in last N hours.
        
        Returns: List of permission block logs
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        block_logs = [
            log.to_dict() for log in self.log_buffer
            if log.decision_type == DecisionType.PERMISSION_BLOCK
            and log.timestamp >= cutoff
        ]
        
        return block_logs
    
    def analyze_decision_chain(self, timestamp: datetime, window_minutes: int = 5) -> List[Dict]:
        """
        Get all decisions around a specific timestamp.
        
        Useful for analyzing what happened around a specific trade or event.
        
        Args:
            timestamp: Timestamp to analyze around
            window_minutes: Window in minutes before/after timestamp
            
        Returns: List of all decisions in window
        """
        start = timestamp - timedelta(minutes=window_minutes)
        end = timestamp + timedelta(minutes=window_minutes)
        
        chain = [
            log.to_dict() for log in self.log_buffer
            if start <= log.timestamp <= end
        ]
        
        # Sort by timestamp
        chain.sort(key=lambda x: x['timestamp'])
        
        return chain
    
    def get_decision_stats(self, hours: int = 24) -> Dict:
        """
        Get decision statistics for last N hours.
        
        Returns: Statistics dictionary
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_logs = [log for log in self.log_buffer if log.timestamp >= cutoff]
        
        if not recent_logs:
            return {"message": "No recent decisions"}
        
        # Count by decision type
        type_counts = {}
        for log in recent_logs:
            dtype = log.decision_type.value
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        # Count approvals vs denials
        approvals = sum(1 for log in recent_logs if log.approved)
        denials = len(recent_logs) - approvals
        
        # Permission block reasons
        block_reasons = {}
        for log in recent_logs:
            if log.decision_type == DecisionType.PERMISSION_BLOCK:
                reason = log.context.get('denial_category', 'unknown')
                block_reasons[reason] = block_reasons.get(reason, 0) + 1
        
        return {
            "total_decisions": len(recent_logs),
            "approvals": approvals,
            "denials": denials,
            "approval_rate": approvals / len(recent_logs) if recent_logs else 0,
            "decision_type_counts": type_counts,
            "permission_block_reasons": block_reasons,
            "timeframe_hours": hours
        }
    
    def _get_log_file_path(self) -> Path:
        """Get current log file path (daily rotation)."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"decisions_{date_str}.jsonl"


# Singleton instance
_decision_logger: Optional[DecisionLogger] = None


def get_decision_logger(log_dir: str = "logs/decisions") -> DecisionLogger:
    """Get the global DecisionLogger instance."""
    global _decision_logger
    if _decision_logger is None:
        _decision_logger = DecisionLogger(log_dir=log_dir)
    return _decision_logger
