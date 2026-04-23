"""
Decision Narrative Engine - v10.4-TRUST-GRADE

Translate system decisions into human-readable explanations.
Replaces opaque "status codes" with clear, English sentences explaining WHY.

Features:
- Structured narratives for common decisions (No Trade, Size Rededuced)
- Gate reference tracking
- Database storage and real-time streaming
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class NarrativeType(Enum):
    """Broad categories of decision narratives."""
    WHY_NO_TRADE = "why_no_trade"           # Vetoed by a gate
    WHY_TRADE_ALLOWED = "why_trade_allowed" # Passed all gates
    WHY_SIZE_REDUCED = "why_size_reduced"   # Risk engine reduced size
    WHY_POS_CLOSED = "why_pos_closed"       # Exit decision
    SYSTEM_EVENT = "system_event"           # Startup, Shutdown, Mode Change


@dataclass
class DecisionNarrative:
    """
    A human-readable explanation of a system decision.
    """
    narrative_id: str
    timestamp: datetime
    narrative_type: NarrativeType
    symbol: str
    summary_text: str           # Single sentence summary
    detailed_text: str          # Full explanation
    
    primary_gate: str           # The main gate involved (e.g., "RiskCheck")
    all_gates_passed: List[str] # List of gates that passed
    blocking_gate: Optional[str] = None # The gate that failed (if any)
    
    data_snapshot: Dict = field(default_factory=dict) # Relevant data context
    
    def to_dict(self) -> Dict:
        return {
            'narrative_id': self.narrative_id,
            'timestamp': self.timestamp.isoformat(),
            'type': self.narrative_type.value,
            'symbol': self.symbol,
            'summary': self.summary_text,
            'details': self.detailed_text,
            'primary_gate': self.primary_gate,
            'blocking_gate': self.blocking_gate,
            'gates_passed': self.all_gates_passed
        }


class NarrativeEngine:
    """
    Generates and distributes decision narratives.
    """
    
    def __init__(self, websocket_manager=None, database_manager=None):
        self._ws = websocket_manager
        self._db = database_manager
        
    def generate_no_trade_narrative(
        self,
        symbol: str,
        gate_name: str,
        reason: str,
        context: Dict
    ) -> DecisionNarrative:
        """
        Create a narrative for a blocked trade.
        """
        import uuid
        
        # Build human-readable text logic here
        summary = f"Trade blocked by {gate_name}."
        detail = f"The {gate_name} vetoed this trade because: {reason}."
        
        narrative = DecisionNarrative(
            narrative_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            narrative_type=NarrativeType.WHY_NO_TRADE,
            symbol=symbol,
            summary_text=summary,
            detailed_text=detail,
            primary_gate=gate_name,
            blocking_gate=gate_name,
            all_gates_passed=[], # In a real implementation we'd pass this in
            data_snapshot=context
        )
        
        self._publish(narrative)
        return narrative

    def generate_trade_allowed_narrative(
        self,
        symbol: str,
        strategy_name: str,
        size: float,
        context: Dict
    ) -> DecisionNarrative:
        """Create a narrative for an allowed trade."""
        import uuid
        
        summary = f"Trade approved for {symbol} ({strategy_name})."
        detail = f"All safety gates passed. Position size calculated at {size}."
        
        gates_passed = context.get("gates_passed", ["Risk", "Bias", "Spread", "Capital"])
        
        narrative = DecisionNarrative(
            narrative_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            narrative_type=NarrativeType.WHY_TRADE_ALLOWED,
            symbol=symbol,
            summary_text=summary,
            detailed_text=detail,
            primary_gate="FinalExecution",
            all_gates_passed=gates_passed,
            blocking_gate=None,
            data_snapshot=context
        )
        
        self._publish(narrative)
        return narrative

    def generate_hold_narrative(
        self,
        symbol: str,
        failed_conditions: Dict[str, str],
        context: Dict,
    ) -> DecisionNarrative:
        """
        Create a narrative explaining why no trade was taken (HOLD).
        
        Args:
            symbol: Trading pair
            failed_conditions: Dict of condition_name -> failure_reason
            context: Market data snapshot
        """
        import uuid
        
        if not failed_conditions:
            reason_text = "No clear directional edge detected."
        else:
            reasons = [f"{k}: {v}" for k, v in failed_conditions.items()]
            reason_text = "; ".join(reasons)
        
        summary = f"HOLD on {symbol} — {len(failed_conditions)} conditions unmet."
        detail = f"No trade taken because: {reason_text}"
        
        narrative = DecisionNarrative(
            narrative_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            narrative_type=NarrativeType.WHY_NO_TRADE,
            symbol=symbol,
            summary_text=summary,
            detailed_text=detail,
            primary_gate="SignalEngine",
            all_gates_passed=[],
            blocking_gate="SignalEngine",
            data_snapshot={**context, "failed_conditions": failed_conditions},
        )
        
        self._publish(narrative)
        return narrative

    def generate_size_reduced_narrative(
        self,
        symbol: str,
        original_size: float,
        reduced_size: float,
        reason: str,
        context: Dict,
    ) -> DecisionNarrative:
        """Create a narrative explaining why position size was reduced."""
        import uuid
        
        reduction_pct = (1 - reduced_size / original_size) * 100 if original_size > 0 else 0
        summary = f"Position size reduced by {reduction_pct:.0f}% on {symbol}."
        detail = (
            f"Original size: {original_size:.4f}, reduced to {reduced_size:.4f}. "
            f"Reason: {reason}"
        )
        
        narrative = DecisionNarrative(
            narrative_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            narrative_type=NarrativeType.WHY_SIZE_REDUCED,
            symbol=symbol,
            summary_text=summary,
            detailed_text=detail,
            primary_gate="CapitalGovernor",
            all_gates_passed=["Risk", "Bias"],
            blocking_gate=None,
            data_snapshot={**context, "original_size": original_size, "reduced_size": reduced_size},
        )
        
        self._publish(narrative)
        return narrative

    def generate_exit_narrative(
        self,
        symbol: str,
        exit_reason: str,
        pnl: float,
        context: Dict,
    ) -> DecisionNarrative:
        """Create a narrative explaining an exit decision."""
        import uuid
        
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        summary = f"Position closed on {symbol} ({pnl_str})."
        detail = f"Exit reason: {exit_reason}. Realized P&L: {pnl_str}."
        
        narrative = DecisionNarrative(
            narrative_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            narrative_type=NarrativeType.WHY_POS_CLOSED,
            symbol=symbol,
            summary_text=summary,
            detailed_text=detail,
            primary_gate="TradeManagement",
            all_gates_passed=["ExitCondition"],
            blocking_gate=None,
            data_snapshot={**context, "pnl": pnl, "exit_reason": exit_reason},
        )
        
        self._publish(narrative)
        return narrative

    def _publish(self, narrative: DecisionNarrative):
        """Persist and stream the narrative."""
        # 1. Log to console (structured)
        log_data = narrative.to_dict()
        if narrative.narrative_type == NarrativeType.WHY_NO_TRADE:
            logger.info(f"NARRATIVE: 🛑 {narrative.summary_text} ({narrative.detailed_text})")
        elif narrative.narrative_type == NarrativeType.WHY_POS_CLOSED:
            logger.info(f"NARRATIVE: 📤 {narrative.summary_text}")
        elif narrative.narrative_type == NarrativeType.WHY_SIZE_REDUCED:
            logger.info(f"NARRATIVE: ⚠️ {narrative.summary_text}")
        else:
            logger.info(f"NARRATIVE: ✅ {narrative.summary_text}")
            
        # 2. Persist to state manager (if available)
        try:
            from .state_manager import get_state_manager
            sm = get_state_manager()
            key = f"narrative:{narrative.narrative_id}"
            sm.save(key, log_data)
        except Exception as e:
            logger.debug(f"Could not persist narrative: {e}")
            
        # 3. Stream to Frontend via WebSocket (if available)
        if self._ws:
            try:
                self._ws.broadcast("narrative", log_data)
            except Exception as e:
                logger.debug(f"Could not stream narrative: {e}")


# Singleton
_narrative_engine: Optional[NarrativeEngine] = None

def get_narrative_engine() -> NarrativeEngine:
    global _narrative_engine
    if _narrative_engine is None:
        _narrative_engine = NarrativeEngine()
    return _narrative_engine

