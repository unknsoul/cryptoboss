"""
Bias Pre-Filter - v10.1-FINAL Component

Eliminates invalid proposals BEFORE scoring:
- NEUTRAL bias → discard ALL proposals
- LONG bias → discard all SHORT proposals
- SHORT bias → discard all LONG proposals

This runs BEFORE the scoring contract to prevent
wasted computation and ensure bias is never bypassed.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class FilterReason(Enum):
    """Reasons for proposal filtering."""
    BIAS_NEUTRAL = "bias_neutral"
    DIRECTION_MISMATCH = "direction_mismatch"
    PASSED = "passed"


@dataclass
class FilteredProposal:
    """Record of a filtered (discarded) proposal."""
    strategy_id: str
    direction: str
    proposed_at: datetime
    filter_reason: FilterReason
    bias_at_filter: str
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'strategy_id': self.strategy_id,
            'direction': self.direction,
            'proposed_at': self.proposed_at.isoformat(),
            'filter_reason': self.filter_reason.value,
            'bias_at_filter': self.bias_at_filter,
            'confidence': self.confidence
        }


@dataclass
class BiasFilterResult:
    """Result of bias pre-filtering."""
    passed_proposals: List[Dict]
    filtered_proposals: List[FilteredProposal]
    current_bias: str
    timestamp: datetime
    
    @property
    def all_filtered(self) -> bool:
        return len(self.passed_proposals) == 0
    
    @property
    def pass_count(self) -> int:
        return len(self.passed_proposals)
    
    @property
    def filter_count(self) -> int:
        return len(self.filtered_proposals)
    
    def to_dict(self) -> Dict:
        return {
            'passed_count': self.pass_count,
            'filtered_count': self.filter_count,
            'current_bias': self.current_bias,
            'timestamp': self.timestamp.isoformat(),
            'all_filtered': self.all_filtered,
            'filtered_details': [f.to_dict() for f in self.filtered_proposals]
        }


class BiasPreFilter:
    """
    Bias Pre-Filter - eliminates invalid proposals BEFORE scoring.
    
    This is a HARD GATE that runs before the scoring contract.
    No proposal can survive if it conflicts with the current bias.
    
    Rules:
    1. NEUTRAL bias → ALL proposals discarded
    2. LONG_BIAS → only LONG proposals pass
    3. SHORT_BIAS → only SHORT proposals pass
    
    Usage:
        pre_filter = BiasPreFilter()
        
        # Get current bias
        bias = get_bias_engine().get_current_bias()
        
        # Filter proposals BEFORE scoring
        result = pre_filter.filter_proposals(
            proposals=raw_proposals,
            current_bias=bias.bias.value
        )
        
        if result.all_filtered:
            logger.info("All proposals filtered by bias gate")
            return  # DO NOT proceed to scoring
        
        # Only passed proposals go to scoring
        scoring_contract.validate_batch(result.passed_proposals)
    """
    
    # Valid directions
    VALID_DIRECTIONS = {'LONG', 'SHORT'}
    
    # Bias to allowed direction mapping
    BIAS_ALLOWED_DIRECTIONS = {
        'long_bias': {'LONG'},
        'long_only': {'LONG'},
        'short_bias': {'SHORT'},
        'short_only': {'SHORT'},
        'neutral': set(),  # No directions allowed
    }
    
    def __init__(self):
        self._filter_history: List[BiasFilterResult] = []
        self._total_filtered: int = 0
        self._total_passed: int = 0
        
        logger.info("BiasPreFilter initialized - HARD GATE active")
    
    def filter_proposals(
        self,
        proposals: List[Dict[str, Any]],
        current_bias: str
    ) -> BiasFilterResult:
        """
        Filter proposals against current bias.
        
        Args:
            proposals: Raw proposals from strategies
            current_bias: Current bias value (e.g., 'long_bias', 'neutral')
            
        Returns:
            BiasFilterResult with passed and filtered proposals
        """
        now = datetime.now()
        passed: List[Dict] = []
        filtered: List[FilteredProposal] = []
        
        # Normalize bias
        bias_lower = current_bias.lower()
        allowed_directions = self.BIAS_ALLOWED_DIRECTIONS.get(bias_lower, set())
        
        # NEUTRAL bias = block everything
        if bias_lower == 'neutral' or not allowed_directions:
            for proposal in proposals:
                filtered.append(FilteredProposal(
                    strategy_id=proposal.get('strategy_id', 'unknown'),
                    direction=proposal.get('direction', 'unknown'),
                    proposed_at=now,
                    filter_reason=FilterReason.BIAS_NEUTRAL,
                    bias_at_filter=current_bias,
                    confidence=proposal.get('confidence', 0.0)
                ))
            
            if proposals:
                logger.warning(
                    f"BiasPreFilter: ALL {len(proposals)} proposals discarded "
                    f"(bias={current_bias})"
                )
        else:
            # Filter by direction
            for proposal in proposals:
                direction = proposal.get('direction', '').upper()
                
                if direction in allowed_directions:
                    passed.append(proposal)
                else:
                    filtered.append(FilteredProposal(
                        strategy_id=proposal.get('strategy_id', 'unknown'),
                        direction=direction,
                        proposed_at=now,
                        filter_reason=FilterReason.DIRECTION_MISMATCH,
                        bias_at_filter=current_bias,
                        confidence=proposal.get('confidence', 0.0)
                    ))
                    
                    logger.debug(
                        f"BiasPreFilter: Discarded {proposal.get('strategy_id')} "
                        f"({direction}) - bias is {current_bias}"
                    )
        
        # Create result
        result = BiasFilterResult(
            passed_proposals=passed,
            filtered_proposals=filtered,
            current_bias=current_bias,
            timestamp=now
        )
        
        # Update stats
        self._total_passed += result.pass_count
        self._total_filtered += result.filter_count
        self._filter_history.append(result)
        
        # Keep history bounded
        if len(self._filter_history) > 1000:
            self._filter_history = self._filter_history[-500:]
        
        # Log summary
        if filtered:
            logger.info(
                f"BiasPreFilter: {result.pass_count} passed, "
                f"{result.filter_count} filtered (bias={current_bias})"
            )
        
        return result
    
    def get_stats(self) -> Dict:
        """Get filtering statistics."""
        return {
            'total_passed': self._total_passed,
            'total_filtered': self._total_filtered,
            'filter_rate': (
                self._total_filtered / (self._total_passed + self._total_filtered)
                if (self._total_passed + self._total_filtered) > 0
                else 0.0
            ),
            'history_size': len(self._filter_history)
        }
    
    def get_recent_filters(self, limit: int = 50) -> List[Dict]:
        """Get recent filter results."""
        return [r.to_dict() for r in self._filter_history[-limit:]]
    
    def clear_history(self):
        """Clear filter history."""
        self._filter_history.clear()
        logger.info("BiasPreFilter history cleared")


# Singleton instance
_bias_pre_filter: Optional[BiasPreFilter] = None


def get_bias_pre_filter() -> BiasPreFilter:
    """Get global BiasPreFilter instance."""
    global _bias_pre_filter
    if _bias_pre_filter is None:
        _bias_pre_filter = BiasPreFilter()
    return _bias_pre_filter
