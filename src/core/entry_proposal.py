"""
Entry Proposal System

Data structures for strategy proposals and ranking logic.
Strategies propose trades, engine selects best proposal.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class EntryProposal:
    """
    Entry proposal from a strategy.
    
    Strategies don't force trades - they propose them.
    Engine decides which proposal (if any) to execute.
    """
    # Identity
    strategy_id: str
    symbol: str
    timestamp: datetime
    
    # Trade parameters
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    
    # Reasoning
    reasoning: str  # Human-readable explanation
    confidence: float  # 0.0 to 1.0
    
    # Context fit
    context_alignment: float  # How well proposal matches current context (0-1)
    bias_alignment: float  # How well proposal matches current bias (0-1)
    
    # Metadata
    metadata: Dict[str, Any]
    
    def get_risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        if self.direction == "LONG":
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:  # SHORT
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        
        if risk <= 0:
            return 0.0
        
        return reward / risk
    
    def get_overall_score(self) -> float:
        """
        Calculate overall proposal score for ranking.
        
        Score components:
        - Confidence (40%)
        - Context alignment (30%)
        - Bias alignment (20%)
        - Risk/reward ratio (10%)
        """
        rr_ratio = min(self.get_risk_reward_ratio() / 3.0, 1.0)  # Normalize to 0-1
        
        score = (
            self.confidence * 0.40 +
            self.context_alignment * 0.30 +
            self.bias_alignment * 0.20 +
            rr_ratio * 0.10
        )
        
        return score


class ProposalRanker:
    """
    Ranks entry proposals to select the best one.
    
    Multiple strategies may propose entries simultaneously.
    This selects the single best proposal to execute.
    """
    
    @staticmethod
    def rank_proposals(proposals: list[EntryProposal]) -> list[EntryProposal]:
        """
        Rank proposals by overall score.
        
        Args:
            proposals: List of entry proposals
            
        Returns:
            Sorted list (best first)
        """
        if not proposals:
            return []
        
        # Sort by overall score (descending)
        ranked = sorted(
            proposals,
            key=lambda p: p.get_overall_score(),
            reverse=True
        )
        
        return ranked
    
    @staticmethod
    def select_best_proposal(proposals: list[EntryProposal]) -> Optional[EntryProposal]:
        """
        Select the best proposal from a list.
        
        Args:
            proposals: List of entry proposals
            
        Returns:
            Best proposal or None
        """
        if not proposals:
            return None
        
        ranked = ProposalRanker.rank_proposals(proposals)
        return ranked[0]
    
    @staticmethod
    def filter_proposals_by_direction(
        proposals: list[EntryProposal],
        allowed_direction: str
    ) -> list[EntryProposal]:
        """
        Filter proposals by allowed direction based on bias.
        
        Args:
            proposals: List of entry proposals
            allowed_direction: 'LONG' or 'SHORT'
            
        Returns:
            Filtered list
        """
        return [
            p for p in proposals
            if p.direction == allowed_direction
        ]
