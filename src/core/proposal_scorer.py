"""
Proposal Scorer - Live Readiness Component

Normalizes and weights strategy proposals with:
- Confidence normalization across strategies
- Context-specific weighting
- Losing strategy decay
- Explainable scoring breakdown

Ensures fair comparison between different strategy proposals.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class StrategyHealth:
    """
    Tracks health metrics for a single strategy.
    
    Health decays on losses, recovers on wins.
    Low health reduces proposal priority.
    """
    strategy_id: str
    
    # Recent performance (rolling window)
    recent_trades: int = 0
    recent_wins: int = 0
    recent_losses: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    
    # Health score
    decay_factor: float = 1.0  # 0.0 to 1.0
    
    # Metadata
    last_trade_time: Optional[datetime] = None
    total_pnl: float = 0.0
    
    def record_win(self, pnl: float = 0.0):
        """Record a winning trade."""
        self.recent_trades += 1
        self.recent_wins += 1
        self.consecutive_wins += 1
        self.consecutive_losses = 0
        self.total_pnl += pnl
        self.last_trade_time = datetime.now()
        
        # Recover health
        self.decay_factor = min(1.0, self.decay_factor * 1.10)
    
    def record_loss(self, pnl: float = 0.0):
        """Record a losing trade."""
        self.recent_trades += 1
        self.recent_losses += 1
        self.consecutive_losses += 1
        self.consecutive_wins = 0
        self.total_pnl += pnl
        self.last_trade_time = datetime.now()
        
        # Decay health
        self.decay_factor *= 0.85
    
    def get_win_rate(self) -> float:
        """Get win rate for recent trades."""
        if self.recent_trades == 0:
            return 0.5  # Neutral for new strategies
        return self.recent_wins / self.recent_trades
    
    def is_disabled(self) -> bool:
        """Check if strategy is disabled due to low health."""
        return self.decay_factor < 0.3
    
    def to_dict(self) -> Dict:
        return {
            'strategy_id': self.strategy_id,
            'recent_trades': self.recent_trades,
            'recent_wins': self.recent_wins,
            'recent_losses': self.recent_losses,
            'consecutive_losses': self.consecutive_losses,
            'decay_factor': self.decay_factor,
            'win_rate': self.get_win_rate(),
            'is_disabled': self.is_disabled(),
            'total_pnl': self.total_pnl
        }


@dataclass
class ScoredProposal:
    """
    A proposal with detailed scoring breakdown.
    
    Provides explainability for ranking decisions.
    """
    strategy_id: str
    direction: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    
    # Original confidence from strategy
    raw_confidence: float
    
    # Score components (all 0.0 to 1.0)
    normalized_confidence: float
    context_weight: float
    bias_alignment: float
    strategy_health: float
    rr_ratio_score: float
    
    # Final score
    final_score: float
    
    # Reasoning
    reasoning: str
    score_breakdown: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            'strategy_id': self.strategy_id,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'raw_confidence': self.raw_confidence,
            'final_score': self.final_score,
            'score_breakdown': self.score_breakdown,
            'reasoning': self.reasoning
        }


class ProposalScorer:
    """
    Scores and ranks strategy proposals.
    
    Scoring Formula:
        final_score = (
            normalized_confidence * 0.30 +
            context_weight * 0.25 +
            bias_alignment * 0.20 +
            strategy_health * 0.15 +
            rr_ratio_score * 0.10
        )
    
    Features:
    - Cross-strategy confidence normalization
    - Context-specific regime weighting
    - Losing strategy decay
    - Explainable score breakdown
    
    Usage:
        scorer = ProposalScorer()
        
        # Score proposals
        scored = scorer.score_proposals(
            proposals,
            context_regime="trending_up",
            current_bias="long_only"
        )
        
        # Get best proposal
        best = scored[0] if scored else None
    """
    
    # Weight configuration
    WEIGHT_CONFIDENCE = 0.30
    WEIGHT_CONTEXT = 0.25
    WEIGHT_BIAS = 0.20
    WEIGHT_HEALTH = 0.15
    WEIGHT_RR = 0.10
    
    # Context preference matrix
    # Strategy type -> preferred contexts
    CONTEXT_PREFERENCES = {
        'dca': ['ranging', 'trending_down'],
        'grid': ['ranging'],
        'momentum': ['trending_up', 'trending_down'],
        'mean_reversion': ['ranging', 'high_volatility'],
        'default': ['ranging', 'trending_up', 'trending_down']
    }
    
    def __init__(self):
        self._strategy_health: Dict[str, StrategyHealth] = {}
        
        logger.info("ProposalScorer initialized")
    
    def get_strategy_health(self, strategy_id: str) -> StrategyHealth:
        """Get or create health tracker for strategy."""
        if strategy_id not in self._strategy_health:
            self._strategy_health[strategy_id] = StrategyHealth(strategy_id=strategy_id)
        return self._strategy_health[strategy_id]
    
    def record_trade_result(self, strategy_id: str, is_win: bool, pnl: float = 0.0):
        """Record trade result to update strategy health."""
        health = self.get_strategy_health(strategy_id)
        
        if is_win:
            health.record_win(pnl)
            logger.debug(f"Strategy {strategy_id}: WIN (+{pnl:.2f}), health={health.decay_factor:.2f}")
        else:
            health.record_loss(pnl)
            logger.debug(f"Strategy {strategy_id}: LOSS ({pnl:.2f}), health={health.decay_factor:.2f}")
            
            if health.is_disabled():
                logger.warning(f"Strategy {strategy_id} DISABLED due to low health ({health.decay_factor:.2f})")
    
    def score_proposals(
        self,
        proposals: List[Dict],
        context_regime: str,
        current_bias: str
    ) -> List[ScoredProposal]:
        """
        Score and rank proposals.
        
        Args:
            proposals: List of proposal dicts from strategies
            context_regime: Current market regime
            current_bias: Current directional bias
            
        Returns:
            Sorted list of scored proposals (best first)
        """
        if not proposals:
            return []
        
        scored_proposals = []
        
        # Get max confidence for normalization
        max_confidence = max(p.get('confidence', 0.5) for p in proposals)
        if max_confidence == 0:
            max_confidence = 1.0
        
        for proposal in proposals:
            strategy_id = proposal.get('strategy_id', 'unknown')
            
            # Check strategy health
            health = self.get_strategy_health(strategy_id)
            if health.is_disabled():
                logger.debug(f"Skipping proposal from disabled strategy: {strategy_id}")
                continue
            
            # Check bias alignment
            direction = proposal.get('direction', 'LONG')
            if not self._is_bias_aligned(direction, current_bias):
                logger.debug(f"Skipping proposal misaligned with bias: {strategy_id} {direction} vs {current_bias}")
                continue
            
            # Calculate score components
            raw_confidence = proposal.get('confidence', 0.5)
            normalized_confidence = raw_confidence / max_confidence
            
            context_weight = self._calculate_context_weight(strategy_id, context_regime)
            bias_alignment = 1.0  # Already filtered above
            strategy_health_score = health.decay_factor
            rr_ratio_score = self._calculate_rr_score(proposal)
            
            # Final score
            final_score = (
                normalized_confidence * self.WEIGHT_CONFIDENCE +
                context_weight * self.WEIGHT_CONTEXT +
                bias_alignment * self.WEIGHT_BIAS +
                strategy_health_score * self.WEIGHT_HEALTH +
                rr_ratio_score * self.WEIGHT_RR
            )
            
            # Build score breakdown for explainability
            score_breakdown = {
                'normalized_confidence': round(normalized_confidence * self.WEIGHT_CONFIDENCE, 4),
                'context_weight': round(context_weight * self.WEIGHT_CONTEXT, 4),
                'bias_alignment': round(bias_alignment * self.WEIGHT_BIAS, 4),
                'strategy_health': round(strategy_health_score * self.WEIGHT_HEALTH, 4),
                'rr_ratio': round(rr_ratio_score * self.WEIGHT_RR, 4)
            }
            
            scored = ScoredProposal(
                strategy_id=strategy_id,
                direction=direction,
                entry_price=proposal.get('entry_price', 0),
                size=proposal.get('size', 0),
                stop_loss=proposal.get('stop_loss', 0),
                take_profit=proposal.get('take_profit', 0),
                raw_confidence=raw_confidence,
                normalized_confidence=normalized_confidence,
                context_weight=context_weight,
                bias_alignment=bias_alignment,
                strategy_health=strategy_health_score,
                rr_ratio_score=rr_ratio_score,
                final_score=final_score,
                reasoning=proposal.get('reasoning', ''),
                score_breakdown=score_breakdown
            )
            
            scored_proposals.append(scored)
        
        # Sort by final score (descending)
        scored_proposals.sort(key=lambda p: p.final_score, reverse=True)
        
        if scored_proposals:
            logger.info(
                f"Scored {len(scored_proposals)} proposals, "
                f"best: {scored_proposals[0].strategy_id} ({scored_proposals[0].final_score:.3f})"
            )
        
        return scored_proposals
    
    def _is_bias_aligned(self, direction: str, bias: str) -> bool:
        """Check if proposal direction matches current bias."""
        direction = direction.upper()
        bias = bias.lower()
        
        if bias == 'no_trade':
            return False
        if bias == 'long_only' and direction == 'SHORT':
            return False
        if bias == 'short_only' and direction == 'LONG':
            return False
        
        return True
    
    def _calculate_context_weight(self, strategy_id: str, context: str) -> float:
        """Calculate context alignment score for strategy."""
        # Determine strategy type from ID
        strategy_type = 'default'
        for known_type in self.CONTEXT_PREFERENCES:
            if known_type in strategy_id.lower():
                strategy_type = known_type
                break
        
        preferred_contexts = self.CONTEXT_PREFERENCES.get(strategy_type, ['ranging'])
        
        context = context.lower()
        
        if context in preferred_contexts:
            return 1.0
        elif context == 'high_volatility':
            return 0.5  # Most strategies work poorly in high vol
        elif context == 'no_trade':
            return 0.0
        else:
            return 0.7  # Neutral
    
    def _calculate_rr_score(self, proposal: Dict) -> float:
        """Calculate risk/reward ratio score."""
        entry = proposal.get('entry_price', 0)
        sl = proposal.get('stop_loss', 0)
        tp = proposal.get('take_profit', 0)
        direction = proposal.get('direction', 'LONG').upper()
        
        if entry == 0 or sl == 0 or tp == 0:
            return 0.5  # Neutral if missing data
        
        if direction == 'LONG':
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp
        
        if risk <= 0:
            return 0.0  # Invalid SL
        
        rr_ratio = reward / risk
        
        # Normalize: 1:1 = 0.5, 2:1 = 0.75, 3:1+ = 1.0
        if rr_ratio >= 3.0:
            return 1.0
        elif rr_ratio >= 2.0:
            return 0.75 + (rr_ratio - 2.0) * 0.25
        elif rr_ratio >= 1.0:
            return 0.5 + (rr_ratio - 1.0) * 0.25
        else:
            return rr_ratio * 0.5
    
    def get_all_health_stats(self) -> Dict[str, Dict]:
        """Get health stats for all tracked strategies."""
        return {
            sid: health.to_dict()
            for sid, health in self._strategy_health.items()
        }
    
    def reset_strategy_health(self, strategy_id: str):
        """Reset health for a specific strategy."""
        if strategy_id in self._strategy_health:
            self._strategy_health[strategy_id] = StrategyHealth(strategy_id=strategy_id)
            logger.info(f"Strategy {strategy_id} health reset")


# Singleton instance
_proposal_scorer: Optional[ProposalScorer] = None


def get_proposal_scorer() -> ProposalScorer:
    """Get global ProposalScorer instance."""
    global _proposal_scorer
    if _proposal_scorer is None:
        _proposal_scorer = ProposalScorer()
    return _proposal_scorer
