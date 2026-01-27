"""
Global Proposal Scoring Contract - v10.0 Component

Enforces strict scoring schema for all strategy proposals:
- All proposals must include 4 mandatory components
- Scores are normalized to [0.0, 1.0]
- Unnormalized proposals are rejected
- Scoring is explainable and logged

This is the foundation for fair, deterministic proposal comparison.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ScoreComponent(Enum):
    """Mandatory score components."""
    CONTEXT_FIT = "context_fit"
    HISTORICAL_EXPECTANCY = "historical_expectancy"
    RECENT_PERFORMANCE_DECAY = "recent_performance_decay"
    RISK_ALIGNMENT = "risk_alignment"


# Component weights (must sum to 1.0)
COMPONENT_WEIGHTS: Dict[ScoreComponent, float] = {
    ScoreComponent.CONTEXT_FIT: 0.30,
    ScoreComponent.HISTORICAL_EXPECTANCY: 0.25,
    ScoreComponent.RECENT_PERFORMANCE_DECAY: 0.25,
    ScoreComponent.RISK_ALIGNMENT: 0.20,
}


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of proposal score."""
    context_fit: float
    historical_expectancy: float
    recent_performance_decay: float
    risk_alignment: float
    
    # Computed
    weighted_score: float = 0.0
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'context_fit': self.context_fit,
            'historical_expectancy': self.historical_expectancy,
            'recent_performance_decay': self.recent_performance_decay,
            'risk_alignment': self.risk_alignment,
            'weighted_score': self.weighted_score,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors
        }


@dataclass
class ContractValidatedProposal:
    """
    A proposal that has passed the scoring contract.
    
    Only these proposals can proceed to permission checking.
    """
    strategy_id: str
    direction: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    
    # Validated score
    score: float
    score_breakdown: ScoreBreakdown
    
    # Metadata
    reasoning: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'strategy_id': self.strategy_id,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'score': self.score,
            'score_breakdown': self.score_breakdown.to_dict(),
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }


class ProposalValidationError(Exception):
    """Raised when a proposal fails contract validation."""
    pass


class ScoringContract:
    """
    Enforces the global proposal scoring contract.
    
    All strategy proposals MUST:
    1. Include all 4 mandatory score components
    2. Have scores in range [0.0, 1.0]
    3. Be explainable (include reasoning)
    
    Proposals that fail validation are REJECTED.
    
    Usage:
        contract = ScoringContract()
        
        # Raw proposal from strategy
        raw_proposal = {
            'strategy_id': 'dca_btc',
            'direction': 'LONG',
            'context_fit': 0.85,
            'historical_expectancy': 0.70,
            'recent_performance_decay': 0.90,
            'risk_alignment': 0.75,
            ...
        }
        
        # Validate and transform
        validated = contract.validate_proposal(raw_proposal)
        if validated:
            # Proceed to permission check
            pass
    """
    
    REQUIRED_COMPONENTS = [
        ScoreComponent.CONTEXT_FIT,
        ScoreComponent.HISTORICAL_EXPECTANCY,
        ScoreComponent.RECENT_PERFORMANCE_DECAY,
        ScoreComponent.RISK_ALIGNMENT,
    ]
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize scoring contract.
        
        Args:
            strict_mode: If True, reject proposals with any validation error.
                        If False, use default values for missing components.
        """
        self.strict_mode = strict_mode
        self._validation_log: List[Dict] = []
        
        logger.info(f"ScoringContract initialized (strict_mode={strict_mode})")
    
    def validate_proposal(
        self,
        raw_proposal: Dict[str, Any]
    ) -> Optional[ContractValidatedProposal]:
        """
        Validate a raw proposal against the scoring contract.
        
        Args:
            raw_proposal: Dictionary containing proposal data
            
        Returns:
            ContractValidatedProposal if valid, None if rejected
        """
        errors: List[str] = []
        
        # 1. Check required fields
        required_fields = ['strategy_id', 'direction', 'entry_price', 'size']
        for field in required_fields:
            if field not in raw_proposal:
                errors.append(f"Missing required field: {field}")
        
        if errors and self.strict_mode:
            self._log_rejection(raw_proposal, errors)
            return None
        
        # 2. Extract and validate score components
        scores = {}
        for component in self.REQUIRED_COMPONENTS:
            key = component.value
            if key in raw_proposal:
                value = raw_proposal[key]
                
                # Validate range
                if not (0.0 <= value <= 1.0):
                    errors.append(f"Score {key}={value} outside [0.0, 1.0]")
                    if self.strict_mode:
                        value = max(0.0, min(1.0, value))  # Clamp
                
                scores[component] = value
            else:
                if self.strict_mode:
                    errors.append(f"Missing score component: {key}")
                else:
                    scores[component] = 0.5  # Default neutral value
        
        # 3. Check for critical errors in strict mode
        if errors and self.strict_mode:
            self._log_rejection(raw_proposal, errors)
            return None
        
        # 4. Calculate weighted score
        weighted_score = sum(
            scores.get(comp, 0.5) * weight
            for comp, weight in COMPONENT_WEIGHTS.items()
        )
        
        # 5. Build score breakdown
        breakdown = ScoreBreakdown(
            context_fit=scores.get(ScoreComponent.CONTEXT_FIT, 0.5),
            historical_expectancy=scores.get(ScoreComponent.HISTORICAL_EXPECTANCY, 0.5),
            recent_performance_decay=scores.get(ScoreComponent.RECENT_PERFORMANCE_DECAY, 0.5),
            risk_alignment=scores.get(ScoreComponent.RISK_ALIGNMENT, 0.5),
            weighted_score=weighted_score,
            is_valid=len(errors) == 0,
            validation_errors=errors
        )
        
        # 6. Create validated proposal
        validated = ContractValidatedProposal(
            strategy_id=raw_proposal.get('strategy_id', 'unknown'),
            direction=raw_proposal.get('direction', 'LONG').upper(),
            entry_price=raw_proposal.get('entry_price', 0.0),
            size=raw_proposal.get('size', 0.0),
            stop_loss=raw_proposal.get('stop_loss', 0.0),
            take_profit=raw_proposal.get('take_profit', 0.0),
            score=weighted_score,
            score_breakdown=breakdown,
            reasoning=raw_proposal.get('reasoning', ''),
            timestamp=datetime.now()
        )
        
        logger.debug(
            f"Proposal validated: {validated.strategy_id} "
            f"score={weighted_score:.3f}"
        )
        
        return validated
    
    def validate_batch(
        self,
        proposals: List[Dict[str, Any]]
    ) -> List[ContractValidatedProposal]:
        """
        Validate multiple proposals and return only valid ones.
        
        Sorted by score (highest first).
        """
        validated = []
        
        for proposal in proposals:
            result = self.validate_proposal(proposal)
            if result:
                validated.append(result)
        
        # Sort by score descending
        validated.sort(key=lambda p: p.score, reverse=True)
        
        logger.info(
            f"Batch validation: {len(validated)}/{len(proposals)} passed"
        )
        
        return validated
    
    def get_score_explanation(
        self,
        proposal: ContractValidatedProposal
    ) -> str:
        """
        Generate human-readable score explanation.
        
        Returns explainable breakdown for logging/UI.
        """
        b = proposal.score_breakdown
        
        lines = [
            f"Score Breakdown for {proposal.strategy_id}:",
            f"  Context Fit:      {b.context_fit:.2f} × 30% = {b.context_fit * 0.30:.3f}",
            f"  Historical Exp:   {b.historical_expectancy:.2f} × 25% = {b.historical_expectancy * 0.25:.3f}",
            f"  Performance Decay:{b.recent_performance_decay:.2f} × 25% = {b.recent_performance_decay * 0.25:.3f}",
            f"  Risk Alignment:   {b.risk_alignment:.2f} × 20% = {b.risk_alignment * 0.20:.3f}",
            f"  ─────────────────────────────────",
            f"  Final Score:      {proposal.score:.3f}",
        ]
        
        if b.validation_errors:
            lines.append(f"  Warnings: {', '.join(b.validation_errors)}")
        
        return "\n".join(lines)
    
    def _log_rejection(self, proposal: Dict, errors: List[str]):
        """Log rejected proposal for debugging."""
        rejection = {
            'timestamp': datetime.now().isoformat(),
            'strategy_id': proposal.get('strategy_id', 'unknown'),
            'errors': errors,
            'raw_data': {k: v for k, v in proposal.items() if k != 'reasoning'}
        }
        self._validation_log.append(rejection)
        
        logger.warning(
            f"Proposal REJECTED: {proposal.get('strategy_id', 'unknown')} - {errors}"
        )
    
    def get_rejection_log(self, limit: int = 50) -> List[Dict]:
        """Get recent rejection log entries."""
        return self._validation_log[-limit:]
    
    def clear_rejection_log(self):
        """Clear rejection log."""
        self._validation_log.clear()


# Singleton instance
_scoring_contract: Optional[ScoringContract] = None


def get_scoring_contract(strict_mode: bool = True) -> ScoringContract:
    """Get global ScoringContract instance."""
    global _scoring_contract
    if _scoring_contract is None:
        _scoring_contract = ScoringContract(strict_mode=strict_mode)
    return _scoring_contract
