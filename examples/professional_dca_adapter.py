"""
DCA Strategy Adapter - Professional Architecture Example

This shows how to adapt the existing DCA strategy to the new
proposal-based architecture.

Key Changes:
1. generate_signal() → propose_entry()
2. Accepts context and bias as inputs
3. Returns EntryProposal instead of direct orders
4. Checks regime suitability before proposing

This is an EXAMPLE adapter. The actual integration should modify
the base DCAStrategy class directly.
"""

import logging
from typing import Optional
from datetime import datetime
import pandas as pd

from src.core.market_context_engine import MarketContext, MarketRegime
from src.core.bias_engine import BiasState, TradeBias
from src.core.entry_proposal import EntryProposal
from src.strategies.dca_strategy import DCAStrategy

logger = logging.getLogger(__name__)


class ProfessionalDCAStrategy:
    """
    Professional DCA Strategy - Proposal-Based Architecture.
    
    Wraps the existing DCAStrategy and adapts it to work with the
    new professional architecture (context → bias → permission → execution).
    
    Instead of forcing trades, this strategy PROPOSES entries that:
    1. Align with current market context
    2. Respect directional bias
    3. Include confidence and reasoning
    
    Usage:
        strategy = ProfessionalDCAStrategy(
            base_order_size=100,
            safety_order_size=200
        )
        
        # Get proposal (not a direct order)
        proposal = strategy.propose_entry(df, context, bias)
        
        if proposal:
            # Engine will rank proposals from multiple strategies
            # and execute the best one (if permission granted)
            pass
    """
    
    def __init__(
        self,
        base_order_size: float = 100.0,
        safety_order_size: float = 200.0,
        max_safety_orders: int = 5,
        price_step_pct: float = 2.5,
        target_profit_pct: float = 3.0,
        safety_order_volume_scale: float = 2.0,
        stop_loss_pct: Optional[float] = 20.0,
        cooldown_bars: int = 24,
        strategy_id: str = "professional_dca"
    ):
        # Wrap existing DCA strategy
        self.dca = DCAStrategy(
            base_order_size=base_order_size,
            safety_order_size=safety_order_size,
            max_safety_orders=max_safety_orders,
            price_step_pct=price_step_pct,
            target_profit_pct=target_profit_pct,
            safety_order_volume_scale=safety_order_volume_scale,
            stop_loss_pct=stop_loss_pct,
            cooldown_bars=cooldown_bars
        )
        
        self.strategy_id = strategy_id
        
        # DCA works best in these regimes
        self.suitable_regimes = [
            MarketRegime.RANGING,
            MarketRegime.TRENDING_DOWN,  # DCA excels at averaging down
        ]
        
        logger.info(f"Professional DCA Strategy initialized: {strategy_id}")
    
    def propose_entry(
        self,
        df: pd.DataFrame,
        context: MarketContext,
        bias: BiasState,
        current_price: float = None,
        index: int = None
    ) -> Optional[EntryProposal]:
        """
        Propose an entry if conditions are suitable.
        
        This is the NEW method signature replacing generate_signal().
        
        Args:
            df: OHLCV dataframe
            context: Current market context
            bias: Current directional bias
            current_price: Current market price
            index: Current candle index
            
        Returns:
            EntryProposal or None if no proposal
        """
        # Default values
        if current_price is None:
            current_price = df['close'].iloc[-1]
        if index is None:
            index = len(df) - 1
        
        # === STEP 1: Check if context allows this strategy ===
        if not self._is_context_suitable(context):
            logger.debug(
                f"{self.strategy_id}: Context not suitable "
                f"(regime: {context.regime.value})"
            )
            return None
        
        # === STEP 2: Check bias alignment ===
        # DCA is a LONG-only strategy in this implementation
        if bias.bias != TradeBias.LONG_BIAS:
            logger.debug(
                f"{self.strategy_id}: Bias not aligned "
                f"(need LONG_BIAS, got {bias.bias.value})"
            )
            return None
        
        # === STEP 3: Get signal from underlying DCA logic ===
        signal = self.dca.generate_signal(df, index, current_price)
        
        # Only propose on BUY signals (base order or safety order)
        if signal.get('action') != 'BUY':
            return None
        
        # === STEP 4: Calculate stop loss and take profit ===
        if self.dca.active_deal:
            # Active deal - use deal's TP
            stop_loss = self._calculate_stop_loss(current_price)
            take_profit = self.dca.active_deal.get_take_profit_price()
        else:
            # New deal - estimate based on parameters
            stop_loss = self._calculate_stop_loss(current_price)
            take_profit = current_price * (1 + self.dca.target_profit_pct / 100)
        
        # === STEP 5: Calculate confidence ===
        confidence = self._calculate_confidence(context, bias, signal)
        
        # === STEP 6: Calculate context and bias alignment ===
        context_alignment = self._calculate_context_alignment(context)
        bias_alignment = 1.0  # Perfect alignment (already checked bias above)
        
        # === STEP 7: Create proposal ===
        proposal = EntryProposal(
            strategy_id=self.strategy_id,
            symbol=context.symbol,
            timestamp=datetime.now(),
            direction="LONG",
            entry_price=current_price,
            size=signal.get('size', 0),
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=self._build_reasoning(signal, context, bias),
            confidence=confidence,
            context_alignment=context_alignment,
            bias_alignment=bias_alignment,
            metadata={
                'signal_reason': signal.get('reason', ''),
                'active_deal': self.dca.active_deal is not None,
                'deal_id': self.dca.active_deal.deal_id if self.dca.active_deal else None
            }
        )
        
        logger.info(
            f"📋 {self.strategy_id} proposal: "
            f"{proposal.direction} @ ${proposal.entry_price:.2f}, "
            f"confidence: {proposal.confidence:.2f}, "
            f"score: {proposal.get_overall_score():.2f}"
        )
        
        return proposal
    
    def _is_context_suitable(self, context: MarketContext) -> bool:
        """Check if current market context is suitable for DCA."""
        # DCA works best in ranging or downtrending markets
        # (where it can average down and profit on recovery)
        return context.regime in self.suitable_regimes
    
    def _calculate_stop_loss(self, entry_price: float) -> float:
        """Calculate stop loss price."""
        if self.dca.stop_loss_pct is None:
            # No stop loss - use very wide level
            return entry_price * 0.5
        
        return entry_price * (1 - self.dca.stop_loss_pct / 100)
    
    def _calculate_confidence(
        self,
        context: MarketContext,
        bias: BiasState,
        signal: dict
    ) -> float:
        """
        Calculate confidence in this proposal.
        
        Higher confidence when:
        - Strong bias conviction
        - Clear market context
        - Base order (vs safety order)
        - Suitable volatility
        """
        confidence = 0.5  # Base confidence
        
        # Bonus for strong bias
        confidence += bias.conviction * 0.2
        
        # Bonus for high context confidence
        confidence += context.confidence * 0.2
        
        # Base orders are higher confidence than safety orders
        if signal.get('reason') == 'BASE_ORDER':
            confidence += 0.15
        else:
            # Safety orders are lower confidence (averaging down is riskier)
            confidence -= 0.10
        
        # Bonus for suitable volatility (not too high)
        if context.volatility_regime in ['normal', 'low']:
            confidence += 0.10
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _calculate_context_alignment(self, context: MarketContext) -> float:
        """
        Calculate how well this proposal fits current market context.
        
        DCA aligns best with:
        - Ranging markets (perfect fit)
        - Downtrending markets (good fit - can average down)
        - Not aligned with strong uptrends (DCA underperforms here)
        """
        regime_scores = {
            MarketRegime.RANGING: 1.0,
            MarketRegime.TRENDING_DOWN: 0.85,
            MarketRegime.TRENDING_UP: 0.5,
            MarketRegime.HIGH_VOLATILITY: 0.4,
            MarketRegime.LOW_LIQUIDITY: 0.0,
            MarketRegime.NO_TRADE: 0.0
        }
        
        return regime_scores.get(context.regime, 0.5)
    
    def _build_reasoning(
        self,
        signal: dict,
        context: MarketContext,
        bias: BiasState
    ) -> str:
        """Build human-readable reasoning for proposal."""
        reason = signal.get('reason', 'UNKNOWN')
        
        if reason == 'BASE_ORDER':
            return (
                f"DCA base order in {context.regime.value} regime with "
                f"{bias.bias.value} (conviction: {bias.conviction:.2f})"
            )
        elif 'SAFETY_ORDER' in reason:
            so_num = reason.split('_')[-1]
            return (
                f"DCA safety order #{so_num} to average down position, "
                f"market regime: {context.regime.value}"
            )
        else:
            return f"DCA signal: {reason}"
    
    def get_metrics(self) -> dict:
        """Pass through to underlying DCA metrics."""
        return self.dca.get_metrics()
    
    def get_deal_history(self) -> list:
        """Pass through to underlying DCA deal history."""
        return self.dca.get_deal_history()


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    """
    Example of how the professional architecture flows:
    """
    print("=" * 70)
    print("PROFESSIONAL ARCHITECTURE DECISION FLOW EXAMPLE")
    print("=" * 70)
    
    # This is how the engine would use the new architecture:
    print("""
    # 1. Get market context
    context = market_context_engine.get_current_context(df_1h, df_4h, df_1d, price)
    
    if context.regime == MarketRegime.NO_TRADE:
        decision_logger.log_no_trade_period(symbol, context.reason)
        return  # HALT - no trading allowed
    
    # 2. Get directional bias
    bias = bias_engine.get_current_bias(df, context)
    
    if bias.bias == TradeBias.NEUTRAL:
        decision_logger.log_bias_decision(symbol, bias, "No directional conviction")
        return  # HALT - no direction
    
    # 3. Collect proposals from all active strategies
    proposals = []
    
    for strategy in active_strategies:
        proposal = strategy.propose_entry(df, context, bias, price)
        if proposal:
            proposals.append(proposal)
            decision_logger.log_entry_proposal(
                symbol, proposal.strategy_id, proposal.direction,
                proposal.entry_price, proposal.reasoning, proposal.confidence
            )
    
    if not proposals:
        return  # No strategies proposed entries
    
    # 4. Rank proposals and select best
    best_proposal = ProposalRanker.select_best_proposal(proposals)
    
    # Log rejected proposals
    for p in proposals:
        if p != best_proposal:
            decision_logger.log_entry_rejected(
                symbol, p.strategy_id, p.direction,
                f"Lower score than {best_proposal.strategy_id}"
            )
    
    # 5. Check final permission
    permission = permission_filter.check_permission(
        context, bias, best_proposal.direction
    )
    
    if not permission.approved:
        decision_logger.log_permission_block(symbol, permission.reason)
        return  # DENIED - permission filter blocked
    
    # 6. EXECUTE TRADE (all gates passed)
    result = await execution_router.execute(best_proposal)
    
    decision_logger.log_entry_executed(
        symbol, best_proposal.strategy_id, best_proposal.direction,
        best_proposal.entry_price, best_proposal.size,
        f"Best proposal with score {best_proposal.get_overall_score():.2f}"
    )
    
    print("✅ Trade executed with complete audit trail")
    """)
    
    print("=" * 70)
    print("KEY DIFFERENCES FROM OLD ARCHITECTURE:")
    print("=" * 70)
    print("""
    OLD (Signal-Based):
    - Strategy generates signal → Risk check → Execute
    - Strategy can FORCE trades
    - Limited context awareness
    - No decision logging
    
    NEW (Context-First):
    - Context gate → Bias gate → Strategies PROPOSE → Permission gate → Execute
    - Strategies SUGGEST, engine DECIDES
    - Full context awareness
    - Complete decision audit trail
    - Trading blocked if context/bias/permission say no
    """)
    
    print("=" * 70)
