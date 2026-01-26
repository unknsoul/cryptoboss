"""
Professional Trading Bot - Complete Example

This demonstrates the full professional architecture decision flow:
1. Market Context Analysis
2. Directional Bias Determination
3. Strategy Proposals
4. Permission Filtering
5. Trade Execution
6. Decision Logging

Run this to see the complete flow in action.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s - %(message)s'
)
logger = logging.getLogger("ProfessionalExample")

# Import professional architecture components
from src.core import (
    get_market_context_engine,
    get_bias_engine,
    get_permission_filter,
    get_decision_logger,
    get_risk_guardian,
    MarketRegime,
    TradeBias
)
from src.core.entry_proposal import ProposalRanker
from examples.professional_dca_adapter import ProfessionalDCAStrategy


def create_sample_data():
    """Create sample market data for demonstration."""
    print("\n📊 Creating sample market data...")
    
    # Generate 1H data (trending up)
    dates_1h = pd.date_range('2024-01-01', periods=500, freq='1h')
    trend_1h = np.linspace(60000, 65000, 500) + np.cumsum(np.random.randn(500) * 50)
    
    df_1h = pd.DataFrame({
        'open': trend_1h + np.random.randn(500) * 20,
        'high': trend_1h + np.abs(np.random.randn(500) * 50),
        'low': trend_1h - np.abs(np.random.randn(500) * 50),
        'close': trend_1h,
        'volume': np.random.uniform(100, 1000, 500)
    }, index=dates_1h)
    
    # Generate 4H data
    dates_4h = pd.date_range('2024-01-01', periods=125, freq='4h')
    trend_4h = np.linspace(60000, 65000, 125) + np.cumsum(np.random.randn(125) * 100)
    
    df_4h = pd.DataFrame({
        'open': trend_4h + np.random.randn(125) * 50,
        'high': trend_4h + np.abs(np.random.randn(125) * 100),
        'low': trend_4h - np.abs(np.random.randn(125) * 100),
        'close': trend_4h,
        'volume': np.random.uniform(500, 5000, 125)
    }, index=dates_4h)
    
    # Generate 1D data
    dates_1d = pd.date_range('2024-01-01', periods=30, freq='1D')
    trend_1d = np.linspace(60000, 65000, 30) + np.cumsum(np.random.randn(30) * 200)
    
    df_1d = pd.DataFrame({
        'open': trend_1d + np.random.randn(30) * 100,
        'high': trend_1d + np.abs(np.random.randn(30) * 200),
        'low': trend_1d - np.abs(np.random.randn(30) * 200),
        'close': trend_1d,
        'volume': np.random.uniform(1000, 10000, 30)
    }, index=dates_1d)
    
    print(f"   1H data: {len(df_1h)} candles")
    print(f"   4H data: {len(df_4h)} candles")
    print(f"   1D data: {len(df_1d)} candles")
    
    return df_1h, df_4h, df_1d


def main():
    """Run complete professional trading example."""
    
    print("=" * 80)
    print(" 🤖 PROFESSIONAL TRADING BOT - COMPLETE EXAMPLE")
    print("=" * 80)
    
    # === STEP 1: Initialize Components ===
    print("\n🔧 Step 1: Initializing Professional Architecture Components...")
    
    context_engine = get_market_context_engine()
    bias_engine = get_bias_engine()
    risk_guardian = get_risk_guardian(portfolio_value=10000)
    permission_filter = get_permission_filter(risk_guardian)
    decision_logger = get_decision_logger(log_dir="logs/decisions")
    
    print("   ✓ MarketContextEngine initialized")
    print("   ✓ BiasEngine initialized")
    print("   ✓ RiskGuardian initialized ($10,000 portfolio)")
    print("   ✓ TradePermissionFilter initialized")
    print("   ✓ DecisionLogger initialized")
    
    # === STEP 2: Create Strategies ===
    print("\n📋 Step 2: Creating Trading Strategies...")
    
    dca_strategy = ProfessionalDCAStrategy(
        base_order_size=100,
        safety_order_size=200,
        max_safety_orders=5,
        price_step_pct=2.5,
        target_profit_pct=3.0,
        strategy_id="professional_dca_btc"
    )
    
    print("   ✓ Professional DCA Strategy created")
    
    # === STEP 3: Load Market Data ===
    print("\n📈 Step 3: Loading Market Data...")
    
    df_1h, df_4h, df_1d = create_sample_data()
    current_price = df_1h['close'].iloc[-1]
    
    print(f"   Current Price: ${current_price:,.2f}")
    
    # === STEP 4: Analyze Market Context ===
    print("\n🔍 Step 4: Analyzing Market Context...")
    
    context = context_engine.get_current_context(
        df_1h=df_1h,
        df_4h=df_4h,
        df_1d=df_1d,
        current_price=current_price,
        orderbook={'bids': [[current_price-5, 10.0]], 'asks': [[current_price+5, 10.0]]},
        volume_24h=1000
    )
    
    decision_logger.log_context_decision(
        symbol="BTC/USDT",
        context=context,
        approved=context.trading_allowed,
        reason=context.reason
    )
    
    print(f"   Market Regime: {context.regime.value}")
    print(f"   Trading Allowed: {'✓' if context.trading_allowed else '✗'}")
    print(f"   Reason: {context.reason}")
    print(f"   Confidence: {context.confidence:.2f}")
    print(f"   Trends: 1H={context.trend_1h}, 4H={context.trend_4h}, 1D={context.trend_1d}")
    print(f"   Volatility: {context.volatility_regime} ({context.atr_percentile:.1f} percentile)")
    print(f"   ADX: {context.adx_value:.2f} ({context.trend_strength})")
    
    if context.regime == MarketRegime.NO_TRADE:
        print("\n❌ TRADING HALTED - NO_TRADE regime detected")
        decision_logger.flush()
        return
    
    if not context.trading_allowed:
        print(f"\n❌ TRADING HALTED - {context.reason}")
        decision_logger.flush()
        return
    
    print("   ✓ Context approved for trading")
    
    # === STEP 5: Determine Directional Bias ===
    print("\n🎯 Step 5: Determining Directional Bias...")
    
    bias = bias_engine.get_current_bias(df_1h, context)
    
    decision_logger.log_bias_decision(
        symbol="BTC/USDT",
        bias=bias,
        reason=bias.reason
    )
    
    print(f"   Bias: {bias.bias.value}")
    print(f"   Conviction: {bias.conviction:.2f}")
    print(f"   HTF Trend: {bias.higher_tf_trend}")
    print(f"   Momentum: {bias.momentum_direction}")
    print(f"   Reason: {bias.reason}")
    
    if bias.bias == TradeBias.NEUTRAL:
        print("\n❌ NO DIRECTIONAL CONVICTION - No entries allowed")
        decision_logger.flush()
        return
    
    print("   ✓ Bias established")
    
    # === STEP 6: Collect Strategy Proposals ===
    print("\n💡 Step 6: Collecting Strategy Proposals...")
    
    proposals = []
    
    # Get proposal from DCA strategy
    proposal = dca_strategy.propose_entry(
        df=df_1h,
        context=context,
        bias=bias,
        current_price=current_price
    )
    
    if proposal:
        proposals.append(proposal)
        
        decision_logger.log_entry_proposal(
            symbol="BTC/USDT",
            strategy_id=proposal.strategy_id,
            direction=proposal.direction,
            entry_price=proposal.entry_price,
            reasoning=proposal.reasoning,
            confidence=proposal.confidence
        )
        
        print(f"   ✓ Proposal from {proposal.strategy_id}:")
        print(f"      Direction: {proposal.direction}")
        print(f"      Entry: ${proposal.entry_price:,.2f}")
        print(f"      Size: {proposal.size:.4f} BTC")
        print(f"      SL: ${proposal.stop_loss:,.2f}")
        print(f"      TP: ${proposal.take_profit:,.2f}")
        print(f"      R:R: {proposal.get_risk_reward_ratio():.2f}")
        print(f"      Confidence: {proposal.confidence:.2f}")
        print(f"      Overall Score: {proposal.get_overall_score():.2f}")
        print(f"      Reasoning: {proposal.reasoning}")
    else:
        print("   ℹ️  No proposal from DCA strategy")
    
    if not proposals:
        print("\n❌ NO PROPOSALS - No strategies suggested entries")
        decision_logger.flush()
        return
    
    # === STEP 7: Rank Proposals ===
    print("\n🏆 Step 7: Ranking Proposals...")
    
    best_proposal = ProposalRanker.select_best_proposal(proposals)
    
    # Log rejected proposals
    for p in proposals:
        if p != best_proposal:
            decision_logger.log_entry_rejected(
                symbol="BTC/USDT",
                strategy_id=p.strategy_id,
                direction=p.direction,
                reason=f"Lower score ({p.get_overall_score():.2f}) than {best_proposal.strategy_id}"
            )
    
    print(f"   Best Proposal: {best_proposal.strategy_id}")
    print(f"   Score: {best_proposal.get_overall_score():.2f}")
    
    # === STEP 8: Check Final Permission ===
    print("\n🚦 Step 8: Checking Final Permission...")
    
    permission = permission_filter.check_permission(
        context=context,
        bias=bias,
        direction=best_proposal.direction,
        orderbook={'bids': [[current_price-5, 10.0]], 'asks': [[current_price+5, 10.0]]}
    )
    
    decision_logger.log_permission_check(
        symbol="BTC/USDT",
        permission=permission
    )
    
    print(f"   Permission: {'✓ APPROVED' if permission.approved else '✗ DENIED'}")
    print(f"   Reason: {permission.reason}")
    
    if not permission.approved:
        print(f"\n   Denial Category: {permission.denial_category.value}")
        print("   Check Results:")
        for check, passed in permission.checks_passed.items():
            print(f"      {check}: {'✓' if passed else '✗'}")
        decision_logger.flush()
        return
    
    print("   All checks passed:")
    for check in permission.checks_passed:
        print(f"      ✓ {check}")
    
    # === STEP 9: EXECUTE TRADE ===
    print("\n" + "=" * 80)
    print(" 🚀 ALL GATES PASSED - EXECUTING TRADE")
    print("=" * 80)
    
    print(f"\n   Symbol: BTC/USDT")
    print(f"   Direction: {best_proposal.direction}")
    print(f"   Size: {best_proposal.size:.4f} BTC")
    print(f"   Entry Price: ${best_proposal.entry_price:,.2f}")
    print(f"   Stop Loss: ${best_proposal.stop_loss:,.2f}")
    print(f"   Take Profit: ${best_proposal.take_profit:,.2f}")
    print(f"   Risk/Reward: {best_proposal.get_risk_reward_ratio():.2f}")
    print(f"   Strategy: {best_proposal.strategy_id}")
    print(f"   Reasoning: {best_proposal.reasoning}")
    
    decision_logger.log_entry_executed(
        symbol="BTC/USDT",
        strategy_id=best_proposal.strategy_id,
        direction=best_proposal.direction,
        entry_price=best_proposal.entry_price,
        size=best_proposal.size,
        reason=f"Best proposal with score {best_proposal.get_overall_score():.2f}"
    )
    
    # === STEP 10: Log Statistics ===
    print("\n" + "=" * 80)
    print(" 📊 DECISION STATISTICS")
    print("=" * 80)
    
    # Flush logs
    decision_logger.flush()
    
    stats = decision_logger.get_decision_stats(hours=24)
    
    print(f"\n   Total Decisions: {stats['total_decisions']}")
    print(f"   Approvals: {stats['approvals']}")
    print(f"   Denials: {stats['denials']}")
    print(f"   Approval Rate: {stats['approval_rate']:.1%}")
    
    print("\n   Decision Type Breakdown:")
    for dtype, count in stats['decision_type_counts'].items():
        print(f"      {dtype}: {count}")
    
    risk_report = risk_guardian.get_risk_report()
    print(f"\n   Risk Status:")
    print(f"      Emergency Stop: {risk_report['emergency_stop_active']}")
    print(f"      Daily P&L: ${risk_report['daily_pnl']:+,.2f}")
    print(f"      Consecutive Errors: {risk_report['consecutive_errors']}")
    
    print("\n" + "=" * 80)
    print(" ✅ EXAMPLE COMPLETE")
    print("=" * 80)
    print(f"\n   Decision logs saved to: logs/decisions/")
    print(f"   Review logs for complete audit trail")
    print("\n")


if __name__ == "__main__":
    main()
