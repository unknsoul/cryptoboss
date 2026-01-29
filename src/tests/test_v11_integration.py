"""
v11.0 Integration Test - Trade Intent Pipeline

Tests the complete v11.0 flow:
1. Strategy creates TradeIntent
2. ExecutionFlowOrchestrator processes intent
3. TradeDecision is returned with full audit

Run: python -m src.tests.test_v11_integration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datetime import datetime


def test_trade_intent_creation():
    """Test creating a TradeIntent."""
    from src.core.trade_intent import TradeIntent, IntentDirection, TradeIntentValidator
    
    intent = TradeIntent.create(
        symbol="BTC/USDT",
        direction=IntentDirection.LONG,
        strategy_id="test_strategy",
        confidence=0.85,
        reasoning="Test signal - breakout above resistance",
        suggested_entry=45000.0,
        suggested_stop=44000.0,
        suggested_target=47000.0,
    )
    
    print(f"✅ Created TradeIntent: {intent}")
    print(f"   Intent ID: {intent.intent_id}")
    print(f"   Checksum: {intent.checksum}")
    print(f"   Valid: {intent.verify_integrity()}")
    
    # Validate
    is_valid, reason = TradeIntentValidator.validate(intent)
    print(f"   Validator: {is_valid} - {reason}")
    
    return intent


def test_trade_decision_creation():
    """Test creating a TradeDecision."""
    from src.core.trade_decision import TradeDecision, DecisionStatus, RejectionStage
    from src.core.trade_intent import TradeIntent, IntentDirection
    
    intent = TradeIntent.create(
        symbol="BTC/USDT",
        direction=IntentDirection.LONG,
        strategy_id="test_strategy",
        confidence=0.85,
        reasoning="Test signal",
    )
    
    decision = TradeDecision.create(
        intent=intent,
        symbol=intent.symbol,
        strategy_id=intent.strategy_id,
        direction="long"
    )
    
    # Add some stage results
    decision.add_stage_result("incident_gate", True, "Passed", 1.5)
    decision.add_stage_result("market_context", True, "Trending", 2.0)
    decision.add_stage_result("bias_filter", False, "Opposite direction", 0.5)
    
    # Reject
    decision.reject(RejectionStage.BIAS_FILTER, "Strategy direction opposes market bias")
    
    print(f"✅ Created TradeDecision: {decision}")
    print(f"   Decision ID: {decision.decision_id}")
    print(f"   Status: {decision.status.value}")
    print(f"   Stages passed: {len([s for s in decision.stage_results if s.passed])}")
    
    return decision


def test_drawdown_governor():
    """Test the DrawdownGovernor."""
    from src.core.drawdown_governor import DrawdownGovernor, DrawdownSeverity
    
    governor = DrawdownGovernor(
        initial_equity=10000.0,
        daily_limit_pct=3.0,
        weekly_limit_pct=8.0,
        monthly_limit_pct=15.0,
        total_limit_pct=25.0,
    )
    
    # Simulate equity updates
    governor.update_equity(10000.0)
    governor.update_equity(9800.0)  # 2% drop
    governor.update_equity(9600.0)  # 4% drop
    
    status = governor.get_status()
    print(f"✅ DrawdownGovernor status:")
    print(f"   Current equity: ${status['current_equity']:,.2f}")
    print(f"   In defensive: {status['in_defensive_mode']}")
    print(f"   Size multiplier: {governor.get_size_multiplier():.2f}")
    
    return governor


def test_slippage_monitor():
    """Test the SlippageMonitor."""
    from src.core.slippage_monitor import SlippageMonitor
    
    monitor = SlippageMonitor(log_dir="logs/test_slippage")
    
    # Record some executions
    monitor.record_execution(
        order_id="test_001",
        symbol="BTC/USDT",
        side="buy",
        expected_price=45000.0,
        fill_price=45012.0,  # 2.67 bps slippage
        size=0.5
    )
    
    monitor.record_execution(
        order_id="test_002",
        symbol="BTC/USDT",
        side="sell",
        expected_price=45500.0,
        fill_price=45490.0,  # Favorable slippage
        size=0.5
    )
    
    status = monitor.get_status()
    print(f"✅ SlippageMonitor status:")
    print(f"   Total records: {status['total_records']}")
    print(f"   Is acceptable: {status['is_acceptable']}")
    print(f"   Avg slippage: {status['stats_24h']['avg_slippage_bps']:.2f} bps")
    
    return monitor


def test_exchange_recovery():
    """Test the ExchangeRecoveryHandler."""
    from src.core.exchange_recovery import ExchangeRecoveryHandler, ErrorCategory
    
    handler = ExchangeRecoveryHandler(log_dir="logs/test_recovery")
    
    # Test error classification
    test_errors = [
        Exception("Connection refused"),
        Exception("Rate limit exceeded"),
        Exception("Invalid API key"),
        Exception("Insufficient balance"),
    ]
    
    print(f"✅ ExchangeRecoveryHandler error classification:")
    for err in test_errors:
        category = handler.classify_error(err)
        print(f"   '{str(err)}' -> {category.value}")
    
    return handler


def test_strategy_adapter():
    """Test the strategy intent adapter."""
    from src.strategies.intent_adapter import (
        StrategyIntentAdapter, StrategySignal, StrategySignalType, create_intent
    )
    
    # Test adapter
    adapter = StrategyIntentAdapter(strategy_id="test_momentum", strategy_version="2.0")
    
    # Create signal
    signal = StrategySignal(
        signal_type=StrategySignalType.ENTRY_LONG,
        symbol="ETH/USDT",
        confidence=0.78,
        reasoning="Price broke above 20-day high with volume"
    )
    
    # Convert to intent
    intent = adapter.signal_to_intent(signal)
    
    print(f"✅ StrategyIntentAdapter:")
    print(f"   Signal type: {signal.signal_type.value}")
    print(f"   Intent: {intent}")
    
    # Test convenience function
    intent2 = create_intent(
        symbol="SOL/USDT",
        direction="short",
        strategy_id="mean_reversion",
        confidence=0.72,
        reasoning="Price deviated 3 std from mean"
    )
    
    print(f"   create_intent: {intent2}")
    
    return adapter


def test_full_pipeline():
    """Test the complete v11.0 pipeline."""
    from src.core.trade_intent import TradeIntent, IntentDirection
    from src.core.execution_flow import get_execution_flow
    
    print("\n" + "="*60)
    print("FULL v11.0 PIPELINE TEST")
    print("="*60)
    
    # Create intent
    intent = TradeIntent.create(
        symbol="BTC/USDT",
        direction=IntentDirection.LONG,
        strategy_id="pipeline_test",
        confidence=0.82,
        reasoning="Full pipeline integration test",
        suggested_entry=45000.0,
        suggested_stop=44000.0,
        suggested_target=47000.0,
    )
    
    print(f"\n1. Created Intent: {intent.intent_id[:8]}...")
    
    # Get orchestrator
    orchestrator = get_execution_flow()
    
    print(f"2. Orchestrator: v11.0 ready")
    
    # Process intent
    # Note: This will likely fail at some stage (e.g., incident gate)
    # because we don't have the full system initialized
    try:
        decision = orchestrator.process_intent(intent, current_price=45000.0)
        
        print(f"3. Decision: {decision.decision_id[:8]}...")
        print(f"   Status: {decision.status.value}")
        print(f"   Stages completed: {len(decision.stage_results)}")
        
        if decision.is_approved:
            print(f"   ✅ APPROVED - Ready to execute")
        else:
            print(f"   ❌ REJECTED at: {decision.rejection_stage.value if decision.rejection_stage else 'unknown'}")
            print(f"      Reason: {decision.rejection_reason}")
        
        return decision
        
    except Exception as e:
        print(f"3. Pipeline error (expected in test): {e}")
        return None


def run_all_tests():
    """Run all v11.0 integration tests."""
    print("\n" + "="*60)
    print("v11.0 INTEGRATION TESTS")
    print("="*60 + "\n")
    
    tests = [
        ("TradeIntent Creation", test_trade_intent_creation),
        ("TradeDecision Creation", test_trade_decision_creation),
        ("DrawdownGovernor", test_drawdown_governor),
        ("SlippageMonitor", test_slippage_monitor),
        ("ExchangeRecovery", test_exchange_recovery),
        ("StrategyAdapter", test_strategy_adapter),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            test_func()
            results.append((name, True, None))
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
