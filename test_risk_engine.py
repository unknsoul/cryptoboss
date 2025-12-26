"""
Test Risk Engine Components
Demonstrates volatility-adjusted sizing and kill-switch.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk import VolatilityAdjustedSizing, KillSwitch


def test_volatility_sizing():
    """Test volatility-adjusted position sizing."""
    print("=" * 70)
    print("VOLATILITY-ADJUSTED POSITION SIZING")
    print("=" * 70)
    print()
    
    sizer = VolatilityAdjustedSizing(
        base_risk_pct=1.0,
        atr_multiplier=2.0
    )
    
    equity = 10000
    price = 40000
    
    # Test different volatility scenarios
    scenarios = [
        ('Low Volatility', 200, 150),
        ('Normal Volatility', 400, 400),
        ('High Volatility', 800, 600)
    ]
    
    print("📊 Position Sizing Scenarios:")
    print("-" * 70)
    for scenario_name, current_atr, hist_atr in scenarios:
        regime = sizer.detect_volatility_regime(current_atr, hist_atr)
        result = sizer.calculate_position_size(equity, price, current_atr, regime)
        
        print(f"\n{scenario_name} (ATR: ${current_atr}, Regime: {regime})")
        print(f"   Position Size:   ${result['position_size_dollars']:,.2f}")
        print(f"   Quantity:        {result['quantity']:.6f} BTC")
        print(f"   Risk:            {result['risk_pct']:.2f}%")
        print(f"   Stop Loss:       ${result['stop_loss']:,.2f}")
        print(f"   Stop Distance:   ${result['stop_distance']:,.2f}")
    
    print()


def test_kill_switch():
    """Test kill-switch logic."""
    print("=" * 70)
    print("KILL SWITCH - EMERGENCY HALT SYSTEM")
    print("=" * 70)
    print()
    
    kill_switch = KillSwitch(initial_equity=10000)
    
    print("🔴 Halt Condition Thresholds:")
    print(f"   Max Daily Loss:       {kill_switch.HALT_CONDITIONS['max_daily_loss_pct']}%")
    print(f"   Consecutive Losses:   {kill_switch.HALT_CONDITIONS['max_consecutive_losses']}")
    print(f"   Max Drawdown:         {kill_switch.HALT_CONDITIONS['max_drawdown_pct']}%")
    print(f"   Min Equity Threshold: {kill_switch.HALT_CONDITIONS['min_equity_threshold_pct']}%")
    print()
    
    print("🧪 Test Scenarios:")
    print("-" * 70)
    
    # Scenario 1: Normal trading
    print("\n1. Normal Trading (small win)")
    result = kill_switch.check_halt_conditions(10100, last_trade_pnl=100)
    print(f"   Should Halt: {result['should_halt']}")
    print(f"   Status: {'✅ OK' if not result['should_halt'] else '🚨 HALTED'}")
    
    # Scenario 2: Small loss
    print("\n2. Small Loss")
    result = kill_switch.check_halt_conditions(9950, last_trade_pnl=-50)
    print(f"   Should Halt: {result['should_halt']}")
    print(f"   Consecutive Losses: {kill_switch.consecutive_losses}")
    
    # Scenario 3: Multiple consecutive losses
    print("\n3. Multiple Consecutive Losses")
    for i in range(5):
        result = kill_switch.check_halt_conditions(9900 - (i*50), last_trade_pnl=-50)
        print(f"   Loss {i+1}: Consecutive = {kill_switch.consecutive_losses}")
    
    if result['should_halt']:
        print(f"   🚨 HALT TRIGGERED: {result['reason']}")
    
    # Reset for next test
    kill_switch.reset_halt()
    
    # Scenario 4: Large daily loss
    print("\n4. Large Daily Loss (-6%)")
    kill_switch.reset_daily(10000)
    result = kill_switch.check_halt_conditions(9400, last_trade_pnl=-600)
    print(f"   Current Equity: $9,400")
    print(f"   Daily Loss: -6.0%")
    if result['should_halt']:
        print(f"   🚨 HALT TRIGGERED: {result['reason']}")
    
    # Scenario 5: Max drawdown
    print("\n5. Max Drawdown Test")
    kill_switch.reset_halt()
    kill_switch.peak_equity = 12000
    result = kill_switch.check_halt_conditions(10500, last_trade_pnl=-500)
    drawdown = ((10500 - 12000) / 12000) * 100
    print(f"   Peak: $12,000")
    print(f"   Current: $10,500")
    print(f"   Drawdown: {drawdown:.1f}%")
    if result['should_halt']:
        print(f"   🚨 HALT TRIGGERED: {result['reason']}")
    
    print()
    print("=" * 70)


def main():
    """Run all risk engine tests."""
    test_volatility_sizing()
    print("\n\n")
    test_kill_switch()
    
    print()
    print("=" * 70)
    print("✅ Risk engine tests complete")
    print("=" * 70)
    print()
    print("🎯 Key Features:")
    print("   ✓ Position size scales with volatility")
    print("   ✓ Kill-switch prevents catastrophic losses")
    print("   ✓ Multiple halt conditions (loss, DD, streak)")
    print("   ✓ Manual reset capability")


if __name__ == "__main__":
    main()
