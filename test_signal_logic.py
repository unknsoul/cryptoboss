
import sys
import os
import pandas as pd
from datetime import datetime

# Mock classes
class MockProbSignal:
    def __init__(self):
        from enum import Enum
        class SignalAction(Enum):
            BUY = "BUY"
            SELL = "SELL"
            HOLD = "HOLD"
        
        self.action = SignalAction.BUY
        self.confidence = 0.6
        self.reasons = ["Test reason"]
        self.should_trade = True

def test_logic():
    print("Testing signal logic...")
    prob_signal = MockProbSignal()
    
    # 7. Convert to Bot Signal Format
    signal = {
        'action': prob_signal.action.value, 
        'confidence': prob_signal.confidence,
        'reasons': prob_signal.reasons.copy(),
        'price': 100000,
        'atr': 100,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"Initial Action: {signal['action']}")
    
    # Map BUY/SELL to LONG/SHORT for compatibility
    if signal['action'] == 'BUY': signal['action'] = 'LONG'
    if signal['action'] == 'SELL': signal['action'] = 'SHORT'
    
    print(f"Mapped Action: {signal['action']}")
    
    if signal['action'] == 'LONG':
        print("Mapping SUCCESS")
    else:
        print("Mapping FAILED")

if __name__ == "__main__":
    test_logic()
