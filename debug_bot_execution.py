
import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock dependencies
sys.path.insert(0, '.')

try:
    from run_trading_bot import EnhancedTradingBot, trading_state
except ImportError as e:
    print(f"Failed to import bot: {e}")
    sys.exit(1)

def run_debug_test():
    print("=== STARTING DEBUG TEST ===")
    
    # Initialize bot
    bot = EnhancedTradingBot(initial_capital=10000)
    print("Bot Initialized")
    
    # Verify State
    print(f"Bot Position: {bot.position}")
    print(f"Trading State Positions: {trading_state['positions']}")
    
    # Create Mock Signal
    print("\nAttempting to EXECUTE mock signal...")
    mock_signal = {
        'action': 'LONG',
        'confidence': 0.8,
        'reasons': ['Debug Force Trade'],
        'price': 90000.0,
        'atr': 500.0,
        'timestamp': datetime.now().isoformat(),
        'sl_multiplier': 1.5, # Normally from advanced features
        'tp_multiplier': 2.5
    }
    
    # Manually add advanced feature overrides if needed
    if bot.advanced_features:
        print("Advanced Features are ENABLED")
        # Ensure we have some outcome
        adj_size = bot.advanced_features.get_size_adjustment(0.1, bot.equity)
        print(f"Size Adjustment check: 0.1 -> {adj_size}")
        
    execution_result = bot.execute_trade(mock_signal)
    print(f"\nExecution Result: {execution_result}")
    
    # Check post-execution state
    print(f"Post-Trade Position: {bot.position}")
    print(f"Post-Trade Trades Count: {len(bot.trades)}")
    
    if bot.position:
        print("SUCCESS: Position created!")
    else:
        print("FAILURE: No position created.")

if __name__ == "__main__":
    run_debug_test()
