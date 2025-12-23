
import logging
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from run_trading_bot import EnhancedTradingBot, logger, trading_state

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_pipeline():
    try:
        print("Initializing Bot...")
        bot = EnhancedTradingBot(initial_capital=10000)
        
        # Mock Data (Strong Bullish Trend) - 5-minute candles
        dates = pd.date_range(end=datetime.now(), periods=200, freq='5min')
        prices = np.linspace(50000, 55000, 200) # Strong up trend
        
        # Add some noise
        noise = np.random.randn(200) * 50
        prices = prices + noise
        
        highs = prices + 100
        lows = prices - 100
        opens = prices - 50
        closes = prices + 50
        volumes = np.random.randint(100, 1000, 200)
        
        df_5m = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        candles_5m = df_5m.to_dict('records')
        
        # INJECT MOCK DATA INTO STATE
        trading_state['candles'] = candles_5m
        trading_state['current_price'] = closes[-1]
        
        with open('debug_output.txt', 'w') as f:
            f.write("\n--- Testing Signal Generation ---\n")
            current_price = closes[-1]
            
            # Force strategy manager active strategies
            f.write(f"Active Strategies: {bot.strategy_manager.active_strategies}\n")
            
            # Run Generate Signal (Fixed call signature)
            signal = bot.generate_signal(current_price)
            
            f.write("\n--- Result ---\n")
            if signal:
                f.write(f"Signal Generated: {signal}\n")
                
                # Test Execution
                f.write("\n--- Testing Execution ---\n")
                executed = bot.execute_trade(signal)
                f.write(f"Trade Executed: {executed}\n")
                f.write(f"Positions: {len(bot.trades)}\n")
                f.write(f"Current Position State: {bot.position}\n")
            else:
                f.write("No Signal Generated\n")

    except Exception:
        print("\n!!! EXCEPTION CAUGHT !!!")
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
