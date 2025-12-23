
import inspect
import run_trading_bot
from run_trading_bot import EnhancedTradingBot
import traceback

print("Checking EnhancedTradingBot...")
try:
    print(f"File: {run_trading_bot.__file__}")
    
    sig = inspect.signature(EnhancedTradingBot.__init__)
    print(f"Signature: {sig}")
    
    print("Attempting init...")
    bot = EnhancedTradingBot(initial_capital=10000)
    print("Init successful!")
    
except Exception:
    print("INIT FAILED!")
    traceback.print_exc()
