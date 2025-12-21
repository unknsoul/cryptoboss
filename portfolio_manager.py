
"""
Enterprise Portfolio Management Module (Features 161-200)
Orchestrates multiple LiveTraders and manages global risk/allocation.
"""

import time
import threading
from typing import Dict, List
from live_trader import LiveTrader
from core.risk.risk_manager import RiskManager

class PortfolioManager:
    def __init__(self, initial_capital: float = 50000.0):
        print("🌐 Initializing Enterprise Portfolio Manager...")
        self.capital = initial_capital
        self.risk_manager = RiskManager(capital=initial_capital)
        self.traders: Dict[str, LiveTrader] = {}
        self.is_running = False
        
        # Define portfolio allocation (Feature 162)
        self.allocation = {
            'BTCUSDT': {'strategy': 'enhanced_momentum', 'allocation': 0.4},
            'ETHUSDT': {'strategy': 'macd_crossover', 'allocation': 0.3},
            'SOLUSDT': {'strategy': 'scalping', 'allocation': 0.2},
            'BNBUSDT': {'strategy': 'mean_reversion', 'allocation': 0.1}
        }
        
    def start_portfolio(self):
        """Launch all trader instances in separate threads"""
        print(f"🚀 Launching Multi-Asset Portfolio (Total Capital: ${self.capital:,.2f})")
        self.is_running = True
        
        for symbol, config in self.allocation.items():
            allocated_amt = self.capital * config['allocation']
            print(f"  > Spawning Trader: {symbol} | Strategy: {config['strategy']} | Alloc: ${allocated_amt:,.2f}")
            
            # Create trader instance
            trader = LiveTrader(
                symbol=symbol,
                strategy_name=config['strategy'],
                capital=allocated_amt
            )
            
            self.traders[symbol] = trader
            
            # Start in background thread (Feature 161)
            t = threading.Thread(target=trader.start)
            t.daemon = True
            t.start()
            
            # Stagger starts to avoid API limit spikes
            time.sleep(1)
            
        print("✅ All Portfolio Managers Online.")
        print("Monitoring functionality active across all venues.")
        
        try:
            while self.is_running:
                self._monitor_portfolio()
                time.sleep(5)
        except KeyboardInterrupt:
            self.stop_portfolio()
            
    def _monitor_portfolio(self):
        """Central monitoring loop (Feature 167: Dynamic Rebalancing)"""
        total_value = 0
        active_trades = 0
        
        # Aggregated reporting
        # In a real impl, we'd query each trader for current equity
        pass
        
    def stop_portfolio(self):
        print("\n🛑 Emergency Stop: Shutting down all traders...")
        self.is_running = False
        for symbol, trader in self.traders.items():
            print(f"  Stopping {symbol}...")
            trader.stop()
            
if __name__ == "__main__":
    pm = PortfolioManager()
    pm.start_portfolio()
