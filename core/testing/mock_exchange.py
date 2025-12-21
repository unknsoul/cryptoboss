"""
Mock Exchange for Testing
Simulates exchange behavior without real API calls
"""

import time
import random
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import threading


class MockExchange:
    """
    Mock exchange client for testing
    Simulates:
    - Order placement and execution
    - Position tracking
    - Market data
    - WebSocket connections
    - Various error scenarios
    """
    
    def __init__(self, initial_balance: float = 10000, latency_ms: float = 50,
                 failure_rate: float = 0.0):
        """
        Args:
            initial_balance: Starting USDT balance
            latency_ms: Simulated API latency in milliseconds
            failure_rate: Probability of API call failure (0.0 to 1.0)
        """
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        
        # State
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.order_id_counter = 1000
        self.is_connected = False
        
        # Market data
        self.current_prices = {
            "BTCUSDT": 45000.0,
            "ETHUSDT": 3000.0,
            "SOLUSDT": 100.0,
            "BNBUSDT": 350.0
        }
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Statistics
        self.api_call_count = 0
        self.order_count = 0
        self.failed_calls = 0
        
        # Price simulation thread
        self._price_thread = None
        self._stop_price_sim = False
    
    def _simulate_latency(self):
        """Simulate network latency"""
        time.sleep(self.latency_ms / 1000.0)
    
    def _maybe_fail(self, operation: str):
        """Randomly fail based on failure_rate"""
        self.api_call_count += 1
        
        if random.random() < self.failure_rate:
            self.failed_calls += 1
            raise Exception(f"MockExchange: Simulated failure for {operation}")
        
        self._simulate_latency()
    
    def connect(self):
        """Simulate WebSocket connection"""
        self._maybe_fail("connect")
        self.is_connected = True
        self._start_price_simulation()
        print("📡 MockExchange: Connected")
    
    def disconnect(self):
        """Disconnect"""
        self.is_connected = False
        self._stop_price_sim = True
        if self._price_thread:
            self._price_thread.join(timeout=1.0)
        print("📡 MockExchange: Disconnected")
    
    def _start_price_simulation(self):
        """Start price simulation thread"""
        def simulate_prices():
            while not self._stop_price_sim:
                for symbol in self.current_prices:
                    # Random walk
                    change_pct = random.gauss(0, 0.001)  # 0.1% std dev
                    self.current_prices[symbol] *= (1 + change_pct)
                
                # Emit ticker updates
                self._emit('ticker', {
                    'symbol': 'BTCUSDT',
                    'price': self.current_prices['BTCUSDT'],
                    'timestamp': int(time.time() * 1000)
                })
                
                time.sleep(1)  # Update every second
        
        self._price_thread = threading.Thread(target=simulate_prices, daemon=True)
        self._price_thread.start()
    
    def _emit(self, event: str, data: Any):
        """Emit event to callbacks"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Error in callback: {e}")
    
    def on(self, event: str, callback: Callable):
        """Register callback"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def subscribe_ticker(self, symbol: str):
        """Subscribe to ticker updates"""
        self._maybe_fail("subscribe_ticker")
        print(f"📊 Subscribed to {symbol}")
    
    def subscribe_orderbook(self, symbol: str):
        """Subscribe to order book"""
        self._maybe_fail("subscribe_orderbook")
        print(f"📖 Subscribed to {symbol} orderbook")
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker"""
        self._maybe_fail("get_ticker")
        
        return {
            'symbol': symbol,
            'price': self.current_prices.get(symbol, 0),
            'volume': random.uniform(1000, 10000),
            'timestamp': int(time.time() * 1000)
        }
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """Fetch historical OHLCV data"""
        self._maybe_fail("fetch_ohlcv")
        
        # Generate fake OHLCV data
        base_price = self.current_prices.get(symbol, 100)
        ohlcv = []
        
        for i in range(limit):
            timestamp = int((time.time() - (limit - i) * 3600) * 1000)
            open_price = base_price * random.uniform(0.95, 1.05)
            high = open_price * random.uniform(1.0, 1.02)
            low = open_price * random.uniform(0.98, 1.0)
            close = random.uniform(low, high)
            volume = random.uniform(100, 1000)
            
            ohlcv.append([timestamp, open_price, high, low, close, volume])
        
        return ohlcv
    
    def place_order(self, symbol: str, side: str, order_type: str, 
                   quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Place an order"""
        self._maybe_fail("place_order")
        
        order_id = f"order_{self.order_id_counter}"
        self.order_id_counter += 1
        self.order_count += 1
        
        # Calculate order cost
        exec_price = price if price else self.current_prices.get(symbol, 0)
        cost = quantity * exec_price
        
        # Check balance
        if side == "BUY" and cost > self.balance:
            raise Exception(f"Insufficient balance: {self.balance} < {cost}")
        
        # Execute order immediately (market order simulation)
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'price': exec_price,
            'status': 'filled',
            'filled_qty': quantity,
            'timestamp': datetime.now().isoformat()
        }
        
        self.orders[order_id] = order
        
        # Update balance and position
        if side == "BUY":
            self.balance -= cost
            self._update_position(symbol, quantity, exec_price)
        else:  # SELL
            self.balance += cost
            self._update_position(symbol, -quantity, exec_price)
        
        print(f"✅ Order executed: {side} {quantity} {symbol} @ {exec_price}")
        return order
    
    def _update_position(self, symbol: str, quantity: float, price: float):
        """Update position"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': 0,
                'entry_price': 0,
                'unrealized_pnl': 0
            }
        
        pos = self.positions[symbol]
        
        # Update position
        old_qty = pos['quantity']
        new_qty = old_qty + quantity
        
        if new_qty == 0:
            # Position closed
            del self.positions[symbol]
        else:
            # Update entry price (weighted average for accumulation)
            if (old_qty >= 0 and quantity > 0) or (old_qty <= 0 and quantity < 0):
                # Accumulating position
                total_cost = (old_qty * pos['entry_price']) + (quantity * price)
                pos['entry_price'] = total_cost / new_qty
            
            pos['quantity'] = new_qty
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        self._maybe_fail("get_positions")
        
        return {
            symbol: data['quantity'] 
            for symbol, data in self.positions.items()
        }
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        self._maybe_fail("get_balance")
        
        # Calculate unrealized PnL
        unrealized_pnl = 0
        for symbol, pos in self.positions.items():
            current_price = self.current_prices.get(symbol, pos['entry_price'])
            pnl = (current_price - pos['entry_price']) * pos['quantity']
            unrealized_pnl += pnl
        
        return {
            'free': self.balance,
            'used': sum(abs(p['quantity'] * p['entry_price']) for p in self.positions.values()),
            'total': self.balance + unrealized_pnl,
            'unrealized_pnl': unrealized_pnl
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        self._maybe_fail("cancel_order")
        
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'cancelled'
            return True
        return False
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order details"""
        self._maybe_fail("get_order")
        return self.orders.get(order_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get exchange statistics"""
        return {
            'api_calls': self.api_call_count,
            'failed_calls': self.failed_calls,
            'success_rate': 1 - (self.failed_calls / max(self.api_call_count, 1)),
            'total_orders': self.order_count,
            'active_positions': len(self.positions),
            'current_balance': self.balance,
            'initial_balance': self.initial_balance,
            'balance_change': self.balance - self.initial_balance
        }
    
    def reset(self):
        """Reset exchange to initial state"""
        self.balance = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.order_id_counter = 1000
        self.api_call_count = 0
        self.order_count = 0
        self.failed_calls = 0
        print("🔄 MockExchange: Reset to initial state")


if __name__ == "__main__":
    # Test mock exchange
    exchange = MockExchange(initial_balance=10000, latency_ms=20, failure_rate=0.1)
    exchange.connect()
    
    # Test ticker
    ticker = exchange.get_ticker("BTCUSDT")
    print(f"Ticker: {ticker}")
    
    # Test order placement
    try:
        order = exchange.place_order("BTCUSDT", "BUY", "MARKET", 0.1)
        print(f"Order: {order}")
    except Exception as e:
        print(f"Order failed: {e}")
    
    # Test positions
    positions = exchange.get_positions()
    print(f"Positions: {positions}")
    
    # Test balance
    balance = exchange.get_balance()
    print(f"Balance: {balance}")
    
    # Stats
    print(f"\nStats: {exchange.get_stats()}")
    
    exchange.disconnect()
