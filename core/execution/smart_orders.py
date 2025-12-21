"""
Smart Order Execution - TWAP (Time-Weighted Average Price)
Splits large orders across time to reduce market impact
"""

import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import threading
from core.monitoring.logger import get_logger
from core.monitoring.metrics import get_metrics


logger = get_logger()
metrics = get_metrics()


class TWAPExecutor:
    """
    TWAP (Time-Weighted Average Price) Order Executor
    
    Splits a large parent order into smaller child orders executed
    at regular intervals to minimize market impact and achieve
    average execution price over time.
    """
    
    def __init__(self, exchange, num_slices: int = 10, 
                 duration_seconds: int = 300):
        """
        Args:
            exchange: Exchange client for order execution
            num_slices: Number of slices to split the order into
            duration_seconds: Total duration for TWAP execution
        """
        self.exchange = exchange
        self.num_slices = num_slices
        self.duration_seconds = duration_seconds
        self.interval_seconds = duration_seconds / num_slices
        
        self.active_twaps: Dict[str, Dict[str, Any]] = {}
        self._stop_threads = {}
        
    def execute_twap(self, symbol: str, side: str, total_quantity: float,
                    on_complete: Optional[Callable] = None,
                    on_slice: Optional[Callable] = None) -> str:
        """
        Execute TWAP order
        
        Args:
            symbol: Trading pair
            side: BUY or SELL
            total_quantity: Total quantity to execute
            on_complete: Callback when TWAP is complete
            on_slice: Callback after each slice execution
        
        Returns:
            TWAP order ID
        """
        twap_id = f"twap_{symbol}_{int(time.time() * 1000)}"
        
        slice_quantity = total_quantity / self.num_slices
        
        twap_order = {
            'id': twap_id,
            'symbol': symbol,
            'side': side,
            'total_quantity': total_quantity,
            'slice_quantity': slice_quantity,
            'num_slices': self.num_slices,
            'slices_executed': 0,
            'total_filled': 0.0,
            'avg_price': 0.0,
            'start_time': datetime.now(),
            'status': 'ACTIVE',
            'fills': []
        }
        
        self.active_twaps[twap_id] = twap_order
        self._stop_threads[twap_id] = False
        
        logger.info(
            f"Starting TWAP execution for {symbol}",
            twap_id=twap_id,
            side=side,
            total_quantity=total_quantity,
            num_slices=self.num_slices,
            duration_seconds=self.duration_seconds
        )
        
        # Start execution in background thread
        thread = threading.Thread(
            target=self._execute_slices,
            args=(twap_id, on_complete, on_slice),
            daemon=True
        )
        thread.start()
        
        return twap_id
    
    def _execute_slices(self, twap_id: str, 
                       on_complete: Optional[Callable],
                       on_slice: Optional[Callable]):
        """Execute TWAP slices"""
        twap = self.active_twaps[twap_id]
        
        for i in range(self.num_slices):
            if self._stop_threads.get(twap_id, False):
                logger.warning(f"TWAP {twap_id} cancelled")
                twap['status'] = 'CANCELLED'
                break
            
            try:
                # Execute slice
                slice_start = time.time()
                
                order = self.exchange.place_order(
                    symbol=twap['symbol'],
                    side=twap['side'],
                    order_type='MARKET',
                    quantity=twap['slice_quantity']
                )
                
                slice_latency = (time.time() - slice_start) * 1000
                
                # Record fill
                fill = {
                    'slice': i + 1,
                    'quantity': order['filled_qty'],
                    'price': order['price'],
                    'timestamp': datetime.now(),
                    'latency_ms': slice_latency
                }
                
                twap['fills'].append(fill)
                twap['slices_executed'] += 1
                twap['total_filled'] += order['filled_qty']
                
                # Update average price (weighted)
                total_value = sum(f['quantity'] * f['price'] for f in twap['fills'])
                twap['avg_price'] = total_value / twap['total_filled']
                
                logger.info(
                    f"TWAP slice {i+1}/{self.num_slices} executed",
                    twap_id=twap_id,
                    quantity=order['filled_qty'],
                    price=order['price'],
                    avg_price=twap['avg_price']
                )
                
                metrics.record_timer(f"twap_slice_latency_{twap['symbol']}", slice_latency)
                
                # Callback
                if on_slice:
                    on_slice(twap_id, fill)
                
                # Wait for next slice (except last one)
                if i < self.num_slices - 1:
                    time.sleep(self.interval_seconds)
                    
            except Exception as e:
                logger.error(
                    f"TWAP slice {i+1} execution failed",
                    twap_id=twap_id,
                    error=str(e)
                )
                twap['status'] = 'FAILED'
                twap['error'] = str(e)
                break
        
        # TWAP complete
        if twap['status'] == 'ACTIVE':
            twap['status'] = 'COMPLETED'
            twap['end_time'] = datetime.now()
            duration = (twap['end_time'] - twap['start_time']).total_seconds()
            
            # Calculate slippage vs benchmark
            if twap['fills']:
                benchmark_price = twap['fills'][0]['price']  # First slice as benchmark
                slippage = (twap['avg_price'] - benchmark_price) / benchmark_price
                if twap['side'] == 'SELL':
                    slippage = -slippage  # Invert for sells
                
                twap['slippage'] = slippage
            
            logger.info(
                f"TWAP execution complete",
                twap_id=twap_id,
                total_filled=twap['total_filled'],
                avg_price=twap['avg_price'],
                duration_seconds=duration,
                slippage=twap.get('slippage', 0)
            )
            
            metrics.increment(f"twap_completed_{twap['symbol']}")
            
            # Callback
            if on_complete:
                on_complete(twap_id, twap)
    
    def cancel_twap(self, twap_id: str) -> bool:
        """Cancel an active TWAP order"""
        if twap_id in self.active_twaps:
            self._stop_threads[twap_id] = True
            logger.info(f"TWAP cancellation requested", twap_id=twap_id)
            return True
        return False
    
    def get_twap_status(self, twap_id: str) -> Optional[Dict[str, Any]]:
        """Get TWAP order status"""
        return self.active_twaps.get(twap_id)
    
    def get_active_twaps(self) -> List[str]:
        """Get list of active TWAP order IDs"""
        return [
            twap_id for twap_id, twap in self.active_twaps.items()
            if twap['status'] == 'ACTIVE'
        ]


class VWAPExecutor:
    """
    VWAP (Volume-Weighted Average Price) Order Executor
    
    Executes orders proportional to historical volume patterns
    to blend with normal market flow.
    """
    
    def __init__(self, exchange, duration_minutes: int = 60,
                 participation_rate: float = 0.10):
        """
        Args:
            exchange: Exchange client
            duration_minutes: Duration to execute over
            participation_rate: Target % of market volume (e.g., 0.10 = 10%)
        """
        self.exchange = exchange
        self.duration_minutes = duration_minutes
        self.participation_rate = participation_rate
        
        logger.info(
            "VWAP Executor initialized",
            duration_minutes=duration_minutes,
            participation_rate=participation_rate
        )
    
    def execute_vwap(self, symbol: str, side: str, total_quantity: float) -> str:
        """
        Execute VWAP order
        
        Note: This is a simplified implementation.
        Production version would analyze real-time volume and adjust.
        """
        # For now, delegate to TWAP with volume-adjusted slicing
        # Real implementation would monitor live volume and adjust dynamically
        
        twap = TWAPExecutor(
            self.exchange,
            num_slices=12,  # 5-minute intervals for 1-hour execution
            duration_seconds=self.duration_minutes * 60
        )
        
        return twap.execute_twap(symbol, side, total_quantity)


if __name__ == "__main__":
    # Test TWAP executor
    from core.testing.mock_exchange import MockExchange
    
    exchange = MockExchange(initial_balance=100000)
    exchange.connect()
    
    twap = TWAPExecutor(
        exchange=exchange,
        num_slices=5,
        duration_seconds=10  # 10 seconds for testing
    )
    
    def on_slice(twap_id, fill):
        print(f"  Slice executed: {fill['quantity']} @ {fill['price']}")
    
    def on_complete(twap_id, twap_data):
        print(f"\n✅ TWAP Complete!")
        print(f"  Total Filled: {twap_data['total_filled']}")
        print(f"  Avg Price: {twap_data['avg_price']:.2f}")
        print(f"  Slippage: {twap_data.get('slippage', 0):.4%}")
    
    print("Executing TWAP order...")
    twap_id = twap.execute_twap(
        symbol="BTCUSDT",
        side="BUY",
        total_quantity=0.5,
        on_complete=on_complete,
        on_slice=on_slice
    )
    
    # Wait for completion
    time.sleep(15)
    
    exchange.disconnect()
