"""
Smart Order Routing with TWAP/VWAP Algorithms
Minimize market impact and optimize execution price
"""

import asyncio
import numpy as np
from typing import Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    """Order execution algorithm type"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"  # Hide full size
    ADAPTIVE = "adaptive"  # Adaptive based on market conditions


@dataclass
class ExecutionResult:
    """Result of order execution"""
    total_size: float
    filled_size: float
    average_price: float
    total_cost: float
    slices_executed: int
    duration_seconds: float
    slippage_bps: float
    success: bool
    error: Optional[str] = None


class SmartOrderRouter:
    """
    Advanced order routing with multiple execution algorithms
    
    Algorithms:
    - TWAP: Split order evenly over time
    - VWAP: Split order based on volume profile
    - Iceberg: Hide order size, execute in chunks
    - Adaptive: Adjust based on market conditions
    """
    
    def __init__(self, exchange_client=None):
        """
        Initialize smart order router
        
        Args:
            exchange_client: Exchange client for order execution
        """
        self.exchange_client = exchange_client
        self.execution_history: List[ExecutionResult] = []
    
    async def execute_twap(
        self,
        symbol: str,
        total_size: float,
        duration_minutes: int,
        side: str = 'BUY',
        num_slices: Optional[int] = None
    ) -> ExecutionResult:
        """
        Time-Weighted Average Price execution
        Splits order evenly over time to reduce market impact
        
        Args:
            symbol: Trading symbol
            total_size: Total size to execute
            duration_minutes: Duration to spread execution over
            side: 'BUY' or 'SELL'
            num_slices: Number of slices (default: duration_minutes)
            
        Returns:
            ExecutionResult with execution details
        """
        if num_slices is None:
            num_slices = duration_minutes
        
        slice_size = total_size / num_slices
        interval_seconds = (duration_minutes * 60) / num_slices
        
        print(f"TWAP Execution: {total_size} {symbol} over {duration_minutes}m in {num_slices} slices")
        print(f"  Slice size: {slice_size:.6f}")
        print(f"  Interval: {interval_seconds:.1f}s")
        
        start_time = datetime.now()
        filled_size = 0
        total_cost = 0
        slices_executed = 0
        
        try:
            for i in range(num_slices):
                # Execute slice
                if self.exchange_client:
                    # Real execution
                    result = await self._execute_market_order(symbol, slice_size, side)
                    filled_size += result['filled_size']
                    total_cost += result['cost']
                else:
                    # Simulated execution
                    simulated_price = 100000 + np.random.randn() * 50  # Mock price
                    filled_size += slice_size
                    total_cost += slice_size * simulated_price
                
                slices_executed += 1
                
                print(f"  Slice {i+1}/{num_slices} executed: {slice_size:.6f} @ ${total_cost/filled_size:.2f}")
                
                # Wait for next interval (except last slice)
                if i < num_slices - 1:
                    await asyncio.sleep(interval_seconds)
            
            duration = (datetime.now() - start_time).total_seconds()
            avg_price = total_cost / filled_size if filled_size > 0 else 0
            
            result = ExecutionResult(
                total_size=total_size,
                filled_size=filled_size,
                average_price=avg_price,
                total_cost=total_cost,
                slices_executed=slices_executed,
                duration_seconds=duration,
                slippage_bps=0,  # Would calculate vs benchmark
                success=True
            )
            
            self.execution_history.append(result)
            return result
            
        except Exception as e:
            return ExecutionResult(
                total_size=total_size,
                filled_size=filled_size,
                average_price=0,
                total_cost=total_cost,
                slices_executed=slices_executed,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                slippage_bps=0,
                success=False,
                error=str(e)
            )
    
    async def execute_vwap(
        self,
        symbol: str,
        total_size: float,
        volume_profile: List[float],
        duration_minutes: int,
        side: str = 'BUY'
    ) -> ExecutionResult:
        """
        Volume-Weighted Average Price execution
        Splits order based on expected volume distribution
        
        Args:
            symbol: Trading symbol
            total_size: Total size to execute
            volume_profile: Volume distribution (e.g., hourly volumes)
            duration_minutes: Duration to spread over
            side: 'BUY' or 'SELL'
            
        Returns:
            ExecutionResult
        """
        # Normalize volume profile
        total_volume = sum(volume_profile)
        volume_weights = [v / total_volume for v in volume_profile]
        
        # Calculate slice sizes based on volume
        slice_sizes = [total_size * weight for weight in volume_weights]
        
        print(f"VWAP Execution: {total_size} {symbol} over {duration_minutes}m")
        print(f"  Slices: {len(slice_sizes)} (volume-weighted)")
        
        start_time = datetime.now()
        filled_size = 0
        total_cost = 0
        slices_executed = 0
        interval_seconds = (duration_minutes * 60) / len(slice_sizes)
        
        try:
            for i, slice_size in enumerate(slice_sizes):
                if slice_size > 0:
                    # Execute slice
                    if self.exchange_client:
                        result = await self._execute_market_order(symbol, slice_size, side)
                        filled_size += result['filled_size']
                        total_cost += result['cost']
                    else:
                        # Simulated
                        simulated_price = 100000 + np.random.randn() * 50
                        filled_size += slice_size
                        total_cost += slice_size * simulated_price
                    
                    slices_executed += 1
                    print(f"  Slice {i+1}/{len(slice_sizes)}: {slice_size:.6f} (weight: {volume_weights[i]:.2%})")
                    
                    if i < len(slice_sizes) - 1:
                        await asyncio.sleep(interval_seconds)
            
            duration = (datetime.now() - start_time).total_seconds()
            avg_price = total_cost / filled_size if filled_size > 0 else 0
            
            result = ExecutionResult(
                total_size=total_size,
                filled_size=filled_size,
                average_price=avg_price,
                total_cost=total_cost,
                slices_executed=slices_executed,
                duration_seconds=duration,
                slippage_bps=0,
                success=True
            )
            
            self.execution_history.append(result)
            return result
            
        except Exception as e:
            return ExecutionResult(
                total_size=total_size,
                filled_size=filled_size,
                average_price=0,
                total_cost=total_cost,
                slices_executed=slices_executed,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                slippage_bps=0,
                success=False,
                error=str(e)
            )
    
    async def execute_iceberg(
        self,
        symbol: str,
        total_size: float,
        visible_size: float,
        side: str = 'BUY',
        price: Optional[float] = None
    ) -> ExecutionResult:
        """
        Iceberg order execution
        Hides total size by showing only small visible portion
        
        Args:
            symbol: Trading symbol
            total_size: Total order size
            visible_size: Size visible to market
            side: 'BUY' or 'SELL'
            price: Limit price (None for market)
            
        Returns:
            ExecutionResult
        """
        print(f"Iceberg Order: {total_size} {symbol} (showing {visible_size})")
        
        start_time = datetime.now()
        filled_size = 0
        total_cost = 0
        slices_executed = 0
        
        remaining = total_size
        
        try:
            while remaining > 0:
                # Place visible portion
                current_slice = min(remaining, visible_size)
                
                if self.exchange_client:
                    # place_order implementation would go here
                    pass
                else:
                    # Simulated
                    simulated_price = price or (100000 + np.random.randn() * 50)
                    filled_size += current_slice
                    total_cost += current_slice * simulated_price
                    await asyncio.sleep(0.1)  # Simulate execution delay
                
                remaining -= current_slice
                slices_executed += 1
                
                print(f"  Filled {current_slice:.6f}, Remaining: {remaining:.6f}")
            
            duration = (datetime.now() - start_time).total_seconds()
            avg_price = total_cost / filled_size if filled_size > 0 else 0
            
            return ExecutionResult(
                total_size=total_size,
                filled_size=filled_size,
                average_price=avg_price,
                total_cost=total_cost,
                slices_executed=slices_executed,
                duration_seconds=duration,
                slippage_bps=0,
                success=True
            )
            
        except Exception as e:
            return ExecutionResult(
                total_size=total_size,
                filled_size=filled_size,
                average_price=0,
                total_cost=total_cost,
                slices_executed=slices_executed,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                slippage_bps=0,
                success=False,
                error=str(e)
            )
    
    async def _execute_market_order(self, symbol: str, size: float, side: str) -> dict:
        """
        Execute single market order slice
        
        Args:
            symbol: Trading symbol
            size: Order size
            side: 'BUY' or 'SELL'
            
        Returns:
            Dictionary with execution result
        """
        # This would interface with actual exchange
        if self.exchange_client:
            return await self.exchange_client.create_market_order(symbol, side, size)
        else:
            # Simulated response
            price = 100000 + np.random.randn() * 50
            return {
                'filled_size': size,
                'price': price,
                'cost': size * price
            }
    
    def get_execution_stats(self) -> dict:
        """Get statistics on past executions"""
        if not self.execution_history:
            return {}
        
        successful = [r for r in self.execution_history if r.success]
        
        return {
            'total_executions': len(self.execution_history),
            'successful': len(successful),
            'success_rate': len(successful) / len(self.execution_history),
            'avg_slippage_bps': np.mean([r.slippage_bps for r in successful]) if successful else 0,
            'avg_duration_seconds': np.mean([r.duration_seconds for r in successful]) if successful else 0
        }


if __name__ == '__main__':
    # Test smart order routing
    
    async def test():
        router = SmartOrderRouter()
        
        print("=" * 60)
        print("TESTING TWAP EXECUTION")
        print("=" * 60)
        result = await router.execute_twap(
            symbol='BTCUSDT',
            total_size=1.0,
            duration_minutes=5,
            side='BUY',
            num_slices=5
        )
        print(f"\n✓ TWAP completed: {result.filled_size} BTC @ ${result.average_price:.2f}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        
        print("\n" + "=" * 60)
        print("TESTING VWAP EXECUTION")
        print("=" * 60)
        # Simulate hourly volume profile (higher volume midday)
        volume_profile = [10, 20, 50, 80, 100, 80, 50, 20, 10]
        result = await router.execute_vwap(
            symbol='BTCUSDT',
            total_size=1.0,
            volume_profile=volume_profile,
            duration_minutes=3,
            side='BUY'
        )
        print(f"\n✓ VWAP completed: {result.filled_size} BTC @ ${result.average_price:.2f}")
        
        print("\n" + "=" * 60)
        print("TESTING ICEBERG EXECUTION")
        print("=" * 60)
        result = await router.execute_iceberg(
            symbol='BTCUSDT',
            total_size=2.0,
            visible_size=0.2,
            side='BUY'
        )
        print(f"\n✓ Iceberg completed: {result.filled_size} BTC in {result.slices_executed} slices")
        
        # Print stats
        print("\n" + "=" * 60)
        stats = router.get_execution_stats()
        print(f"Execution Stats: {stats['successful']}/{stats['total_executions']} successful")
    
    asyncio.run(test())
