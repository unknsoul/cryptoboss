"""
Smart Execution Engine
Advanced order execution algorithms to minimize market impact and slippage

Features:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Iceberg Orders (hidden quantity)
- Adaptive Execution
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np

from core.monitoring.logger import get_logger
from core.monitoring.metrics import get_metrics

logger = get_logger()
metrics = get_metrics()


class ExecutionAlgorithm(Enum):
    """Available execution algorithms"""
    MARKET = "market"           # Immediate market order
    TWAP = "twap"               # Time-Weighted Average Price
    VWAP = "vwap"               # Volume-Weighted Average Price
    ICEBERG = "iceberg"         # Hidden quantity orders
    ADAPTIVE = "adaptive"       # Smart adaptive execution


@dataclass
class ExecutionOrder:
    """Represents an order to be executed"""
    symbol: str
    side: str               # 'BUY' or 'SELL'
    total_quantity: float
    algorithm: ExecutionAlgorithm
    limit_price: Optional[float] = None
    duration_seconds: int = 300      # Default 5 minutes
    max_slippage_pct: float = 0.5    # Max acceptable slippage
    urgency: float = 0.5             # 0 = patient, 1 = aggressive


@dataclass
class ExecutionSlice:
    """A single slice of a larger execution"""
    quantity: float
    target_time: datetime
    executed: bool = False
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None


class SmartOrderRouter:
    """
    Smart Order Router with Execution Algorithms
    
    Breaks down large orders into smaller pieces to:
    - Minimize market impact
    - Achieve better average price
    - Reduce slippage
    """
    
    def __init__(self, 
                 exchange_client=None,
                 default_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.TWAP):
        """
        Args:
            exchange_client: Exchange API client for order execution
            default_algorithm: Default execution algorithm
        """
        self.exchange = exchange_client
        self.default_algorithm = default_algorithm
        
        # Active executions
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
        # Execution history
        self.execution_history: List[Dict[str, Any]] = []
        
        # Volume profile for VWAP (hourly, normalized)
        # This is a simplified profile - in production, load historical data
        self.volume_profile = self._generate_default_volume_profile()
        
        logger.info(
            "SmartOrderRouter initialized",
            default_algorithm=default_algorithm.value
        )
    
    def _generate_default_volume_profile(self) -> np.ndarray:
        """
        Generate default intraday volume profile.
        Real implementation should use historical volume data.
        
        Returns:
            24-hour normalized volume profile
        """
        # Typical crypto volume profile (higher at market opens/closes)
        profile = np.array([
            0.03, 0.02, 0.02, 0.02, 0.02, 0.03,  # 00:00 - 05:00
            0.04, 0.05, 0.06, 0.05, 0.04, 0.04,  # 06:00 - 11:00
            0.04, 0.05, 0.06, 0.07, 0.06, 0.05,  # 12:00 - 17:00
            0.05, 0.05, 0.05, 0.04, 0.03, 0.03   # 18:00 - 23:00
        ])
        
        # Normalize to sum to 1
        return profile / profile.sum()
    
    def execute_twap(self, 
                     order: ExecutionOrder,
                     on_slice_complete: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute order using TWAP (Time-Weighted Average Price)
        
        Slices the order into equal parts over the duration.
        
        Args:
            order: ExecutionOrder to execute
            on_slice_complete: Optional callback when each slice executes
        
        Returns:
            Execution result with average fill price
        """
        execution_id = f"TWAP_{order.symbol}_{int(time.time())}"
        
        logger.info(
            f"🕐 Starting TWAP execution",
            execution_id=execution_id,
            symbol=order.symbol,
            quantity=order.total_quantity,
            duration_seconds=order.duration_seconds
        )
        
        # Calculate number of slices (minimum 1 every 30 seconds)
        num_slices = max(2, order.duration_seconds // 30)
        slice_quantity = order.total_quantity / num_slices
        slice_interval = order.duration_seconds / num_slices
        
        # Create slices
        slices: List[ExecutionSlice] = []
        start_time = datetime.now()
        
        for i in range(num_slices):
            target_time = start_time + timedelta(seconds=i * slice_interval)
            slices.append(ExecutionSlice(
                quantity=slice_quantity,
                target_time=target_time
            ))
        
        # Track execution
        self.active_executions[execution_id] = {
            'order': order,
            'slices': slices,
            'start_time': start_time,
            'algorithm': 'TWAP',
            'status': 'active'
        }
        
        # Execute slices (synchronous simulation)
        fill_prices = []
        
        for i, slice_order in enumerate(slices):
            # Simulate execution (in production, this would submit real orders)
            fill_price = self._simulate_execution(
                order.symbol, 
                order.side, 
                slice_order.quantity,
                order.limit_price
            )
            
            slice_order.executed = True
            slice_order.fill_price = fill_price
            slice_order.fill_time = datetime.now()
            fill_prices.append(fill_price)
            
            logger.info(
                f"  📊 TWAP Slice {i+1}/{num_slices}: {slice_order.quantity:.4f} @ ${fill_price:.2f}"
            )
            
            if on_slice_complete:
                on_slice_complete(slice_order)
            
            metrics.increment("twap_slice_executed")
        
        # Calculate results
        avg_fill_price = np.mean(fill_prices)
        
        result = {
            'execution_id': execution_id,
            'algorithm': 'TWAP',
            'symbol': order.symbol,
            'side': order.side,
            'total_quantity': order.total_quantity,
            'avg_fill_price': avg_fill_price,
            'num_slices': num_slices,
            'duration_seconds': order.duration_seconds,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        # Update tracking
        self.active_executions[execution_id]['status'] = 'completed'
        self.active_executions[execution_id]['result'] = result
        self.execution_history.append(result)
        
        logger.info(
            f"✅ TWAP execution complete",
            avg_price=avg_fill_price,
            slices=num_slices
        )
        
        metrics.increment("twap_execution_complete")
        
        return result
    
    def execute_vwap(self, 
                     order: ExecutionOrder,
                     on_slice_complete: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute order using VWAP (Volume-Weighted Average Price)
        
        Sizes slices based on historical volume profile.
        
        Args:
            order: ExecutionOrder to execute
            on_slice_complete: Optional callback when each slice executes
        
        Returns:
            Execution result with average fill price
        """
        execution_id = f"VWAP_{order.symbol}_{int(time.time())}"
        
        logger.info(
            f"📊 Starting VWAP execution",
            execution_id=execution_id,
            symbol=order.symbol,
            quantity=order.total_quantity
        )
        
        # Get current hour and relevant volume profile segment
        current_hour = datetime.now().hour
        hours_to_execute = min(order.duration_seconds // 3600, 4) + 1
        
        # Get volume weights for execution period
        end_hour = (current_hour + hours_to_execute) % 24
        if end_hour > current_hour:
            volume_weights = self.volume_profile[current_hour:end_hour]
        else:
            volume_weights = np.concatenate([
                self.volume_profile[current_hour:],
                self.volume_profile[:end_hour]
            ])
        
        # Normalize weights
        if len(volume_weights) == 0:
            volume_weights = np.array([1.0])
        volume_weights = volume_weights / volume_weights.sum()
        
        # Create slices based on volume profile
        slices: List[ExecutionSlice] = []
        start_time = datetime.now()
        
        for i, weight in enumerate(volume_weights):
            slice_quantity = order.total_quantity * weight
            target_time = start_time + timedelta(hours=i)
            
            slices.append(ExecutionSlice(
                quantity=slice_quantity,
                target_time=target_time
            ))
        
        # Track execution
        self.active_executions[execution_id] = {
            'order': order,
            'slices': slices,
            'start_time': start_time,
            'algorithm': 'VWAP',
            'status': 'active'
        }
        
        # Execute slices
        fill_prices = []
        fill_quantities = []
        
        for i, slice_order in enumerate(slices):
            fill_price = self._simulate_execution(
                order.symbol,
                order.side,
                slice_order.quantity,
                order.limit_price
            )
            
            slice_order.executed = True
            slice_order.fill_price = fill_price
            slice_order.fill_time = datetime.now()
            
            fill_prices.append(fill_price)
            fill_quantities.append(slice_order.quantity)
            
            logger.info(
                f"  📊 VWAP Slice {i+1}/{len(slices)}: "
                f"{slice_order.quantity:.4f} ({volume_weights[i]*100:.1f}% of volume) @ ${fill_price:.2f}"
            )
            
            if on_slice_complete:
                on_slice_complete(slice_order)
            
            metrics.increment("vwap_slice_executed")
        
        # Calculate volume-weighted average price
        avg_fill_price = np.average(fill_prices, weights=fill_quantities)
        
        result = {
            'execution_id': execution_id,
            'algorithm': 'VWAP',
            'symbol': order.symbol,
            'side': order.side,
            'total_quantity': order.total_quantity,
            'avg_fill_price': avg_fill_price,
            'num_slices': len(slices),
            'volume_profile_used': volume_weights.tolist(),
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        self.active_executions[execution_id]['status'] = 'completed'
        self.active_executions[execution_id]['result'] = result
        self.execution_history.append(result)
        
        logger.info(
            f"✅ VWAP execution complete",
            avg_price=avg_fill_price
        )
        
        metrics.increment("vwap_execution_complete")
        
        return result
    
    def execute_iceberg(self, 
                        order: ExecutionOrder,
                        visible_quantity: float = None) -> Dict[str, Any]:
        """
        Execute iceberg order (hidden quantity)
        
        Only shows a portion of the order to the market at a time.
        
        Args:
            order: ExecutionOrder to execute
            visible_quantity: Amount to show at a time (default: 10% of total)
        
        Returns:
            Execution result
        """
        execution_id = f"ICEBERG_{order.symbol}_{int(time.time())}"
        
        if visible_quantity is None:
            visible_quantity = order.total_quantity * 0.1
        
        logger.info(
            f"🧊 Starting Iceberg execution",
            execution_id=execution_id,
            total_quantity=order.total_quantity,
            visible_quantity=visible_quantity
        )
        
        remaining_quantity = order.total_quantity
        fill_prices = []
        fill_quantities = []
        slice_count = 0
        
        while remaining_quantity > 0:
            slice_qty = min(visible_quantity, remaining_quantity)
            
            fill_price = self._simulate_execution(
                order.symbol,
                order.side,
                slice_qty,
                order.limit_price
            )
            
            fill_prices.append(fill_price)
            fill_quantities.append(slice_qty)
            remaining_quantity -= slice_qty
            slice_count += 1
            
            logger.info(
                f"  🧊 Iceberg Slice {slice_count}: "
                f"{slice_qty:.4f} @ ${fill_price:.2f} (hidden: {remaining_quantity:.4f})"
            )
            
            metrics.increment("iceberg_slice_executed")
        
        avg_fill_price = np.average(fill_prices, weights=fill_quantities)
        
        result = {
            'execution_id': execution_id,
            'algorithm': 'ICEBERG',
            'symbol': order.symbol,
            'side': order.side,
            'total_quantity': order.total_quantity,
            'visible_quantity': visible_quantity,
            'avg_fill_price': avg_fill_price,
            'num_slices': slice_count,
            'status': 'completed'
        }
        
        self.execution_history.append(result)
        
        logger.info(f"✅ Iceberg execution complete", avg_price=avg_fill_price)
        
        return result
    
    def execute(self, order: ExecutionOrder) -> Dict[str, Any]:
        """
        Execute order using specified algorithm
        
        Args:
            order: ExecutionOrder to execute
        
        Returns:
            Execution result
        """
        if order.algorithm == ExecutionAlgorithm.TWAP:
            return self.execute_twap(order)
        elif order.algorithm == ExecutionAlgorithm.VWAP:
            return self.execute_vwap(order)
        elif order.algorithm == ExecutionAlgorithm.ICEBERG:
            return self.execute_iceberg(order)
        elif order.algorithm == ExecutionAlgorithm.ADAPTIVE:
            # Choose algorithm based on order characteristics
            if order.urgency > 0.7:
                return self.execute_twap(order)  # Faster
            else:
                return self.execute_vwap(order)  # Better price
        else:
            # Market order - immediate execution
            fill_price = self._simulate_execution(
                order.symbol,
                order.side,
                order.total_quantity,
                order.limit_price
            )
            
            return {
                'execution_id': f"MARKET_{order.symbol}_{int(time.time())}",
                'algorithm': 'MARKET',
                'symbol': order.symbol,
                'side': order.side,
                'total_quantity': order.total_quantity,
                'avg_fill_price': fill_price,
                'status': 'completed'
            }
    
    def _simulate_execution(self, 
                           symbol: str, 
                           side: str, 
                           quantity: float,
                           limit_price: Optional[float] = None) -> float:
        """
        Simulate order execution with slippage
        
        In production, this would submit actual orders via exchange API.
        
        Returns:
            Simulated fill price
        """
        # Use limit price or generate random price for simulation
        if limit_price:
            base_price = limit_price
        else:
            # Simulate BTC price for demo
            base_price = 45000 + np.random.randn() * 500
        
        # Add slippage (0.01% to 0.1%)
        slippage = np.random.uniform(0.0001, 0.001)
        
        if side == 'BUY':
            fill_price = base_price * (1 + slippage)
        else:
            fill_price = base_price * (1 - slippage)
        
        return round(fill_price, 2)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {'total_executions': 0}
        
        return {
            'total_executions': len(self.execution_history),
            'by_algorithm': {
                algo.value: sum(1 for e in self.execution_history 
                               if e.get('algorithm') == algo.value)
                for algo in ExecutionAlgorithm
            },
            'active_executions': len([e for e in self.active_executions.values() 
                                      if e.get('status') == 'active'])
        }


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 SMART EXECUTION ENGINE TEST")
    print("=" * 70)
    
    router = SmartOrderRouter()
    
    # Test TWAP
    print("\n1. Testing TWAP Execution...")
    twap_order = ExecutionOrder(
        symbol="BTCUSDT",
        side="BUY",
        total_quantity=1.0,
        algorithm=ExecutionAlgorithm.TWAP,
        duration_seconds=60,
        limit_price=45000
    )
    
    twap_result = router.execute_twap(twap_order)
    print(f"   TWAP Result: Avg Price = ${twap_result['avg_fill_price']:.2f}")
    
    # Test VWAP
    print("\n2. Testing VWAP Execution...")
    vwap_order = ExecutionOrder(
        symbol="BTCUSDT",
        side="SELL",
        total_quantity=0.5,
        algorithm=ExecutionAlgorithm.VWAP,
        duration_seconds=3600,
        limit_price=45000
    )
    
    vwap_result = router.execute_vwap(vwap_order)
    print(f"   VWAP Result: Avg Price = ${vwap_result['avg_fill_price']:.2f}")
    
    # Test Iceberg
    print("\n3. Testing Iceberg Execution...")
    iceberg_order = ExecutionOrder(
        symbol="BTCUSDT",
        side="BUY",
        total_quantity=2.0,
        algorithm=ExecutionAlgorithm.ICEBERG,
        limit_price=45000
    )
    
    iceberg_result = router.execute_iceberg(iceberg_order, visible_quantity=0.2)
    print(f"   Iceberg Result: Avg Price = ${iceberg_result['avg_fill_price']:.2f}")
    
    # Stats
    print("\n" + "=" * 70)
    print("Execution Statistics:")
    print(router.get_execution_stats())
    print("=" * 70)
