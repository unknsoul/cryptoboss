"""
Position Reconciliation System
Ensures internal positions match exchange positions
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from core.monitoring.logger import get_logger
from core.monitoring.alerting import get_alerts


logger = get_logger()
alerts = get_alerts()


class PositionDiscrepancy:
    """Represents a position discrepancy"""
    
    def __init__(self, symbol: str, internal_qty: float, exchange_qty: float, 
                 price: float, timestamp: datetime):
        self.symbol = symbol
        self.internal_qty = internal_qty
        self.exchange_qty = exchange_qty
        self.difference = exchange_qty - internal_qty
        self.price = price
        self.timestamp = timestamp
        self.notional_difference = abs(self.difference * price)
    
    def __str__(self) -> str:
        return (f"Discrepancy in {self.symbol}: "
                f"Internal={self.internal_qty}, Exchange={self.exchange_qty}, "
                f"Diff={self.difference:.4f}")


class PositionReconciler:
    """
    Reconciles internal position tracking with exchange positions
    Features:
    - Periodic position sync
    - Discrepancy detection and alerting
    - Auto-correction (optional)
    - Phantom position detection
    """
    
    def __init__(self, exchange, risk_manager, auto_correct: bool = False,
                 tolerance: float = 0.001):
        """
        Args:
            exchange: Exchange client instance
            risk_manager: Risk manager instance
            auto_correct: Whether to automatically correct discrepancies
            tolerance: Tolerance for floating point comparison (e.g., 0.001 BTC)
        """
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.auto_correct = auto_correct
        self.tolerance = tolerance
        
        self.last_reconciliation = None
        self.discrepancies_history: List[PositionDiscrepancy] = []
        self.reconciliation_count = 0
        
        logger.info(
            "Position Reconciler initialized",
            auto_correct=auto_correct,
            tolerance=tolerance
        )
    
    def reconcile(self) -> Dict[str, Any]:
        """
        Perform position reconciliation
        
        Returns:
            Dictionary with reconciliation results
        """
        start_time = time.time()
        self.reconciliation_count += 1
        
        logger.info("Starting position reconciliation", count=self.reconciliation_count)
        
        try:
            # Get positions from both sources
            exchange_positions = self._get_exchange_positions()
            internal_positions = self._get_internal_positions()
            
            # Find discrepancies
            discrepancies = self._find_discrepancies(exchange_positions, internal_positions)
            
            # Handle discrepancies
            if discrepancies:
                logger.warning(
                    f"Found {len(discrepancies)} position discrepancies",
                    count=len(discrepancies)
                )
                
                for disc in discrepancies:
                    self._handle_discrepancy(disc)
                
                # Send alert for significant discrepancies
                significant = [d for d in discrepancies if d.notional_difference > 100]
                if significant:
                    alerts.send_alert(
                        "position_discrepancy",
                        f"Found {len(significant)} significant position discrepancies",
                        {
                            "total_discrepancies": len(discrepancies),
                            "significant_discrepancies": len(significant),
                            "symbols": [d.symbol for d in significant],
                            "total_notional": sum(d.notional_difference for d in significant)
                        }
                    )
            
            self.last_reconciliation = datetime.now()
            duration = time.time() - start_time
            
            result = {
                "timestamp": self.last_reconciliation.isoformat(),
                "duration_ms": round(duration * 1000, 2),
                "total_positions_checked": len(set(list(exchange_positions.keys()) + 
                                                   list(internal_positions.keys()))),
                "discrepancies_found": len(discrepancies),
                "auto_corrected": sum(1 for d in discrepancies if self.auto_correct),
                "success": True
            }
            
            logger.info("Position reconciliation complete", **result)
            return result
            
        except Exception as e:
            logger.error(
                "Position reconciliation failed",
                error=str(e),
                error_type=type(e).__name__
            )
            
            alerts.send_alert(
                "reconciliation_failed",
                f"Position reconciliation failed: {str(e)}",
                {"error": str(e)}
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            }
    
    def _get_exchange_positions(self) -> Dict[str, float]:
        """Get current positions from exchange"""
        try:
            # This depends on your exchange client implementation
            # For now, return empty dict - should be implemented based on exchange API
            # Real implementation would call: self.exchange.get_positions()
            
            positions = {}
            logger.debug("Retrieved exchange positions", count=len(positions))
            return positions
            
        except Exception as e:
            logger.error("Failed to get exchange positions", error=str(e))
            raise
    
    def _get_internal_positions(self) -> Dict[str, float]:
        """Get positions from internal risk manager"""
        positions = {}
        
        for symbol, pos_data in self.risk_manager.active_positions.items():
            qty = pos_data.get('size', 0)
            if abs(qty) > self.tolerance:
                positions[symbol] = qty
        
        logger.debug("Retrieved internal positions", count=len(positions))
        return positions
    
    def _find_discrepancies(self, exchange_positions: Dict[str, float],
                           internal_positions: Dict[str, float]) -> List[PositionDiscrepancy]:
        """Find discrepancies between exchange and internal positions"""
        discrepancies = []
        
        # Get all symbols
        all_symbols = set(list(exchange_positions.keys()) + list(internal_positions.keys()))
        
        for symbol in all_symbols:
            exchange_qty = exchange_positions.get(symbol, 0.0)
            internal_qty = internal_positions.get(symbol, 0.0)
            
            # Check if difference exceeds tolerance
            if abs(exchange_qty - internal_qty) > self.tolerance:
                # Get current price (simplified - should come from exchange)
                price = 1.0  # Placeholder - implement actual price fetch
                
                disc = PositionDiscrepancy(
                    symbol=symbol,
                    internal_qty=internal_qty,
                    exchange_qty=exchange_qty,
                    price=price,
                    timestamp=datetime.now()
                )
                
                discrepancies.append(disc)
                self.discrepancies_history.append(disc)
                
                logger.warning(
                    f"Position discrepancy detected: {disc}",
                    symbol=symbol,
                    internal_qty=internal_qty,
                    exchange_qty=exchange_qty,
                    difference=disc.difference
                )
        
        return discrepancies
    
    def _handle_discrepancy(self, discrepancy: PositionDiscrepancy):
        """Handle a position discrepancy"""
        
        if self.auto_correct:
            logger.info(
                f"Auto-correcting position for {discrepancy.symbol}",
                from_qty=discrepancy.internal_qty,
                to_qty=discrepancy.exchange_qty
            )
            
            # Update internal position to match exchange
            if discrepancy.symbol in self.risk_manager.active_positions:
                self.risk_manager.active_positions[discrepancy.symbol]['size'] = discrepancy.exchange_qty
            elif abs(discrepancy.exchange_qty) > self.tolerance:
                # Phantom position on exchange
                self.risk_manager.active_positions[discrepancy.symbol] = {
                    'size': discrepancy.exchange_qty,
                    'entry_price': discrepancy.price,
                    'timestamp': datetime.now()
                }
        else:
            logger.warning(
                f"Manual intervention required for {discrepancy.symbol}",
                discrepancy=str(discrepancy)
            )
            
            # Alert for manual intervention
            if discrepancy.notional_difference > 50:  # $50 threshold
                alerts.send_alert(
                    "position_stuck",
                    f"Position mismatch detected for {discrepancy.symbol}",
                    {
                        "symbol": discrepancy.symbol,
                        "internal_qty": discrepancy.internal_qty,
                        "exchange_qty": discrepancy.exchange_qty,
                        "difference": discrepancy.difference,
                        "notional": discrepancy.notional_difference
                    }
                )
    
    def get_reconciliation_stats(self) -> Dict[str, Any]:
        """Get reconciliation statistics"""
        recent_discrepancies = [
            d for d in self.discrepancies_history
            if d.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "total_reconciliations": self.reconciliation_count,
            "last_reconciliation": self.last_reconciliation.isoformat() if self.last_reconciliation else None,
            "total_discrepancies_all_time": len(self.discrepancies_history),
            "discrepancies_last_24h": len(recent_discrepancies),
            "auto_correct_enabled": self.auto_correct,
            "tolerance": self.tolerance
        }
    
    def should_reconcile(self, interval_seconds: int = 300) -> bool:
        """
        Check if it's time to reconcile
        
        Args:
            interval_seconds: Reconciliation interval (default: 5 minutes)
        
        Returns:
            True if reconciliation should be performed
        """
        if self.last_reconciliation is None:
            return True
        
        time_since_last = datetime.now() - self.last_reconciliation
        return time_since_last.total_seconds() >= interval_seconds


if __name__ == "__main__":
    # Test position reconciler
    from core.risk.risk_manager import RiskManager
    
    # Mock exchange
    class MockExchange:
        def get_positions(self):
            return {"BTCUSDT": 0.5, "ETHUSDT": 2.0}
    
    # Create instances
    exchange = MockExchange()
    risk_manager = RiskManager(capital=10000)
    
    # Add some internal positions (different from exchange)
    risk_manager.active_positions = {
        "BTCUSDT": {"size": 0.6, "entry_price": 45000},
        "ETHUSDT": {"size": 2.0, "entry_price": 3000}
    }
    
    # Create reconciler
    reconciler = PositionReconciler(
        exchange=exchange,
        risk_manager=risk_manager,
        auto_correct=True
    )
    
    # Run reconciliation
    result = reconciler.reconcile()
    print("\n" + "=" * 60)
    print("Position Reconciliation Test")
    print("=" * 60)
    print(f"Result: {result}")
    print(f"Stats: {reconciler.get_reconciliation_stats()}")
