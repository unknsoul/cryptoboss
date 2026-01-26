"""
Tax Integration with StateManager - Upgrade M

Automatic tax tracking from trade executions.
"""

from typing import Dict, Optional
from datetime import datetime
import logging

from .state_manager import get_state_manager
from ..tax.tax_reporter import TaxReporter, CostBasisMethod

logger = logging.getLogger(__name__)


class IntegratedTaxTracker:
    """
    Tax tracking integrated with StateManager.
    
    Automatically records trades for tax purposes when orders are filled.
    
    Usage:
        tracker = IntegratedTaxTracker()
        
        # Called automatically by ExecutionRouter on fill
        tracker.record_trade_from_result(order_result)
        
        # Generate tax report
        report = tracker.generate_report(tax_year=2024)
    """
    
    def __init__(
        self,
        cost_basis_method: str = "fifo",
        tax_jurisdiction: str = "US"
    ):
        self.tax_reporter = TaxReporter(
            cost_basis_method=cost_basis_method,
            tax_jurisdiction=tax_jurisdiction
        )
        self.state_manager = get_state_manager()
        
        # Track last processed order to avoid duplicates
        self._processed_orders: set = set()
        
        logger.info(f"IntegratedTaxTracker initialized: {cost_basis_method}, {tax_jurisdiction}")
    
    def record_trade_from_result(self, order_result) -> bool:
        """
        Record a trade from an OrderResult.
        
        Args:
            order_result: OrderResult from ExecutionRouter
        
        Returns:
            True if recorded successfully
        """
        if not order_result.success:
            return False
        
        if order_result.order_id in self._processed_orders:
            return False
        
        try:
            timestamp = order_result.timestamp
            symbol = order_result.symbol
            side = order_result.side
            quantity = order_result.filled_quantity
            price = order_result.average_price
            fees = order_result.fees
            
            # Extract base currency from symbol
            base = symbol.split("/")[0] if "/" in symbol else symbol.replace("USDT", "")
            
            if side == "buy":
                self.tax_reporter.record_purchase(
                    symbol=base,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    fees=fees
                )
            else:  # sell
                self.tax_reporter.record_sale(
                    symbol=base,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    fees=fees
                )
            
            self._processed_orders.add(order_result.order_id)
            
            # Persist to state
            self.state_manager.save_order(order_result.order_id, {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "fees": fees,
                "status": "filled",
                "timestamp": timestamp.isoformat(),
                "tax_recorded": True
            })
            
            logger.debug(f"Tax recorded: {side} {quantity} {base} @ ${price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record trade for tax: {e}")
            return False
    
    def sync_from_state(self):
        """
        Sync tax records from StateManager.
        
        Useful after restart to rebuild tax state.
        """
        # Load all filled orders from state
        orders = self.state_manager.load_pending_orders()  # Would need method for all orders
        
        for order in orders:
            if order.get('status') == 'filled' and not order.get('tax_recorded'):
                # Create minimal result object
                class MinimalResult:
                    pass
                
                result = MinimalResult()
                result.success = True
                result.order_id = order.get('order_id', '')
                result.symbol = order.get('symbol', '')
                result.side = order.get('side', '')
                result.filled_quantity = order.get('quantity', 0)
                result.average_price = order.get('price', 0)
                result.fees = order.get('fees', 0)
                result.timestamp = datetime.fromisoformat(order.get('timestamp', datetime.now().isoformat()))
                
                self.record_trade_from_result(result)
    
    def calculate_tax_liability(self, tax_year: int) -> Dict:
        """Calculate tax liability for a year."""
        return self.tax_reporter.calculate_tax_liability(tax_year)
    
    def find_harvesting_opportunities(self, current_prices: Dict[str, float]) -> list:
        """Find tax-loss harvesting opportunities."""
        return self.tax_reporter.find_tax_loss_harvesting_opportunities(current_prices)
    
    def generate_report(self, tax_year: int) -> str:
        """Generate human-readable tax report."""
        return self.tax_reporter.get_summary_report(tax_year)
    
    def export_csv(self, tax_year: int, filename: str = None):
        """Export transactions for tax software."""
        if filename is None:
            filename = f"tax_report_{tax_year}.csv"
        self.tax_reporter.export_for_turbotax(tax_year, filename)
        logger.info(f"Exported tax report to {filename}")


# Singleton
_tax_tracker: Optional[IntegratedTaxTracker] = None

def get_tax_tracker() -> IntegratedTaxTracker:
    global _tax_tracker
    if _tax_tracker is None:
        _tax_tracker = IntegratedTaxTracker()
    return _tax_tracker
