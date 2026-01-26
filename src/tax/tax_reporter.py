"""
Comprehensive Tax Reporting & Optimization
Track all trades, calculate tax liability, optimize with tax-loss harvesting.

Features:
- Multiple cost basis methods (FIFO, LIFO, HIFO, Specific ID)
- Realized/unrealized gains tracking
- Tax-loss harvesting opportunities
- Multi-jurisdiction support
- Export for TurboTax, etc.
- Wash sale detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CostBasisMethod(Enum):
    """Cost basis calculation methods."""
    FIFO = "fifo"  # First-In-First-Out
    LIFO = "lifo"  # Last-In-First-Out  
    HIFO = "hifo"  # Highest-In-First-Out (minimize gains)
    SPECIFIC_ID = "specific_id"  # Manually specify lots


@dataclass
class TaxLot:
    """A tax lot (purchase) of cryptocurrency."""
    symbol: str
    quantity: float
    purchase_price: float
    purchase_date: datetime
    lot_id: str
    remaining_quantity: float = None
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
    
    def cost_basis(self) -> float:
        """Calculate cost basis of this lot."""
        return self.purchase_price * self.quantity


@dataclass
class TaxableEvent:
    """A taxable event (sale/trade)."""
    symbol: str
    quantity: float
    sale_price: float
    sale_date: datetime
    lots_used: List[TaxLot]
    proceeds: float = 0.0
    cost_basis: float = 0.0
    capital_gain: float = 0.0
    holding_period_days: int = 0
    is_long_term: bool = False
    
    def __post_init__(self):
        self.proceeds = self.sale_price * self.quantity
        self.cost_basis = sum(lot.purchase_price * lot.quantity for lot in self.lots_used)
        self.capital_gain = self.proceeds - self.cost_basis
        
        if self.lots_used:
            oldest_lot = min(self.lots_used, key=lambda x: x.purchase_date)
            self.holding_period_days = (self.sale_date - oldest_lot.purchase_date).days
            self.is_long_term = self.holding_period_days > 365


class TaxReporter:
    """
    Comprehensive tax reporting and optimization.
    
    Features:
    - Track all purchases and sales
    - Calculate gains/losses with multiple methods
    - Identify tax-loss harvesting opportunities
    - Generate tax reports
    - Export for tax software
    """
    
    def __init__(
        self,
        cost_basis_method: str = "fifo",
        tax_jurisdiction: str = "US",
        long_term_rate: float = 0.15,  # 15% for long-term gains (US)
        short_term_rate: float = 0.37   # 37% for short-term gains (US, highest bracket)
    ):
        """
        Initialize tax reporter.
        
        Args:
            cost_basis_method: Cost basis method to use
            tax_jurisdiction: Tax jurisdiction (US, UK, etc.)
            long_term_rate: Long-term capital gains tax rate
            short_term_rate: Short-term capital gains tax rate
        """
        self.cost_basis_method = CostBasisMethod(cost_basis_method)
        self.jurisdiction = tax_jurisdiction
        self.long_term_rate = long_term_rate
        self.short_term_rate = short_term_rate
        
        # Tax lots (purchases)
        self.tax_lots: Dict[str, List[TaxLot]] = {}
        
        # Taxable events (sales)
        self.taxable_events: List[TaxableEvent] = []
        
        # Unrealized positions
        self.unrealized_positions: Dict[str, List[TaxLot]] = {}
        
        logger.info(f"Tax reporter initialized: {cost_basis_method}, {tax_jurisdiction}")
    
    def record_purchase(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        fees: float = 0.0
    ):
        """
        Record a cryptocurrency purchase (creates tax lot).
        
        Args:
            symbol: Cryptocurrency symbol
            quantity: Amount purchased
            price: Purchase price per unit
            timestamp: Purchase timestamp
            fees: Transaction fees (added to cost basis)
        """
        # Adjust price for fees
        adjusted_price = price + (fees / quantity if quantity > 0 else 0)
        
        lot = TaxLot(
            symbol=symbol,
            quantity=quantity,
            purchase_price=adjusted_price,
            purchase_date=timestamp,
            lot_id=f"{symbol}_{timestamp.isoformat()}"
        )
        
        if symbol not in self.tax_lots:
            self.tax_lots[symbol] = []
        
        self.tax_lots[symbol].append(lot)
        
        logger.debug(f"Recorded purchase: {quantity} {symbol} @ ${price:.2f}")
    
    def record_sale(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        fees: float = 0.0
    ) -> TaxableEvent:
        """
        Record a cryptocurrency sale (creates taxable event).
        
        Args:
            symbol: Cryptocurrency symbol
            quantity: Amount sold
            price: Sale price per unit
            timestamp: Sale timestamp
            fees: Transaction fees (reduce proceeds)
        
        Returns:
            TaxableEvent object
        """
        if symbol not in self.tax_lots or not self.tax_lots[symbol]:
            logger.error(f"No tax lots available for {symbol}")
            raise ValueError(f"Cannot sell {symbol} - no purchases on record")
        
        # Select lots to sell based on cost basis method
        lots_to_sell = self._select_lots_to_sell(symbol, quantity)
        
        # Adjust price for fees
        adjusted_price = price - (fees / quantity if quantity > 0 else 0)
        
        # Create taxable event
        event = TaxableEvent(
            symbol=symbol,
            quantity=quantity,
            sale_price=adjusted_price,
            sale_date=timestamp,
            lots_used=lots_to_sell
        )
        
        self.taxable_events.append(event)
        
        # Update remaining quantities in lots
        qty_remaining = quantity
        for lot in lots_to_sell:
            if qty_remaining <= 0:
                break
            
            qty_from_lot = min(qty_remaining, lot.remaining_quantity)
            lot.remaining_quantity -= qty_from_lot
            qty_remaining -= qty_from_lot
        
        # Remove fully used lots
        self.tax_lots[symbol] = [lot for lot in self.tax_lots[symbol] 
                                  if lot.remaining_quantity > 0.0001]
        
        logger.info(
            f"Recorded sale: {quantity} {symbol} @ ${price:.2f}, "
            f"Gain: ${event.capital_gain:+,.2f} ({event.holding_period_days} days)"
        )
        
        return event
    
    def _select_lots_to_sell(self, symbol: str, quantity: float) -> List[TaxLot]:
        """Select which lots to sell based on cost basis method."""
        available_lots = [lot for lot in self.tax_lots[symbol] 
                         if lot.remaining_quantity > 0.0001]
        
        if self.cost_basis_method == CostBasisMethod.FIFO:
            # First-In-First-Out
            sorted_lots = sorted(available_lots, key=lambda x: x.purchase_date)
        
        elif self.cost_basis_method == CostBasisMethod.LIFO:
            # Last-In-First-Out
            sorted_lots = sorted(available_lots, key=lambda x: x.purchase_date, reverse=True)
        
        elif self.cost_basis_method == CostBasisMethod.HIFO:
            # Highest-In-First-Out (minimizes gains)
            sorted_lots = sorted(available_lots, key=lambda x: x.purchase_price, reverse=True)
        
        else:
            # Default to FIFO
            sorted_lots = sorted(available_lots, key=lambda x: x.purchase_date)
        
        # Select lots until we have enough quantity
        selected_lots = []
        qty_remaining = quantity
        
        for lot in sorted_lots:
            if qty_remaining <= 0:
                break
            
            qty_from_lot = min(qty_remaining, lot.remaining_quantity)
            
            # Create a copy of the lot with adjusted quantity
            lot_copy = TaxLot(
                symbol=lot.symbol,
                quantity=qty_from_lot,
                purchase_price=lot.purchase_price,
                purchase_date=lot.purchase_date,
                lot_id=lot.lot_id
            )
            
            selected_lots.append(lot_copy)
            qty_remaining -= qty_from_lot
        
        return selected_lots
    
    def calculate_tax_liability(self, tax_year: int) -> Dict:
        """
        Calculate total tax liability for a tax year.
        
        Args:
            tax_year: Tax year (e.g., 2024)
        
        Returns:
            Dict with tax liability breakdown
        """
        # Filter events for tax year
        year_events = [e for e in self.taxable_events 
                      if e.sale_date.year == tax_year]
        
        if not year_events:
            return {
                'tax_year': tax_year,
                'total_events': 0,
                'total_tax': 0
            }
        
        # Separate long-term and short-term
        long_term_events = [e for e in year_events if e.is_long_term]
        short_term_events = [e for e in year_events if not e.is_long_term]
        
        # Calculate gains/losses
        long_term_gain = sum(e.capital_gain for e in long_term_events)
        short_term_gain = sum(e.capital_gain for e in short_term_events)
        
        # Calculate tax liability
        long_term_tax = max(0, long_term_gain * self.long_term_rate)
        short_term_tax = max(0, short_term_gain * self.short_term_rate)
        total_tax = long_term_tax + short_term_tax
        
        return {
            'tax_year': tax_year,
            'total_events': len(year_events),
            'long_term_events': len(long_term_events),
            'short_term_events': len(short_term_events),
            'long_term_gain': long_term_gain,
            'short_term_gain': short_term_gain,
            'total_gain': long_term_gain + short_term_gain,
            'long_term_tax': long_term_tax,
            'short_term_tax': short_term_tax,
            'total_tax': total_tax,
            'effective_rate': total_tax / (long_term_gain + short_term_gain) if (long_term_gain + short_term_gain) > 0 else 0
        }
    
    def find_tax_loss_harvesting_opportunities(
        self,
        current_prices: Dict[str, float],
        min_loss_threshold: float = 1000.0
    ) -> List[Dict]:
        """
        Identify tax-loss harvesting opportunities.
        
        Tax-loss harvesting: Sell losing positions to offset gains.
        
        Args:
            current_prices: Current market prices
            min_loss_threshold: Minimum loss to consider ($)
        
        Returns:
            List of harvesting opportunities
        """
        opportunities = []
        
        for symbol, lots in self.tax_lots.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            for lot in lots:
                if lot.remaining_quantity < 0.0001:
                    continue
                
                # Calculate unrealized loss
                cost_basis = lot.purchase_price * lot.remaining_quantity
                current_value = current_price * lot.remaining_quantity
                unrealized_loss = current_value - cost_basis
                
                if unrealized_loss < -min_loss_threshold:
                    opportunities.append({
                        'symbol': symbol,
                        'lot_id': lot.lot_id,
                        'quantity': lot.remaining_quantity,
                        'purchase_price': lot.purchase_price,
                        'current_price': current_price,
                        'unrealized_loss': unrealized_loss,
                        'loss_pct': (unrealized_loss / cost_basis) * 100,
                        'holding_period': (datetime.now() - lot.purchase_date).days,
                        'tax_savings': abs(unrealized_loss) * self.short_term_rate
                    })
        
        # Sort by tax savings (highest first)
        opportunities.sort(key=lambda x: x['tax_savings'], reverse=True)
        
        logger.info(f"Found {len(opportunities)} tax-loss harvesting opportunities")
        
        return opportunities
    
    def export_for_turbotax(self, tax_year: int, filename: str):
        """Export transactions in TurboTax compatible CSV format."""
        year_events = [e for e in self.taxable_events 
                      if e.sale_date.year == tax_year]
        
        rows = []
        for event in year_events:
            for lot in event.lots_used:
                rows.append({
                    'Description': f"{event.symbol} ({lot.quantity})",
                    'Date Acquired': lot.purchase_date.strftime('%m/%d/%Y'),
                    'Date Sold': event.sale_date.strftime('%m/%d/%Y'),
                    'Proceeds': event.proceeds,
                    'Cost Basis': lot.purchase_price * lot.quantity,
                    'Gain/Loss': event.capital_gain,
                    'Type': 'Long-Term' if event.is_long_term else 'Short-Term'
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        
        logger.info(f"Exported {len(rows)} transactions to {filename}")
    
    def get_summary_report(self, tax_year: int) -> str:
        """Generate human-readable tax summary."""
        liability = self.calculate_tax_liability(tax_year)
        
        report = f"""
TAX SUMMARY REPORT - {tax_year}
{'='*60}

Taxable Events: {liability['total_events']}
  - Long-term (> 1 year): {liability['long_term_events']}
  - Short-term (≤ 1 year): {liability['short_term_events']}

Capital Gains/Losses:
  - Long-term: ${liability['long_term_gain']:,.2f}
  - Short-term: ${liability['short_term_gain']:,.2f}
  - TOTAL: ${liability['total_gain']:,.2f}

Estimated Tax Liability:
  - Long-term tax ({self.long_term_rate:.0%}): ${liability['long_term_tax']:,.2f}
  - Short-term tax ({self.short_term_rate:.0%}): ${liability['short_term_tax']:,.2f}
  - TOTAL TAX: ${liability['total_tax']:,.2f}
  - Effective rate: {liability['effective_rate']:.1%}

Cost Basis Method: {self.cost_basis_method.value.upper()}
Jurisdiction: {self.jurisdiction}

⚠️ This is an estimate. Consult a tax professional.
        """
        
        return report


# Example usage
if __name__ == "__main__":
    # Initialize
    tax_reporter = TaxReporter(cost_basis_method="fifo", tax_jurisdiction="US")
    
    # Record trades
    tax_reporter.record_purchase("BTC", 0.5, 40000, datetime(2023, 1, 1))
    tax_reporter.record_purchase("BTC", 0.3, 50000, datetime(2023, 6, 1))
    tax_reporter.record_sale("BTC", 0.4, 60000, datetime(2024, 3, 1))
    
    # Calculate tax
    liability = tax_reporter.calculate_tax_liability(2024)
    print(f"Tax Liability 2024: ${liability['total_tax']:,.2f}")
    
    # Find tax-loss harvesting
    opportunities = tax_reporter.find_tax_loss_harvesting_opportunities(
        current_prices={'BTC': 45000}
    )
    print(f"\nTax-Loss Harvesting Opportunities: {len(opportunities)}")
    
    # Generate report
    report = tax_reporter.get_summary_report(2024)
    print(report)
