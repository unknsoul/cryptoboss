"""
Shadow Mode & Position Reconciliation
Production safety features for testing and validation

Shadow Mode: Run strategies without executing real trades
Position Reconciliation: Verify internal state matches exchange
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ShadowMode:
    """
    Run trading strategies without executing real trades
    
    Perfect for:
    - Testing new strategies
    - Validation before going live
    - Performance comparison
    - Debug mode
    """
    
    def __init__(self, enabled: bool = False):
        """
        Initialize shadow mode
        
        Args:
            enabled: If True, trades are logged but not executed
        """
        self.enabled = enabled
        self.shadow_trades = []
        self.shadow_equity = 10000.0  # Starting virtual equity
        
    def enable(self):
        """Enable shadow mode"""
        self.enabled = True
        logger.info("✓ Shadow mode ENABLED - trades will be simulated only")
    
    def disable(self):
        """Disable shadow mode"""
        self.enabled = False
        logger.info("Shadow mode DISABLED - live trading resumed")
    
    def log_trade(self, trade_signal: Dict) -> Dict:
        """
        Log a trade that would have been executed
        
        Args:
            trade_signal: Trade signal dict
            
        Returns:
            Shadow trade result
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        shadow_trade = {
            'timestamp': datetime.now(),
            'action': trade_signal.get('action'),
            'price': trade_signal.get('price'),
            'size': trade_signal.get('size', 0.1),
            'confidence': trade_signal.get('confidence'),
            'reasons': trade_signal.get('reasons', []),
            'mode': 'SHADOW'
        }
        
        self.shadow_trades.append(shadow_trade)
        
        logger.info(f"[SHADOW] Trade logged: {shadow_trade['action']} @ ${shadow_trade['price']:,.2f}")
        
        return {
            'trade_id': len(self.shadow_trades),
            'status': 'shadow_logged',
            'trade': shadow_trade
        }
    
    def get_statistics(self) -> Dict:
        """Get shadow mode statistics"""
        if not self.shadow_trades:
            return {'status': 'no_shadow_trades'}
        
        return {
            'enabled': self.enabled,
            'total_shadow_trades': len(self.shadow_trades),
            'shadow_equity': round(self.shadow_equity, 2),
            'recent_trades': self.shadow_trades[-5:],
            'status': 'ok'
        }


class PositionReconciliation:
    """
    Verify internal position state matches exchange
    
    Critical for:
    - Detecting desyncs
    - Preventing duplicate orders
    - Ensuring data integrity
    """
    
    def __init__(self, tolerance_pct: float = 0.01):
        """
        Initialize position reconciliation
        
        Args:
            tolerance_pct: Allowed difference % (0.01 = 1%)
        """
        self.tolerance_pct = tolerance_pct
        self.internal_positions = {}
        self.mismatches = []
        
    def update_internal(self, symbol: str, size: float, entry_price: float):
        """Update internal position record"""
        self.internal_positions[symbol] = {
            'size': size,
            'entry_price': entry_price,
            'timestamp': datetime.now()
        }
    
    def clear_internal(self, symbol: str):
        """Clear internal position"""
        if symbol in self.internal_positions:
            del self.internal_positions[symbol]
    
    def reconcile(self, exchange_positions: Dict[str, Dict]) -> Dict:
        """
        Reconcile internal positions with exchange
        
        Args:
            exchange_positions: Dict of {symbol: {'size': float, 'entry_price': float}}
            
        Returns:
            Reconciliation result with mismatches
        """
        mismatches = []
        matched = []
        
        # Check all internal positions
        for symbol, internal in self.internal_positions.items():
            if symbol not in exchange_positions:
                mismatches.append({
                    'symbol': symbol,
                    'type': 'missing_on_exchange',
                    'internal_size': internal['size'],
                    'exchange_size': 0,
                    'severity': 'high'
                })
            else:
                exchange = exchange_positions[symbol]
                size_diff = abs(internal['size'] - exchange['size'])
                size_diff_pct = size_diff / max(abs(internal['size']), 0.001)
                
                if size_diff_pct > self.tolerance_pct:
                    mismatches.append({
                        'symbol': symbol,
                        'type': 'size_mismatch',
                        'internal_size': internal['size'],
                        'exchange_size': exchange['size'],
                        'difference_pct': round(size_diff_pct * 100, 2),
                        'severity': 'medium'
                    })
                else:
                    matched.append(symbol)
        
        # Check for positions on exchange not in internal
        for symbol in exchange_positions:
            if symbol not in self.internal_positions:
                mismatches.append({
                    'symbol': symbol,
                    'type': 'missing_internal',
                    'internal_size': 0,
                    'exchange_size': exchange_positions[symbol]['size'],
                    'severity': 'high'
                })
        
        if mismatches:
            self.mismatches.extend(mismatches)
            logger.warning(f"Position reconciliation found {len(mismatches)} mismatches")
            for mismatch in mismatches:
                logger.warning(f"  {mismatch}")
        
        return {
            'timestamp': datetime.now(),
            'matched': matched,
            'mismatches': mismatches,
            'total_checked': len(self.internal_positions) + len(exchange_positions),
            'is_clean': len(mismatches) == 0,
            'status': 'ok'
        }
    
    def auto_correct(self, reconciliation_result: Dict) -> List[str]:
        """
        Automatically correct mismatches
        
        Returns:
            List of corrections made
        """
        corrections = []
        
        for mismatch in reconciliation_result.get('mismatches', []):
            symbol = mismatch['symbol']
            mismatch_type = mismatch['type']
            
            if mismatch_type == 'missing_on_exchange':
                # Internal shows position but exchange doesn't - clear internal
                self.clear_internal(symbol)
                corrections.append(f"Cleared phantom internal position for {symbol}")
                logger.info(f"Auto-corrected: Cleared {symbol} from internal")
                
            elif mismatch_type == 'missing_internal':
                # Exchange has position but internal doesn't - add to internal
                exchange_size = mismatch['exchange_size']
                self.update_internal(symbol, exchange_size, 0)  # Price unknown
                corrections.append(f"Added missing {symbol} to internal positions")
                logger.info(f"Auto-corrected: Added {symbol} to internal")
                
            elif mismatch_type == 'size_mismatch':
                # Sizes don't match - trust exchange
                exchange_size = mismatch['exchange_size']
                if exchange_size == 0:
                    self.clear_internal(symbol)
                    corrections.append(f"Cleared {symbol} (exchange shows 0)")
                else:
                    self.update_internal(symbol, exchange_size, 0)
                    corrections.append(f"Updated {symbol} size to match exchange")
                logger.info(f"Auto-corrected: Synced {symbol} size")
        
        return corrections


# Singletons
_shadow_mode: Optional[ShadowMode] = None
_position_reconciliation: Optional[PositionReconciliation] = None


def get_shadow_mode() -> ShadowMode:
    global _shadow_mode
    if _shadow_mode is None:
        _shadow_mode = ShadowMode()
    return _shadow_mode


def get_position_reconciliation() -> PositionReconciliation:
    global _position_reconciliation
    if _position_reconciliation is None:
        _position_reconciliation = PositionReconciliation()
    return _position_reconciliation


if __name__ == '__main__':
    print("=" * 70)
    print("PRODUCTION SAFETY FEATURES - TEST")
    print("=" * 70)
    
    # Test shadow mode
    print("\n👻 Testing Shadow Mode...")
    shadow = ShadowMode()
    shadow.enable()
    
    # Simulate shadow trades
    for i in range(5):
        signal = {
            'action': 'LONG' if i % 2 == 0 else 'SHORT',
            'price': 50000 + i * 100,
            'size': 0.1,
            'confidence': 0.75,
            'reasons': ['Test signal']
        }
        result = shadow.log_trade(signal)
        print(f"  Shadow trade {result['trade_id']}: {signal['action']}")
    
    stats = shadow.get_statistics()
    print(f"\nShadow Statistics:")
    print(f"  Enabled: {stats['enabled']}")
    print(f"  Total shadow trades: {stats['total_shadow_trades']}")
    
    # Test position reconciliation
    print("\n🔄 Testing Position Reconciliation...")
    recon = PositionReconciliation(tolerance_pct=0.01)
    
    # Setup internal positions
    recon.update_internal('BTC', 0.5, 50000)
    recon.update_internal('ETH', 2.0, 3000)
    recon.update_internal('SOL', 10.0, 150)  # This one is phantom
    
    # Setup exchange positions (with mismatch)
    exchange_positions = {
        'BTC': {'size': 0.5, 'entry_price': 50000},  # Match
        'ETH': {'size': 2.05, 'entry_price': 3000},  # Size mismatch
        'ADA': {'size': 1000, 'entry_price': 0.5}    # Missing from internal
    }
    
    result = recon.reconcile(exchange_positions)
    print(f"\nReconciliation Result:")
    print(f"  Matched: {result['matched']}")
    print(f"  Mismatches: {len(result['mismatches'])}")
    
    for mismatch in result['mismatches']:
        print(f"    {mismatch['type']}: {mismatch['symbol']}")
    
    # Auto-correct
    print(f"\n🔧 Auto-correcting mismatches...")
    corrections = recon.auto_correct(result)
    for correction in corrections:
        print(f"  ✓ {correction}")
    
    # Verify correction
    result2 = recon.reconcile(exchange_positions)
    print(f"\nAfter correction:")
    print(f"  Is clean: {result2['is_clean']}")
    print(f"  Mismatches: {len(result2['mismatches'])}")
    
    print("\n" + "=" * 70)
    print("✅ Production safety features working!")
    print("=" * 70)
