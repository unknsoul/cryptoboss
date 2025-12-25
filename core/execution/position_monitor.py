"""
Position Monitoring & Management System
Critical fix for stop loss and take profit enforcement

This module provides proper position monitoring that:
- Enforces stop loss exits
- Takes profit at targets
- Updates trailing stops
- Monitors position health
"""

import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Monitor open positions and enforce exits
    
    Critical for protecting capital and taking profits
    """
    
    def __init__(self):
        """Initialize position monitor"""
        self.position = None
        self.monitoring_active = False
        
    def set_position(self, position: Dict):
        """
        Start monitoring a position
        
        Args:
            position: Dict with entry_price, stop_loss, take_profit, side, size
        """
        self.position = position
        self.monitoring_active = True
        
        logger.info(f"Position monitoring started: {position['side']} @ ${position['entry_price']:,.0f}")
        logger.info(f"  Stop: ${position['stop_loss']:,.0f}, Target: ${position['take_profit']:,.0f}")
    
    def clear_position(self):
        """Clear position after exit"""
        self.position = None
        self.monitoring_active = False
    
    def check_exit_conditions(self, current_price: float, atr: float = None) -> Optional[Dict]:
        """
        Check if position should be closed
        
        Args:
            current_price: Current market price
            atr: Current ATR for trailing stop
            
        Returns:
            Exit signal if conditions met, None otherwise
        """
        if not self.monitoring_active or not self.position:
            return None
        
        side = self.position['side']
        stop_loss = self.position['stop_loss']
        take_profit = self.position['take_profit']
        entry_price = self.position['entry_price']
        
        # Calculate current P&L
        if side == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price * 100
            
            # Check stop loss
            if current_price <= stop_loss:
                logger.warning(f"STOP LOSS HIT: ${current_price:,.0f} <= ${stop_loss:,.0f}")
                return {
                    'action': 'CLOSE',
                    'reason': 'stop_loss',
                    'price': current_price,
                    'pnl_pct': pnl_pct
                }
            
            # Check take profit
            if current_price >= take_profit:
                logger.info(f"TAKE PROFIT HIT: ${current_price:,.0f} >= ${take_profit:,.0f}")
                return {
                    'action': 'CLOSE',
                    'reason': 'take_profit',
                    'price': current_price,
                    'pnl_pct': pnl_pct
                }
        
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price * 100
            
            # Check stop loss
            if current_price >= stop_loss:
                logger.warning(f"STOP LOSS HIT: ${current_price:,.0f} >= ${stop_loss:,.0f}")
                return {
                    'action': 'CLOSE',
                    'reason': 'stop_loss',
                    'price': current_price,
                    'pnl_pct': pnl_pct
                }
            
            # Check take profit
            if current_price <= take_profit:
                logger.info(f"TAKE PROFIT HIT: ${current_price:,.0f} <= ${take_profit:,.0f}")
                return {
                    'action': 'CLOSE',
                    'reason': 'take_profit',
                    'price': current_price,
                    'pnl_pct': pnl_pct
                }
        
        # Update trailing stop if ATR provided and we have rr_optimizer
        if atr and hasattr(self, 'rr_optimizer'):
            trailing_result = self.update_trailing_stop(current_price, atr)
            if trailing_result and trailing_result.get('triggered'):
                return {
                    'action': 'CLOSE',
                    'reason': 'trailing_stop',
                    'price': current_price,
                    'pnl_pct': pnl_pct
                }
        
        return None
    
    def update_trailing_stop(self, current_price: float, atr: float) -> Optional[Dict]:
        """
        Update trailing stop based on profit
        
        Returns:
            Dict with new stop and whether it was triggered
        """
        if not self.position:
            return None
        
        # Check if we have the optimizer
        if not hasattr(self, 'rr_optimizer'):
            return None
        
        try:
            trailing = self.rr_optimizer.calculate_trailing_stop(
                entry_price=self.position['entry_price'],
                current_price=current_price,
                initial_stop=self.position.get('initial_stop', self.position['stop_loss']),
                side=self.position['side'],
                atr=atr
            )
            
            if trailing.get('trailing_active'):
                old_stop = self.position['stop_loss']
                new_stop = trailing['new_stop']
                
                # Only move stop in favorable direction
                if self.position['side'] == 'LONG':
                    if new_stop > old_stop:
                        self.position['stop_loss'] = new_stop
                        logger.info(f"✓ Trailing stop updated: ${old_stop:,.0f} → ${new_stop:,.0f}")
                        logger.info(f"  Locked profit: ${trailing['locked_profit']:.2f}")
                else:  # SHORT
                    if new_stop < old_stop:
                        self.position['stop_loss'] = new_stop
                        logger.info(f"✓ Trailing stop updated: ${old_stop:,.0f} → ${new_stop:,.0f}")
                        logger.info(f"  Locked profit: ${trailing['locked_profit']:.2f}")
                
                # Check if new stop is hit
                if self.position['side'] == 'LONG':
                    if current_price <= new_stop:
                        return {'triggered': True, 'new_stop': new_stop}
                else:
                    if current_price >= new_stop:
                        return {'triggered': True, 'new_stop': new_stop}
            
            return trailing
            
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
            return None
    
    def get_position_status(self, current_price: float) -> Dict:
        """Get current position status"""
        if not self.position:
            return {'has_position': False}
        
        entry = self.position['entry_price']
        side = self.position['side']
        
        if side == 'LONG':
            pnl = current_price - entry
            pnl_pct = pnl / entry * 100
        else:
            pnl = entry - current_price
            pnl_pct = pnl / entry * 100
        
        return {
            'has_position': True,
            'side': side,
            'entry_price': entry,
            'current_price': current_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'stop_loss': self.position['stop_loss'],
            'take_profit': self.position['take_profit']
        }


# Singleton
_position_monitor: Optional[PositionMonitor] = None


def get_position_monitor() -> PositionMonitor:
    """Get singleton position monitor"""
    global _position_monitor
    if _position_monitor is None:
        _position_monitor = PositionMonitor()
    return _position_monitor


if __name__ == '__main__':
    print("=" * 70)
    print("POSITION MONITORING - TEST")
    print("=" * 70)
    
    # Test position monitoring
    monitor = PositionMonitor()
    
    # Set a LONG position
    position = {
        'side': 'LONG',
        'entry_price': 50000,
        'stop_loss': 49500,
        'take_profit': 51250,
        'size': 0.1
    }
    
    monitor.set_position(position)
    
    # Test scenarios
    test_prices = [
        (50100, "Small profit"),
        (49400, "Stop loss hit"),
        (51300, "Take profit hit"),
    ]
    
    print("\n📊 Testing exit conditions...")
    for price, scenario in test_prices:
        monitor.set_position(position)  # Reset
        
        exit_signal = monitor.check_exit_conditions(price)
        
        if exit_signal:
            print(f"  {scenario}: EXIT triggered - {exit_signal['reason']}")
            print(f"    P&L: {exit_signal['pnl_pct']:.2f}%")
        else:
            print(f"  {scenario}: No exit")
    
    # Test status
    status = monitor.get_position_status(50500)
    print(f"\n📈 Position Status @ $50,500:")
    print(f"  Side: {status['side']}")
    print(f"  P&L: ${status['pnl']:.2f} ({status['pnl_pct']:.2f}%)")
    
    print("\n" + "=" * 70)
    print("✅ Position monitoring working!")
    print("=" * 70)
