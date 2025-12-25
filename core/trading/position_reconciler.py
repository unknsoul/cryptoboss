"""
Position Reconciliation Service
Ensures bot's internal position state matches exchange positions.
"""
import logging
from typing import Dict, Optional,List
from datetime import datetime

logger = logging.getLogger(__name__)


class PositionReconciler:
    """
    Reconciles internal position state with exchange.
    
    Features:
    - Startup position sync
    - Periodic reconciliation
    - Position drift detection
    - Alert on mismatches
    """
    
    def __init__(self, exchange_client=None, alert_manager=None):
        """
        Initialize position reconciler.
        
        Args:
            exchange_client: Exchange client for fetching positions
            alert_manager: Alert manager for notifications
        """
        self.exchange_client = exchange_client
        self.alert_manager = alert_manager
        self.last_sync = None
        self.drift_count = 0
        
        logger.info("Position Reconciler initialized")
    
    def reconcile_on_startup(self, bot) -> Dict:
        """
        Reconcile positions on bot startup.
        
        Args:
            bot: Bot instance with position state
            
        Returns:
            Reconciliation result
        """
        if not self.exchange_client:
            logger.warning("No exchange client - skipping position reconciliation")
            return {'status': 'skipped', 'reason': 'no_exchange_client'}
        
        try:
            # Fetch exchange positions
            exchange_positions = self.exchange_client.get_open_positions()
            
            # Check for open positions on exchange
            if exchange_positions:
                logger.warning(f"⚠️  Found {len(exchange_positions)} open position(s) on exchange")
                
                for pos in exchange_positions:
                    symbol = pos.get('symbol', 'UNKNOWN')
                    size = float(pos.get('positionAmt', 0))
                    entry_price = float(pos.get('entryPrice', 0))
                    
                    logger.warning(f"   {symbol}: {size} @ ${entry_price:.2f}")
                
                # Check if bot has matching position
                if not bot.position:
                    logger.critical("🚨 POSITION DRIFT: Exchange has positions but bot doesn't!")
                    if self.alert_manager:
                        self.alert_manager.send_alert(
                            "position_drift",
                            f"Found {len(exchange_positions)} positions on exchange but bot has none",
                            {'exchange_positions': len(exchange_positions)}
                        )
                    
                    return {
                        'status': 'drift_detected',
                        'exchange_positions': len(exchange_positions),
                        'bot_position': None
                    }
            else:
                logger.info("✓ No open positions on exchange")
            
            self.last_sync = datetime.now()
            return {'status': 'success', 'timestamp': self.last_sync.isoformat()}
            
        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def periodic_sync(self, bot, interval_seconds: int = 60) -> Optional[Dict]:
        """
        Perform per iodic position sync.
        
        Args:
            bot: Bot instance
            interval_seconds: Sync interval
            
        Returns:
            Sync result if sync performed
        """
        now = datetime.now()
        
        if self.last_sync is None:
            return self.reconcile_on_startup(bot)
        
        elapsed = (now - self.last_sync).total_seconds()
        if elapsed >= interval_seconds:
            return self.reconcile_on_startup(bot)
        
        return None
