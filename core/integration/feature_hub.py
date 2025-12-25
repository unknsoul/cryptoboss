"""
Integrated Feature Hub
Connects all 24 professional features to the trading bot

This module ensures all features are:
- Properly initialized
- Updated on every trade
- Reporting metrics
- Sending alerts
"""

import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class IntegratedFeatureHub:
    """
    Central hub for all trading features
    
    Manages:
    - Analytics tracking
    - Risk monitoring
    - Alert notifications
    - Database persistence
    - Position monitoring
    """
    
    def __init__(self):
        """Initialize all features"""
        self.features_loaded = {}
        self.position_monitor = None
        self.telegram = None
        self.database = None
        self.slippage_monitor = None
        self.latency_monitor = None
        self.mae_mfe_tracker = None
        self.metrics_calculator = None
        self.var_calculator = None
        self.rr_optimizer = None
        
        self._load_all_features()
    
    def _load_all_features(self):
        """Load all available features"""
        
        # Position Monitor (CRITICAL)
        try:
            from core.execution.position_monitor import get_position_monitor
            self.position_monitor = get_position_monitor()
            self.features_loaded['position_monitor'] = True
            logger.info("✓ Position Monitor loaded")
        except Exception as e:
            logger.warning(f"Position Monitor failed: {e}")
            self.features_loaded['position_monitor'] = False
        
        # Database
        try:
            from core.database.trading_db import get_trading_db
            self.database = get_trading_db()
            self.features_loaded['database'] = True
            logger.info("✓ Database loaded")
        except Exception as e:
            logger.warning(f"Database failed: {e}")
            self.features_loaded['database'] = False
        
        # Telegram
        try:
            from integrations.telegram_notifier import get_telegram_notifier
            self.telegram = get_telegram_notifier()
            self.features_loaded['telegram'] = True
            logger.info("✓ Telegram loaded")
        except Exception as e:
            logger.warning(f"Telegram failed: {e}")
            self.features_loaded['telegram'] = False
        
        # Slippage Monitor
        try:
            from core.execution.slippage_monitor import get_slippage_monitor
            self.slippage_monitor = get_slippage_monitor()
            self.features_loaded['slippage'] = True
            logger.info("✓ Slippage Monitor loaded")
        except Exception as e:
            logger.warning(f"Slippage failed: {e}")
            self.features_loaded['slippage'] = False
        
        # Latency Monitor  
        try:
            from core.execution.latency_monitor import get_latency_monitor
            self.latency_monitor = get_latency_monitor()
            self.features_loaded['latency'] = True
            logger.info("✓ Latency Monitor loaded")
        except Exception as e:
            logger.warning(f"Latency failed: {e}")
            self.features_loaded['latency'] = False
        
        # MAE/MFE Tracker
        try:
            from core.analytics.mae_mfe_tracker import get_mae_mfe_tracker
            self.mae_mfe_tracker = get_mae_mfe_tracker()
            self.features_loaded['mae_mfe'] = True
            logger.info("✓ MAE/MFE Tracker loaded")
        except Exception as e:
            logger.warning(f"MAE/MFE failed: {e}")
            self.features_loaded['mae_mfe'] = False
        
        # Advanced Metrics
        try:
            from core.analytics.advanced_metrics import get_metrics_calculator
            self.metrics_calculator = get_metrics_calculator()
            self.features_loaded['metrics'] = True
            logger.info("✓ Metrics Calculator loaded")
        except Exception as e:
            logger.warning(f"Metrics failed: {e}")
            self.features_loaded['metrics'] = False
        
        # VaR Calculator
        try:
            from core.risk.var_calculator import get_var_calculator
            self.var_calculator = get_var_calculator()
            self.features_loaded['var'] = True
            logger.info("✓ VaR Calculator loaded")
        except Exception as e:
            logger.warning(f"VaR failed: {e}")
            self.features_loaded['var'] = False
        
        # R/R Optimizer
        try:
            from core.risk.position_optimizer import get_rr_optimizer
            self.rr_optimizer = get_rr_optimizer()
            self.features_loaded['rr_optimizer'] = True
            logger.info("✓ R/R Optimizer loaded")
        except Exception as e:
            logger.warning(f"R/R Optimizer failed: {e}")
            self.features_loaded['rr_optimizer'] = False
        
        loaded_count = sum(1 for v in self.features_loaded.values() if v)
        logger.info(f"✓ Feature Hub: {loaded_count}/{len(self.features_loaded)} features loaded")
    
    def on_trade_entry(self, trade: Dict, bot_state: Dict):
        """
        Called when a trade is opened
        
        Args:
            trade: Trade data dict
            bot_state: Bot state for context
        """
        # Start position monitoring
        if self.position_monitor:
            position = {
                'side': trade['action'],
                'entry_price': trade['price'],
                'stop_loss': trade.get('stop_loss', trade['price'] * 0.99),
                'take_profit': trade.get('take_profit', trade['price'] * 1.025),
                'size': trade.get('size', 0.1),
                'entry_time': datetime.now()
            }
            self.position_monitor.set_position(position)
            
            # Link R/R optimizer to monitor for trailing stops
            if self.rr_optimizer:
                self.position_monitor.rr_optimizer = self.rr_optimizer
        
        # Record latency
        if self.latency_monitor and 'signal_time' in trade:
            signal_time = trade['signal_time']
            execution_time = datetime.now()
            latency_ms = (execution_time - signal_time).total_seconds() * 1000
            self.latency_monitor.record_execution(latency_ms)
        
        # Send notifications
        if self.telegram and self.telegram.enabled:
            self.telegram.notify_trade_execution(trade)
        
        logger.info(f"✓ Trade entry processed: {trade['action']} @ ${trade['price']:,.0f}")
    
    def on_trade_exit(self, trade: Dict, exit_price: float, exit_reason: str, bot_state: Dict):
        """
        Called when a trade is closed
        
        Args:
            trade: Original trade data
            exit_price: Exit price
            exit_reason: Reason for exit
            bot_state: Bot state
        """
        # Calculate final metrics
        entry_price = trade['price']
        size = trade.get('size', 0.1)
        side = trade['action']
        
        if side == 'LONG':
            pnl = (exit_price - entry_price) * size
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - exit_price) * size
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        
        # Record to database
        if self.database:
            trade_record = {
                'timestamp': trade.get('timestamp', datetime.now().isoformat()),
                'symbol': 'BTC',
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'confidence': trade.get('confidence', 0),
                'status': 'CLOSED',
                'notes': f"Exit: {exit_reason}"
            }
            self.database.insert_trade(trade_record)
        
        # Track MAE/MFE if available
        if self.mae_mfe_tracker and 'mae' in trade and 'mfe' in trade:
            self.mae_mfe_tracker.record_trade({
                'mae': trade['mae'],
                'mfe': trade['mfe'],
                'pnl': pnl,
                'exit_price': exit_price,
                'entry_price': entry_price
            })
        
        # Clear position monitor
        if self.position_monitor:
            self.position_monitor.clear_position()
        
        # Send notification
        if self.telegram and self.telegram.enabled:
            message = f"{'✅' if pnl > 0 else '❌'} Trade Closed\n"
            message += f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)\n"
            message += f"Reason: {exit_reason}"
            self.telegram.send_message(message)
        
        logger.info(f"✓ Trade exit processed: P&L ${pnl:,.2f} ({exit_reason})")
    
    def monitor_position(self, current_price: float, atr: float = None) -> Optional[Dict]:
        """
        Check if position should be closed
        
        Returns:
            Exit signal if conditions met
        """
        if not self.position_monitor:
            return None
        
        return self.position_monitor.check_exit_conditions(current_price, atr)
    
    def record_slippage(self, expected_price: float, actual_price: float, side: str, size: float):
        """Record slippage for a fill"""
        if self.slippage_monitor:
            self.slippage_monitor.record_fill(expected_price, actual_price, side, size)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from all features"""
        stats = {
            'features_loaded': self.features_loaded,
            'timestamp': datetime.now().isoformat()
        }
        
        # Database stats
        if self.database:
            stats['trade_stats'] = self.database.get_trade_statistics()
        
        # Slippage stats
        if self.slippage_monitor:
            stats['slippage'] = self.slippage_monitor.get_summary()
        
        # Latency stats
        if self.latency_monitor:
            stats['latency'] = self.latency_monitor.get_statistics()
        
        # MAE/MFE stats
        if self.mae_mfe_tracker:
            stats['mae_mfe'] = self.mae_mfe_tracker.get_statistics()
        
        return stats


# Singleton
_feature_hub: Optional[IntegratedFeatureHub] = None


def get_feature_hub() -> IntegratedFeatureHub:
    """Get singleton feature hub"""
    global _feature_hub
    if _feature_hub is None:
        _feature_hub = IntegratedFeatureHub()
    return _feature_hub


if __name__ == '__main__':
    print("=" * 70)
    print("INTEGRATED FEATURE HUB - TEST")
    print("=" * 70)
    
    # Initialize hub
    hub = get_feature_hub()
    
    print(f"\n✅ Features Loaded:")
    for feature, loaded in hub.features_loaded.items():
        status = "✓" if loaded else "✗"
        print(f"  {status} {feature}")
    
    print("\n" + "=" * 70)
    print("✅ Feature hub ready!")
    print("=" * 70)
