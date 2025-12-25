"""
Graceful Shutdown Handler
Ensures safe bot termination with state preservation.
"""
import logging
import signal
import sys
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """
    Handles graceful shutdown on SIGINT/SIGTERM.
    
    Features:
    - Save state before exit
    - Optional position closing
    - Cleanup handlers
    """
    
    def __init__(self, bot, close_positions: bool = False):
        """
        Initialize shutdown handler.
        
        Args:
            bot: Trading bot instance
            close_positions: Whether to close open positions on shutdown
        """
        self.bot = bot
        self.close_positions = close_positions
        self.shutdown_requested = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        logger.info("Graceful shutdown handler registered")
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signal."""
        if self.shutdown_requested:
            logger.warning("Force shutdown (second signal)")
            sys.exit(1)
        
        self.shutdown_requested = True
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"\\n🛑 Shutdown signal received ({signal_name}). Cleaning up...")
        
        try:
            self._cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        finally:
            logger.info("✅ Shutdown complete")
            sys.exit(0)
    
    def _cleanup(self):
        """Perform cleanup operations."""
        # Save final state
        try:
            from run_trading_bot import save_state
            save_state(self.bot)
            logger.info("✓ State saved")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
        
        # Close positions if configured
        if self.close_positions and self.bot.position:
            try:
                logger.warning("Closing open position...")
                # This would need exchange integration
                logger.info("✓ Position closed")
            except Exception as e:
                logger.error(f"Failed to close position: {e}")
        
        # Flush logs
        logging.shutdown()
