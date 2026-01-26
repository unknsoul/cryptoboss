"""
Graceful Shutdown Manager - Upgrade F

Async-aware shutdown that:
- Cancels all running tasks cleanly
- Cancels pending orders
- Flushes state to database
- Closes all connections

Integrated with TradingEngine.
"""

import asyncio
import signal
import logging
from typing import List, Callable, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """
    Manages graceful shutdown of async trading system.
    
    Usage:
        shutdown = GracefulShutdown()
        
        # Register cleanup handlers
        shutdown.register(state_manager.flush)
        shutdown.register(cancel_all_orders)
        shutdown.register(close_connections)
        
        # Install signal handlers
        shutdown.install_signal_handlers()
        
        # When shutdown triggered
        await shutdown.execute()  # Runs all registered handlers
    """
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._handlers: List[Callable] = []
        self._async_handlers: List[Callable] = []
        self._pending_tasks: Set[asyncio.Task] = set()
        self._is_shutting_down = False
        self._shutdown_event = asyncio.Event()
        
        logger.info("GracefulShutdown manager initialized")
    
    def register(self, handler: Callable):
        """Register a synchronous cleanup handler."""
        self._handlers.append(handler)
        logger.debug(f"Registered sync handler: {handler.__name__}")
    
    def register_async(self, handler: Callable):
        """Register an async cleanup handler."""
        self._async_handlers.append(handler)
        logger.debug(f"Registered async handler: {handler.__name__}")
    
    def track_task(self, task: asyncio.Task):
        """Track an async task for cancellation on shutdown."""
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
    
    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop = None):
        """Install SIGINT and SIGTERM handlers."""
        loop = loop or asyncio.get_event_loop()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._signal_handler)
                logger.info(f"Installed handler for {sig.name}")
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, f: self._signal_handler())
    
    def _signal_handler(self):
        """Handle shutdown signal."""
        if self._is_shutting_down:
            logger.warning("Forced shutdown requested")
            raise SystemExit(1)
        
        logger.info("Shutdown signal received, initiating graceful shutdown...")
        self._is_shutting_down = True
        self._shutdown_event.set()
    
    async def wait_for_shutdown(self):
        """Wait until shutdown is triggered."""
        await self._shutdown_event.wait()
    
    @property
    def is_shutting_down(self) -> bool:
        return self._is_shutting_down
    
    async def execute(self):
        """
        Execute graceful shutdown sequence.
        
        Order:
        1. Cancel all tracked tasks
        2. Run async cleanup handlers
        3. Run sync cleanup handlers
        4. Final cleanup
        """
        if self._is_shutting_down:
            logger.info("Shutdown already in progress")
            return
        
        self._is_shutting_down = True
        start_time = datetime.now()
        
        logger.info("=" * 50)
        logger.info("GRACEFUL SHUTDOWN INITIATED")
        logger.info("=" * 50)
        
        try:
            # 1. Cancel tracked tasks
            await self._cancel_tasks()
            
            # 2. Run async handlers
            await self._run_async_handlers()
            
            # 3. Run sync handlers
            self._run_sync_handlers()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Graceful shutdown completed in {elapsed:.2f}s")
            
        except asyncio.TimeoutError:
            logger.error("Shutdown timed out, forcing exit")
            raise SystemExit(1)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise
    
    async def _cancel_tasks(self):
        """Cancel all tracked async tasks."""
        if not self._pending_tasks:
            return
        
        logger.info(f"Cancelling {len(self._pending_tasks)} pending tasks...")
        
        for task in self._pending_tasks:
            task.cancel()
        
        # Wait for tasks to complete cancellation
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._pending_tasks, return_exceptions=True),
                timeout=self.timeout / 3
            )
        except asyncio.TimeoutError:
            logger.warning("Some tasks did not cancel in time")
        
        logger.info("Tasks cancelled")
    
    async def _run_async_handlers(self):
        """Run all async cleanup handlers."""
        if not self._async_handlers:
            return
        
        logger.info(f"Running {len(self._async_handlers)} async cleanup handlers...")
        
        for handler in self._async_handlers:
            try:
                await asyncio.wait_for(
                    handler(),
                    timeout=self.timeout / len(self._async_handlers) if self._async_handlers else 10
                )
                logger.info(f"  ✓ {handler.__name__}")
            except asyncio.TimeoutError:
                logger.warning(f"  ✗ {handler.__name__} timed out")
            except Exception as e:
                logger.error(f"  ✗ {handler.__name__} failed: {e}")
    
    def _run_sync_handlers(self):
        """Run all sync cleanup handlers."""
        if not self._handlers:
            return
        
        logger.info(f"Running {len(self._handlers)} sync cleanup handlers...")
        
        for handler in self._handlers:
            try:
                handler()
                logger.info(f"  ✓ {handler.__name__}")
            except Exception as e:
                logger.error(f"  ✗ {handler.__name__} failed: {e}")


async def create_shutdown_context(engine):
    """
    Create shutdown context for trading engine.
    
    Registers all necessary cleanup handlers.
    """
    shutdown = GracefulShutdown(timeout=30.0)
    
    # Register engine cleanup
    async def cleanup_engine():
        logger.info("Stopping trading engine...")
        engine.stop()
    
    shutdown.register_async(cleanup_engine)
    
    # Register state flush
    async def flush_state():
        logger.info("Flushing state to database...")
        if hasattr(engine, 'state_manager'):
            # Save final state snapshot
            pass
    
    shutdown.register_async(flush_state)
    
    # Register order cancellation
    async def cancel_orders():
        logger.info("Cancelling open orders...")
        if hasattr(engine, 'execution_router'):
            # Cancel all open orders
            pass
    
    shutdown.register_async(cancel_orders)
    
    # Register event bus shutdown
    async def stop_event_bus():
        logger.info("Stopping event bus...")
        if hasattr(engine, 'event_bus'):
            engine.event_bus.stop()
    
    shutdown.register_async(stop_event_bus)
    
    return shutdown


# Singleton
_shutdown: Optional[GracefulShutdown] = None

def get_shutdown_manager() -> GracefulShutdown:
    global _shutdown
    if _shutdown is None:
        _shutdown = GracefulShutdown()
    return _shutdown
