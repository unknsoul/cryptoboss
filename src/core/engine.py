"""
CryptoBoss Trading Engine - Clean Architecture Entry Point

This is the main orchestrator that ties together:
- StateManager: Crash-proof persistence
- ExecutionRouter: Unified order execution
- RiskGuardian: Global risk protection
- EventBus: Event-driven communication
- Strategies: DCA, Grid, Market Making, etc.

Usage:
    engine = TradingEngine(mode="paper")
    engine.add_strategy("dca", DCAStrategy(...))
    engine.start()
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .state_manager import StateManager, get_state_manager
from .execution_router import ExecutionRouter, ExecutionMode, OrderIntent, OrderSide, OrderType
from .risk_guardian import RiskGuardian, get_risk_guardian
from .event_bus import EventBus, get_event_bus, EventType, Event, emit_price_tick

logger = logging.getLogger(__name__)


class EngineStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EngineConfig:
    """Engine configuration."""
    mode: str = "paper"  # "paper", "live", "testnet"
    portfolio_value: float = 10000.0
    auto_recover: bool = True  # Recover strategies on restart
    heartbeat_interval: int = 60  # Seconds
    log_level: str = "INFO"


class TradingEngine:
    """
    Main trading engine orchestrator.
    
    This is the single entry point for the entire trading system.
    It manages the lifecycle of all components and strategies.
    
    Example:
        from src.core.engine import TradingEngine
        from src.strategies.dca_strategy import DCAStrategy
        
        engine = TradingEngine(mode="testnet", portfolio_value=10000)
        
        dca = DCAStrategy(
            base_order_size=100,
            safety_order_size=200,
            max_safety_orders=5
        )
        engine.add_strategy("dca_btc", dca, symbol="BTC/USDT", allocation=2500)
        
        engine.start()
    """
    
    def __init__(
        self,
        mode: str = "testnet",
        portfolio_value: float = 10000.0,
        exchange_client = None,
        config: EngineConfig = None
    ):
        self.config = config or EngineConfig(mode=mode, portfolio_value=portfolio_value)
        self.status = EngineStatus.STOPPED
        
        # v11.1: Environment Truth - Generate signature at startup
        try:
            from .environment_guard import get_environment_guard
            self._env_guard = get_environment_guard()
            if not self._env_guard._initialized:
                exchange_url = "https://api.binance.com" if mode == "live" else "https://testnet.binance.vision"
                self._env_signature = self._env_guard.initialize(
                    mode=mode,
                    exchange_id="binance",
                    exchange_url=exchange_url,
                    config={"portfolio_value": portfolio_value, "mode": mode}
                )
            else:
                self._env_signature = self._env_guard.get_signature()
        except Exception as e:
            logger.warning(f"Environment guard init skipped: {e}")
            self._env_guard = None
            self._env_signature = None
        
        # v11.1: Incident State Machine
        try:
            from .incident_state_machine import get_incident_state_machine
            self._incident_sm = get_incident_state_machine()
        except Exception as e:
            logger.warning(f"Incident state machine init skipped: {e}")
            self._incident_sm = None
        
        # v11.1: Decision Narrative Engine
        try:
            from .decision_narrative import get_narrative_engine
            self._narrative_engine = get_narrative_engine()
        except Exception as e:
            logger.warning(f"Narrative engine init skipped: {e}")
            self._narrative_engine = None
        
        # Core components
        self.state_manager = get_state_manager()
        self.event_bus = get_event_bus()
        self.risk_guardian = get_risk_guardian(portfolio_value)
        
        # Execution router - supports paper, testnet, and live
        if mode == "live":
            exec_mode = ExecutionMode.LIVE
        elif mode == "testnet":
            exec_mode = ExecutionMode.TESTNET
        else:
            exec_mode = ExecutionMode.PAPER
        
        self.execution_router = ExecutionRouter(
            mode=exec_mode,
            exchange_client=exchange_client,
            state_manager=self.state_manager,
            risk_guardian=self.risk_guardian,
            portfolio_value=portfolio_value,
        )
        
        # Strategy management
        self.strategies: Dict[str, Dict] = {}  # strategy_id -> {strategy, symbol, allocation, active}
        
        # Runtime
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        logger.info(f"TradingEngine initialized in {mode.upper()} mode with ${portfolio_value:,.2f}")
    
    def add_strategy(
        self,
        strategy_id: str,
        strategy: Any,
        symbol: str,
        allocation: float,
        auto_start: bool = False
    ):
        """
        Add a strategy to the engine.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            strategy: Strategy object (DCAStrategy, GridStrategy, etc.)
            symbol: Trading symbol (e.g., "BTC/USDT")
            allocation: Capital allocated to this strategy
            auto_start: Start strategy immediately when engine starts
        """
        if strategy_id in self.strategies:
            raise ValueError(f"Strategy {strategy_id} already exists")
        
        self.strategies[strategy_id] = {
            "strategy": strategy,
            "symbol": symbol,
            "allocation": allocation,
            "active": False,
            "auto_start": auto_start,
            "pnl": 0.0
        }
        
        # Update risk guardian
        self.risk_guardian.update_strategy_allocation(strategy_id, allocation)
        
        logger.info(f"Added strategy: {strategy_id} on {symbol} with ${allocation:,.2f}")
    
    def remove_strategy(self, strategy_id: str):
        """Remove a strategy from the engine."""
        if strategy_id in self.strategies:
            self.stop_strategy(strategy_id)
            del self.strategies[strategy_id]
            logger.info(f"Removed strategy: {strategy_id}")
    
    def start_strategy(self, strategy_id: str) -> bool:
        """Start a specific strategy."""
        if strategy_id not in self.strategies:
            logger.error(f"Strategy {strategy_id} not found")
            return False
        
        self.strategies[strategy_id]["active"] = True
        
        # Save state
        strategy_info = self.strategies[strategy_id]
        self.state_manager.save_strategy_state(
            strategy_id=strategy_id,
            strategy_type=type(strategy_info["strategy"]).__name__,
            symbol=strategy_info["symbol"],
            status="active",
            capital_allocated=strategy_info["allocation"],
            current_pnl=strategy_info["pnl"],
            custom_state={}
        )
        
        # Emit event
        self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_STARTED,
            source="TradingEngine",
            data={"strategy_id": strategy_id}
        ))
        
        logger.info(f"Started strategy: {strategy_id}")
        return True
    
    def stop_strategy(self, strategy_id: str) -> bool:
        """Stop a specific strategy."""
        if strategy_id not in self.strategies:
            return False
        
        self.strategies[strategy_id]["active"] = False
        self.state_manager.mark_strategy_stopped(strategy_id)
        
        # Emit event
        self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_STOPPED,
            source="TradingEngine",
            data={"strategy_id": strategy_id}
        ))
        
        logger.info(f"Stopped strategy: {strategy_id}")
        return True
    
    async def _process_price_update(self, symbol: str, price: float, df=None, index: int = 0):
        """Process a price update for all relevant strategies."""
        # Update broker price (used by PaperBroker for fill simulation)
        try:
            self.execution_router.set_price(symbol, price)
        except Exception:
            pass
        
        # Emit price tick
        emit_price_tick(self.event_bus, symbol, price)
        
        # Process each active strategy for this symbol
        for strategy_id, info in self.strategies.items():
            if not info["active"] or info["symbol"] != symbol:
                continue
            
            strategy = info["strategy"]
            
            try:
                # Generate signal
                if hasattr(strategy, "generate_signal"):
                    signal = strategy.generate_signal(df, index, price)
                    signal_payload: Dict[str, Any]

                    # Support both legacy dict outputs and v11+ SignalResult objects.
                    if isinstance(signal, dict):
                        signal_payload = signal
                    elif hasattr(strategy, "signal_to_legacy_dict"):
                        signal_payload = strategy.signal_to_legacy_dict(signal)
                    elif hasattr(signal, "action"):
                        signal_payload = {
                            "action": getattr(signal, "action", "HOLD"),
                            "size": getattr(signal, "size", 0.0),
                            "price": getattr(signal, "price", price),
                            "reason": getattr(signal, "reason", ""),
                        }
                    else:
                        signal_payload = {"action": "HOLD"}
                    
                    if signal_payload.get("action") in ("BUY", "SELL"):
                        # Create order intent
                        intent = OrderIntent(
                            symbol=symbol,
                            side=OrderSide.BUY if signal_payload["action"] == "BUY" else OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=signal_payload.get("size", 0),
                            price=signal_payload.get("price", price),
                            strategy_id=strategy_id
                        )
                        
                        # Execute via router
                        result = await self.execution_router.execute(intent)
                        
                        if result.success:
                            # Update strategy P&L
                            pnl = signal_payload.get("pnl", 0)
                            info["pnl"] += pnl
                            self.risk_guardian.record_trade(pnl, strategy_id)
                            
                            # Update state
                            self.state_manager.save_strategy_state(
                                strategy_id=strategy_id,
                                strategy_type=type(strategy).__name__,
                                symbol=info["symbol"],
                                status="active",
                                capital_allocated=info["allocation"],
                                current_pnl=info["pnl"],
                                custom_state=getattr(strategy, "get_state", lambda: {})()
                            )
            except Exception as e:
                logger.error(f"Error processing strategy {strategy_id}: {e}")
                self.risk_guardian.record_error(str(e))
    
    async def _heartbeat_loop(self):
        """Periodic heartbeat for monitoring."""
        while self._running:
            try:
                self.event_bus.publish(Event(
                    event_type=EventType.HEARTBEAT,
                    source="TradingEngine",
                    data={
                        "status": self.status.value,
                        "active_strategies": sum(1 for s in self.strategies.values() if s["active"]),
                        "total_pnl": sum(s["pnl"] for s in self.strategies.values())
                    }
                ))
                await asyncio.sleep(self.config.heartbeat_interval)
            except asyncio.CancelledError:
                break
    
    def _recover_strategies(self):
        """Recover strategies from persistent state."""
        if not self.config.auto_recover:
            return
        
        states = self.state_manager.load_all_active_strategies()
        
        for state in states:
            if state.strategy_id in self.strategies:
                # Restore state to existing strategy
                info = self.strategies[state.strategy_id]
                info["pnl"] = state.current_pnl
                info["active"] = state.status == "active"
                
                # Restore custom state if strategy supports it
                if hasattr(info["strategy"], "restore_state"):
                    info["strategy"].restore_state(state.custom_state)
                
                logger.info(f"Recovered strategy: {state.strategy_id}")
    
    def start(self):
        """Start the trading engine."""
        if self.status == EngineStatus.RUNNING:
            logger.warning("Engine already running")
            return
        
        self.status = EngineStatus.STARTING
        self._running = True
        
        # Start event bus
        self.event_bus.start()
        
        # Recover strategies
        self._recover_strategies()
        
        # Auto-start strategies
        for strategy_id, info in self.strategies.items():
            if info.get("auto_start"):
                self.start_strategy(strategy_id)
        
        self.status = EngineStatus.RUNNING
        
        # Emit startup event
        self.event_bus.publish(Event(
            event_type=EventType.SYSTEM_STARTUP,
            source="TradingEngine",
            data={"mode": self.config.mode}
        ))
        
        logger.info("Trading engine started")
    
    def stop(self):
        """Stop the trading engine gracefully."""
        if self.status == EngineStatus.STOPPED:
            return
        
        self.status = EngineStatus.STOPPING
        self._running = False
        
        # Stop all strategies
        for strategy_id in list(self.strategies.keys()):
            self.stop_strategy(strategy_id)
        
        # Cancel async tasks
        for task in self._tasks:
            task.cancel()
        
        # Stop event bus
        self.event_bus.stop()
        
        self.status = EngineStatus.STOPPED
        
        logger.info("Trading engine stopped")
    
    def emergency_stop(self, reason: str = "Manual trigger"):
        """Emergency stop - halt everything immediately."""
        logger.critical(f"EMERGENCY STOP: {reason}")
        
        self.risk_guardian.emergency_stop(reason)
        
        self.event_bus.publish_sync(Event(
            event_type=EventType.EMERGENCY_STOP,
            source="TradingEngine",
            data={"reason": reason}
        ))
        
        self.stop()
    
    def get_status(self) -> Dict:
        """Get comprehensive engine status."""
        return {
            "status": self.status.value,
            "mode": self.config.mode,
            "portfolio_value": self.config.portfolio_value,
            "strategies": {
                sid: {
                    "symbol": info["symbol"],
                    "allocation": info["allocation"],
                    "active": info["active"],
                    "pnl": info["pnl"],
                    "type": type(info["strategy"]).__name__
                }
                for sid, info in self.strategies.items()
            },
            "total_pnl": sum(s["pnl"] for s in self.strategies.values()),
            "risk_report": self.risk_guardian.get_risk_report(),
            "event_bus_stats": self.event_bus.get_stats()
        }
    
    async def get_balance(self) -> Dict:
        """Get current account balance."""
        return await self.execution_router.get_balance()


# Factory function for easy instantiation
def create_engine(
    mode: str = "paper",
    portfolio_value: float = 10000.0,
    exchange_client = None
) -> TradingEngine:
    """Create a trading engine with sensible defaults."""
    return TradingEngine(
        mode=mode,
        portfolio_value=portfolio_value,
        exchange_client=exchange_client
    )
