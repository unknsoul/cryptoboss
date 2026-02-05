"""
Execution Router - Unified Order Execution Layer

CRYPTOBOSS 2.0: PAPER TRADING REMOVED
Only TESTNET and LIVE environments are supported.
All trading goes through the exchange (testnet or live).

Architecture:
    Strategy -> OrderIntent -> ExecutionRouter -> [TestnetBroker | LiveBroker]
                                    |
                                    v
                              OrderResult (fill price, fees, slippage)

CRITICAL RULES:
    - NO paper trading, NO local simulation
    - All balances come from exchange
    - All trades execute on exchange
    - TESTNET = Binance Testnet (realistic testing)
    - LIVE = Binance Live (real money)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """
    Execution mode for the router.
    
    PAPER MODE IS PERMANENTLY REMOVED.
    Only TESTNET and LIVE are valid.
    """
    TESTNET = "testnet"
    LIVE = "live"
    
    # PAPER REMOVED - Do not add back


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class OrderIntent:
    """
    An intent to place an order. Strategies produce these.
    The ExecutionRouter converts them to actual orders.
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # Required for limit orders
    strategy_id: str = ""
    client_order_id: str = ""
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    reduce_only: bool = False
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.client_order_id:
            self.client_order_id = f"{self.strategy_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


@dataclass
class OrderResult:
    """Result of order execution."""
    success: bool
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    filled_quantity: float
    average_price: float
    fees: float
    slippage_bps: float
    timestamp: datetime
    raw_response: Dict = field(default_factory=dict)
    error_message: str = ""
    
    @property
    def total_cost(self) -> float:
        return self.filled_quantity * self.average_price + self.fees


class BaseBroker(ABC):
    """Abstract broker interface - exchange-based only."""
    
    @abstractmethod
    async def execute_order(self, intent: OrderIntent) -> OrderResult:
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict:
        pass


class ExchangeBroker(BaseBroker):
    """
    Exchange broker for both TESTNET and LIVE.
    
    The same broker handles both modes - the only difference
    is which exchange endpoint is used.
    """
    
    def __init__(self, exchange_client, mode: ExecutionMode, max_retries: int = 3):
        self.exchange = exchange_client
        self.mode = mode
        self.max_retries = max_retries
        
        if mode == ExecutionMode.LIVE:
            logger.warning("⚠️ LIVE BROKER INITIALIZED - REAL MONEY MODE")
        else:
            logger.info("📋 TESTNET BROKER INITIALIZED - Testing Mode")
    
    async def execute_order(self, intent: OrderIntent) -> OrderResult:
        for attempt in range(self.max_retries):
            try:
                if intent.order_type == OrderType.MARKET:
                    if intent.side == OrderSide.BUY:
                        order = await self.exchange.create_market_buy_order(
                            intent.symbol, intent.quantity
                        )
                    else:
                        order = await self.exchange.create_market_sell_order(
                            intent.symbol, intent.quantity
                        )
                else:  # LIMIT
                    if intent.side == OrderSide.BUY:
                        order = await self.exchange.create_limit_buy_order(
                            intent.symbol, intent.quantity, intent.price
                        )
                    else:
                        order = await self.exchange.create_limit_sell_order(
                            intent.symbol, intent.quantity, intent.price
                        )
                
                return OrderResult(
                    success=True,
                    order_id=order['id'],
                    client_order_id=intent.client_order_id,
                    symbol=intent.symbol,
                    side=intent.side.value,
                    filled_quantity=order.get('filled', intent.quantity),
                    average_price=order.get('average', order.get('price', 0)),
                    fees=order.get('fee', {}).get('cost', 0),
                    slippage_bps=0,  # Calculated post-fill
                    timestamp=datetime.now(),
                    raw_response=order
                )
                
            except Exception as e:
                logger.warning(f"Order attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return OrderResult(
                        success=False,
                        order_id="",
                        client_order_id=intent.client_order_id,
                        symbol=intent.symbol,
                        side=intent.side.value,
                        filled_quantity=0,
                        average_price=0,
                        fees=0,
                        slippage_bps=0,
                        timestamp=datetime.now(),
                        error_message=str(e)
                    )
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        return await self.exchange.fetch_open_orders(symbol)
    
    async def get_balance(self) -> Dict:
        """
        Get balance from EXCHANGE ONLY.
        No local simulation, no fake balances.
        """
        balance = await self.exchange.fetch_balance()
        return balance.get('total', {})


class ExecutionRouter:
    """
    Central execution router that handles all order flow.
    
    CRYPTOBOSS 2.0: PAPER TRADING PERMANENTLY REMOVED
    
    Only two modes:
        - TESTNET: All testing via Binance Testnet
        - LIVE: Real trading via Binance Live
    
    Usage:
        router = ExecutionRouter(
            mode=ExecutionMode.TESTNET,
            exchange_client=binance_testnet_client
        )
        
        intent = OrderIntent(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            strategy_id="dca_btc_1"
        )
        
        result = await router.execute(intent)
    """
    
    # FORBIDDEN MODES - fail fast if attempted
    _FORBIDDEN_MODES = ["paper", "PAPER", "backtest", "BACKTEST", "simulation", "demo"]
    
    def __init__(
        self,
        mode: ExecutionMode,
        exchange_client,
        state_manager = None,
        risk_guardian = None
    ):
        # CRITICAL: Reject forbidden modes
        if isinstance(mode, str):
            if mode.lower() in [m.lower() for m in self._FORBIDDEN_MODES]:
                raise ValueError(
                    f"❌ FORBIDDEN MODE: '{mode}' is not allowed. "
                    "Paper trading has been permanently removed. "
                    "Use TESTNET for testing or LIVE for real trading."
                )
            # Convert string to enum
            mode = ExecutionMode(mode.lower())
        
        if not exchange_client:
            raise ValueError(
                "exchange_client is REQUIRED. "
                "No local simulation allowed - all trading goes through exchange."
            )
        
        self.mode = mode
        self.state_manager = state_manager
        self.risk_guardian = risk_guardian
        self._exchange_client = exchange_client
        
        # Initialize exchange broker (same for both modes)
        self.broker = ExchangeBroker(exchange_client, mode)
        
        # Callbacks
        self.on_fill: Optional[Callable[[OrderResult], None]] = None
        self.on_error: Optional[Callable[[OrderIntent, str], None]] = None
        
        # Lazy-loaded components
        self._slippage_monitor = None
        self._recovery_handler = None
        
        mode_emoji = "🔴" if mode == ExecutionMode.LIVE else "🟡"
        logger.info(f"{mode_emoji} ExecutionRouter initialized in {mode.value.upper()} mode")
    
    def _get_slippage_monitor(self):
        """Get or create SlippageMonitor."""
        if self._slippage_monitor is None:
            try:
                from .slippage_monitor import get_slippage_monitor
                self._slippage_monitor = get_slippage_monitor()
            except ImportError:
                pass
        return self._slippage_monitor
    
    def _get_recovery_handler(self):
        """Get or create ExchangeRecoveryHandler."""
        if self._recovery_handler is None:
            try:
                from .exchange_recovery import get_recovery_handler
                self._recovery_handler = get_recovery_handler()
            except ImportError:
                pass
        return self._recovery_handler
    
    async def execute(self, intent: OrderIntent) -> OrderResult:
        """
        Execute an order intent on the EXCHANGE.
        
        Flow:
        1. Validate with RiskGuardian (if present)
        2. Save pending order state
        3. Execute via exchange broker
        4. Track slippage quality
        5. Update state with result
        6. Trigger callbacks
        """
        expected_price = intent.price or 0
        
        # Risk check
        if self.risk_guardian:
            approved, reason = self.risk_guardian.approve_order(intent)
            if not approved:
                logger.warning(f"Order rejected by RiskGuardian: {reason}")
                return OrderResult(
                    success=False,
                    order_id="",
                    client_order_id=intent.client_order_id,
                    symbol=intent.symbol,
                    side=intent.side.value,
                    filled_quantity=0,
                    average_price=0,
                    fees=0,
                    slippage_bps=0,
                    timestamp=datetime.now(),
                    error_message=f"Risk rejected: {reason}"
                )
        
        # Save pending state
        if self.state_manager:
            self.state_manager.save_order(intent.client_order_id, {
                'symbol': intent.symbol,
                'side': intent.side.value,
                'quantity': intent.quantity,
                'status': 'pending',
                'strategy_id': intent.strategy_id,
                'mode': self.mode.value
            })
        
        # Execute on exchange
        result = None
        recovery = self._get_recovery_handler()
        
        if recovery:
            # Use recovery handler for robust execution
            async def execute_with_recovery():
                return await self.broker.execute_order(intent)
            
            recovery_result = await recovery.execute_with_recovery(
                execute_with_recovery,
                f"{intent.symbol}_{intent.client_order_id}"
            )
            
            if recovery_result.success and recovery_result.result:
                result = recovery_result.result
            else:
                # No fallback to paper mode - just fail
                result = OrderResult(
                    success=False,
                    order_id="",
                    client_order_id=intent.client_order_id,
                    symbol=intent.symbol,
                    side=intent.side.value,
                    filled_quantity=0,
                    average_price=0,
                    fees=0,
                    slippage_bps=0,
                    timestamp=datetime.now(),
                    error_message=recovery_result.error_message or "Exchange execution failed"
                )
        else:
            # Standard execution
            try:
                result = await self.broker.execute_order(intent)
            except Exception as e:
                logger.error(f"Order execution failed: {e}")
                result = OrderResult(
                    success=False,
                    order_id="",
                    client_order_id=intent.client_order_id,
                    symbol=intent.symbol,
                    side=intent.side.value,
                    filled_quantity=0,
                    average_price=0,
                    fees=0,
                    slippage_bps=0,
                    timestamp=datetime.now(),
                    error_message=str(e)
                )
        
        # Track slippage
        if result.success and expected_price > 0:
            slippage_monitor = self._get_slippage_monitor()
            if slippage_monitor:
                slippage_monitor.record_execution(
                    order_id=result.order_id,
                    symbol=intent.symbol,
                    side=intent.side.value,
                    expected_price=expected_price,
                    fill_price=result.average_price,
                    size=result.filled_quantity
                )
        
        # Update state
        if self.state_manager:
            self.state_manager.save_order(intent.client_order_id, {
                'symbol': intent.symbol,
                'side': intent.side.value,
                'quantity': intent.quantity,
                'status': 'filled' if result.success else 'failed',
                'fill_price': result.average_price,
                'fees': result.fees,
                'strategy_id': intent.strategy_id,
                'slippage_bps': result.slippage_bps,
                'mode': self.mode.value
            })
        
        # Callbacks
        if result.success and self.on_fill:
            self.on_fill(result)
        elif not result.success and self.on_error:
            self.on_error(intent, result.error_message)
        
        return result
    
    async def cancel(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order on exchange."""
        return await self.broker.cancel_order(order_id, symbol)
    
    async def get_balance(self) -> Dict:
        """
        Get current balances FROM EXCHANGE.
        No local simulation, no fake balances.
        """
        return await self.broker.get_balance()
    
    def switch_mode(self, mode: ExecutionMode, exchange_client):
        """
        Switch execution mode.
        
        CRITICAL: exchange_client is REQUIRED for both modes.
        There is no paper mode fallback.
        """
        if isinstance(mode, str):
            if mode.lower() in [m.lower() for m in self._FORBIDDEN_MODES]:
                raise ValueError(f"❌ FORBIDDEN MODE: '{mode}' is not allowed.")
            mode = ExecutionMode(mode.lower())
        
        if not exchange_client:
            raise ValueError("exchange_client is REQUIRED for mode switch.")
        
        self.mode = mode
        self._exchange_client = exchange_client
        self.broker = ExchangeBroker(exchange_client, mode)
        
        mode_emoji = "🔴" if mode == ExecutionMode.LIVE else "🟡"
        logger.warning(f"{mode_emoji} Execution mode switched to {mode.value.upper()}")


def validate_environment(mode: str) -> ExecutionMode:
    """
    Validate and convert environment string to ExecutionMode.
    
    Raises ValueError for forbidden modes (paper, backtest, etc.)
    """
    forbidden = ["paper", "backtest", "simulation", "demo", "mock"]
    
    if mode.lower() in forbidden:
        raise ValueError(
            f"❌ ENVIRONMENT '{mode}' IS NOT ALLOWED.\n"
            f"Paper trading and simulation modes have been permanently removed.\n"
            f"Valid environments: TESTNET, LIVE"
        )
    
    if mode.lower() == "testnet":
        return ExecutionMode.TESTNET
    elif mode.lower() == "live":
        return ExecutionMode.LIVE
    else:
        raise ValueError(f"Unknown environment: {mode}. Must be TESTNET or LIVE.")
