"""
Execution Router - Unified Order Execution Layer

Supports three execution modes:
    - PAPER: Local simulation with configurable slippage (no exchange needed)
    - TESTNET: Real orders on Binance Testnet
    - LIVE: Real orders on Binance Live

Architecture:
    Strategy -> OrderIntent -> ExecutionRouter -> [PaperBroker | ExchangeBroker]
                                    |
                                    v
                              OrderResult (fill price, fees, slippage)

CRITICAL RULES:
    - PAPER mode does NOT require an exchange client
    - TESTNET and LIVE modes REQUIRE a real exchange client
    - All trades pass through RiskGuardian before execution
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
    """Execution mode for the router."""
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


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
    """Abstract broker interface."""

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


class PaperBroker(BaseBroker):
    """
    Paper trading broker — simulates order execution locally.

    Features:
        - Configurable slippage (default 0.05%)
        - Configurable fee rate (default 0.1%)
        - Position tracking
        - Balance tracking
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        slippage_pct: float = 0.05,
        fee_rate: float = 0.001,
    ):
        self._balance = {"USDT": initial_balance}
        self._positions: Dict[str, float] = {}  # symbol -> quantity
        self._open_orders: List[Dict] = []
        self._order_counter = 0
        self.slippage_pct = slippage_pct
        self.slippage_bps = max(float(slippage_pct) * 100.0, 5.0)
        self.fee_rate = fee_rate
        self._prices: Dict[str, float] = {}

        logger.info(
            f"📄 PAPER BROKER initialized — balance=${initial_balance:,.2f}, "
            f"slippage={slippage_pct}%, fee_rate={fee_rate * 100}%"
        )

    def set_price(self, symbol: str, price: float):
        """Update the current price for a symbol (called by engine on each tick)."""
        self._prices[symbol] = price

    async def execute_order(self, intent: OrderIntent) -> OrderResult:
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"

        # Get current price
        base_price = intent.price or self._prices.get(intent.symbol, 0)
        if base_price <= 0:
            return OrderResult(
                success=False,
                order_id=order_id,
                client_order_id=intent.client_order_id,
                symbol=intent.symbol,
                side=intent.side.value,
                filled_quantity=0,
                average_price=0,
                fees=0,
                slippage_bps=0,
                timestamp=datetime.now(),
                error_message=f"No price available for {intent.symbol}",
            )

        # Apply adverse-only slippage in paper mode.
        slip_ratio = self.slippage_bps / 10000.0
        if intent.side == OrderSide.BUY:
            fill_price = base_price * (1.0 + slip_ratio)
        else:
            fill_price = base_price * (1.0 - slip_ratio)
        slippage_bps = self.slippage_bps

        # Calculate fees
        trade_value = intent.quantity * fill_price
        fees = trade_value * self.fee_rate

        # Validate balance
        if intent.side == OrderSide.BUY:
            required = trade_value + fees
            available = self._balance.get("USDT", 0)
            if required > available:
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    client_order_id=intent.client_order_id,
                    symbol=intent.symbol,
                    side=intent.side.value,
                    filled_quantity=0,
                    average_price=0,
                    fees=0,
                    slippage_bps=0,
                    timestamp=datetime.now(),
                    error_message=f"Insufficient balance: need ${required:.2f}, have ${available:.2f}",
                )

        # Execute fill
        base_asset = intent.symbol.split("/")[0] if "/" in intent.symbol else intent.symbol.replace("USDT", "")

        if intent.side == OrderSide.BUY:
            self._balance["USDT"] = self._balance.get("USDT", 0) - trade_value - fees
            self._positions[base_asset] = self._positions.get(base_asset, 0) + intent.quantity
        else:
            current_pos = self._positions.get(base_asset, 0)
            if intent.quantity > current_pos and not intent.reduce_only:
                # Allow short selling in paper mode
                pass
            self._positions[base_asset] = current_pos - intent.quantity
            self._balance["USDT"] = self._balance.get("USDT", 0) + trade_value - fees

        logger.info(
            f"📄 PAPER FILL: {intent.side.value.upper()} {intent.quantity} {intent.symbol} "
            f"@ ${fill_price:,.2f} (slip={slippage_bps:.1f}bps, fee=${fees:.2f})"
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side.value,
            filled_quantity=intent.quantity,
            average_price=fill_price,
            fees=fees,
            slippage_bps=slippage_bps,
            timestamp=datetime.now(),
            raw_response={"mode": "paper", "balance": dict(self._balance)},
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        self._open_orders = [o for o in self._open_orders if o.get("id") != order_id]
        return True

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        if symbol:
            return [o for o in self._open_orders if o.get("symbol") == symbol]
        return list(self._open_orders)

    async def get_balance(self) -> Dict:
        result = dict(self._balance)
        # Add position values
        for asset, qty in self._positions.items():
            if qty != 0:
                result[asset] = qty
        return result


class ExchangeBroker(BaseBroker):
    """
    Exchange broker for both TESTNET and LIVE.

    The same broker handles both modes — the only difference
    is which exchange endpoint is used.
    """

    def __init__(self, exchange_client, mode: ExecutionMode, max_retries: int = 3):
        self.exchange = exchange_client
        self.mode = mode
        self.max_retries = max_retries

        if mode == ExecutionMode.LIVE:
            logger.warning("⚠️ LIVE BROKER INITIALIZED — REAL MONEY MODE")
        else:
            logger.info("📋 TESTNET BROKER INITIALIZED — Testing Mode")

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
                    slippage_bps=0,
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
                await asyncio.sleep(1 * (attempt + 1))

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

    Three modes:
        - PAPER: Simulated execution (no exchange needed)
        - TESTNET: All testing via Binance Testnet
        - LIVE: Real trading via Binance Live

    Usage:
        router = ExecutionRouter(
            mode=ExecutionMode.PAPER,
            exchange_client=None,  # paper mode — no client needed
            portfolio_value=10000.0
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

    # Minimum order sizes per symbol (quote currency)
    MIN_ORDER_VALUE_USD = 10.0

    def __init__(
        self,
        mode: ExecutionMode,
        exchange_client=None,
        state_manager=None,
        risk_guardian=None,
        portfolio_value: float = 10000.0,
    ):
        # Convert string to enum if needed
        if isinstance(mode, str):
            mode = ExecutionMode(mode.lower())

        # Validate: TESTNET/LIVE require exchange_client
        if mode in (ExecutionMode.TESTNET, ExecutionMode.LIVE) and not exchange_client:
            raise ValueError(
                f"exchange_client is REQUIRED for {mode.value.upper()} mode. "
                "Use PAPER mode for trading without an exchange connection."
            )

        self.mode = mode
        self.state_manager = state_manager
        self.risk_guardian = risk_guardian
        self._exchange_client = exchange_client

        # Position tracking
        self._positions: Dict[str, float] = {}  # symbol -> net quantity

        # Initialize broker based on mode
        if mode == ExecutionMode.PAPER:
            self.broker = PaperBroker(initial_balance=portfolio_value)
        else:
            self.broker = ExchangeBroker(exchange_client, mode)

        # Callbacks
        self.on_fill: Optional[Callable[[OrderResult], None]] = None
        self.on_error: Optional[Callable[[OrderIntent, str], None]] = None

        # Lazy-loaded components
        self._slippage_monitor = None
        self._recovery_handler = None

        mode_emoji = {"paper": "📄", "testnet": "🟡", "live": "🔴"}
        logger.info(
            f"{mode_emoji.get(mode.value, '❓')} ExecutionRouter initialized "
            f"in {mode.value.upper()} mode"
        )

    def set_price(self, symbol: str, price: float):
        """Update price in paper broker. No-op for exchange modes."""
        if isinstance(self.broker, PaperBroker):
            self.broker.set_price(symbol, price)

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

    def _validate_order(self, intent: OrderIntent) -> Optional[str]:
        """
        Validate an order intent before execution.

        Returns:
            None if valid, error message string if invalid.
        """
        if not intent.symbol:
            return "Symbol is required"

        if intent.quantity <= 0:
            return f"Invalid quantity: {intent.quantity}"

        if intent.order_type == OrderType.LIMIT and (intent.price is None or intent.price <= 0):
            return "Limit orders require a positive price"

        # Min order value check
        est_price = intent.price or 0
        if est_price > 0:
            order_value = intent.quantity * est_price
            if order_value < self.MIN_ORDER_VALUE_USD:
                return f"Order value ${order_value:.2f} below minimum ${self.MIN_ORDER_VALUE_USD}"

        return None

    async def execute(self, intent: OrderIntent) -> OrderResult:
        """
        Execute an order intent.

        Flow:
        1. Validate order parameters
        2. Check with RiskGuardian (if present)
        3. Save pending order state
        4. Execute via broker
        5. Track slippage quality
        6. Update positions
        7. Update state with result
        8. Trigger callbacks
        """
        expected_price = intent.price or 0

        # Order validation
        validation_error = self._validate_order(intent)
        if validation_error:
            logger.warning(f"Order validation failed: {validation_error}")
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
                error_message=f"Validation failed: {validation_error}",
            )

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
                    error_message=f"Risk rejected: {reason}",
                )

        # Save pending state
        if self.state_manager:
            self.state_manager.save_order(intent.client_order_id, {
                'symbol': intent.symbol,
                'side': intent.side.value,
                'quantity': intent.quantity,
                'status': 'pending',
                'strategy_id': intent.strategy_id,
                'mode': self.mode.value,
            })

        # Execute via broker
        result = None

        if self.mode == ExecutionMode.PAPER:
            # Paper mode: direct execution, no recovery handler needed
            try:
                result = await self.broker.execute_order(intent)
            except Exception as e:
                logger.error(f"Paper order execution failed: {e}")
                result = OrderResult(
                    success=False, order_id="", client_order_id=intent.client_order_id,
                    symbol=intent.symbol, side=intent.side.value,
                    filled_quantity=0, average_price=0, fees=0, slippage_bps=0,
                    timestamp=datetime.now(), error_message=str(e),
                )
        else:
            # Exchange mode: use recovery handler if available
            recovery = self._get_recovery_handler()

            if recovery:
                async def execute_with_recovery():
                    return await self.broker.execute_order(intent)

                recovery_result = await recovery.execute_with_recovery(
                    execute_with_recovery,
                    f"{intent.symbol}_{intent.client_order_id}",
                )

                if recovery_result.success and recovery_result.result:
                    result = recovery_result.result
                else:
                    result = OrderResult(
                        success=False, order_id="", client_order_id=intent.client_order_id,
                        symbol=intent.symbol, side=intent.side.value,
                        filled_quantity=0, average_price=0, fees=0, slippage_bps=0,
                        timestamp=datetime.now(),
                        error_message=recovery_result.error_message or "Exchange execution failed",
                    )
            else:
                try:
                    result = await self.broker.execute_order(intent)
                except Exception as e:
                    logger.error(f"Order execution failed: {e}")
                    result = OrderResult(
                        success=False, order_id="", client_order_id=intent.client_order_id,
                        symbol=intent.symbol, side=intent.side.value,
                        filled_quantity=0, average_price=0, fees=0, slippage_bps=0,
                        timestamp=datetime.now(), error_message=str(e),
                    )

        # Update position tracking
        if result.success:
            base_asset = intent.symbol.split("/")[0] if "/" in intent.symbol else intent.symbol
            if intent.side == OrderSide.BUY:
                self._positions[base_asset] = self._positions.get(base_asset, 0) + result.filled_quantity
            else:
                self._positions[base_asset] = self._positions.get(base_asset, 0) - result.filled_quantity

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
                    size=result.filled_quantity,
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
                'mode': self.mode.value,
            })

        # Callbacks
        if result.success and self.on_fill:
            self.on_fill(result)
        elif not result.success and self.on_error:
            self.on_error(intent, result.error_message)

        return result

    async def cancel(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        return await self.broker.cancel_order(order_id, symbol)

    async def get_balance(self) -> Dict:
        """Get current balances."""
        return await self.broker.get_balance()

    def get_positions(self) -> Dict[str, float]:
        """Get current tracked positions."""
        return dict(self._positions)

    def switch_mode(self, mode: ExecutionMode, exchange_client=None, portfolio_value: float = 10000.0):
        """
        Switch execution mode.

        CRITICAL: exchange_client is REQUIRED for TESTNET/LIVE modes.
        """
        if isinstance(mode, str):
            mode = ExecutionMode(mode.lower())

        if mode in (ExecutionMode.TESTNET, ExecutionMode.LIVE) and not exchange_client:
            raise ValueError(f"exchange_client is REQUIRED for {mode.value.upper()} mode.")

        self.mode = mode
        self._exchange_client = exchange_client

        if mode == ExecutionMode.PAPER:
            self.broker = PaperBroker(initial_balance=portfolio_value)
        else:
            self.broker = ExchangeBroker(exchange_client, mode)

        mode_emoji = {"paper": "📄", "testnet": "🟡", "live": "🔴"}
        logger.warning(
            f"{mode_emoji.get(mode.value, '❓')} Execution mode switched to {mode.value.upper()}"
        )


def validate_environment(mode: str) -> ExecutionMode:
    """
    Validate and convert environment string to ExecutionMode.
    """
    valid_modes = {"paper": ExecutionMode.PAPER, "testnet": ExecutionMode.TESTNET, "live": ExecutionMode.LIVE}

    if mode.lower() not in valid_modes:
        raise ValueError(
            f"Unknown environment: {mode}. Must be one of: {', '.join(valid_modes.keys())}"
        )

    return valid_modes[mode.lower()]
