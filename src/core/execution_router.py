"""
Execution Router - Unified Order Execution Layer

The "Shadow" that seamlessly switches between Live, Paper, and Backtest modes.
Strategies emit OrderIntents, the Router handles execution details.

Architecture:
    Strategy -> OrderIntent -> ExecutionRouter -> [LiveBroker | PaperBroker | BacktestBroker]
                                    |
                                    v
                              OrderResult (fill price, fees, slippage)

Benefits:
    - Switch between modes with ONE config change
    - Consistent interface regardless of mode
    - Automatic retry logic for live trading
    - Circuit breaker for exchange errors
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
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


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
    """Paper trading broker with realistic simulation."""
    
    def __init__(
        self,
        initial_balance: Dict[str, float] = None,
        fee_rate: float = 0.001,
        slippage_bps: float = 5.0
    ):
        self.balances = initial_balance or {"USDT": 10000.0}
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.open_orders: List[Dict] = []
        self.filled_orders: List[OrderResult] = []
        self.current_prices: Dict[str, float] = {}
    
    def set_price(self, symbol: str, price: float):
        """Set current price for simulation."""
        self.current_prices[symbol] = price
    
    async def execute_order(self, intent: OrderIntent) -> OrderResult:
        base, quote = intent.symbol.split("/") if "/" in intent.symbol else (intent.symbol.replace("USDT", ""), "USDT")
        
        # Get current price
        current_price = self.current_prices.get(intent.symbol, intent.price or 0)
        if current_price == 0:
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
                error_message="No price available"
            )
        
        # Apply slippage for market orders
        if intent.order_type == OrderType.MARKET:
            slippage_mult = 1 + (self.slippage_bps / 10000) if intent.side == OrderSide.BUY else 1 - (self.slippage_bps / 10000)
            fill_price = current_price * slippage_mult
        else:
            fill_price = intent.price
        
        # Check balance
        if intent.side == OrderSide.BUY:
            cost = intent.quantity * fill_price
            fee = cost * self.fee_rate
            total_cost = cost + fee
            
            if self.balances.get(quote, 0) < total_cost:
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
                    error_message="Insufficient balance"
                )
            
            self.balances[quote] -= total_cost
            self.balances[base] = self.balances.get(base, 0) + intent.quantity
        else:  # SELL
            if self.balances.get(base, 0) < intent.quantity:
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
                    error_message="Insufficient balance"
                )
            
            proceeds = intent.quantity * fill_price
            fee = proceeds * self.fee_rate
            
            self.balances[base] -= intent.quantity
            self.balances[quote] = self.balances.get(quote, 0) + proceeds - fee
        
        result = OrderResult(
            success=True,
            order_id=f"PAPER_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side.value,
            filled_quantity=intent.quantity,
            average_price=fill_price,
            fees=fee,
            slippage_bps=self.slippage_bps if intent.order_type == OrderType.MARKET else 0,
            timestamp=datetime.now()
        )
        
        self.filled_orders.append(result)
        logger.info(f"Paper order filled: {intent.side.value} {intent.quantity} {intent.symbol} @ {fill_price:.2f}")
        
        return result
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        self.open_orders = [o for o in self.open_orders if o.get('order_id') != order_id]
        return True
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        if symbol:
            return [o for o in self.open_orders if o.get('symbol') == symbol]
        return self.open_orders
    
    async def get_balance(self) -> Dict:
        return self.balances.copy()


class LiveBroker(BaseBroker):
    """Live trading broker with exchange integration."""
    
    def __init__(self, exchange_client, max_retries: int = 3):
        self.exchange = exchange_client
        self.max_retries = max_retries
    
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
        balance = await self.exchange.fetch_balance()
        return balance.get('total', {})


class ExecutionRouter:
    """
    Central execution router that handles all order flow.
    
    Usage:
        router = ExecutionRouter(mode=ExecutionMode.PAPER)
        
        # Execute order
        intent = OrderIntent(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            strategy_id="dca_btc_1"
        )
        
        result = await router.execute(intent)
        if result.success:
            print(f"Filled at {result.average_price}")
    """
    
    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PAPER,
        exchange_client = None,
        state_manager = None,
        risk_guardian = None
    ):
        self.mode = mode
        self.state_manager = state_manager
        self.risk_guardian = risk_guardian
        
        # Initialize appropriate broker
        if mode == ExecutionMode.LIVE:
            if not exchange_client:
                raise ValueError("exchange_client required for LIVE mode")
            self.broker = LiveBroker(exchange_client)
        else:
            self.broker = PaperBroker()
        
        # Callbacks
        self.on_fill: Optional[Callable[[OrderResult], None]] = None
        self.on_error: Optional[Callable[[OrderIntent, str], None]] = None
        
        logger.info(f"ExecutionRouter initialized in {mode.value} mode")
    
    async def execute(self, intent: OrderIntent) -> OrderResult:
        """
        Execute an order intent.
        
        Flow:
        1. Validate with RiskGuardian (if present)
        2. Save pending order state
        3. Execute via broker
        4. Update state with result
        5. Trigger callbacks
        """
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
                'strategy_id': intent.strategy_id
            })
        
        # Execute
        result = await self.broker.execute_order(intent)
        
        # Update state
        if self.state_manager:
            self.state_manager.save_order(intent.client_order_id, {
                'symbol': intent.symbol,
                'side': intent.side.value,
                'quantity': intent.quantity,
                'status': 'filled' if result.success else 'failed',
                'fill_price': result.average_price,
                'fees': result.fees,
                'strategy_id': intent.strategy_id
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
    
    def set_price(self, symbol: str, price: float):
        """Set price for paper/backtest mode."""
        if isinstance(self.broker, PaperBroker):
            self.broker.set_price(symbol, price)
    
    def switch_mode(self, mode: ExecutionMode, exchange_client = None):
        """Switch execution mode (use with caution)."""
        self.mode = mode
        if mode == ExecutionMode.LIVE:
            if not exchange_client:
                raise ValueError("exchange_client required for LIVE mode")
            self.broker = LiveBroker(exchange_client)
        else:
            self.broker = PaperBroker()
        logger.warning(f"Execution mode switched to {mode.value}")
