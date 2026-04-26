"""Unit coverage for paper execution routing and fills."""

import asyncio

from src.core.execution_router import ExecutionMode, ExecutionRouter, OrderIntent, OrderSide, OrderType


def test_paper_mode_no_exchange_client_needed() -> None:
    """Paper mode should initialize without an exchange client."""
    router = ExecutionRouter(mode=ExecutionMode.PAPER, exchange_client=None, portfolio_value=10000.0)
    assert router.mode == ExecutionMode.PAPER


def test_paper_buy_order_fills() -> None:
    """A paper buy order should fill and produce a positive quantity."""
    async def run():
        router = ExecutionRouter(mode=ExecutionMode.PAPER, exchange_client=None, portfolio_value=10000.0)
        router.set_price("BTC/USDT", 100.0)
        result = await router.execute(
            OrderIntent(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.2,
                strategy_id="test",
            )
        )
        assert result.success is True
        assert result.filled_quantity == 0.2
        assert result.average_price >= 100.0

    asyncio.run(run())


def test_paper_sell_order_fills_with_adverse_slippage() -> None:
    """A paper sell order should fill below the mid price in adverse-only mode."""
    async def run():
        router = ExecutionRouter(mode=ExecutionMode.PAPER, exchange_client=None, portfolio_value=10000.0)
        router.set_price("BTC/USDT", 100.0)
        result = await router.execute(
            OrderIntent(
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=0.2,
                strategy_id="test",
            )
        )
        assert result.success is True
        assert result.average_price <= 100.0
        assert result.slippage_bps == 5.0

    asyncio.run(run())


def test_slippage_always_adverse() -> None:
    """Repeated paper fills should never improve on the quoted mid price."""
    async def run():
        router = ExecutionRouter(mode=ExecutionMode.PAPER, exchange_client=None, portfolio_value=10000.0)
        router.set_price("BTC/USDT", 100.0)
        buy_prices = []
        for _ in range(25):
            result = await router.execute(
                OrderIntent(
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=0.1,
                    strategy_id="test",
                )
            )
            buy_prices.append(result.average_price)
        assert all(price >= 100.0 for price in buy_prices)

    asyncio.run(run())


def test_fees_deducted() -> None:
    """Paper fills should charge the configured taker fee on notional value."""
    async def run():
        router = ExecutionRouter(mode=ExecutionMode.PAPER, exchange_client=None, portfolio_value=10000.0)
        router.set_price("BTC/USDT", 100.0)
        result = await router.execute(
            OrderIntent(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.5,
                strategy_id="test",
            )
        )
        expected_fee = result.filled_quantity * result.average_price * 0.001
        assert round(result.fees, 8) == round(expected_fee, 8)

    asyncio.run(run())
