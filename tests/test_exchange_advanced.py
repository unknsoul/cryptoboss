"""Phase 1 exchange layer tests for v13 master foundation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from src.exchange.binance_client import BinanceClient
from src.exchange.binance_futures import BinanceFuturesClient, TESTNET_REST
from src.exchange.binance_order_manager import AdvancedOrderManager
from src.exchange.binance_websocket import BinanceWebSocketManager
from src.exchange.rate_limiter import BinanceRateLimiter


class TestExchangeAdvanced:
    def test_websocket_reconnect_on_disconnect(self):
        manager = BinanceWebSocketManager(max_reconnect_delay=60)

        delays = [manager._handle_reconnect("btcusdt@kline_1m") for _ in range(8)]

        assert delays[0] == 1
        assert delays[1] == 2
        assert delays[2] == 4
        assert delays[3] == 8
        assert delays[-1] <= 60

    def test_rate_limiter_blocks_above_limit(self):
        async def _run():
            limiter = BinanceRateLimiter(
                request_weight_per_minute=2,
                raw_requests_per_5min=100,
                orders_per_second=10,
                orders_per_day=200000,
            )

            await limiter.acquire(weight=2)
            snapshot = limiter.snapshot()

            assert snapshot.request_weight_used_1m == 2
            assert limiter._wait_for_weight(__import__("time").monotonic(), 1) > 0

        asyncio.run(_run())

    def test_oco_order_structure_valid(self):
        async def _run():
            manager = AdvancedOrderManager(api_key="k", api_secret="s", testnet=True)

            captured = {}

            async def fake_request(method, path, params=None, **kwargs):
                captured["method"] = method
                captured["path"] = path
                captured["params"] = dict(params or {})
                captured["kwargs"] = dict(kwargs)
                return {"orderListId": 123}

            manager._request = fake_request  # type: ignore[assignment]

            result = await manager.place_oco(
                symbol="BTCUSDT",
                side="SELL",
                quantity=0.01,
                price=105000,
                stop_price=98000,
                stop_limit_price=97900,
            )

            assert result["orderListId"] == 123
            assert captured["method"] == "POST"
            assert captured["path"] == "/api/v3/order/oco"
            assert captured["params"]["symbol"] == "BTCUSDT"
            assert captured["params"]["stopLimitTimeInForce"] == "GTC"

        asyncio.run(_run())

    def test_futures_client_testnet_connectivity(self):
        async def _run():
            client = BinanceFuturesClient(api_key="k", api_secret="s", testnet=True)
            assert client.base_rest_url == TESTNET_REST

            async def fake_request(*args, **kwargs):
                return {}

            client._request = fake_request  # type: ignore[assignment]
            ok = await client.ping()
            assert ok is True

        asyncio.run(_run())

    def test_circuit_breaker_opens_on_errors(self):
        client = BinanceClient.__new__(BinanceClient)
        client._error_count = 0
        client._circuit_open = False
        client._circuit_open_at = None
        client._circuit_error_threshold = 2
        client._circuit_reset_seconds = 30

        client._register_request_failure()
        client._register_request_failure()

        assert client._circuit_open is True
        assert client._circuit_open_at is not None

        client._circuit_open_at = datetime.utcnow() - timedelta(seconds=35)
        assert client._circuit_breaker_check() is True
        assert client._circuit_open is False

    def test_trailing_stop_delta_calculation(self):
        async def _run():
            manager = AdvancedOrderManager(api_key="k", api_secret="s", testnet=True)

            captured = {}

            async def fake_reference_price(symbol: str) -> float:
                return 100.0

            async def fake_request(method, path, params=None, **kwargs):
                captured["method"] = method
                captured["path"] = path
                captured["params"] = dict(params or {})
                return {"orderId": 456}

            manager._get_reference_price = fake_reference_price  # type: ignore[assignment]
            manager._request = fake_request  # type: ignore[assignment]

            result = await manager.place_trailing_stop(
                symbol="BTCUSDT",
                side="SELL",
                quantity=0.02,
                trailing_delta_bips=100,
            )

            assert result["orderId"] == 456
            assert captured["path"] == "/api/v3/order"
            assert captured["params"]["type"] == "STOP_LOSS_LIMIT"
            assert captured["params"]["trailingDelta"] == 100

        asyncio.run(_run())

    def test_iceberg_order_qty_split(self):
        async def _run():
            manager = AdvancedOrderManager(api_key="k", api_secret="s", testnet=True)

            with pytest.raises(ValueError):
                await manager.place_iceberg(
                    symbol="BTCUSDT",
                    side="BUY",
                    total_qty=1.0,
                    iceberg_qty=1.2,
                    price=95000,
                )

        asyncio.run(_run())
