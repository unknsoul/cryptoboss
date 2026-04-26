"""Unit tests for BinanceClient normalization across SDK and ccxt paths."""

from __future__ import annotations

import pytest

from src.exchange.binance_client import BinanceClient


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio only; trio is not part of this repo runtime."""
    return "asyncio"


class FakeSdkExchange:
    """Small python-binance style stub for client normalization tests."""

    API_URL = "https://testnet.binance.vision/api"

    def __init__(self) -> None:
        self.timestamp_offset = 0

    def get_server_time(self) -> dict:
        return {"serverTime": 1700000000000}

    def get_account(self) -> dict:
        return {
            "balances": [
                {"asset": "USDT", "free": "125.50", "locked": "4.50"},
                {"asset": "BTC", "free": "0.00000000", "locked": "0.00000000"},
            ]
        }

    def get_exchange_info(self) -> dict:
        return {"symbols": [{"symbol": "BTCUSDT"}]}

    def get_ticker(self, symbol: str) -> dict:
        assert symbol == "BTCUSDT"
        return {
            "symbol": symbol,
            "lastPrice": "100.50",
            "bidPrice": "100.40",
            "askPrice": "100.60",
            "highPrice": "103.00",
            "lowPrice": "97.00",
            "volume": "999.0",
            "priceChangePercent": "1.2",
        }

    def get_klines(self, symbol: str, interval: str, limit: int) -> list[list[str | int]]:
        assert symbol == "BTCUSDT"
        assert interval == "1m"
        assert limit == 2
        return [
            [1700000000000, "99.0", "101.0", "98.5", "100.5", "12.0"],
            [1700000060000, "100.5", "102.0", "100.0", "101.5", "14.0"],
        ]

    def create_order(self, **kwargs: str | float) -> dict:
        return {
            "orderId": 42,
            "symbol": kwargs["symbol"],
            "side": kwargs["side"],
            "type": kwargs["type"],
            "status": "FILLED",
            "executedQty": "0.01000000",
            "cummulativeQuoteQty": "1.00500000",
            "fills": [{"price": "100.5", "qty": "0.01", "commission": "0.001"}],
        }

    def get_open_orders(self, **kwargs: str) -> list[dict]:
        _ = kwargs
        return [
            {
                "orderId": 43,
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "LIMIT",
                "origQty": "0.01000000",
                "executedQty": "0.00000000",
                "price": "101.0",
            }
        ]

    def get_order(self, **kwargs: str) -> dict:
        return {
            "orderId": kwargs["orderId"],
            "symbol": kwargs["symbol"],
            "side": "BUY",
            "type": "MARKET",
            "origQty": "0.01000000",
            "executedQty": "0.01000000",
            "cummulativeQuoteQty": "1.00500000",
            "price": "0.0",
        }

    def cancel_order(self, **kwargs: str) -> dict:
        return {"orderId": kwargs["orderId"], "symbol": kwargs["symbol"], "status": "CANCELED"}


class FakeCcxtExchange:
    """Small ccxt-style stub for balance normalization tests."""

    async def fetch_balance(self) -> dict:
        return {
            "free": {"USDT": 50.0, "BTC": 0.0},
            "used": {"USDT": 5.0, "BTC": 0.0},
            "total": {"USDT": 55.0, "BTC": 0.0},
        }


def _patch_sdk_client(monkeypatch: pytest.MonkeyPatch) -> BinanceClient:
    """Construct a BinanceClient with a fake python-binance exchange."""

    def fake_init_exchange(self: BinanceClient) -> None:
        self._use_binance_sdk = True
        self.exchange = FakeSdkExchange()

    monkeypatch.setattr(BinanceClient, "_init_exchange", fake_init_exchange)
    return BinanceClient(api_key="key", api_secret="secret", testnet=True)


@pytest.mark.anyio
async def test_sdk_validate_and_market_data_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK path should validate credentials and normalize ticker/candle payloads."""
    client = _patch_sdk_client(monkeypatch)

    validation = await client.validate_credentials()
    ticker = await client.get_ticker("BTC/USDT")
    candles = await client.get_ohlcv("BTC/USDT", timeframe="1m", limit=2)

    assert validation["success"] is True
    assert validation["balances"]["USDT"] == pytest.approx(130.0)
    assert ticker["last"] == pytest.approx(100.5)
    assert candles[-1][4] == pytest.approx(101.5)


@pytest.mark.anyio
async def test_sdk_order_paths_return_ccxt_like_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK path should normalize orders into the shape the dashboard expects."""
    client = _patch_sdk_client(monkeypatch)

    order = await client.create_order("BTC/USDT", "buy", "market", 0.01)
    open_orders = await client.get_open_orders()
    fetched = await client.fetch_order("43", "BTC/USDT")
    cancelled = await client.cancel_order("43", "BTC/USDT")

    assert order["id"] == "42"
    assert order["average"] == pytest.approx(100.5)
    assert open_orders[0]["symbol"] == "BTC/USDT"
    assert fetched["filled"] == pytest.approx(0.01)
    assert cancelled["id"] == "43"


@pytest.mark.anyio
async def test_ccxt_balance_normalization_uses_free_used_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ccxt path should preserve free, used, and total values for each asset."""
    client = _patch_sdk_client(monkeypatch)
    client._use_binance_sdk = False
    client.exchange = FakeCcxtExchange()

    balances = await client.get_balance()

    assert balances["USDT"]["free"] == pytest.approx(50.0)
    assert balances["USDT"]["used"] == pytest.approx(5.0)
    assert balances["USDT"]["total"] == pytest.approx(55.0)
