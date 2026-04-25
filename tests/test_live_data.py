import asyncio
import os

import pytest

from src.exchange.binance_client import MarketDataService


pytestmark = pytest.mark.integration


def test_live_price_feed_smoke():
    if os.getenv("RUN_LIVE_EXCHANGE_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_EXCHANGE_TESTS=1 to run the live exchange smoke test")

    events = []
    service = MarketDataService(
        exchange_account_id="test-suite",
        testnet=False,
        poll_interval=0.25,
    )

    def on_price(event):
        events.append(event)

    async def run_smoke():
        service.subscribe("BTCUSDT", on_price)
        started = await service.start()
        assert started is True

        try:
            for _ in range(20):
                if events:
                    break
                await asyncio.sleep(0.25)
        finally:
            await service.stop()

    asyncio.run(run_smoke())

    assert events
    assert events[0]["symbol"] == "BTCUSDT"
    assert events[0]["price"] > 0
