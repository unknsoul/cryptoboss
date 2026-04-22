import asyncio

from src.core.event_bus import EventType
from src.data.pipeline import DataPipeline, DataPipelineConfig


class FakeExchange:
    async def fetch_ohlcv(self, symbol, timeframe, limit=500):
        base = 1710000000000
        rows = []
        for i in range(limit):
            price = 50000 + i
            rows.append(
                [
                    base + (i * 60_000),
                    price,
                    price + 5,
                    price - 5,
                    price + 2,
                    100 + i,
                ]
            )
        return rows


class RecordingBus:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)


class FlakyPipeline(DataPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempts = 0

    async def start_websocket_streams(self, symbols, timeframes):
        self.attempts += 1
        if self.attempts < 3:
            raise RuntimeError("simulated websocket disconnect")


def test_historical_fetch_returns_correct_shape():
    pipeline = DataPipeline(
        event_bus=RecordingBus(),
        exchange_client=FakeExchange(),
        config=DataPipelineConfig(historical_lookback_bars=5),
    )

    df = asyncio.run(pipeline.fetch_historical_ohlcv("BTC/USDT", "1m", lookback_bars=5))

    assert list(df.columns) == ["open", "high", "low", "close", "volume", "timestamp"]
    assert len(df) == 5
    assert df["timestamp"].is_monotonic_increasing


def test_candle_close_fires_event():
    bus = RecordingBus()
    pipeline = DataPipeline(
        event_bus=bus,
        exchange_client=FakeExchange(),
        config=DataPipelineConfig(historical_lookback_bars=5),
    )

    candle = [1710000600000, 50100, 50110, 50090, 50105, 123]
    asyncio.run(pipeline.on_candle_close("BTC/USDT", "1m", candle))

    assert len(bus.events) == 1
    assert bus.events[0].event_type == EventType.OHLCV_UPDATED

    latest = pipeline.get_latest("BTC/USDT", "1m", n_bars=1)
    assert len(latest) == 1
    assert latest.iloc[-1]["close"] == 50105


def test_websocket_reconnect_on_disconnect():
    pipeline = FlakyPipeline(
        event_bus=RecordingBus(),
        exchange_client=FakeExchange(),
        config=DataPipelineConfig(
            max_reconnect_attempts=5,
            reconnect_backoff_base_seconds=0,
            historical_lookback_bars=5,
        ),
    )

    recovered = asyncio.run(pipeline.reconnect_on_failure())

    assert recovered is True
    assert pipeline.attempts == 3
