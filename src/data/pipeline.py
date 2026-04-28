"""Unified multi-timeframe OHLCV data pipeline."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import pandas as pd

try:
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover - optional dependency fallback
    AsyncRetrying = None
    retry_if_exception_type = None
    stop_after_attempt = None
    wait_exponential = None

from src.core.event_bus import Event, EventBus, EventType, get_event_bus
from src.data.schema import OHLCV_COLUMNS, standardize_ohlcv

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DataPipelineConfig:
    """Configuration for DataPipeline runtime behavior."""

    cache_backend: str = "redis"
    redis_url: str = "redis://localhost:6379/0"
    max_reconnect_attempts: int = 10
    reconnect_backoff_base_seconds: float = 2.0
    historical_lookback_bars: int = 500
    max_bars_in_memory: int = 2_000


class DataPipeline:
    """Asynchronous OHLCV ingestion and distribution pipeline."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        exchange_client: Optional[Any] = None,
        redis_client: Optional[Any] = None,
        config: Optional[DataPipelineConfig] = None,
    ) -> None:
        """Create a new data pipeline instance."""
        self.config = config or DataPipelineConfig()
        self.event_bus = event_bus or get_event_bus()
        self.exchange_client = exchange_client
        self.redis_client = redis_client

        self._cache: dict[str, dict[str, pd.DataFrame]] = {}
        self._stream_tasks: list[asyncio.Task] = []
        self._running = False

        self._active_symbols: list[str] = []
        self._active_timeframes: list[str] = []

    async def start_websocket_streams(self, symbols: list[str], timeframes: list[str]) -> None:
        """Start websocket streams and seed each stream with historical candles."""
        self._active_symbols = list(symbols)
        self._active_timeframes = list(timeframes)

        if self.exchange_client is None:
            raise RuntimeError("exchange_client is required for DataPipeline")

        for symbol in symbols:
            for timeframe in timeframes:
                historical = await self.fetch_historical_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_bars=self.config.historical_lookback_bars,
                )
                self._set_cache(symbol, timeframe, historical)

        if not hasattr(self.exchange_client, "watch_ohlcv"):
            logger.info("Exchange client does not expose watch_ohlcv; historical-only mode active")
            return

        self._running = True
        for symbol in symbols:
            for timeframe in timeframes:
                task = asyncio.create_task(self._stream_loop(symbol, timeframe))
                self._stream_tasks.append(task)

    async def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 500,
    ) -> pd.DataFrame:
        """Fetch and normalize historical OHLCV data."""
        if self.exchange_client is None or not hasattr(self.exchange_client, "fetch_ohlcv"):
            raise RuntimeError("exchange_client with fetch_ohlcv is required")

        async def _fetch() -> Any:
            return await self.exchange_client.fetch_ohlcv(
                symbol,
                timeframe,
                limit=lookback_bars,
            )

        raw_data = await self._call_exchange(_fetch, max_attempts=3)
        normalized = self.normalize_ohlcv(raw_data)
        self._set_cache(symbol, timeframe, normalized)
        await self._cache_to_redis(symbol, timeframe, normalized)
        return normalized

    def normalize_ohlcv(self, raw_data: Any) -> pd.DataFrame:
        """Normalize raw OHLCV rows into a canonical DataFrame schema."""
        return standardize_ohlcv(raw_data)

    async def on_candle_close(self, symbol: str, timeframe: str, candle: Any) -> None:
        """Handle a closed candle update and emit the OHLCV update event."""
        incoming = self.normalize_ohlcv([candle])

        existing = self._cache.get(symbol, {}).get(timeframe)
        if existing is None or existing.empty:
            merged = incoming
        else:
            merged = pd.concat([existing, incoming], ignore_index=True)
            merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
            merged = merged.sort_values("timestamp")

        merged = merged.tail(self.config.max_bars_in_memory).reset_index(drop=True)
        self._set_cache(symbol, timeframe, merged)
        await self._cache_to_redis(symbol, timeframe, merged)

        event = Event(
            event_type=EventType.OHLCV_UPDATED,
            source="DataPipeline",
            data={
                "symbol": symbol,
                "timeframe": timeframe,
                "latest": incoming.iloc[-1].to_dict(),
                "rows": len(merged),
            },
        )
        self.event_bus.publish(event)

    def get_latest(self, symbol: str, timeframe: str, n_bars: int = 200) -> pd.DataFrame:
        """Return latest candles from in-memory cache."""
        frame = self._cache.get(symbol, {}).get(timeframe)
        if frame is None:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        return frame.tail(n_bars).reset_index(drop=True)

    async def reconnect_on_failure(self) -> bool:
        """Reconnect websocket streams with exponential backoff."""
        symbols = self._active_symbols
        timeframes = self._active_timeframes

        for attempt in range(1, self.config.max_reconnect_attempts + 1):
            try:
                await self.start_websocket_streams(symbols, timeframes)
                logger.info("DataPipeline reconnect succeeded on attempt %s", attempt)
                return True
            except Exception as exc:  # pylint: disable=broad-except
                delay = self.config.reconnect_backoff_base_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Reconnect attempt %s/%s failed: %s",
                    attempt,
                    self.config.max_reconnect_attempts,
                    exc,
                )
                if attempt >= self.config.max_reconnect_attempts:
                    break
                await asyncio.sleep(delay)

        logger.error("DataPipeline reconnect failed after %s attempts", self.config.max_reconnect_attempts)
        return False

    async def stop(self) -> None:
        """Stop stream tasks and close exchange client if supported."""
        self._running = False
        for task in self._stream_tasks:
            task.cancel()

        if self._stream_tasks:
            await asyncio.gather(*self._stream_tasks, return_exceptions=True)
        self._stream_tasks = []

        if self.exchange_client is not None and hasattr(self.exchange_client, "close"):
            close_result = self.exchange_client.close()
            if asyncio.iscoroutine(close_result):
                await close_result

    async def _stream_loop(self, symbol: str, timeframe: str) -> None:
        """Consume websocket OHLCV stream for one symbol/timeframe pair."""
        while self._running:
            try:
                async def _watch() -> Any:
                    return await self.exchange_client.watch_ohlcv(symbol, timeframe)

                candles = await self._call_exchange(_watch, max_attempts=3)
                if candles:
                    await self.on_candle_close(symbol, timeframe, candles[-1])
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Stream error for %s %s: %s", symbol, timeframe, exc)
                await asyncio.sleep(0.5)

    async def _call_exchange(self, fn: Callable[[], Awaitable[Any]], max_attempts: int = 3) -> Any:
        """Execute exchange I/O with exponential retry semantics."""
        if AsyncRetrying is None:
            return await fn()

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=0.1, max=4),
            retry=retry_if_exception_type(Exception),
        ):
            with attempt:
                return await fn()

        raise RuntimeError("Exchange call retries exhausted")

    def _set_cache(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
        """Store normalized frame in in-memory cache."""
        self._cache.setdefault(symbol, {})[timeframe] = frame

    async def _cache_to_redis(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
        """Best-effort cache write to Redis backend."""
        if self.redis_client is None:
            return

        key = f"ohlcv:{symbol}:{timeframe}"
        payload = frame.to_json(orient="records", date_format="iso")

        try:
            result = self.redis_client.set(key, payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to cache OHLCV in Redis for %s %s: %s", symbol, timeframe, exc)
