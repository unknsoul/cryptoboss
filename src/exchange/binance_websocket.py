"""Unified Binance WebSocket manager with reconnect and stream callbacks."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, DefaultDict, Dict, List, Optional

logger = logging.getLogger(__name__)


StreamCallback = Callable[[Dict[str, Any]], Optional[Awaitable[None]]]


class BinanceWebSocketManager:
    """Manages multiple Binance streams with per-stream callbacks."""

    def __init__(
        self,
        testnet: bool = False,
        futures: bool = False,
        max_streams: int = 50,
        max_reconnect_delay: int = 60,
        heartbeat_seconds: int = 30,
        queue_maxsize: int = 10000,
    ):
        self.testnet = bool(testnet)
        self.futures = bool(futures)
        self.max_streams = int(max_streams)
        self.max_reconnect_delay = int(max_reconnect_delay)
        self.heartbeat_seconds = int(heartbeat_seconds)

        if self.futures:
            self.base_url = (
                "wss://stream.binancefuture.com" if self.testnet else "wss://fstream.binance.com"
            )
        else:
            self.base_url = (
                "wss://testnet.binance.vision" if self.testnet else "wss://stream.binance.com:9443"
            )

        self._callbacks: DefaultDict[str, List[StreamCallback]] = defaultdict(list)
        self._tasks: Dict[str, asyncio.Task] = {}
        self._reconnect_attempts: Dict[str, int] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)

        self._running = False
        self._stop_event = asyncio.Event()

    @property
    def message_queue(self) -> asyncio.Queue:
        return self._message_queue

    @staticmethod
    def build_kline_stream(symbol: str, interval: str) -> str:
        return f"{symbol.lower()}@kline_{interval}"

    @staticmethod
    def build_depth_stream(symbol: str) -> str:
        return f"{symbol.lower()}@depth20@100ms"

    @staticmethod
    def build_agg_trade_stream(symbol: str) -> str:
        return f"{symbol.lower()}@aggTrade"

    @staticmethod
    def build_ticker_stream(symbol: str) -> str:
        return f"{symbol.lower()}@ticker"

    @staticmethod
    def build_mark_price_stream(symbol: str) -> str:
        return f"{symbol.lower()}@markPrice@1s"

    @staticmethod
    def build_force_order_stream(symbol: Optional[str] = None) -> str:
        if symbol:
            return f"{symbol.lower()}@forceOrder"
        return "!forceOrder@arr"

    async def start(self):
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        for stream_name in list(self._callbacks.keys()):
            if stream_name not in self._tasks:
                self._tasks[stream_name] = asyncio.create_task(self._stream_loop(stream_name))

    async def stop(self):
        self._running = False
        self._stop_event.set()

        tasks = list(self._tasks.values())
        self._tasks.clear()

        for task in tasks:
            task.cancel()

        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("WebSocket task shutdown error: %s", exc)

    async def subscribe(self, stream_name: str, callback: StreamCallback):
        stream_key = stream_name.strip()
        if not stream_key:
            raise ValueError("stream_name cannot be empty")

        if stream_key not in self._callbacks and len(self._callbacks) >= self.max_streams:
            raise RuntimeError("maximum stream count reached")

        if callback not in self._callbacks[stream_key]:
            self._callbacks[stream_key].append(callback)

        if self._running and stream_key not in self._tasks:
            self._tasks[stream_key] = asyncio.create_task(self._stream_loop(stream_key))

    async def unsubscribe(self, stream_name: str):
        stream_key = stream_name.strip()
        self._callbacks.pop(stream_key, None)

        task = self._tasks.pop(stream_key, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _handle_reconnect(self, stream_name: str) -> int:
        attempts = self._reconnect_attempts.get(stream_name, 0) + 1
        self._reconnect_attempts[stream_name] = attempts
        return min(2 ** (attempts - 1), self.max_reconnect_delay)

    async def next_message(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if timeout is None:
            return await self._message_queue.get()
        return await asyncio.wait_for(self._message_queue.get(), timeout=timeout)

    async def _stream_loop(self, stream_name: str):
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError("websockets package is required for BinanceWebSocketManager") from exc

        url = f"{self.base_url}/ws/{stream_name}"

        while self._running and stream_name in self._callbacks:
            heartbeat_task: Optional[asyncio.Task] = None
            try:
                async with websockets.connect(
                    url,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                    max_queue=2048,
                ) as ws:
                    self._reconnect_attempts[stream_name] = 0
                    heartbeat_task = asyncio.create_task(self._heartbeat(ws))

                    async for raw_message in ws:
                        if not self._running or stream_name not in self._callbacks:
                            break

                        payload = self._decode_message(raw_message)
                        await self._enqueue_message(payload)
                        await self._dispatch(stream_name, payload)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._running or stream_name not in self._callbacks:
                    break
                delay = self._handle_reconnect(stream_name)
                logger.warning(
                    "WebSocket stream %s disconnected (%s). Reconnecting in %ss",
                    stream_name,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            finally:
                if heartbeat_task:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

    async def _heartbeat(self, ws) -> None:
        while self._running and not self._stop_event.is_set():
            await asyncio.sleep(self.heartbeat_seconds)
            try:
                pong_waiter = await ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=10)
            except Exception:
                return

    async def _enqueue_message(self, payload: Dict[str, Any]) -> None:
        if self._message_queue.full():
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._message_queue.put(payload)

    async def _dispatch(self, stream_name: str, payload: Dict[str, Any]) -> None:
        callbacks = list(self._callbacks.get(stream_name, []))
        for callback in callbacks:
            try:
                result = callback(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                logger.error("Stream callback failed (%s): %s", stream_name, exc)

    @staticmethod
    def _decode_message(raw_message: Any) -> Dict[str, Any]:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")

        if isinstance(raw_message, str):
            try:
                return json.loads(raw_message)
            except json.JSONDecodeError:
                return {"raw": raw_message}

        if isinstance(raw_message, dict):
            return raw_message

        return {"raw": raw_message}
