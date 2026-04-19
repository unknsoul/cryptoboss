"""Binance API rate limiter with async token-window controls."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple


@dataclass
class RateLimitSnapshot:
    request_weight_used_1m: int
    raw_requests_used_5m: int
    orders_used_1s: int
    orders_used_1d: int


class BinanceRateLimiter:
    """
    Enforces key Binance limits using sliding windows.

    Defaults match public Binance docs:
    - request weight per minute: 1200
    - orders per second: 10
    - orders per day: 200000
    - raw requests per 5 minutes: 61000
    """

    def __init__(
        self,
        request_weight_per_minute: int = 1200,
        orders_per_second: int = 10,
        orders_per_day: int = 200000,
        raw_requests_per_5min: int = 61000,
    ):
        self.request_weight_per_minute = int(request_weight_per_minute)
        self.orders_per_second = int(orders_per_second)
        self.orders_per_day = int(orders_per_day)
        self.raw_requests_per_5min = int(raw_requests_per_5min)

        self._weight_requests: Deque[Tuple[float, int]] = deque()
        self._raw_requests: Deque[float] = deque()
        self._order_second_requests: Deque[float] = deque()
        self._order_day_requests: Deque[float] = deque()

        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        while self._weight_requests and now - self._weight_requests[0][0] >= 60.0:
            self._weight_requests.popleft()

        while self._raw_requests and now - self._raw_requests[0] >= 300.0:
            self._raw_requests.popleft()

        while self._order_second_requests and now - self._order_second_requests[0] >= 1.0:
            self._order_second_requests.popleft()

        while self._order_day_requests and now - self._order_day_requests[0] >= 86400.0:
            self._order_day_requests.popleft()

    def _wait_for_weight(self, now: float, weight: int) -> float:
        used = sum(w for _, w in self._weight_requests)
        if used + weight <= self.request_weight_per_minute:
            return 0.0
        oldest_ts = self._weight_requests[0][0]
        return max(0.0, 60.0 - (now - oldest_ts))

    def _wait_for_raw(self, now: float) -> float:
        if len(self._raw_requests) < self.raw_requests_per_5min:
            return 0.0
        oldest_ts = self._raw_requests[0]
        return max(0.0, 300.0 - (now - oldest_ts))

    def _wait_for_orders(self, now: float) -> float:
        waits = []

        if len(self._order_second_requests) >= self.orders_per_second:
            waits.append(max(0.0, 1.0 - (now - self._order_second_requests[0])))

        if len(self._order_day_requests) >= self.orders_per_day:
            waits.append(max(0.0, 86400.0 - (now - self._order_day_requests[0])))

        return max(waits) if waits else 0.0

    async def acquire(self, weight: int = 1, order: bool = False) -> None:
        """Wait until all relevant limits allow the request."""
        req_weight = max(1, int(weight))

        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)

                wait_weight = self._wait_for_weight(now, req_weight)
                wait_raw = self._wait_for_raw(now)
                wait_order = self._wait_for_orders(now) if order else 0.0
                wait_for = max(wait_weight, wait_raw, wait_order)

                if wait_for <= 0:
                    self._weight_requests.append((now, req_weight))
                    self._raw_requests.append(now)
                    if order:
                        self._order_second_requests.append(now)
                        self._order_day_requests.append(now)
                    return

            await asyncio.sleep(min(wait_for, 1.0))

    def snapshot(self) -> RateLimitSnapshot:
        now = time.monotonic()
        self._prune(now)
        return RateLimitSnapshot(
            request_weight_used_1m=sum(w for _, w in self._weight_requests),
            raw_requests_used_5m=len(self._raw_requests),
            orders_used_1s=len(self._order_second_requests),
            orders_used_1d=len(self._order_day_requests),
        )
