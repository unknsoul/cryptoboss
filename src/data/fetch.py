"""Market data fetching helpers."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import pandas as pd

from src.data.schema import standardize_ohlcv
from src.data.validation import validate_ohlcv


async def _maybe_await(result: Any) -> Any:
    if asyncio.iscoroutine(result):
        return await result
    return result


def _normalize_ohlcv(raw_data: Any) -> pd.DataFrame:
    return standardize_ohlcv(raw_data)


class MarketDataFetcher:
    """Fetches OHLCV data via a supplied exchange client."""

    def __init__(self, exchange_client: Optional[Any] = None) -> None:
        self.exchange_client = exchange_client

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV data with async-aware exchange clients."""
        if self.exchange_client is None or not hasattr(self.exchange_client, "fetch_ohlcv"):
            raise RuntimeError("exchange_client with fetch_ohlcv is required")

        raw = await _maybe_await(self.exchange_client.fetch_ohlcv(symbol, timeframe, limit=limit))
        df = _normalize_ohlcv(raw)
        validate_ohlcv(df)
        return df

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV data with sync exchange clients."""
        if self.exchange_client is None or not hasattr(self.exchange_client, "fetch_ohlcv"):
            raise RuntimeError("exchange_client with fetch_ohlcv is required")

        raw = self.exchange_client.fetch_ohlcv(symbol, timeframe, limit=limit)
        if asyncio.iscoroutine(raw):
            raise RuntimeError("fetch_ohlcv returned coroutine; use fetch_ohlcv_async")

        df = _normalize_ohlcv(raw)
        validate_ohlcv(df)
        return df

    async def fetch_multi_timeframe_async(
        self,
        symbol: str,
        timeframes: list[str],
        limit: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple timeframes concurrently."""
        tasks = [self.fetch_ohlcv_async(symbol, timeframe, limit=limit) for timeframe in timeframes]
        results = await asyncio.gather(*tasks)
        return {tf: df for tf, df in zip(timeframes, results)}

    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: list[str],
        limit: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple timeframes sequentially (sync)."""
        data: Dict[str, pd.DataFrame] = {}
        for timeframe in timeframes:
            data[timeframe] = self.fetch_ohlcv(symbol, timeframe, limit=limit)
        return data
