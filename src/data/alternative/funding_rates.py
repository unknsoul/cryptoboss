"""Funding rate, open interest, and liquidation data fetchers."""

from __future__ import annotations

from typing import Optional, List

import pandas as pd
import requests


def _to_millis(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return int(pd.Timestamp(value).timestamp() * 1000)


def fetch_binance_funding_rates(
    symbol: str,
    start_time: Optional[object] = None,
    end_time: Optional[object] = None,
    limit: int = 1000,
    base_url: str = "https://fapi.binance.com",
) -> pd.DataFrame:
    """Fetch funding rate history from Binance Futures."""
    symbol = symbol.replace("/", "")
    url = f"{base_url}/fapi/v1/fundingRate"

    params = {
        "symbol": symbol,
        "limit": min(max(limit, 1), 1000),
    }
    start_ms = _to_millis(start_time)
    end_ms = _to_millis(end_time)
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    results: List[dict] = []
    while True:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json() or []
        if not payload:
            break
        results.extend(payload)

        last_time = int(payload[-1].get("fundingTime", 0))
        if end_ms is not None and last_time >= end_ms:
            break
        if len(payload) < params["limit"] or len(results) >= limit:
            break
        params["startTime"] = last_time + 1

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["symbol"] = symbol
    return df[["timestamp", "symbol", "funding_rate"]].dropna()


def fetch_open_interest_history(
    symbol: str,
    period: str = "5m",
    limit: int = 500,
    base_url: str = "https://fapi.binance.com",
) -> pd.DataFrame:
    """Fetch open interest history from Binance Futures."""
    symbol = symbol.replace("/", "")
    url = f"{base_url}/futures/data/openInterestHist"
    params = {
        "symbol": symbol,
        "period": period,
        "limit": min(max(limit, 1), 500),
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json() or []
    df = pd.DataFrame(payload)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df["symbol"] = symbol
    return df[["timestamp", "symbol", "open_interest"]].dropna()


def fetch_liquidation_history(
    symbol: str,
    start_time: Optional[object] = None,
    end_time: Optional[object] = None,
    limit: int = 500,
    base_url: str = "https://fapi.binance.com",
) -> pd.DataFrame:
    """Fetch liquidation orders from Binance Futures public endpoint."""
    symbol = symbol.replace("/", "")
    url = f"{base_url}/futures/data/liquidationOrders"
    params = {
        "symbol": symbol,
        "limit": min(max(limit, 1), 1000),
    }
    start_ms = _to_millis(start_time)
    end_ms = _to_millis(end_time)
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json() or []
    df = pd.DataFrame(payload)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["origQty"], errors="coerce")
    df["side"] = df.get("side", "").astype(str)
    df["symbol"] = symbol
    return df[["timestamp", "symbol", "side", "price", "quantity"]].dropna()
