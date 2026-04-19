"""Advanced Binance spot order manager for complex execution patterns."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp

from .rate_limiter import BinanceRateLimiter


SPOT_TESTNET_REST = "https://testnet.binance.vision"
SPOT_LIVE_REST = "https://api.binance.com"


class AdvancedOrderManager:
    """Handles advanced order types like OCO, trailing, iceberg, TWAP, and VWAP."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        rate_limiter: Optional[BinanceRateLimiter] = None,
        timeout_seconds: int = 15,
    ):
        if testnet is None:
            mode = os.getenv("BINANCE_MODE", "testnet").strip().lower()
            testnet = mode != "live"

        self.testnet = bool(testnet)

        if api_key is None:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY" if self.testnet else "BINANCE_API_KEY", "")
        if api_secret is None:
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET" if self.testnet else "BINANCE_API_SECRET", "")

        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = SPOT_TESTNET_REST if self.testnet else SPOT_LIVE_REST
        self.timeout_seconds = int(timeout_seconds)

        self.rate_limiter = rate_limiter or BinanceRateLimiter()
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AdvancedOrderManager":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def _timestamp(self) -> int:
        return int(time.time() * 1000)

    def _sign(self, params: Dict[str, Any]) -> str:
        query = urlencode(params, doseq=True)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = True,
        weight: int = 1,
        order: bool = False,
    ) -> Any:
        await self.rate_limiter.acquire(weight=weight, order=order)
        session = await self._ensure_session()

        payload = dict(params or {})
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}

        if signed:
            payload["timestamp"] = self._timestamp()
            payload["recvWindow"] = int(payload.get("recvWindow", 5000))
            payload["signature"] = self._sign(payload)

        url = f"{self.base_url}{path}"
        method_u = method.upper()

        kwargs: Dict[str, Any] = {"headers": headers}
        if method_u in {"GET", "DELETE"}:
            kwargs["params"] = payload
        else:
            kwargs["data"] = payload

        async with session.request(method_u, url, **kwargs) as response:
            data = await response.json(content_type=None)
            if response.status >= 400:
                code = data.get("code", response.status) if isinstance(data, dict) else response.status
                msg = data.get("msg", str(data)) if isinstance(data, dict) else str(data)
                raise RuntimeError(f"Spot order request failed ({code}): {msg}")
            return data

    async def _get_reference_price(self, symbol: str) -> float:
        ticker = await self._request(
            "GET",
            "/api/v3/ticker/price",
            params={"symbol": symbol.upper()},
            signed=False,
            weight=1,
            order=False,
        )
        return float(ticker.get("price", 0.0) or 0.0)

    async def place_oco(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_price: float,
        stop_limit_price: float,
    ) -> Dict[str, Any]:
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": quantity,
            "price": price,
            "stopPrice": stop_price,
            "stopLimitPrice": stop_limit_price,
            "stopLimitTimeInForce": "GTC",
        }
        return await self._request("POST", "/api/v3/order/oco", params=params, signed=True, order=True)

    async def place_trailing_stop(
        self,
        symbol: str,
        side: str,
        quantity: float,
        trailing_delta_bips: int,
    ) -> Dict[str, Any]:
        reference_price = await self._get_reference_price(symbol)
        if reference_price <= 0:
            raise RuntimeError("unable to fetch reference price for trailing stop")

        price_multiplier = 0.995 if side.upper() == "SELL" else 1.005
        limit_price = round(reference_price * price_multiplier, 6)

        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "STOP_LOSS_LIMIT",
            "timeInForce": "GTC",
            "quantity": quantity,
            "price": limit_price,
            "stopPrice": limit_price,
            "trailingDelta": int(trailing_delta_bips),
        }
        return await self._request("POST", "/api/v3/order", params=params, signed=True, order=True)

    async def place_iceberg(
        self,
        symbol: str,
        side: str,
        total_qty: float,
        iceberg_qty: float,
        price: float,
    ) -> Dict[str, Any]:
        if iceberg_qty <= 0 or total_qty <= 0:
            raise ValueError("total_qty and iceberg_qty must be positive")
        if iceberg_qty > total_qty:
            raise ValueError("iceberg_qty cannot exceed total_qty")

        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": total_qty,
            "icebergQty": iceberg_qty,
            "price": price,
        }
        return await self._request("POST", "/api/v3/order", params=params, signed=True, order=True)

    async def execute_twap(
        self,
        symbol: str,
        side: str,
        total_qty: float,
        n_slices: int,
        interval_secs: int,
    ) -> List[Dict[str, Any]]:
        if n_slices <= 0:
            raise ValueError("n_slices must be > 0")
        if total_qty <= 0:
            raise ValueError("total_qty must be > 0")

        slice_qty = total_qty / n_slices
        responses: List[Dict[str, Any]] = []

        for index in range(n_slices):
            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": "MARKET",
                "quantity": slice_qty,
            }
            response = await self._request("POST", "/api/v3/order", params=params, signed=True, order=True)
            responses.append(response)

            if index < n_slices - 1 and interval_secs > 0:
                await asyncio.sleep(interval_secs)

        return responses

    async def execute_vwap(
        self,
        symbol: str,
        side: str,
        total_qty: float,
        window_minutes: int = 60,
    ) -> List[Dict[str, Any]]:
        if total_qty <= 0:
            raise ValueError("total_qty must be > 0")

        trades = await self._request(
            "GET",
            "/api/v3/trades",
            params={"symbol": symbol.upper(), "limit": 500},
            signed=False,
            weight=25,
            order=False,
        )

        n_slices = 5
        volume_weights = [1.0] * n_slices

        if isinstance(trades, list) and trades:
            chunk_size = max(1, len(trades) // n_slices)
            weights: List[float] = []
            for idx in range(n_slices):
                chunk = trades[idx * chunk_size : (idx + 1) * chunk_size]
                chunk_volume = sum(float(item.get("qty", 0.0) or 0.0) for item in chunk)
                weights.append(chunk_volume)
            if any(w > 0 for w in weights):
                volume_weights = weights

        total_weight = sum(volume_weights) or float(n_slices)
        interval_secs = max(1, int((window_minutes * 60) / n_slices)) if window_minutes > 0 else 0

        responses: List[Dict[str, Any]] = []
        for index, weight in enumerate(volume_weights):
            qty = total_qty * (weight / total_weight)
            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": "MARKET",
                "quantity": qty,
            }
            response = await self._request("POST", "/api/v3/order", params=params, signed=True, order=True)
            responses.append(response)

            if index < len(volume_weights) - 1 and interval_secs > 0:
                await asyncio.sleep(interval_secs)

        return responses
