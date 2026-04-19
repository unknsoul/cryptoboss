"""Async Binance Futures client for REST endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp


TESTNET_REST = "https://testnet.binancefuture.com"
LIVE_REST = "https://fapi.binance.com"
TESTNET_WS = "wss://stream.binancefuture.com"
LIVE_WS = "wss://fstream.binance.com"


class BinanceFuturesClient:
    """Thin async client around Binance USDT-M Futures REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        timeout_seconds: int = 15,
    ):
        self.api_key = api_key or os.getenv("BINANCE_FUTURES_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_FUTURES_SECRET", "")

        if testnet is None:
            testnet = os.getenv("BINANCE_FUTURES_TESTNET", "true").lower() in {
                "1",
                "true",
                "yes",
            }

        self.testnet = bool(testnet)
        self.base_rest_url = TESTNET_REST if self.testnet else LIVE_REST
        self.base_ws_url = TESTNET_WS if self.testnet else LIVE_WS
        self.timeout_seconds = int(timeout_seconds)

        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "BinanceFuturesClient":
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
        signed: bool = False,
    ) -> Any:
        session = await self._ensure_session()
        method_u = method.upper()
        payload = dict(params or {})

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        if signed:
            payload["timestamp"] = self._timestamp()
            payload["recvWindow"] = int(payload.get("recvWindow", 5000))
            payload["signature"] = self._sign(payload)

        url = f"{self.base_rest_url}{path}"

        request_kwargs: Dict[str, Any] = {"headers": headers}
        if method_u == "GET" or method_u == "DELETE":
            request_kwargs["params"] = payload
        else:
            request_kwargs["data"] = payload

        async with session.request(method_u, url, **request_kwargs) as response:
            data = await response.json(content_type=None)
            if response.status >= 400:
                code = data.get("code", response.status) if isinstance(data, dict) else response.status
                msg = data.get("msg", str(data)) if isinstance(data, dict) else str(data)
                raise RuntimeError(f"Binance futures request failed ({code}): {msg}")
            return data

    async def get_position_risk(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"symbol": symbol} if symbol else {}
        return await self._request("GET", "/fapi/v2/positionRisk", params=params, signed=True)

    async def get_account_info(self) -> Dict[str, Any]:
        return await self._request("GET", "/fapi/v2/account", signed=True)

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        params = {"symbol": symbol.upper(), "leverage": int(leverage)}
        return await self._request("POST", "/fapi/v1/leverage", params=params, signed=True)

    async def set_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        params = {
            "symbol": symbol.upper(),
            "marginType": margin_type.upper(),
        }
        return await self._request("POST", "/fapi/v1/marginType", params=params, signed=True)

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        order_type_upper = order_type.upper()
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type_upper,
            "quantity": quantity,
            "reduceOnly": "true" if reduce_only else "false",
        }

        if order_type_upper in {"LIMIT", "STOP", "TAKE_PROFIT"}:
            if price is None:
                raise ValueError("price is required for limit/stop/take-profit orders")
            params["price"] = price
            params["timeInForce"] = "GTC"

        return await self._request("POST", "/fapi/v1/order", params=params, signed=True)

    async def place_bracket_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry: float,
        stop_loss: float,
        take_profit: float,
    ) -> Dict[str, Any]:
        side_upper = side.upper()
        closing_side = "SELL" if side_upper == "BUY" else "BUY"

        entry_order = await self.place_order(
            symbol=symbol,
            side=side_upper,
            order_type="LIMIT",
            quantity=quantity,
            price=entry,
            reduce_only=False,
        )

        stop_order = await self._request(
            "POST",
            "/fapi/v1/order",
            params={
                "symbol": symbol.upper(),
                "side": closing_side,
                "type": "STOP_MARKET",
                "stopPrice": stop_loss,
                "closePosition": "true",
                "workingType": "MARK_PRICE",
            },
            signed=True,
        )

        take_profit_order = await self._request(
            "POST",
            "/fapi/v1/order",
            params={
                "symbol": symbol.upper(),
                "side": closing_side,
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": take_profit,
                "closePosition": "true",
                "workingType": "MARK_PRICE",
            },
            signed=True,
        )

        return {
            "entry_order": entry_order,
            "stop_order": stop_order,
            "take_profit_order": take_profit_order,
        }

    async def close_position(self, symbol: str, position_side: str = "BOTH") -> Dict[str, Any]:
        positions = await self.get_position_risk(symbol=symbol)
        if not positions:
            return {"closed": False, "reason": "no_position"}

        target = None
        for position in positions:
            if position.get("symbol", "").upper() == symbol.upper():
                target = position
                break

        if not target:
            return {"closed": False, "reason": "symbol_not_found"}

        amount = float(target.get("positionAmt", 0.0))
        if amount == 0:
            return {"closed": False, "reason": "already_flat"}

        close_side = "SELL" if amount > 0 else "BUY"
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": close_side,
            "type": "MARKET",
            "quantity": abs(amount),
            "reduceOnly": "true",
            "positionSide": position_side.upper(),
        }
        order = await self._request("POST", "/fapi/v1/order", params=params, signed=True)
        return {"closed": True, "order": order}

    async def get_funding_rate(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        params = {"symbol": symbol.upper(), "limit": int(limit)}
        return await self._request("GET", "/fapi/v1/fundingRate", params=params)

    async def get_mark_price(self, symbol: str) -> Dict[str, Any]:
        return await self._request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol.upper()})

    async def get_liquidation_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return await self._request("GET", "/fapi/v1/forceOrders", params=params, signed=True)

    async def get_open_interest(self, symbol: str) -> Dict[str, Any]:
        return await self._request("GET", "/fapi/v1/openInterest", params={"symbol": symbol.upper()})

    async def get_long_short_ratio(self, symbol: str, period: str = "5m") -> List[Dict[str, Any]]:
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": 100,
        }
        return await self._request(
            "GET",
            "/futures/data/globalLongShortAccountRatio",
            params=params,
        )

    async def ping(self) -> bool:
        try:
            await self._request("GET", "/fapi/v1/ping")
            return True
        except Exception:
            return False
