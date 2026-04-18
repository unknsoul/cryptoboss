"""Unified Binance client with testnet/live switching via BINANCE_MODE."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    ccxt = None
    CCXT_AVAILABLE = False
    logger.error("ccxt is not installed. Install with: pip install ccxt")


TESTNET_SPOT_REST = "https://testnet.binance.vision"


@dataclass
class BinanceCredentials:
    api_key: str
    api_secret: str
    mode: str  # live | testnet

    @classmethod
    def from_env(cls) -> "BinanceCredentials":
        mode = os.getenv("BINANCE_MODE", "testnet").strip().lower()
        if mode not in {"live", "testnet"}:
            raise ValueError("BINANCE_MODE must be 'live' or 'testnet'")

        if mode == "live":
            api_key = os.getenv("BINANCE_API_KEY", "")
            api_secret = os.getenv("BINANCE_API_SECRET", "")
            key_prefix = "BINANCE_"
        else:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
            key_prefix = "BINANCE_TESTNET_"

        if not api_key or not api_secret:
            raise EnvironmentError(
                f"{key_prefix}API_KEY and {key_prefix}API_SECRET must be set when BINANCE_MODE={mode}"
            )

        return cls(api_key=api_key, api_secret=api_secret, mode=mode)


class BinanceUnified:
    """Unified account and execution client for Binance spot in testnet/live."""

    def __init__(self, credentials: BinanceCredentials):
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt is required for BinanceUnified")

        self.creds = credentials
        self.mode = credentials.mode

        options: Dict[str, Any] = {
            "apiKey": credentials.api_key,
            "secret": credentials.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
                "adjustForTimeDifference": True,
            },
        }

        if self.mode == "testnet":
            options["urls"] = {
                "api": {
                    "public": f"{TESTNET_SPOT_REST}/api",
                    "private": f"{TESTNET_SPOT_REST}/api",
                }
            }

        self._exchange = ccxt.binance(options)

        if self.mode == "testnet":
            self._exchange.set_sandbox_mode(True)

        logger.info("BinanceUnified initialized in %s mode", self.mode)

    @classmethod
    def from_env(cls) -> "BinanceUnified":
        return cls(BinanceCredentials.from_env())

    @classmethod
    def testnet(cls, api_key: str, api_secret: str) -> "BinanceUnified":
        return cls(BinanceCredentials(api_key=api_key, api_secret=api_secret, mode="testnet"))

    @classmethod
    def live(cls, api_key: str, api_secret: str) -> "BinanceUnified":
        return cls(BinanceCredentials(api_key=api_key, api_secret=api_secret, mode="live"))

    def get_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 500) -> pd.DataFrame:
        raw = self._exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df.index.name = "timestamp"

        df["tick_volume"] = df["volume"].astype(float)
        df["spread"] = ((df["high"] - df["low"]) / df["close"]).replace([pd.NA], 0.0).fillna(0.0) * 100.0
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df

    def get_multi_timeframe(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        limit: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        return {
            tf: self.get_ohlcv(symbol=symbol, timeframe=tf, limit=limit)
            for tf in (timeframes or ["1m", "5m", "15m"])
        }

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        t = self._exchange.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "bid": t.get("bid"),
            "ask": t.get("ask"),
            "last": t.get("last"),
            "volume_24h": t.get("quoteVolume"),
            "change_pct": t.get("percentage"),
            "mode": self.mode,
        }

    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        book = self._exchange.fetch_order_book(symbol=symbol, limit=limit)
        return {
            "symbol": symbol,
            "bids": book.get("bids", [])[:limit],
            "asks": book.get("asks", [])[:limit],
            "timestamp": book.get("timestamp"),
            "mode": self.mode,
        }

    def get_balance(self) -> Dict[str, Any]:
        bal = self._exchange.fetch_balance()
        totals = bal.get("total", {})

        assets = {}
        for key, value in totals.items():
            if isinstance(value, (int, float)) and float(value) > 0:
                assets[key] = float(value)

        usdt = bal.get("USDT", {}) if isinstance(bal.get("USDT"), dict) else {}
        return {
            "mode": self.mode,
            "total_usdt": float(usdt.get("total", 0) or 0),
            "free_usdt": float(usdt.get("free", 0) or 0),
            "assets": assets,
        }

    def get_open_positions(self) -> List[Dict[str, Any]]:
        orders = self._exchange.fetch_open_orders()
        return [self._format_order(order) for order in orders]

    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._log_order_intent("MARKET", symbol, side, amount)
        order = self._exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params=params or {},
        )
        return self._format_order(order)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._log_order_intent("LIMIT", symbol, side, amount, price)
        order = self._exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=amount,
            price=price,
            params=params or {},
        )
        return self._format_order(order)

    def place_stop_loss_order(self, symbol: str, side: str, amount: float, stop_price: float) -> Dict[str, Any]:
        order = self._exchange.create_order(
            symbol=symbol,
            type="STOP_LOSS_LIMIT",
            side=side,
            amount=amount,
            price=float(stop_price) * 0.999,
            params={"stopPrice": float(stop_price)},
        )
        return self._format_order(order)

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return self._exchange.cancel_order(order_id, symbol)

    def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        order = self._exchange.fetch_order(order_id, symbol)
        return self._format_order(order)

    def _format_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": order.get("id"),
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "type": order.get("type"),
            "price": order.get("price"),
            "amount": order.get("amount"),
            "filled": order.get("filled"),
            "cost": order.get("cost"),
            "status": order.get("status"),
            "timestamp": order.get("timestamp"),
            "mode": self.mode,
        }

    def _log_order_intent(
        self,
        order_type: str,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> None:
        at_price = f" @ {price}" if price is not None else ""
        logger.info("[%s] %s %s %s %s%s", self.mode.upper(), order_type, side.upper(), amount, symbol, at_price)
        if self.mode == "live":
            logger.warning("LIVE MODE: real funds are at risk")
