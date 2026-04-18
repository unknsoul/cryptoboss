"""Unified price feed that prefers MT5 and falls back to Binance."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from .binance_unified import BinanceUnified
from .mt5_connector import MT5Config, MT5Connector

logger = logging.getLogger(__name__)


SYMBOL_MAP_TO_MT5 = {
    "BTC/USDT": "BTCUSD",
    "ETH/USDT": "ETHUSD",
    "SOL/USDT": "SOLUSD",
    "BNB/USDT": "BNBUSD",
    "XRP/USDT": "XRPUSD",
    "ADA/USDT": "ADAUSD",
    "LTC/USDT": "LTCUSD",
    "LINK/USDT": "LINKUSD",
}

SYMBOL_MAP_TO_BINANCE = {mt5_symbol: binance_symbol for binance_symbol, mt5_symbol in SYMBOL_MAP_TO_MT5.items()}


class PriceFeed:
    """Price source abstraction: MT5 first, Binance fallback."""

    def __init__(
        self,
        mt5: Optional[MT5Connector] = None,
        binance: Optional[BinanceUnified] = None,
        prefer_mt5: bool = True,
    ):
        self.mt5 = mt5
        self.binance = binance
        self.prefer_mt5 = prefer_mt5
        self._mt5_ok = False

        if self.mt5 is not None:
            try:
                self.mt5.connect()
                self._mt5_ok = True
                logger.info("PriceFeed initialized with MT5")
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("PriceFeed could not connect MT5 (%s), fallback enabled", exc)

        if self.binance is None:
            try:
                self.binance = BinanceUnified.from_env()
                logger.info("PriceFeed initialized with Binance")
            except Exception as exc:
                logger.warning("PriceFeed could not initialize Binance (%s)", exc)

    @classmethod
    def from_env(cls) -> "PriceFeed":
        mt5_client: Optional[MT5Connector] = None
        binance_client: Optional[BinanceUnified] = None

        try:
            mt5_client = MT5Connector(MT5Config.from_env())
        except Exception:
            mt5_client = None

        try:
            binance_client = BinanceUnified.from_env()
        except Exception:
            binance_client = None

        return cls(mt5=mt5_client, binance=binance_client)

    def get_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 500) -> pd.DataFrame:
        if self.prefer_mt5 and self._mt5_ok and self.mt5 is not None:
            mt5_symbol = SYMBOL_MAP_TO_MT5.get(symbol, symbol.replace("/", ""))
            try:
                return self.mt5.get_ohlcv(symbol=mt5_symbol, timeframe=timeframe, n_bars=limit)
            except Exception as exc:
                logger.warning("MT5 get_ohlcv failed for %s (%s). Falling back to Binance", symbol, exc)
                self._mt5_ok = False

        if self.binance is not None:
            return self.binance.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)

        raise RuntimeError("No active price source. Configure MT5 and/or Binance credentials")

    def get_multi_timeframe(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        limit: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        tf_list = timeframes or ["1m", "5m", "15m"]
        if self.prefer_mt5 and self._mt5_ok and self.mt5 is not None:
            mt5_symbol = SYMBOL_MAP_TO_MT5.get(symbol, symbol.replace("/", ""))
            try:
                return self.mt5.get_multi_timeframe(symbol=mt5_symbol, timeframes=tf_list, n_bars=limit)
            except Exception as exc:
                logger.warning("MT5 get_multi_timeframe failed for %s (%s). Falling back to Binance", symbol, exc)
                self._mt5_ok = False

        if self.binance is not None:
            return self.binance.get_multi_timeframe(symbol=symbol, timeframes=tf_list, limit=limit)

        raise RuntimeError("No active price source")

    def get_last_price(self, symbol: str) -> float:
        if self.prefer_mt5 and self._mt5_ok and self.mt5 is not None:
            mt5_symbol = SYMBOL_MAP_TO_MT5.get(symbol, symbol.replace("/", ""))
            try:
                tick = self.mt5.get_last_tick(mt5_symbol)
                return float(tick["bid"] + tick["ask"]) / 2.0
            except Exception:
                self._mt5_ok = False

        if self.binance is not None:
            ticker = self.binance.get_ticker(symbol)
            return float(ticker.get("last") or 0.0)

        return 0.0

    def get_spread(self, symbol: str) -> float:
        if self.prefer_mt5 and self._mt5_ok and self.mt5 is not None:
            mt5_symbol = SYMBOL_MAP_TO_MT5.get(symbol, symbol.replace("/", ""))
            try:
                tick = self.mt5.get_last_tick(mt5_symbol)
                return float(tick.get("spread_pct") or 0.0)
            except Exception:
                self._mt5_ok = False

        if self.binance is not None:
            ticker = self.binance.get_ticker(symbol)
            ask = ticker.get("ask")
            bid = ticker.get("bid")
            if ask and bid:
                return round((float(ask) - float(bid)) / float(ask) * 100.0, 6)

        return 0.0

    @property
    def active_source(self) -> str:
        if self.prefer_mt5 and self._mt5_ok:
            return "mt5"
        return "binance"
