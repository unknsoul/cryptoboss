"""Factory for v4 dual-source architecture wiring."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

from .binance_unified import BinanceUnified
from .price_feed import PriceFeed

logger = logging.getLogger(__name__)


@dataclass
class ExchangeFactory:
    price_feed: PriceFeed
    executor: BinanceUnified
    mode: str

    @classmethod
    def from_env(cls) -> "ExchangeFactory":
        mode = os.getenv("BINANCE_MODE", "testnet").strip().lower()
        price_feed = PriceFeed.from_env()
        executor = BinanceUnified.from_env()
        logger.info("ExchangeFactory ready: mode=%s, price_source=%s", mode, price_feed.active_source)
        return cls(price_feed=price_feed, executor=executor, mode=mode)

    def status(self) -> Dict[str, Any]:
        mt5_ok = False
        mt5_account = None
        if self.price_feed.mt5 and self.price_feed._mt5_ok:
            try:
                mt5_account = self.price_feed.mt5.get_account_info()
                mt5_ok = True
            except Exception:
                mt5_ok = False

        binance_ok = False
        binance_balance = None
        try:
            binance_balance = self.executor.get_balance()
            binance_ok = True
        except Exception:
            binance_ok = False

        return {
            "mode": self.mode,
            "price_source": self.price_feed.active_source,
            "mt5": {
                "connected": mt5_ok,
                "account": mt5_account,
            },
            "binance": {
                "connected": binance_ok,
                "mode": self.executor.mode,
                "balance": binance_balance,
            },
        }
