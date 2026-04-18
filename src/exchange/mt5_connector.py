"""MT5 connector used as broker-grade market data source for CryptoBoss v4."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    mt5 = None
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not installed. Install with: pip install MetaTrader5")


def _tf_map() -> Dict[str, int]:
    if MT5_AVAILABLE:
        return {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
            "1w": mt5.TIMEFRAME_W1,
        }
    # Fallback constants only used for validation before import.
    return {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 16385,
        "4h": 16388,
        "1d": 16408,
        "1w": 32769,
    }


TF_MAP = _tf_map()


@dataclass
class MT5Config:
    login: int
    password: str
    server: str
    path: Optional[str] = None

    @classmethod
    def from_env(cls) -> "MT5Config":
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        path = os.getenv("MT5_PATH")

        if not login or not password or not server:
            raise EnvironmentError("MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER must be set")

        return cls(login=int(login), password=password, server=server, path=path)


class MT5Error(RuntimeError):
    """Raised when MT5 initialization or data retrieval fails."""


class MT5Connector:
    """Provides OHLCV and tick data from MetaTrader 5. No execution methods."""

    def __init__(self, config: Optional[MT5Config] = None):
        self.config = config or MT5Config.from_env()
        self._connected = False

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            raise MT5Error("MetaTrader5 package not installed")

        kwargs = {
            "login": self.config.login,
            "password": self.config.password,
            "server": self.config.server,
        }
        if self.config.path:
            kwargs["path"] = self.config.path

        if not mt5.initialize(**kwargs):
            raise MT5Error(f"MT5 initialize failed: {mt5.last_error()}")

        info = mt5.account_info()
        if info is None:
            mt5.shutdown()
            raise MT5Error("MT5 login failed. Verify credentials and server")

        self._connected = True
        logger.info("MT5 connected: %s | %s", info.server, info.name)
        return True

    def disconnect(self) -> None:
        if MT5_AVAILABLE:
            mt5.shutdown()
        self._connected = False

    def is_connected(self) -> bool:
        return bool(MT5_AVAILABLE and self._connected and mt5.terminal_info() is not None)

    def _ensure_connected(self) -> None:
        if not self.is_connected():
            self.connect()

    def get_ohlcv(self, symbol: str, timeframe: str = "5m", n_bars: int = 500) -> pd.DataFrame:
        self._ensure_connected()
        tf_const = TF_MAP.get(timeframe)
        if tf_const is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Allowed: {sorted(TF_MAP.keys())}")

        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n_bars)
        if rates is None or len(rates) == 0:
            raise MT5Error(f"No MT5 data for {symbol} {timeframe}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df.index.name = "timestamp"

        if "real_volume" not in df.columns:
            df["real_volume"] = 0.0
        if "spread" not in df.columns:
            df["spread"] = 0.0

        out = df[["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]].copy()
        for col in out.columns:
            out[col] = out[col].astype(float)
        return out

    def get_multi_timeframe(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        n_bars: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for tf in timeframes or ["1m", "5m", "15m"]:
            result[tf] = self.get_ohlcv(symbol=symbol, timeframe=tf, n_bars=n_bars)
        return result

    def get_last_tick(self, symbol: str) -> Dict[str, object]:
        self._ensure_connected()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MT5Error(f"No tick data for {symbol}")

        spread = float(tick.ask - tick.bid)
        spread_pct = (spread / tick.ask) * 100.0 if tick.ask else 0.0

        return {
            "symbol": symbol,
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "spread": round(spread, 8),
            "spread_pct": round(float(spread_pct), 6),
            "last": float(tick.last),
            "volume": float(getattr(tick, "volume", 0.0)),
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
        }

    def get_symbol_info(self, symbol: str) -> Dict[str, object]:
        self._ensure_connected()
        info = mt5.symbol_info(symbol)
        if info is None:
            raise MT5Error(f"Symbol not found on MT5 broker: {symbol}")

        return {
            "symbol": info.name,
            "description": info.description,
            "digits": info.digits,
            "point": info.point,
            "spread": info.spread,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_mode": info.trade_mode,
            "currency_base": info.currency_base,
            "currency_profit": info.currency_profit,
        }

    def get_account_info(self) -> Dict[str, object]:
        self._ensure_connected()
        info = mt5.account_info()
        if info is None:
            raise MT5Error("Could not fetch MT5 account info")

        return {
            "login": info.login,
            "server": info.server,
            "name": info.name,
            "balance": float(info.balance),
            "equity": float(info.equity),
            "margin": float(info.margin),
            "margin_free": float(info.margin_free),
            "margin_level": float(info.margin_level),
            "profit": float(info.profit),
            "currency": info.currency,
            "leverage": int(info.leverage),
        }

    def get_available_symbols(self, filter_crypto: bool = True) -> List[str]:
        self._ensure_connected()
        symbols = mt5.symbols_get()
        if not symbols:
            return []

        names = [s.name for s in symbols]
        if filter_crypto:
            keys = ["BTC", "ETH", "LTC", "XRP", "ADA", "SOL", "BNB", "DOT", "LINK"]
            names = [name for name in names if any(k in name for k in keys)]

        return sorted(names)

    def __enter__(self) -> "MT5Connector":
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.disconnect()
