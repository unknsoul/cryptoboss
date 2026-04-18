"""Data engine for v3 intraday scalper microservice architecture."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import DataEngineConfig


class DataEngine:
    """Normalizes and enriches market data across 1m/5m/15m timeframes."""

    REQUIRED_COLUMNS = ("open", "high", "low", "close")

    def __init__(self, config: Optional[DataEngineConfig] = None):
        self.config = config or DataEngineConfig()
        self.sources = set(self.config.sources)

    def normalize_frame(self, frame: pd.DataFrame, source: str = "binance") -> pd.DataFrame:
        if source not in self.sources:
            raise ValueError(f"Unsupported source: {source}")

        if frame is None or frame.empty:
            raise ValueError("Input frame is empty")

        df = frame.copy()
        df.columns = [str(column).lower() for column in df.columns]

        for column in self.REQUIRED_COLUMNS:
            if column not in df.columns:
                raise ValueError(f"Missing required OHLC column: {column}")

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Frame must include a timestamp column or DatetimeIndex")

        df = df.sort_index().dropna(subset=list(self.REQUIRED_COLUMNS))

        if "tick_volume" not in df.columns:
            if "volume" in df.columns:
                df["tick_volume"] = df["volume"].astype(float)
            else:
                proxy_volume = (df["high"] - df["low"]).abs() * 1000.0
                df["tick_volume"] = proxy_volume.clip(lower=1.0)

        if "spread" not in df.columns:
            # Estimate spread in percentage points from candle range when L2 data is unavailable.
            df["spread"] = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).fillna(0.0) * 100.0

        return self._enrich_features(df)

    def prepare_multi_timeframe(self, frames_by_timeframe: Dict[str, pd.DataFrame], source: str = "binance") -> Dict[str, pd.DataFrame]:
        normalized: Dict[str, pd.DataFrame] = {}

        for timeframe in self.config.timeframes:
            raw_frame = frames_by_timeframe.get(timeframe)
            if raw_frame is not None and not raw_frame.empty:
                normalized[timeframe] = self.normalize_frame(raw_frame, source=source)

        if "1m" not in normalized:
            raise ValueError("At least 1m frame is required for v3 intraday pipeline")

        if "5m" not in normalized:
            normalized["5m"] = self._resample_timeframe(normalized["1m"], "5m")

        if "15m" not in normalized:
            normalized["15m"] = self._resample_timeframe(normalized["1m"], "15m")

        # Keep the latest required candles, but do not fail hard when historical depth is smaller.
        for timeframe, frame in normalized.items():
            normalized[timeframe] = frame.tail(self.config.candles_required)

        return normalized

    def _resample_timeframe(self, frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        rule_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
        }
        if timeframe not in rule_map:
            raise ValueError(f"Unsupported timeframe for resampling: {timeframe}")

        resampled = pd.DataFrame(
            {
                "open": frame["open"].resample(rule_map[timeframe]).first(),
                "high": frame["high"].resample(rule_map[timeframe]).max(),
                "low": frame["low"].resample(rule_map[timeframe]).min(),
                "close": frame["close"].resample(rule_map[timeframe]).last(),
                "tick_volume": frame["tick_volume"].resample(rule_map[timeframe]).sum(),
                "spread": frame["spread"].resample(rule_map[timeframe]).mean(),
            }
        ).dropna(subset=["open", "high", "low", "close"])

        return self._enrich_features(resampled)

    def _enrich_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        close_returns = df["close"].pct_change().fillna(0.0)
        df["volatility"] = close_returns.rolling(20, min_periods=5).std().fillna(0.0)
        df["momentum"] = df["close"].pct_change(5).fillna(0.0)

        rolling_high = df["high"].rolling(20, min_periods=5).max()
        rolling_low = df["low"].rolling(20, min_periods=5).min()
        zone_width = (rolling_high - rolling_low).replace(0, np.nan)
        df["liquidity_zones"] = ((df["close"] - rolling_low) / zone_width).clip(lower=0.0, upper=1.0).fillna(0.5)

        return df
