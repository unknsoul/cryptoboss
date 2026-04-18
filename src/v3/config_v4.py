"""V4 system configuration extending v3 with dual-source and pro-builder settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataSourceConfig:
    primary: str = "mt5"
    fallback: str = "binance"
    mt5_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    binance_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    candles_per_tf: int = 500
    auto_resample: bool = True

    @classmethod
    def from_env(cls) -> "DataSourceConfig":
        primary = os.getenv("PRICE_SOURCE", "mt5").strip().lower()
        fallback = os.getenv("PRICE_SOURCE_FALLBACK", "binance").strip().lower()
        return cls(primary=primary, fallback=fallback)


@dataclass
class BinanceModeConfig:
    mode: str = "testnet"
    market_type: str = "spot"
    max_order_size_usdt: float = 1000.0
    order_timeout_ms: int = 5000

    @classmethod
    def from_env(cls) -> "BinanceModeConfig":
        return cls(
            mode=os.getenv("BINANCE_MODE", "testnet").strip().lower(),
            market_type=os.getenv("BINANCE_MARKET_TYPE", "spot").strip().lower(),
        )

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def is_testnet(self) -> bool:
        return self.mode == "testnet"


@dataclass
class ProBuilderConfig:
    max_conditions_per_side: int = 10
    min_ai_score_to_backtest: float = 40.0
    auto_score_on_change: bool = True
    default_presets: List[str] = field(
        default_factory=lambda: [
            "smc_scalper_pro",
            "ema_ribbon_momentum",
            "vwap_mean_reversion",
            "choch_reversal_pro",
            "squeeze_breakout",
        ]
    )
    save_path: str = "data/strategies/"


@dataclass
class V4SystemConfig:
    version: str = "4.0_dual_source_pro"
    architecture: str = "modular_microservices_v4"
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    binance_mode: BinanceModeConfig = field(default_factory=BinanceModeConfig)
    pro_builder: ProBuilderConfig = field(default_factory=ProBuilderConfig)
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "V4SystemConfig":
        return cls(
            data_source=DataSourceConfig.from_env(),
            binance_mode=BinanceModeConfig.from_env(),
        )

    def summary(self) -> dict:
        return {
            "version": self.version,
            "architecture": self.architecture,
            "price_source": self.data_source.primary,
            "execution_mode": self.binance_mode.mode,
            "market_type": self.binance_mode.market_type,
            "symbols": self.symbols,
        }
