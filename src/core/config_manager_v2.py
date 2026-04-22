"""Centralized bot configuration with runtime validation and hot reload."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TradingConfig(BaseModel):
    """Trading runtime configuration."""

    symbols: list[str] = Field(min_length=1)
    base_timeframes: list[str] = Field(min_length=1)
    mode: Literal["paper", "live"] = "paper"
    exchange: str = "binance"
    leverage: int = Field(default=1, ge=1, le=125)
    max_trades: int = Field(default=3, ge=1, le=20)


class RiskConfig(BaseModel):
    """Portfolio and per-trade risk controls."""

    max_risk_per_trade_pct: float = Field(default=2.0, gt=0.0, le=2.0)
    max_portfolio_risk_pct: float = Field(default=6.0, gt=0.0, le=6.0)
    max_daily_loss_pct: float = Field(default=3.0, gt=0.0, le=10.0)
    max_drawdown_halt_pct: float = Field(default=15.0, gt=0.0, le=50.0)
    max_sl_pct: float = Field(default=2.5, gt=0.0, le=5.0)


class MLConfig(BaseModel):
    """ML runtime behavior tuning."""

    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    retrain_interval_days: int = Field(default=7, ge=1, le=90)
    model_dir: str = "models/"
    use_lstm: bool = True
    use_sentiment: bool = True
    use_onchain: bool = False


class NotificationsConfig(BaseModel):
    """Notification routing configuration."""

    telegram_enabled: bool = True
    discord_enabled: bool = False
    notify_on_entry: bool = True
    notify_on_exit: bool = True
    daily_summary_hour_utc: int = Field(default=0, ge=0, le=23)


class BotConfig(BaseModel):
    """Top-level bot configuration schema."""

    model_config = ConfigDict(extra="forbid")

    trading: TradingConfig
    risk: RiskConfig
    ml: MLConfig
    notifications: NotificationsConfig

    @model_validator(mode="after")
    def validate_risk_consistency(self) -> "BotConfig":
        """Validate cross-section constraints."""
        if self.risk.max_portfolio_risk_pct < self.risk.max_risk_per_trade_pct:
            raise ValueError("max_portfolio_risk_pct must be >= max_risk_per_trade_pct")
        if self.trading.max_trades <= 0:
            raise ValueError("max_trades must be positive")
        return self


class ConfigManagerV2:
    """Hot-reloadable config manager backed by a pydantic schema."""

    def __init__(self, config_path: Path | str = "configs/bot_config.yaml") -> None:
        """Load and validate configuration from disk."""
        self.config_path = Path(config_path)
        self._config: BotConfig | None = None
        self._last_mtime: float | None = None
        self._load()

    def _load(self) -> None:
        """Load and validate config from file."""
        raw = self._read_yaml(self.config_path)
        self._config = BotConfig.model_validate(raw)
        self._last_mtime = self.config_path.stat().st_mtime

    def get_config(self) -> BotConfig:
        """Return validated BotConfig object."""
        if self._config is None:
            self._load()
        return self._config

    def get_dict(self) -> dict[str, Any]:
        """Return configuration as a plain dictionary."""
        return self.get_config().model_dump()

    def reload_if_changed(self, force: bool = False) -> bool:
        """Reload config when the file changed or when force=True."""
        if force:
            self._load()
            return True

        current_mtime = self.config_path.stat().st_mtime
        if self._last_mtime is None or current_mtime > self._last_mtime:
            self._load()
            return True

        return False

    def update(self, updates: dict[str, Any], persist: bool = True) -> BotConfig:
        """Apply updates, validate, and optionally persist to disk."""
        merged = self.get_dict()
        self._deep_merge(merged, updates)
        updated = BotConfig.model_validate(merged)
        self._config = updated

        if persist:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(
                yaml.safe_dump(updated.model_dump(), sort_keys=False),
                encoding="utf-8",
            )
            self._last_mtime = self.config_path.stat().st_mtime

        return updated

    @staticmethod
    def default_dict() -> dict[str, Any]:
        """Return a default config dictionary matching the schema."""
        return deepcopy(
            {
                "trading": {
                    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                    "base_timeframes": ["15m", "1h", "4h", "1d"],
                    "mode": "paper",
                    "exchange": "binance",
                    "leverage": 1,
                    "max_trades": 3,
                },
                "risk": {
                    "max_risk_per_trade_pct": 2.0,
                    "max_portfolio_risk_pct": 6.0,
                    "max_daily_loss_pct": 3.0,
                    "max_drawdown_halt_pct": 15.0,
                    "max_sl_pct": 2.5,
                },
                "ml": {
                    "confidence_threshold": 0.65,
                    "retrain_interval_days": 7,
                    "model_dir": "models/",
                    "use_lstm": True,
                    "use_sentiment": True,
                    "use_onchain": False,
                },
                "notifications": {
                    "telegram_enabled": True,
                    "discord_enabled": False,
                    "notify_on_entry": True,
                    "notify_on_exit": True,
                    "daily_summary_hour_utc": 0,
                },
            }
        )

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        """Read YAML file as dictionary."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Config file must contain a YAML object")
        return payload

    @staticmethod
    def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> None:
        """Deep-merge updates into base dict in place."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigManagerV2._deep_merge(base[key], value)
            else:
                base[key] = value


_config_v2: Optional[ConfigManagerV2] = None


def get_config_v2(config_path: Path | str = "configs/bot_config.yaml") -> ConfigManagerV2:
    """Return singleton ConfigManagerV2 instance."""
    global _config_v2
    if _config_v2 is None:
        _config_v2 = ConfigManagerV2(config_path=config_path)
    return _config_v2
