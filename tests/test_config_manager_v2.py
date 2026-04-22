from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.core.config_manager_v2 import ConfigManagerV2


def _valid_config(mode: str = "paper") -> dict:
    return {
        "trading": {
            "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "base_timeframes": ["15m", "1h", "4h", "1d"],
            "mode": mode,
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


def test_loads_and_validates_default_config(tmp_path: Path):
    cfg = _valid_config()
    cfg_path = tmp_path / "bot_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    manager = ConfigManagerV2(config_path=cfg_path)
    loaded = manager.get_config()

    assert loaded.trading.exchange == "binance"
    assert loaded.risk.max_risk_per_trade_pct == 2.0


def test_hot_reload_detects_change(tmp_path: Path):
    cfg_path = tmp_path / "bot_config.yaml"
    cfg_path.write_text(yaml.safe_dump(_valid_config(mode="paper")), encoding="utf-8")

    manager = ConfigManagerV2(config_path=cfg_path)
    assert manager.get_config().trading.mode == "paper"

    cfg_path.write_text(yaml.safe_dump(_valid_config(mode="live")), encoding="utf-8")
    reloaded = manager.reload_if_changed(force=True)

    assert reloaded is True
    assert manager.get_config().trading.mode == "live"


def test_invalid_config_raises_validation_error(tmp_path: Path):
    bad = _valid_config()
    bad["risk"]["max_risk_per_trade_pct"] = 3.5

    cfg_path = tmp_path / "bad_config.yaml"
    cfg_path.write_text(yaml.safe_dump(bad), encoding="utf-8")

    with pytest.raises(ValidationError):
        ConfigManagerV2(config_path=cfg_path)
