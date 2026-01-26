"""
Config Versioning - Upgrade N

Environment-specific configuration with validation:
- dev, staging, prod environments
- Schema validation
- Override support
- Secrets integration
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    name: str  # dev, staging, prod
    debug: bool
    log_level: str
    
    # Trading
    trading_mode: str  # paper, live
    max_position_size: float
    default_leverage: int
    
    # Exchange
    exchange: str
    use_testnet: bool
    
    # Risk
    max_daily_loss_pct: float
    max_drawdown_pct: float
    
    # Features
    enable_telegram: bool
    enable_email: bool
    enable_metrics: bool


class ConfigManager:
    """
    Environment-aware configuration management.
    
    Usage:
        config = ConfigManager(env="prod")
        
        # Get config values
        max_loss = config.get("risk.max_daily_loss_pct")
        
        # Get typed config
        trading_config = config.get_section("trading")
    """
    
    DEFAULT_CONFIG = {
        "environment": "dev",
        "debug": True,
        "log_level": "DEBUG",
        
        "trading": {
            "mode": "paper",
            "max_position_size_usd": 1000,
            "default_leverage": 1,
            "symbols": ["BTC/USDT", "ETH/USDT"],
        },
        
        "exchange": {
            "name": "binance",
            "use_testnet": True,
            "rate_limit_per_min": 1200,
        },
        
        "risk": {
            "max_daily_loss_pct": 5.0,
            "max_drawdown_pct": 15.0,
            "max_position_concentration_pct": 25.0,
            "enable_circuit_breaker": True,
        },
        
        "strategies": {
            "enabled": ["dca", "grid"],
            "dca": {"preset": "balanced"},
            "grid": {"preset": "neutral"},
        },
        
        "notifications": {
            "telegram": {"enabled": False},
            "email": {"enabled": False},
            "discord": {"enabled": False},
        },
        
        "observability": {
            "metrics_enabled": True,
            "log_json": False,
            "log_file": "logs/cryptoboss.log",
        },
        
        "database": {
            "state_db": "data/state.db",
            "events_db": "data/events.db",
        }
    }
    
    ENV_OVERRIDES = {
        "dev": {
            "debug": True,
            "log_level": "DEBUG",
            "trading": {"mode": "paper"},
            "exchange": {"use_testnet": True},
            "risk": {"max_daily_loss_pct": 10.0},
        },
        "staging": {
            "debug": True,
            "log_level": "INFO",
            "trading": {"mode": "paper"},
            "exchange": {"use_testnet": True},
            "risk": {"max_daily_loss_pct": 5.0},
        },
        "prod": {
            "debug": False,
            "log_level": "WARNING",
            "trading": {"mode": "live"},
            "exchange": {"use_testnet": False},
            "risk": {"max_daily_loss_pct": 3.0, "max_drawdown_pct": 10.0},
            "observability": {"log_json": True},
        }
    }
    
    def __init__(self, env: str = None, config_dir: str = "configs"):
        self.env = env or os.getenv("CRYPTOBOSS_ENV", "dev")
        self.config_dir = Path(config_dir)
        self._config: Dict = {}
        
        self._load_config()
        logger.info(f"ConfigManager initialized for environment: {self.env}")
    
    def _load_config(self):
        """Load configuration with proper precedence."""
        # 1. Start with defaults
        self._config = self._deep_copy(self.DEFAULT_CONFIG)
        
        # 2. Apply environment overrides
        if self.env in self.ENV_OVERRIDES:
            self._merge(self._config, self.ENV_OVERRIDES[self.env])
        
        # 3. Load base config file if exists
        base_file = self.config_dir / "base.yaml"
        if base_file.exists():
            with open(base_file) as f:
                base_config = yaml.safe_load(f)
                if base_config:
                    self._merge(self._config, base_config)
        
        # 4. Load environment-specific file if exists
        env_file = self.config_dir / f"{self.env}.yaml"
        if env_file.exists():
            with open(env_file) as f:
                env_config = yaml.safe_load(f)
                if env_config:
                    self._merge(self._config, env_config)
        
        # 5. Apply environment variable overrides
        self._apply_env_overrides()
        
        # Set environment in config
        self._config["environment"] = self.env
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            "TRADING_MODE": "trading.mode",
            "MAX_DAILY_LOSS": "risk.max_daily_loss_pct",
            "MAX_DRAWDOWN": "risk.max_drawdown_pct",
            "LOG_LEVEL": "log_level",
            "DEBUG": "debug",
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(f"CRYPTOBOSS_{env_var}")
            if value is not None:
                self._set_nested(config_path, self._parse_value(value))
    
    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value
    
    def _deep_copy(self, d: Dict) -> Dict:
        """Deep copy a dict."""
        import copy
        return copy.deepcopy(d)
    
    def _merge(self, base: Dict, override: Dict):
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge(base[key], value)
            else:
                base[key] = value
    
    def _set_nested(self, path: str, value: Any):
        """Set a nested config value by dot-path."""
        keys = path.split(".")
        d = self._config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get config value by dot-path."""
        keys = path.split(".")
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict:
        """Get entire config section."""
        return self._config.get(section, {})
    
    def get_all(self) -> Dict:
        """Get entire config."""
        return self._deep_copy(self._config)
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.env == "prod"
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get("debug", False)
    
    def validate(self) -> tuple[bool, list]:
        """Validate configuration."""
        errors = []
        
        # Required fields
        required = [
            "trading.mode",
            "exchange.name",
            "risk.max_daily_loss_pct",
        ]
        
        for path in required:
            if self.get(path) is None:
                errors.append(f"Missing required config: {path}")
        
        # Value ranges
        if self.get("risk.max_daily_loss_pct", 0) > 20:
            errors.append("risk.max_daily_loss_pct should not exceed 20%")
        
        if self.get("risk.max_drawdown_pct", 0) > 50:
            errors.append("risk.max_drawdown_pct should not exceed 50%")
        
        # Production checks
        if self.is_production():
            if self.get("debug"):
                errors.append("Debug should be disabled in production")
            if self.get("exchange.use_testnet"):
                errors.append("Testnet should be disabled in production")
        
        return len(errors) == 0, errors
    
    def save_snapshot(self, path: str = None):
        """Save current config snapshot."""
        path = path or f"configs/snapshot_{self.env}.yaml"
        with open(path, 'w') as f:
            yaml.safe_dump(self._config, f, default_flow_style=False)


# Singleton
_config: Optional[ConfigManager] = None

def get_config(env: str = None) -> ConfigManager:
    global _config
    if _config is None:
        _config = ConfigManager(env=env)
    return _config
