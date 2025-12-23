"""
Configuration Management
Handles environment-based settings with validation
"""

# Pydantic v2 compatibility
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator
    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseSettings, Field, validator as field_validator
        PYDANTIC_V2 = False
    except ImportError:
        # Fallback to simple dataclass-based config
        from dataclasses import dataclass, field
        BaseSettings = object
        Field = field
        field_validator = lambda *a, **k: lambda f: f
        PYDANTIC_V2 = False

from typing import Optional, List, Dict, Any
import os
from pathlib import Path


class ExchangeConfig(BaseSettings):
    """Exchange connection settings"""
    api_key: str = Field(default="", env='BINANCE_API_KEY')
    api_secret: str = Field(default="", env='BINANCE_API_SECRET')
    use_testnet: bool = Field(default=True, env='USE_TESTNET')
    exchange_name: str = Field(default="binance")
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


class TradingConfig(BaseSettings):
    """Trading parameters"""
    initial_capital: float = Field(default=10000.0, gt=0)
    max_position_size_pct: float = Field(default=0.20, gt=0, le=1.0)  # 20% max per position
    base_risk_pct: float = Field(default=0.02, gt=0, le=0.1)  # 2% risk per trade
    fee_rate: float = Field(default=0.001, ge=0)  # 0.1% trading fee
    
    # Symbols to trade
    trading_symbols: List[str] = Field(default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    default_interval: str = Field(default="1h")
    
    # Strategy settings
    default_strategy: str = Field(default="enhanced_momentum")
    enable_ml_signals: bool = Field(default=True)
    enable_sentiment: bool = Field(default=True)
    enable_regime_detection: bool = Field(default=True)


class RiskConfig(BaseSettings):
    """Risk management settings"""
    target_volatility: float = Field(default=0.15, gt=0, le=1.0)  # 15% annual target
    max_drawdown_limit: float = Field(default=0.15, gt=0, le=0.5)  # 15% max drawdown
    max_leverage: float = Field(default=1.0, ge=1.0, le=5.0)  # No leverage by default
    
    # Circuit breakers
    max_daily_loss_pct: float = Field(default=0.05, gt=0, le=0.2)  # 5% daily loss limit
    max_daily_trades: int = Field(default=100, gt=0)
    
    # Position limits
    max_positions: int = Field(default=5, gt=0)
    max_correlation: float = Field(default=0.7, gt=0, le=1.0)  # Max 70% correlation
    
    # Order book guards
    min_orderbook_imbalance: float = Field(default=-0.6)
    max_orderbook_imbalance: float = Field(default=0.6)


class MLConfig(BaseSettings):
    """Machine Learning settings"""
    model_path: str = Field(default="models/ensemble_v1.pkl")
    retrain_interval_hours: int = Field(default=168, gt=0)  # Weekly retraining
    min_training_samples: int = Field(default=1000, gt=0)
    
    # Feature engineering
    lookback_period: int = Field(default=200, gt=0)
    feature_columns: List[str] = Field(default=[
        'z_score_price', 'rsi_norm', 'volatility', 
        'ob_imbalance', 'ret_lag_1', 'ret_lag_2', 'ret_lag_3'
    ])
    
    # Model parameters
    enable_walk_forward: bool = Field(default=True)
    train_test_split: float = Field(default=0.7, gt=0, lt=1.0)


class MonitoringConfig(BaseSettings):
    """Monitoring and alerting settings"""
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="logs")
    metrics_retention_hours: int = Field(default=24, gt=0)
    
    # Email alerts
    email_enabled: bool = Field(default=False)
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_user: Optional[str] = Field(default=None, env='SMTP_USER')
    smtp_password: Optional[str] = Field(default=None, env='SMTP_PASSWORD')
    email_to: List[str] = Field(default=[])
    
    # Slack alerts
    slack_enabled: bool = Field(default=False)
    slack_webhook: Optional[str] = Field(default=None, env='SLACK_WEBHOOK')
    
    # Discord alerts
    discord_enabled: bool = Field(default=False)
    discord_webhook: Optional[str] = Field(default=None, env='DISCORD_WEBHOOK')
    
    class Config:
        env_file = ".env"


class DatabaseConfig(BaseSettings):
    """Database connection settings"""
    db_type: str = Field(default="sqlite")  # sqlite, postgresql, etc.
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="trading_bot")
    db_user: Optional[str] = Field(default=None, env='DB_USER')
    db_password: Optional[str] = Field(default=None, env='DB_PASSWORD')
    
    # SQLite specific
    sqlite_path: str = Field(default="data/trading_bot.db")
    
    class Config:
        env_file = ".env"


class Settings(BaseSettings):
    """Main settings container"""
    # Environment
    environment: str = Field(default="development")  # development, staging, production
    debug: bool = Field(default=True)
    
    # Component configs
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # Google Gemini API (for sentiment analysis)
    google_api_key: Optional[str] = Field(default=None, env='GOOGLE_API_KEY')
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "exchange": self.exchange.dict(),
            "trading": self.trading.dict(),
            "risk": self.risk.dict(),
            "ml": self.ml.dict(),
            "monitoring": {
                k: v for k, v in self.monitoring.dict().items() 
                if k not in ['smtp_password', 'email_to']  # Hide sensitive data
            },
            "database": {
                k: v for k, v in self.database.dict().items()
                if k not in ['db_password']  # Hide sensitive data
            }
        }


def load_settings(env_file: Optional[str] = None) -> Settings:
    """
    Load settings from environment or .env file
    
    Args:
        env_file: Path to .env file (optional, defaults to .env in current directory)
    
    Returns:
        Settings instance
    """
    if env_file:
        os.environ['ENV_FILE'] = env_file
    
    return Settings()


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance"""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


if __name__ == "__main__":
    # Test configuration
    import json
    
    settings = get_settings()
    
    print("=" * 70)
    print("TRADING BOT CONFIGURATION")
    print("=" * 70)
    print(json.dumps(settings.to_dict(), indent=2))
    
    # Validate settings
    print("\n✅ Configuration loaded successfully")
    print(f"📍 Environment: {settings.environment}")
    print(f"💰 Initial Capital: ${settings.trading.initial_capital:,.2f}")
    print(f"📊 Trading Symbols: {', '.join(settings.trading.trading_symbols)}")
    print(f"⚠️ Max Daily Loss: {settings.risk.max_daily_loss_pct:.1%}")
    print(f"🤖 ML Enabled: {settings.trading.enable_ml_signals}")
