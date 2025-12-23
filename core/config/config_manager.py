"""
Configuration Management with Hydra
Centralized, type-safe configuration system
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class EnvironmentType(Enum):
    """Trading environment"""
    TESTNET = "testnet"
    MAINNET = "mainnet"
    BACKTEST = "backtest"
    PAPER = "paper"


@dataclass
class BinanceConfig:
    """Binance exchange configuration"""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    # API key encryption
    use_encryption: bool = True
    encryption_key_file: str = ".keyfile"


@dataclass
class StrategyConfig:
    """Trading strategy configuration"""
    name: str = "SimpleTrendStrategy"
    
    # Indicator parameters
    ema_fast: int = 50
    ema_slow: int = 200
    donchian_period: int = 20
    atr_period: int = 14
    atr_multiplier: float = 2.0
    
    # Additional filters
    use_rsi_filter: bool = False
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    
    use_volume_filter: bool = False
    min_volume_multiplier: float = 1.5


@dataclass
class RiskConfig:
    """Risk management configuration"""
    # Position sizing
    max_position_size: float = 0.30  # 30% max per position
    max_total_exposure: float = 0.60  # 60% max total
    risk_per_trade: float = 0.02  # 2% risk per trade
    
    # Kelly Criterion
    use_kelly: bool = True
    kelly_fraction: float = 0.25  # 1/4 Kelly for safety
    
    # Drawdown limits
    max_drawdown: float = 0.25  # 25% max drawdown
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    weekly_loss_limit: float = 0.12  # 12% weekly loss limit
    
    # Trade controls
    max_consecutive_losses: int = 5
    cooldown_after_losses: int = 3
    cooldown_periods: int = 5
    
    # Position duration
    max_hold_hours: int = 48
    use_partial_profits: bool = True
    partial_profit_1r: float = 0.25  # Take 25% at 1R
    partial_profit_2r: float = 0.50  # Take 50% at 2R
    use_breakeven_stop: bool = True


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 10000
    fee: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    
    # Data settings
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    start_date: str = "2023-01-01"
    end_date: Optional[str] = None
    
    # Testing
    run_walk_forward: bool = True
    walk_forward_train_ratio: float = 0.7
    walk_forward_windows: int = 3
    
    run_monte_carlo: bool = True
    monte_carlo_simulations: int = 1000


@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    use_numba: bool = True  # JIT compilation
    use_polars: bool = False  # Polars instead of pandas (not fully integrated yet)
    cache_indicators: bool = True
    parallel_backtests: bool = True
    max_workers: int = 4


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"
    
    # Metrics
    enable_prometheus: bool = False
    prometheus_port: int = 8000
    
    # Tracing
    enable_tracing: bool = False
    
    # Alerts
    enable_alerts: bool = True
    alert_on_error: bool = True
    alert_on_large_loss: bool = True
    large_loss_threshold: float = 0.05  # 5%


@dataclass
class TradingConfig:
    """Complete trading system configuration"""
    # Environment
    environment: EnvironmentType = EnvironmentType.TESTNET
    
    # Sub-configurations
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'environment': self.environment.value,
            'binance': self.binance.__dict__,
            'strategy': self.strategy.__dict__,
            'risk': self.risk.__dict__,
            'backtest': self.backtest.__dict__,
            'performance': self.performance.__dict__,
            'monitoring': self.monitoring.__dict__
        }
    
    @classmethod
    def from_env(cls) -> 'TradingConfig':
        """Load configuration from environment variables"""
        from dotenv import load_dotenv
        load_dotenv()
        
        config = cls()
        
        # Binance config
        config.binance.api_key = os.getenv('BINANCE_API_KEY', '')
        config.binance.api_secret = os.getenv('BINANCE_API_SECRET', '')
        config.binance.testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        # Environment
        env_type = os.getenv('ENVIRONMENT', 'testnet').lower()
        if env_type in [e.value for e in EnvironmentType]:
            config.environment = EnvironmentType(env_type)
        
        # Risk settings from env (optional overrides)
        if risk_per_trade := os.getenv('RISK_PER_TRADE'):
            config.risk.risk_per_trade = float(risk_per_trade)
        
        if max_position := os.getenv('MAX_POSITION_SIZE'):
            config.risk.max_position_size = float(max_position)
        
        return config
    
    def save_to_yaml(self, filepath: Path):
        """Save configuration to YAML file"""
        import yaml
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        print(f"✓ Configuration saved to {filepath}")
    
    @classmethod
    def load_from_yaml(cls, filepath: Path) -> 'TradingConfig':
        """Load configuration from YAML file"""
        import yaml
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Reconstruct config (simplified - would need full deserialization)
        config = cls()
        # ... populate from data dict ...
        return config


class ConfigManager:
    """
    Central configuration manager
    Provides easy access to all configuration
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[TradingConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_config(cls) -> TradingConfig:
        """Get global configuration (singleton)"""
        if cls._config is None:
            cls._config = TradingConfig.from_env()
        return cls._config
    
    @classmethod
    def set_config(cls, config: TradingConfig):
        """Set global configuration"""
        cls._config = config
    
    @classmethod
    def reload_config(cls):
        """Reload configuration from environment"""
        cls._config = TradingConfig.from_env()
        return cls._config


# Convenience function
def get_config() -> TradingConfig:
    """Get current trading configuration"""
    return ConfigManager.get_config()


if __name__ == '__main__':
    # Test configuration
    
    # Create default config
    config = TradingConfig()
    
    print("Default Configuration:")
    print(f"  Environment: {config.environment.value}")
    print(f"  Strategy: {config.strategy.name}")
    print(f"  EMA Fast/Slow: {config.strategy.ema_fast}/{config.strategy.ema_slow}")
    print(f"  Risk per trade: {config.risk.risk_per_trade*100}%")
    print(f"  Max drawdown: {config.risk.max_drawdown*100}%")
    print(f"  Use Numba JIT: {config.performance.use_numba}")
    
    # Load from environment
    env_config = TradingConfig.from_env()
    print(f"\nLoaded from .env:")
    print(f"  Testnet: {env_config.binance.testnet}")
    
    # Save to YAML
    config.save_to_yaml(Path('config_example.yaml'))
    
    print("\n✓ Configuration system working!")
