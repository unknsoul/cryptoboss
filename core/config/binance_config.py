"""
Configuration Manager for Binance Trading
Handles testnet vs mainnet selection and API credentials
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class BinanceConfig:
    """Binance API configuration"""
    environment: str  # 'testnet' or 'mainnet'
    api_key: str
    secret_key: str
    ws_url: str
    rest_url: str


class ConfigManager:
    """
    Manages configuration for Binance trading
    
    Supports both testnet and mainnet with easy switching
    """
    
    # Binance endpoints
    TESTNET_WS = "wss://testnet.binance.vision/ws"
    TESTNET_REST = "https://testnet.binance.vision/api"
    
    MAINNET_WS = "wss://stream.binance.com:9443/ws"
    MAINNET_REST = "https://api.binance.com/api"
    
    def __init__(self):
        """Initialize configuration from environment variables"""
        self.env = os.getenv('BINANCE_ENV', 'testnet').lower()
        
        if self.env not in ['testnet', 'mainnet']:
            raise ValueError(f"Invalid BINANCE_ENV: {self.env}. Must be 'testnet' or 'mainnet'")
        
        logger.info(f"🔧 Configuring for Binance {self.env.upper()}")
        
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> BinanceConfig:
        """Load configuration based on environment"""
        if self.env == 'testnet':
            return BinanceConfig(
                environment='testnet',
                api_key=os.getenv('BINANCE_TESTNET_API_KEY', ''),
                secret_key=os.getenv('BINANCE_TESTNET_SECRET_KEY', ''),
                ws_url=self.TESTNET_WS,
                rest_url=self.TESTNET_REST
            )
        else:  # mainnet
            return BinanceConfig(
                environment='mainnet',
                api_key=os.getenv('BINANCE_MAINNET_API_KEY', ''),
                secret_key=os.getenv('BINANCE_MAINNET_SECRET_KEY', ''),
                ws_url=self.MAINNET_WS,
                rest_url=self.MAINNET_REST
            )
    
    def _validate_config(self):
        """Validate configuration"""
        if not self.config.api_key or not self.config.secret_key:
            logger.warning(f"⚠️ API keys not configured for {self.env}")
            logger.warning(f"   WebSocket data will work, but order execution requires API keys")
            logger.warning(f"   Set BINANCE_{self.env.upper()}_API_KEY and SECRET_KEY in .env")
        else:
            logger.info(f"✅ API keys configured for {self.env}")
    
    def get_websocket_url(self) -> str:
        """Get WebSocket URL"""
        return self.config.ws_url
    
    def get_rest_url(self) -> str:
        """Get REST API URL"""
        return self.config.rest_url
    
    def get_api_credentials(self) -> tuple:
        """Get (api_key, secret_key) tuple"""
        return (self.config.api_key, self.config.secret_key)
    
    def is_testnet(self) -> bool:
        """Check if using testnet"""
        return self.env == 'testnet'
    
    def is_mainnet(self) -> bool:
        """Check if using mainnet"""
        return self.env == 'mainnet'
    
    def print_status(self):
        """Print current configuration status"""
        print("\n" + "="*60)
        print("📡 BINANCE CONFIGURATION")
        print("="*60)
        print(f"Environment:  {self.config.environment.upper()}")
        print(f"WebSocket:    {self.config.ws_url}")
        print(f"REST API:     {self.config.rest_url}")
        
        if self.config.api_key:
            masked_key = self.config.api_key[:8] + "..." + self.config.api_key[-4:]
            print(f"API Key:      {masked_key} ✅")
        else:
            print(f"API Key:      Not configured ⚠️")
        
        if self.is_testnet():
            print("\n⚠️  TESTNET MODE - Safe for testing")
        else:
            print("\n🚨 MAINNET MODE - REAL MONEY AT RISK!")
        
        print("="*60 + "\n")
    
    @staticmethod
    def get_trading_config():
        """Get trading configuration from environment"""
        return {
            'initial_capital': float(os.getenv('INITIAL_CAPITAL', '10000')),
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.02')),
            'symbol': os.getenv('SYMBOL', 'btcusdt'),
            'timeframes': os.getenv('TIMEFRAMES', '1h').split(','),
            'max_drawdown_limit': float(os.getenv('MAX_DRAWDOWN_LIMIT', '0.20')),
            'daily_loss_limit': float(os.getenv('DAILY_LOSS_LIMIT', '0.05'))
        }


# Singleton instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get configuration manager singleton"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# Example usage
if __name__ == "__main__":
    import sys
    
    # Test configuration
    config = get_config()
    config.print_status()
    
    # Show trading config
    trading_config = ConfigManager.get_trading_config()
    print("\n📊 TRADING CONFIGURATION")
    print("="*60)
    for key, value in trading_config.items():
        print(f"{key:20s}: {value}")
    print("="*60)
