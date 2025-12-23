"""
Input Validation with Pydantic V2
Ensures all trading requests are valid and safe
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal, Optional
from datetime import datetime
from decimal import Decimal


class TradeRequest(BaseModel):
    """Validated trade request model"""
    
    model_config = ConfigDict(frozen=True, strict=True)
    
    symbol: str = Field(pattern=r'^[A-Z]{3,10}USDT$', description="Trading pair (e.g., BTCUSDT)")
    side: Literal['LONG', 'SHORT', 'BUY', 'SELL']
    size: float = Field(gt=0, le=1.0, description="Position size as fraction of capital (0-1)")
    price: Optional[float] = Field(None, gt=0, description="Limit price (None for market order)")
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[float] = Field(None, gt=0, description="Take profit price")
    
    @field_validator('size')
    @classmethod
    def validate_size(cls, v: float) -> float:
        """Ensure size is reasonable"""
        if v < 0.001:  # Minimum 0.1%
            raise ValueError('Size too small (min 0.1%)')
        if v > 0.95:  # Maximum 95% (safety margin)
            raise ValueError('Size too large (max 95%)')
        return v
    
    @field_validator('stop_loss', 'take_profit')
    @classmethod
    def validate_prices(cls, v: Optional[float]) -> Optional[float]:
        """Ensure prices are positive if provided"""
        if v is not None and v <= 0:
            raise ValueError('Price must be positive')
        return v
    
    def validate_logic(self, current_price: float):
        """
        Validate trade logic (stop loss below entry for LONG, etc.)
        
        Args:
            current_price: Current market price
            
        Raises:
            ValueError: If trade logic is invalid
        """
        if self.side in ['LONG', 'BUY']:
            if self.stop_loss and self.stop_loss >= current_price:
                raise ValueError(f'Stop loss ({self.stop_loss}) must be below entry ({current_price}) for LONG')
            if self.take_profit and self.take_profit <= current_price:
                raise ValueError(f'Take profit ({self.take_profit}) must be above entry ({current_price}) for LONG')
        
        elif self.side in ['SHORT', 'SELL']:
            if self.stop_loss and self.stop_loss <= current_price:
                raise ValueError(f'Stop loss ({self.stop_loss}) must be above entry ({current_price}) for SHORT')
            if self.take_profit and self.take_profit >= current_price:
                raise ValueError(f'Take profit ({self.take_profit}) must be below entry ({current_price}) for SHORT')


class BacktestConfig(BaseModel):
    """Validated backtest configuration"""
    
    model_config = ConfigDict(frozen=True)
    
    capital: float = Field(default=10000, gt=0, description="Initial capital")
    risk_per_trade: float = Field(default=0.02, gt=0, le=0.1, description="Risk per trade (0-0.1)")
    fee: float = Field(default=0.001, ge=0, le=0.01, description="Trading fee (0-0.01)")
    slippage: float = Field(default=0.0005, ge=0, le=0.01, description="Slippage (0-0.01)")
    max_drawdown_limit: float = Field(default=0.20, gt=0, le=0.50, description="Max drawdown limit")
    daily_loss_limit: float = Field(default=0.05, gt=0, le=0.20, description="Daily loss limit")
    
    @field_validator('risk_per_trade')
    @classmethod
    def validate_risk(cls, v: float) -> float:
        """Ensure risk is reasonable"""
        if v > 0.05:  # More than 5% is aggressive
            import warnings
            warnings.warn(f'Risk per trade {v*100:.1f}% is very aggressive. Consider reducing.')
        return v


class StrategyConfig(BaseModel):
    """Validated strategy configuration"""
    
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(description="Strategy name")
    ema_fast: int = Field(default=50, ge=5, le=200, description="Fast EMA period")
    ema_slow: int = Field(default=200, ge=20, le=500, description="Slow EMA period")
    donchian_period: int = Field(default=20, ge=10, le=100, description="Donchian period")
    atr_period: int = Field(default=14, ge=5, le=50, description="ATR period")
    atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0, description="ATR multiplier for stops")
    
    @field_validator('ema_slow')
    @classmethod
    def validate_ema_relationship(cls, v: int, info) -> int:
        """Ensure slow EMA is greater than fast EMA"""
        if 'ema_fast' in info.data and v <= info.data['ema_fast']:
            raise ValueError(f'Slow EMA ({v}) must be greater than fast EMA ({info.data["ema_fast"]})')
        return v


class RiskLimits(BaseModel):
    """Validated risk limits"""
    
    model_config = ConfigDict(frozen=True)
    
    max_position_size: float = Field(default=0.30, gt=0, le=0.95)
    max_total_exposure: float = Field(default=0.60, gt=0, le=1.0)
    max_drawdown: float = Field(default=0.25, gt=0, le=0.50)
    max_daily_loss: float = Field(default=0.05, gt=0, le=0.20)
    max_consecutive_losses: int = Field(default=5, ge=1, le=20)
    
    @field_validator('max_total_exposure')
    @classmethod
    def validate_exposure(cls, v: float, info) -> float:
        """Ensure total exposure is greater than single position"""
        if 'max_position_size' in info.data and v < info.data['max_position_size']:
            raise ValueError('Total exposure must be >= max position size')
        return v


class TradeRequestValidator:
    """Helper class for validating trade requests"""
    
    @staticmethod
    def validate_trade(
        symbol: str,
        side: str,
        size: float,
        current_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeRequest:
        """
        Validate and create trade request
        
        Returns:
            Validated TradeRequest object
            
        Raises:
            ValidationError: If validation fails
        """
        request = TradeRequest(
            symbol=symbol,
            side=side,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Validate trade logic
        request.validate_logic(current_price)
        
        return request


if __name__ == '__main__':
    # Test validation
    
    # Valid trade
    try:
        trade = TradeRequest(
            symbol='BTCUSDT',
            side='LONG',
            size=0.02,
            stop_loss=95000,
            take_profit=105000
        )
        trade.validate_logic(current_price=100000)
        print("✓ Valid trade request")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Invalid trade (stop loss above entry for LONG)
    try:
        trade = TradeRequest(
            symbol='BTCUSDT',
            side='LONG',
            size=0.02,
            stop_loss=105000,  # Wrong - above entry
            take_profit=110000
        )
        trade.validate_logic(current_price=100000)
        print("✗ Should have failed validation")
    except ValueError as e:
        print(f"✓ Correctly rejected invalid trade: {e}")
    
    # Test configuration
    config = BacktestConfig(
        capital=10000,
        risk_per_trade=0.02,
        fee=0.001
    )
    print(f"✓ Valid backtest config: {config.capital} capital, {config.risk_per_trade*100}% risk")
