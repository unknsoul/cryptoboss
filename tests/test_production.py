"""
Test Suite for Production Components
"""

import pytest
from src.backtest import RealBacktestEngine, SlippageModel
from src.data import FeatureEngine
from src.models import MLPipeline, ModelRegistry
from src.risk import VolatilityAdjustedSizing, KillSwitch
from src.exchange import MockExchange, get_exchange
import pandas as pd
import numpy as np


class TestBacktestEngine:
    """Test backtesting engine."""
    
    def test_engine_initialization(self):
        engine = RealBacktestEngine(initial_capital=10000)
        assert engine.initial_capital == 10000
        assert engine.equity == 10000
    
    def test_slippage_calculation(self):
        slippage = SlippageModel.adaptive_slippage(
            price=40000,
            size=0.5,
            side='BUY',
            volatility=0.02
        )
        assert slippage > 0  # Should have positive slippage for buys


class TestFeatureEngine:
    """Test feature engineering."""
    
    def test_feature_generation(self):
        engine = FeatureEngine()
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=200, freq='1h')
        df = pd.DataFrame({
            'open': np.random.uniform(39000, 41000, 200),
            'high': np.random.uniform(40000, 42000, 200),
            'low': np.random.uniform(38000, 40000, 200),
            'close': np.random.uniform(39500, 40500, 200),
            'volume': np.random.uniform(1000, 5000, 200)
        }, index=dates)
        
        features = engine.generate_features(df)
        
        assert len(features) > 0
        assert 'returns_1' in features.columns
        assert 'volatility_atr_14' in features.columns
    
    def test_feature_consistency(self):
        engine = FeatureEngine()
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=200, freq='1h')
        df = pd.DataFrame({
            'open': np.random.uniform(39000, 41000, 200),
            'high': np.random.uniform(40000, 42000, 200),
            'low': np.random.uniform(38000, 40000, 200),
            'close': np.random.uniform(39500, 40500, 200),
            'volume': np.random.uniform(1000, 5000, 200)
        }, index=dates)
        
        features1 = engine.generate_features(df)
        features2 = engine.generate_features(df)
        
        # Should be identical
        assert features1.equals(features2)


class TestRiskEngine:
    """Test risk management."""
    
    def test_volatility_sizing(self):
        sizer = VolatilityAdjustedSizing(base_risk_pct=1.0)
        
        result = sizer.calculate_position_size(
            equity=10000,
            price=40000,
            atr=400,
            volatility_regime='normal'
        )
        
        assert result['quantity'] > 0
        assert result['risk_pct'] > 0
        assert result['stop_loss'] > 0
    
    def test_kill_switch(self):
        kill_switch = KillSwitch(initial_equity=10000)
        
        # Test normal conditions
        result = kill_switch.check_halt_conditions(10100)
        assert not result['should_halt']
        
        # Test daily loss trigger
        result = kill_switch.check_halt_conditions(9400)  # -6%
        assert result['should_halt']
        assert 'Daily loss' in result['reason']


class TestExchange:
    """Test exchange abstraction."""
    
    def test_mock_exchange(self):
        exchange = MockExchange()
        
        # Test ticker
        ticker = exchange.get_ticker('BTCUSDT')
        assert 'price' in ticker
        assert ticker['price'] > 0
        
        # Test order placement
        order = exchange.place_order(
            symbol='BTCUSDT',
            side='BUY',
            order_type='MARKET',
            quantity=0.1
        )
        assert order['status'] == 'FILLED'
    
    def test_exchange_factory(self):
        exchange = get_exchange('mock')
        assert isinstance(exchange, MockExchange)


class TestMLPipeline:
    """Test ML pipeline."""
    
    def test_label_creation(self):
        ml = MLPipeline()
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=200, freq='1h')
        df = pd.DataFrame({
            'close': np.random.uniform(39500, 40500, 200)
        }, index=dates)
        
        labels = ml.create_labels(df, horizon=4, threshold_pct=0.5)
        
        assert len(labels) == len(df)
        assert labels.dtype == int


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
