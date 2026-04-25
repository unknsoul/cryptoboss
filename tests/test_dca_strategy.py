"""
Unit Tests for DCA Strategy
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.strategies.dca_strategy import DCAStrategy, DCADeal


class TestDCADeal:
    """Test DCADeal class."""
    
    def test_deal_initialization(self):
        """Test deal is initialized correctly."""
        deal = DCADeal(
            deal_id="TEST_1",
            symbol="BTCUSDT",
            start_time=datetime.now(),
            base_order_price=60000.0,
            base_order_size=0.001,
            target_profit_pct=3.0,
            max_safety_orders=5,
            price_step_pct=2.5,
            safety_order_volume_scale=2.0
        )
        
        assert deal.deal_id == "TEST_1"
        assert deal.total_invested == pytest.approx(60.0, rel=0.01)  # 60000 * 0.001
        assert deal.total_quantity == 0.001
        assert deal.average_price == 60000.0
        assert deal.is_active == True
        assert len(deal.safety_orders_filled) == 0
    
    def test_take_profit_calculation(self):
        """Test take profit price calculation."""
        deal = DCADeal(
            deal_id="TEST_1",
            symbol="BTCUSDT",
            start_time=datetime.now(),
            base_order_price=60000.0,
            base_order_size=0.001,
            target_profit_pct=3.0,
            max_safety_orders=5,
            price_step_pct=2.5,
            safety_order_volume_scale=2.0
        )
        
        tp_price = deal.get_take_profit_price()
        assert tp_price == pytest.approx(61800.0, rel=0.01)  # 60000 * 1.03
    
    def test_safety_order_trigger_price(self):
        """Test safety order trigger price calculation."""
        deal = DCADeal(
            deal_id="TEST_1",
            symbol="BTCUSDT",
            start_time=datetime.now(),
            base_order_price=60000.0,
            base_order_size=0.001,
            target_profit_pct=3.0,
            max_safety_orders=5,
            price_step_pct=2.5,
            safety_order_volume_scale=2.0
        )
        
        # First safety order at -2.5%
        so1_price = deal.get_next_safety_order_price()
        assert so1_price == pytest.approx(58500.0, rel=0.01)  # 60000 * (1 - 0.025)
        
        # After adding safety order, next should be at -5%
        deal.add_safety_order(58500.0, 0.002, datetime.now())
        so2_price = deal.get_next_safety_order_price()
        assert so2_price == pytest.approx(57000.0, rel=0.01)  # 60000 * (1 - 0.05)
    
    def test_safety_order_size_martingale(self):
        """Test safety order size with Martingale."""
        deal = DCADeal(
            deal_id="TEST_1",
            symbol="BTCUSDT",
            start_time=datetime.now(),
            base_order_price=60000.0,
            base_order_size=0.001,
            target_profit_pct=3.0,
            max_safety_orders=5,
            price_step_pct=2.5,
            safety_order_volume_scale=2.0
        )
        
        # First SO: base_size * scale^0 = 0.001 * 1 = 0.001
        so1_size = deal.get_next_safety_order_size()
        assert so1_size == pytest.approx(0.001, rel=0.01)
        
        deal.add_safety_order(58500.0, so1_size, datetime.now())
        
        # Second SO: base_size * scale^1 = 0.001 * 2 = 0.002
        so2_size = deal.get_next_safety_order_size()
        assert so2_size == pytest.approx(0.002, rel=0.01)
    
    def test_average_price_after_safety_orders(self):
        """Test average price calculation after safety orders."""
        deal = DCADeal(
            deal_id="TEST_1",
            symbol="BTCUSDT",
            start_time=datetime.now(),
            base_order_price=60000.0,
            base_order_size=0.001,  # $60
            target_profit_pct=3.0,
            max_safety_orders=5,
            price_step_pct=2.5,
            safety_order_volume_scale=1.0  # Linear for easy calculation
        )
        
        # Add one safety order
        deal.add_safety_order(58500.0, 0.001, datetime.now())  # Another $58.50
        
        # Average should be (60000 + 58500) / 2 = 59250
        assert deal.average_price == pytest.approx(59250.0, rel=0.01)
        assert deal.total_quantity == pytest.approx(0.002, rel=0.01)
        assert deal.total_invested == pytest.approx(118.50, rel=0.01)
    
    def test_max_safety_orders_limit(self):
        """Test that we can't exceed max safety orders."""
        deal = DCADeal(
            deal_id="TEST_1",
            symbol="BTCUSDT",
            start_time=datetime.now(),
            base_order_price=60000.0,
            base_order_size=0.001,
            target_profit_pct=3.0,
            max_safety_orders=2,  # Only 2 allowed
            price_step_pct=2.5,
            safety_order_volume_scale=1.0
        )
        
        # Add 2 safety orders
        deal.add_safety_order(58500.0, 0.001, datetime.now())
        deal.add_safety_order(57000.0, 0.001, datetime.now())
        
        # Should be at limit
        assert len(deal.safety_orders_filled) == 2
        
        # Next order should be None
        assert deal.get_next_safety_order_price() is None
        assert deal.get_next_safety_order_size() is None
    
    def test_close_deal_profit(self):
        """Test closing deal with profit."""
        deal = DCADeal(
            deal_id="TEST_1",
            symbol="BTCUSDT",
            start_time=datetime.now(),
            base_order_price=60000.0,
            base_order_size=0.001,  # Invested $60
            target_profit_pct=3.0,
            max_safety_orders=5,
            price_step_pct=2.5,
            safety_order_volume_scale=1.0
        )
        
        # Close at profit target
        deal.close_deal(61800.0, datetime.now(), "TAKE_PROFIT")
        
        assert deal.is_active == False
        assert deal.exit_price == 61800.0
        assert deal.realized_pnl == pytest.approx(1.8, rel=0.01)  # (61800 - 60000) * 0.001
        assert deal.realized_pnl_pct == pytest.approx(3.0, rel=0.01)
        assert deal.exit_reason == "TAKE_PROFIT"
    
    def test_close_deal_loss(self):
        """Test closing deal with loss."""
        deal = DCADeal(
            deal_id="TEST_1",
            symbol="BTCUSDT",
            start_time=datetime.now(),
            base_order_price=60000.0,
            base_order_size=0.001,
            target_profit_pct=3.0,
            max_safety_orders=5,
            price_step_pct=2.5,
            safety_order_volume_scale=1.0
        )
        
        # Close at loss
        deal.close_deal(54000.0, datetime.now(), "STOP_LOSS")
        
        assert deal.is_active == False
        assert deal.realized_pnl < 0
        assert deal.realized_pnl == pytest.approx(-6.0, rel=0.01)  # (54000 - 60000) * 0.001
        assert deal.realized_pnl_pct == pytest.approx(-10.0, rel=0.01)


class TestDCAStrategy:
    """Test DCA Strategy."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        dates = pd.date_range(start='2024-01-01', periods=300, freq='H')
        
        # Create downtrend then recovery pattern (good for DCA)
        prices = []
        base_price = 60000
        for i in range(300):
            if i < 100:  # Downtrend
                price = base_price - (i * 100)
            elif i < 200:  # Sideways
                price = base_price - 10000 + np.random.normal(0, 500)
            else:  # Recovery
                price = base_price - 10000 + ((i - 200) * 200)
            prices.append(max(price, 40000))  # Floor at 40k
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [100] * 300
        })
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def test_strategy_initialization(self):
        """Test strategy initializes correctly."""
        dca = DCAStrategy(
            base_order_size=100,
            safety_order_size=200,
            max_safety_orders=5,
            price_step_pct=2.5,
            target_profit_pct=3.0
        )
        
        assert dca.base_order_size == 100
        assert dca.max_safety_orders == 5
        assert dca.active_deal is None
        assert len(dca.closed_deals) == 0
    
    def test_capital_calculation(self):
        """Test total capital requirement calculation."""
        dca = DCAStrategy(
            base_order_size=100,
            safety_order_size=200,
            max_safety_orders=5,
            safety_order_volume_scale=2.0  # Martingale
        )
        
        # Base: 100
        # SO1: 200 * 2^0 = 200
        # SO2: 200 * 2^1 = 400
        # SO3: 200 * 2^2 = 800
        # SO4: 200 * 2^3 = 1600
        # SO5: 200 * 2^4 = 3200
        # Total: 100 + 200 + 400 + 800 + 1600 + 3200 = 6300
        
        total = dca.calculate_total_investment()
        assert total == pytest.approx(6300, rel=0.01)
    
    def test_no_deal_during_cooldown(self, sample_data):
        """Test that no new deal starts during cooldown."""
        dca = DCAStrategy(
            base_order_size=100,
            safety_order_size=200,
            max_safety_orders=3,
            cooldown_bars=24
        )
        
        # Manually set cooldown
        dca.bars_since_last_deal = 10
        
        signal = dca.generate_signal(sample_data, 250, sample_data['close'].iloc[250])
        
        assert signal['action'] == 'HOLD'
        assert 'Cooldown' in signal['reason']
    
    def test_deal_lifecycle(self, sample_data):
        """Test complete deal lifecycle."""
        dca = DCAStrategy(
            base_order_size=100,
            safety_order_size=100,
            max_safety_orders=2,
            price_step_pct=5.0,  # Wide steps
            target_profit_pct=5.0,  # Easy target
            cooldown_bars=0
        )
        
        # Simulate trading
        for i in range(200, min(250, len(sample_data))):
            price = sample_data['close'].iloc[i]
            signal = dca.generate_signal(sample_data, i, price)
            
            if signal['action'] == 'BUY':
                print(f"BUY @ ${price:.2f}: {signal.get('reason')}")
            elif signal['action'] == 'SELL':
                print(f"SELL @ ${price:.2f}: P&L={signal.get('pnl_pct'):+.2f}%")
                break
        
        # Should have at least attempted a deal
        metrics = dca.get_metrics()
        print(f"\nMetrics: {metrics}")
        
        # Basic checks
        assert 'total_deals' in metrics
        assert 'win_rate_pct' in metrics


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
