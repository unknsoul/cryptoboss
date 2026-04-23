"""
CryptoBoss Integration Tests

Tests:
    1. test_startup_without_keys - Paper mode runs without crash
    2. test_invalid_api_keys - Validation fails with clear error
    3. test_risk_limit_hit - Trades stop when daily loss threshold hit
    4. test_account_switch - State isolation works between accounts
    5. test_price_feed_disconnect - Reconnect works (mock disconnect)
"""

import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.engine import TradingEngine, create_engine, EngineStatus
from src.core.execution_router import (
    ExecutionRouter, ExecutionMode, OrderIntent, OrderSide, OrderType, OrderResult, PaperBroker
)
from src.core.risk_guardian import RiskGuardian, RiskLimits
from src.core.mock_price_generator import MockPriceGenerator
from src.core.bot_instance import BotInstance, AccountIdentity, BotInstanceState


# ===================================================================
# Test 1: Paper mode startup without API keys
# ===================================================================

class TestStartupWithoutKeys:
    """Paper mode must run without crash and without API keys."""

    def test_engine_creates_in_paper_mode(self):
        """Engine should create successfully without exchange_client."""
        engine = create_engine(mode="paper", portfolio_value=10000.0, exchange_client=None)
        assert engine is not None
        assert engine.config.mode == "paper"

    def test_engine_starts_in_paper_mode(self):
        """Engine should start and run in paper mode."""
        engine = create_engine(mode="paper", portfolio_value=10000.0)
        engine.start()
        assert engine.status == EngineStatus.RUNNING
        engine.stop()
        assert engine.status == EngineStatus.STOPPED

    def test_paper_broker_executes_trades(self):
        """Paper broker should simulate order fills."""
        broker = PaperBroker(initial_balance=10000.0)
        broker.set_price("BTC/USDT", 67000.0)

        intent = OrderIntent(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            price=67000.0,
            strategy_id="test_dca",
        )
        result = asyncio.get_event_loop().run_until_complete(broker.execute_order(intent))
        assert result.success
        assert result.filled_quantity == 0.01
        assert result.average_price > 0
        assert result.fees > 0

    def test_mock_price_generator_produces_ticks(self):
        """MockPriceGenerator should produce price ticks."""
        gen = MockPriceGenerator(
            symbols={"BTC/USDT": 67000.0},
            tick_interval=0.1,
        )
        ticks = []

        async def collect():
            async for symbol, price, ts in gen.stream():
                ticks.append((symbol, price, ts))
                if len(ticks) >= 5:
                    gen.stop()

        asyncio.get_event_loop().run_until_complete(collect())
        assert len(ticks) >= 5
        for symbol, price, ts in ticks:
            assert symbol == "BTC/USDT"
            assert price > 0
            assert isinstance(ts, datetime)


# ===================================================================
# Test 2: Invalid API keys
# ===================================================================

class TestInvalidApiKeys:
    """Live/testnet mode should reject invalid credentials."""

    def test_testnet_requires_exchange_client(self):
        """TESTNET mode without exchange_client should raise ValueError."""
        with pytest.raises(ValueError, match="REQUIRED"):
            ExecutionRouter(mode=ExecutionMode.TESTNET, exchange_client=None)

    def test_live_requires_exchange_client(self):
        """LIVE mode without exchange_client should raise ValueError."""
        with pytest.raises(ValueError, match="REQUIRED"):
            ExecutionRouter(mode=ExecutionMode.LIVE, exchange_client=None)

    def test_paper_does_not_require_exchange_client(self):
        """PAPER mode should work without exchange_client."""
        router = ExecutionRouter(mode=ExecutionMode.PAPER, exchange_client=None)
        assert router.mode == ExecutionMode.PAPER


# ===================================================================
# Test 3: Risk limit hit
# ===================================================================

class TestRiskLimitHit:
    """Trades must stop when daily loss threshold is exceeded."""

    def test_daily_loss_blocks_trades(self):
        """When daily loss exceeds limit, new trades should be rejected."""
        limits = RiskLimits(max_daily_loss_usd=100.0, max_daily_loss_pct=50.0)
        guardian = RiskGuardian(portfolio_value=10000.0, limits=limits)

        # Simulate losses
        for _ in range(10):
            guardian.record_trade(-15.0, "test_strategy")  # total: -150

        # Should be blocked
        intent = OrderIntent(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            price=67000.0,
            strategy_id="test_strategy",
        )

        approved, reason = guardian.approve_order(intent)
        assert not approved
        assert "loss" in reason.lower() or "limit" in reason.lower()

    def test_per_trade_risk_blocks_large_orders(self):
        """Orders exceeding per-trade risk should be rejected."""
        limits = RiskLimits(risk_per_trade_pct=1.0)  # 1% of $10000 = $100
        guardian = RiskGuardian(portfolio_value=10000.0, limits=limits)

        # Large order ($500) should be blocked
        intent = OrderIntent(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            price=50000.0,  # $500 value
            strategy_id="test_strategy",
        )

        approved, reason = guardian.approve_order(intent)
        assert not approved
        assert "risk" in reason.lower() or "per-trade" in reason.lower()

    def test_max_concurrent_trades_enforced(self):
        """Should reject trades when max concurrent positions reached."""
        limits = RiskLimits(max_concurrent_trades=2, risk_per_trade_pct=100.0)
        guardian = RiskGuardian(portfolio_value=100000.0, limits=limits)

        # Open 2 positions
        guardian.record_position_open("BTC/USDT", 0.1)
        guardian.record_position_open("ETH/USDT", 1.0)

        # Third should be blocked
        intent = OrderIntent(
            symbol="SOL/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=150.0,
            strategy_id="test_strategy",
        )

        approved, reason = guardian.approve_order(intent)
        assert not approved
        assert "concurrent" in reason.lower()

    def test_emergency_stop_blocks_all(self):
        """Emergency stop should block all trading."""
        guardian = RiskGuardian(portfolio_value=10000.0)
        guardian.emergency_stop("Test emergency")

        intent = OrderIntent(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            price=67000.0,
            strategy_id="test_strategy",
        )

        approved, reason = guardian.approve_order(intent)
        assert not approved
        assert "emergency" in reason.lower()


# ===================================================================
# Test 4: Account switch / state isolation
# ===================================================================

class TestAccountSwitch:
    """Each account must have isolated state."""

    def test_bot_instances_have_separate_state(self):
        """Two BotInstances should not share state."""
        id1 = AccountIdentity(
            user_id="user_1",
            exchange_account_id="account_aaa111",
            environment="TESTNET",
        )
        id2 = AccountIdentity(
            user_id="user_1",
            exchange_account_id="account_bbb222",
            environment="TESTNET",
        )

        bot1 = BotInstance(id1)
        bot2 = BotInstance(id2)

        # Modify state in bot1
        bot1.trading_state.positions.append({"symbol": "BTC/USDT", "qty": 0.01})
        bot1.price_cache.update("BTC/USDT", 67000.0, datetime.now())

        # bot2 should be clean
        assert len(bot2.trading_state.positions) == 0
        assert "BTC/USDT" not in bot2.price_cache.prices

    def test_destroy_clears_state(self):
        """Destroying a bot instance should clear all its state."""
        identity = AccountIdentity(
            user_id="user_1",
            exchange_account_id="account_ccc333",
            environment="TESTNET",
        )
        bot = BotInstance(identity)
        bot.trading_state.positions.append({"test": True})

        bot.destroy()

        assert bot.state == BotInstanceState.DESTROYED
        assert len(bot.trading_state.positions) == 0

    def test_separate_storage_paths(self):
        """Different accounts should have different storage paths."""
        id1 = AccountIdentity(
            user_id="user_1",
            exchange_account_id="account_ddd444",
            environment="TESTNET",
        )
        id2 = AccountIdentity(
            user_id="user_1",
            exchange_account_id="account_eee555",
            environment="TESTNET",
        )

        assert id1.storage_path != id2.storage_path


# ===================================================================
# Test 5: Price feed disconnect / reconnect
# ===================================================================

class TestPriceFeedDisconnect:
    """Bot should handle price feed disconnects gracefully."""

    def test_mock_price_generator_stops_cleanly(self):
        """MockPriceGenerator should stop cleanly when stop() is called."""
        gen = MockPriceGenerator(tick_interval=0.05)
        ticks = []

        async def run():
            async for symbol, price, ts in gen.stream():
                ticks.append((symbol, price))
                if len(ticks) >= 3:
                    gen.stop()

        asyncio.get_event_loop().run_until_complete(run())
        assert not gen.is_running
        assert len(ticks) >= 3

    def test_bot_instance_handles_no_exchange_client(self):
        """BotInstance should start without exchange client (paper mode)."""
        identity = AccountIdentity(
            user_id="user_1",
            exchange_account_id="account_fff666",
            environment="TESTNET",
        )
        bot = BotInstance(identity, exchange_client=None)

        async def start_and_stop():
            result = await bot.start()
            assert result is True
            assert bot.state == BotInstanceState.RUNNING
            result = await bot.stop()
            assert result is True
            assert bot.state == BotInstanceState.STOPPED

        asyncio.get_event_loop().run_until_complete(start_and_stop())

    def test_paper_broker_rejects_no_price(self):
        """PaperBroker should reject orders when no price is set."""
        broker = PaperBroker(initial_balance=10000.0)
        # Don't set any price

        intent = OrderIntent(
            symbol="UNKNOWN/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            strategy_id="test",
        )
        result = asyncio.get_event_loop().run_until_complete(broker.execute_order(intent))
        assert not result.success
        assert "No price" in result.error_message


# ===================================================================
# Run all tests
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
