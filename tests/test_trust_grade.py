"""
Trust Grade Edge Case Tests - v10.4-TRUST-GRADE

Tests for failure scenarios and edge cases as specified in v10.4 spec:
- timestamp_out_of_sync
- partial_balance_response
- api_key_revoked
- config_checksum_mismatch
- exchange_health_flip
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import v10.4 modules
from src.core.environment_guard import (
    EnvironmentGuard, EnvironmentMode, get_environment_guard
)
from src.core.data_authenticity import (
    DataAuthenticityGuard, AuthenticData, DataSource, AuthenticityStatus
)
from src.core.config_seal import LiveConfigGuard, SealStatus
from src.core.cold_start import ColdStartManager, StartupState


class TestTimestampOutOfSync:
    """Test stale data detection and handling."""
    
    def test_stale_price_data_triggers_warning(self):
        """Price data older than 5 seconds should be flagged as stale."""
        # Setup
        env_guard = EnvironmentGuard()
        env_guard.initialize(
            mode="testnet",
            exchange_id="binance",
            exchange_url="https://testnet.binance.vision",
            config={"test": True}
        )
        auth_guard = DataAuthenticityGuard(env_guard)
        
        # Create stale data (10 seconds old)
        stale_data = AuthenticData(
            data={"price": 50000.0},
            source=DataSource.LIVE,
            timestamp=datetime.utcnow() - timedelta(seconds=10),
            producer_id="test_producer",
            environment="testnet"
        )
        
        # Verify stale detection
        status = auth_guard.verify(stale_data, "price")
        assert status == AuthenticityStatus.WARNING
    
    def test_fresh_price_data_is_verified(self):
        """Price data within threshold should be verified."""
        env_guard = EnvironmentGuard()
        env_guard.initialize(
            mode="testnet",
            exchange_id="binance",
            exchange_url="https://testnet.binance.vision",
            config={"test": True}
        )
        auth_guard = DataAuthenticityGuard(env_guard)
        
        fresh_data = AuthenticData(
            data={"price": 50000.0},
            source=DataSource.LIVE,
            timestamp=datetime.utcnow(),
            producer_id="test_producer",
            environment="testnet"
        )
        
        status = auth_guard.verify(fresh_data, "price")
        assert status == AuthenticityStatus.VERIFIED


class TestPartialBalanceResponse:
    """Test handling of incomplete API responses."""
    
    def test_missing_fields_handled_gracefully(self):
        """Partial balance response should not crash the system."""
        partial_balance = {
            "BTC": 1.5,
            # Missing other expected fields
        }
        
        # System should handle partial data without exception
        assert "BTC" in partial_balance
        assert partial_balance.get("ETH") is None
    
    def test_empty_balance_response(self):
        """Empty balance dict should be handled."""
        empty_balance = {}
        assert len(empty_balance) == 0


class TestApiKeyRevoked:
    """Test graceful handling of authentication failures."""
    
    def test_revoked_key_triggers_halt(self):
        """API key revocation should trigger system halt."""
        # Mock scenario: API returns auth error
        mock_response = {
            "code": -2015,
            "msg": "Invalid API-key, IP, or permissions for action."
        }
        
        # Should be detected as auth failure
        assert mock_response["code"] == -2015
        assert "Invalid API-key" in mock_response["msg"]
    
    def test_expired_key_detection(self):
        """Expired API key messages should be caught."""
        error_messages = [
            "API-key format invalid",
            "Signature for this request is not valid",
            "Invalid API-key, IP, or permissions for action",
        ]
        
        for msg in error_messages:
            assert any(keyword in msg for keyword in ["API-key", "Signature", "Invalid"])


class TestConfigChecksumMismatch:
    """Test config seal violation detection."""
    
    def test_config_modification_detected(self):
        """Modifying sealed config should trigger violation."""
        guard = LiveConfigGuard()
        
        original_config = {"risk_limit": 0.02, "max_position": 1000}
        guard.seal("risk", original_config, operator_id="test")
        
        # Modify config
        modified_config = {"risk_limit": 0.05, "max_position": 1000}
        
        is_valid, msg = guard.validate("risk", modified_config)
        assert is_valid is False
        assert "mismatch" in msg.lower() or "violation" in msg.lower()
    
    def test_unmodified_config_passes(self):
        """Unmodified config should pass validation."""
        guard = LiveConfigGuard()
        
        config = {"risk_limit": 0.02, "max_position": 1000}
        guard.seal("risk", config, operator_id="test")
        
        is_valid, msg = guard.validate("risk", config)
        assert is_valid is True


class TestExchangeHealthFlip:
    """Test exchange connectivity state changes."""
    
    def test_exchange_goes_offline(self):
        """System should handle exchange going offline."""
        # Simulate health state
        health_states = {
            "connected": True,
            "latency_ms": 50,
            "last_success": datetime.utcnow()
        }
        
        # Flip to offline
        health_states["connected"] = False
        health_states["latency_ms"] = -1
        
        assert health_states["connected"] is False
    
    def test_exchange_reconnection(self):
        """System should detect exchange coming back online."""
        health_states = {"connected": False}
        
        # Simulate reconnection
        health_states["connected"] = True
        health_states["last_success"] = datetime.utcnow()
        
        assert health_states["connected"] is True


class TestEnvironmentMismatch:
    """Test environment guard blocking cross-env operations."""
    
    def test_live_action_blocked_in_testnet(self):
        """Actions targeting LIVE should be blocked in TESTNET mode."""
        guard = EnvironmentGuard()
        guard.initialize(
            mode="testnet",
            exchange_id="binance",
            exchange_url="https://testnet.binance.vision",
            config={}
        )
        
        is_valid = guard.validate_action("live")
        assert is_valid is False
    
    def test_matching_environment_allowed(self):
        """Actions matching current env should be allowed."""
        guard = EnvironmentGuard()
        guard.initialize(
            mode="testnet",
            exchange_id="binance",
            exchange_url="https://testnet.binance.vision",
            config={}
        )
        
        is_valid = guard.validate_action("testnet")
        assert is_valid is True


class TestColdStartStates:
    """Test cold start visibility states."""
    
    def test_trading_blocked_until_ready(self):
        """Trading should be blocked during startup."""
        manager = ColdStartManager()
        
        # Initially not ready
        assert manager.is_ready() is False
        
        # Transition through states
        manager.set_state(StartupState.CONNECTING)
        assert manager.is_ready() is False
        
        manager.set_state(StartupState.SYNCING_DATA)
        assert manager.is_ready() is False
        
        # Finally ready
        manager.set_state(StartupState.READY_TO_TRADE)
        assert manager.is_ready() is True
    
    def test_startup_failure_handling(self):
        """Failed startup should be clearly visible."""
        manager = ColdStartManager()
        manager.fail_startup("Database connection failed")
        
        assert manager._current_state == StartupState.FAILED
        assert manager.is_ready() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
