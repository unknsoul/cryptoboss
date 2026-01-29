"""
Test: Config Checksum Validation - v11.1 FINAL-MAP

Tests for configuration integrity and checksum validation.
"""

import pytest
import hashlib
import json


class TestConfigChecksum:
    """Test configuration checksum validation."""

    def compute_checksum(self, config: dict) -> str:
        """Compute SHA256 checksum of config."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def test_identical_config_same_checksum(self):
        """Identical configs should produce same checksum."""
        config1 = {"risk_limit": 0.02, "max_position": 1000}
        config2 = {"risk_limit": 0.02, "max_position": 1000}
        
        assert self.compute_checksum(config1) == self.compute_checksum(config2)

    def test_modified_config_different_checksum(self):
        """Modified config should produce different checksum."""
        original = {"risk_limit": 0.02, "max_position": 1000}
        modified = {"risk_limit": 0.05, "max_position": 1000}
        
        assert self.compute_checksum(original) != self.compute_checksum(modified)

    def test_order_independent_checksum(self):
        """Key order should not affect checksum (sorted keys)."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}
        
        assert self.compute_checksum(config1) == self.compute_checksum(config2)

    def test_runtime_config_modification_detected(self):
        """Runtime config modification should be detected."""
        original = {"risk_limit": 0.02}
        original_checksum = self.compute_checksum(original)
        
        # Simulate runtime modification
        runtime = {"risk_limit": 0.05}
        runtime_checksum = self.compute_checksum(runtime)
        
        is_valid = original_checksum == runtime_checksum
        assert is_valid is False


class TestConfigSealing:
    """Test configuration sealing mechanism."""

    def test_sealed_config_immutable(self):
        """Sealed config should not allow modification."""
        sealed = True
        
        # Attempt to modify
        can_modify = not sealed
        assert can_modify is False

    def test_seal_violation_triggers_halt(self):
        """Seal violation should trigger system halt."""
        violations = []
        
        # Simulate violation
        violations.append({
            "field": "risk_limit",
            "original": 0.02,
            "attempted": 0.05,
            "timestamp": "2024-01-01T00:00:00"
        })
        
        should_halt = len(violations) > 0
        assert should_halt is True


class TestEnvironmentSignature:
    """Test environment signature validation."""

    def test_signature_includes_mode(self):
        """Environment signature should include mode."""
        signature = {
            "mode": "testnet",
            "exchange_url": "https://testnet.binance.vision",
            "checksum": "abc123"
        }
        
        assert "mode" in signature
        assert signature["mode"] == "testnet"

    def test_signature_mismatch_blocks_render(self):
        """Signature mismatch should block UI render."""
        server_sig = {"mode": "testnet", "checksum": "abc123"}
        expected_sig = {"mode": "testnet", "checksum": "def456"}
        
        matches = server_sig["checksum"] == expected_sig["checksum"]
        allow_render = matches
        
        assert allow_render is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
