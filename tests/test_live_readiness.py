"""
Live Readiness Hardening Tests

Comprehensive tests for Phase 6 components:
- Context State Machine
- Risk State Persistence
- Trade Budget Manager
- Proposal Scorer
- Exchange Health Monitor
- Cold Start Controller
- Replay Engine
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

# Import components
from src.core.context_state_machine import (
    ContextStateMachine, ContextState, VALID_TRANSITIONS,
    reset_context_state_machine
)
from src.core.risk_state_persistence import (
    RiskStatePersistence, PersistedRiskState
)
from src.core.trade_budget_manager import (
    TradeBudgetManager, BudgetLimits, BudgetType
)
from src.core.proposal_scorer import (
    ProposalScorer, StrategyHealth
)
from src.core.exchange_health_monitor import (
    ExchangeHealthMonitor, HealthLevel
)
from src.core.cold_start_controller import (
    ColdStartController, ColdStartPhase
)
from src.core.replay_engine import (
    DeterministicReplayEngine, ReplayEvent
)


# ============= Context State Machine Tests =============

class TestContextStateMachine:
    """Tests for finite state machine with transitions and cooldowns."""
    
    def test_initial_state(self):
        """State machine starts with initial state."""
        machine = ContextStateMachine(initial_state=ContextState.RANGING)
        assert machine.current_state == ContextState.RANGING
    
    def test_valid_transition(self):
        """Valid transitions are allowed after min duration."""
        machine = ContextStateMachine(
            initial_state=ContextState.RANGING,
            min_state_duration_hours=0.0001,  # Nearly instant
            transition_cooldown_minutes=0.0001
        )
        
        # Wait for min duration
        machine._entered_at = datetime.now() - timedelta(hours=1)
        
        success, event = machine.transition_to(
            ContextState.TRENDING_UP,
            reason="Test transition"
        )
        
        assert success is True
        assert machine.current_state == ContextState.TRENDING_UP
        assert not event.was_blocked
    
    def test_invalid_transition_blocked(self):
        """Invalid transitions are blocked."""
        machine = ContextStateMachine(
            initial_state=ContextState.NO_TRADE,
            min_state_duration_hours=0.0001
        )
        machine._entered_at = datetime.now() - timedelta(hours=1)
        
        # NO_TRADE can only go to RANGING
        success, event = machine.transition_to(
            ContextState.TRENDING_UP,
            reason="Invalid"
        )
        
        assert success is False
        assert event.was_blocked
        assert "Invalid transition" in event.reason
    
    def test_min_duration_enforced(self):
        """Transitions blocked before min duration."""
        machine = ContextStateMachine(
            initial_state=ContextState.RANGING,
            min_state_duration_hours=2.0  # 2 hour minimum
        )
        
        # Just created, not enough time
        success, event = machine.transition_to(
            ContextState.TRENDING_UP,
            reason="Too soon"
        )
        
        assert success is False
        assert "Min duration" in event.reason
    
    def test_cooldown_enforced(self):
        """Cooldown period prevents rapid transitions."""
        machine = ContextStateMachine(
            initial_state=ContextState.RANGING,
            min_state_duration_hours=0.0001,
            transition_cooldown_minutes=30.0  # 30 min cooldown
        )
        machine._entered_at = datetime.now() - timedelta(hours=1)
        
        # First transition
        success1, _ = machine.transition_to(ContextState.TRENDING_UP, reason="First")
        assert success1
        
        # Immediate second transition should be blocked by cooldown
        machine._entered_at = datetime.now() - timedelta(hours=1)  # Pretend state is old enough
        success2, event2 = machine.transition_to(ContextState.RANGING, reason="Second")
        
        assert success2 is False
        assert "cooldown" in event2.reason.lower()
    
    def test_max_transitions_per_day(self):
        """Max transitions per 24h enforced."""
        machine = ContextStateMachine(
            initial_state=ContextState.RANGING,
            min_state_duration_hours=0.0001,
            transition_cooldown_minutes=0.0001,
            max_transitions_per_24h=2
        )
        
        # Perform 2 transitions
        machine._entered_at = datetime.now() - timedelta(hours=1)
        machine.transition_to(ContextState.TRENDING_UP, reason="1", force=True)
        machine.transition_to(ContextState.RANGING, reason="2", force=True)
        
        # 3rd should be blocked
        success, event = machine.transition_to(ContextState.TRENDING_UP, reason="3")
        
        assert success is False
        assert "Max transitions" in event.reason
    
    def test_force_no_trade(self):
        """Force transition to NO_TRADE works."""
        machine = ContextStateMachine(initial_state=ContextState.TRENDING_UP)
        
        event = machine.force_no_trade("Emergency")
        
        assert machine.current_state == ContextState.NO_TRADE
        assert "FORCED" in event.reason
    
    def test_serialization(self):
        """State machine can be serialized and restored."""
        machine1 = ContextStateMachine(initial_state=ContextState.TRENDING_UP)
        machine1._entered_at = datetime.now() - timedelta(hours=3)
        
        data = machine1.to_dict()
        machine2 = ContextStateMachine.from_dict(data)
        
        assert machine2.current_state == ContextState.TRENDING_UP


# ============= Risk State Persistence Tests =============

class TestRiskStatePersistence:
    """Tests for risk state persistence across restarts."""
    
    def test_create_default_state(self):
        """Creates default state when no file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = RiskStatePersistence(
                state_file=os.path.join(tmpdir, "risk.json")
            )
            
            state = persistence.load_state()
            
            assert state.daily_drawdown == 0.0
            assert state.kill_switch_active is False
            assert persistence.is_loaded
    
    def test_save_and_load(self):
        """State is saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = RiskStatePersistence(
                state_file=os.path.join(tmpdir, "risk.json")
            )
            
            # Load and modify
            state = persistence.load_state()
            state.daily_drawdown = -500.0
            state.consecutive_losses = 3
            persistence.save_state(state)
            
            # Create new persistence and load
            persistence2 = RiskStatePersistence(
                state_file=os.path.join(tmpdir, "risk.json")
            )
            state2 = persistence2.load_state()
            
            assert state2.daily_drawdown == -500.0
            assert state2.consecutive_losses == 3
    
    def test_checksum_validation(self):
        """Corrupted state is detected via checksum."""
        state = PersistedRiskState(daily_drawdown=-100.0)
        state.checksum = state.compute_checksum()
        
        assert state.validate_checksum() is True
        
        # Corrupt the data
        state.daily_drawdown = -999.0
        assert state.validate_checksum() is False
    
    def test_update_drawdown(self):
        """Drawdown updates correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = RiskStatePersistence(
                state_file=os.path.join(tmpdir, "risk.json")
            )
            persistence.load_state()
            
            # Record losses
            persistence.update_drawdown(-100.0)
            persistence.update_drawdown(-50.0)
            
            state = persistence.state
            assert state.daily_drawdown == -150.0
            assert state.consecutive_losses == 2
    
    def test_kill_switch_persists(self):
        """Kill switch state persists across restarts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = RiskStatePersistence(
                state_file=os.path.join(tmpdir, "risk.json")
            )
            persistence.load_state()
            persistence.activate_kill_switch("Test emergency")
            
            # Reload
            persistence2 = RiskStatePersistence(
                state_file=os.path.join(tmpdir, "risk.json")
            )
            state2 = persistence2.load_state()
            
            assert state2.kill_switch_active is True
            assert state2.kill_switch_reason == "Test emergency"


# ============= Trade Budget Manager Tests =============

class TestTradeBudgetManager:
    """Tests for trade budgeting and limits."""
    
    def test_initial_budgets(self):
        """Initial budgets are set correctly."""
        manager = TradeBudgetManager()
        status = manager.get_status()
        
        assert status.trades_remaining_day == 10
        assert status.trades_remaining_context == 3
        assert status.trading_allowed is True
    
    def test_daily_budget_exhaustion(self):
        """Trading blocked when daily budget exhausted."""
        manager = TradeBudgetManager(limits=BudgetLimits(max_trades_per_day=3))
        
        # Use up budget
        manager.record_trade()
        manager.record_trade()
        manager.record_trade()
        
        status = manager.get_status()
        assert status.trading_allowed is False
        assert BudgetType.DAILY_TRADES in status.exhausted_budgets
    
    def test_context_budget_reset(self):
        """Context budget resets on context change."""
        manager = TradeBudgetManager()
        
        manager.record_trade()
        manager.record_trade()
        assert manager.get_status().trades_remaining_context == 1
        
        manager.on_context_change()
        assert manager.get_status().trades_remaining_context == 3
    
    def test_bias_loss_budget(self):
        """Bias budget tracks losses correctly."""
        manager = TradeBudgetManager(limits=BudgetLimits(max_losses_per_bias=2))
        
        manager.record_loss()
        manager.record_loss()
        
        status = manager.get_status()
        assert BudgetType.BIAS_LOSSES in status.exhausted_budgets
        
        # Reset on bias change
        manager.on_bias_change()
        assert manager.get_status().losses_remaining_bias == 2


# ============= Proposal Scorer Tests =============

class TestProposalScorer:
    """Tests for proposal scoring and ranking."""
    
    def test_score_proposals(self):
        """Proposals are scored and ranked correctly."""
        scorer = ProposalScorer()
        
        proposals = [
            {
                'strategy_id': 'dca_btc',
                'direction': 'LONG',
                'confidence': 0.8,
                'entry_price': 40000,
                'stop_loss': 39000,
                'take_profit': 43000,
                'reasoning': 'DCA entry'
            },
            {
                'strategy_id': 'momentum_btc',
                'direction': 'LONG',
                'confidence': 0.6,
                'entry_price': 40000,
                'stop_loss': 38000,
                'take_profit': 44000,
                'reasoning': 'Momentum entry'
            }
        ]
        
        scored = scorer.score_proposals(
            proposals,
            context_regime='ranging',
            current_bias='long_only'
        )
        
        assert len(scored) == 2
        assert scored[0].strategy_id == 'dca_btc'  # Higher confidence
    
    def test_bias_filtering(self):
        """Proposals misaligned with bias are filtered."""
        scorer = ProposalScorer()
        
        proposals = [
            {'strategy_id': 'test', 'direction': 'SHORT', 'confidence': 0.9}
        ]
        
        scored = scorer.score_proposals(
            proposals,
            context_regime='trending_up',
            current_bias='long_only'
        )
        
        assert len(scored) == 0
    
    def test_strategy_health_decay(self):
        """Losing strategies decay in health."""
        scorer = ProposalScorer()
        
        health = scorer.get_strategy_health('test_strategy')
        initial_decay = health.decay_factor
        
        scorer.record_trade_result('test_strategy', is_win=False)
        scorer.record_trade_result('test_strategy', is_win=False)
        
        assert health.decay_factor < initial_decay
        assert health.consecutive_losses == 2
    
    def test_disabled_strategy_filtered(self):
        """Strategies with low health are filtered out."""
        scorer = ProposalScorer()
        
        # Decay to disabled threshold
        for _ in range(10):
            scorer.record_trade_result('bad_strategy', is_win=False)
        
        health = scorer.get_strategy_health('bad_strategy')
        assert health.is_disabled()
        
        proposals = [
            {'strategy_id': 'bad_strategy', 'direction': 'LONG', 'confidence': 0.9}
        ]
        
        scored = scorer.score_proposals(proposals, 'ranging', 'long_only')
        assert len(scored) == 0


# ============= Exchange Health Monitor Tests =============

class TestExchangeHealthMonitor:
    """Tests for exchange health monitoring."""
    
    def test_healthy_state(self):
        """Monitor reports healthy when all metrics good."""
        monitor = ExchangeHealthMonitor()
        
        # Record successful orders
        for i in range(10):
            monitor.record_order_result(
                order_id=f"order_{i}",
                latency_ms=100,
                was_rejected=False,
                fill_ratio=1.0
            )
        
        health = monitor.get_health()
        assert health.health_level == HealthLevel.HEALTHY
        assert health.execution_allowed is True
    
    def test_warning_on_high_latency(self):
        """Warning level on high latency."""
        monitor = ExchangeHealthMonitor()
        
        for i in range(10):
            monitor.record_order_result(
                order_id=f"order_{i}",
                latency_ms=800,  # Above 500ms warning threshold
                was_rejected=False
            )
        
        health = monitor.get_health()
        assert health.health_level == HealthLevel.WARNING
        assert any("Latency" in issue for issue in health.issues)
    
    def test_critical_on_rejections(self):
        """Critical level on high rejection rate."""
        monitor = ExchangeHealthMonitor()
        
        # Record mostly rejections
        for i in range(10):
            monitor.record_order_result(
                order_id=f"order_{i}",
                latency_ms=100,
                was_rejected=(i < 5)  # 50% rejection
            )
        
        health = monitor.get_health()
        assert health.health_level == HealthLevel.CRITICAL
        assert health.execution_allowed is False
    
    def test_manual_recovery_required(self):
        """After max consecutive failures, manual recovery required."""
        monitor = ExchangeHealthMonitor()
        
        for i in range(5):
            monitor.record_order_result(
                order_id=f"order_{i}",
                latency_ms=100,
                was_rejected=True
            )
        
        health = monitor.get_health()
        assert health.requires_manual_recovery is True
        assert health.execution_allowed is False
    
    def test_size_multiplier(self):
        """Size multiplier based on health."""
        monitor = ExchangeHealthMonitor()
        
        # Healthy
        for i in range(5):
            monitor.record_order_result(f"o{i}", 100, False)
        assert monitor.get_size_multiplier() == 1.0


# ============= Cold Start Controller Tests =============

class TestColdStartController:
    """Tests for cold start safety."""
    
    def test_not_ready_initially(self):
        """System not ready immediately after start."""
        controller = ColdStartController(min_warm_up_minutes=5.0)
        
        status = controller.get_status()
        assert status.is_ready is False
        assert status.phase == ColdStartPhase.INITIALIZING
    
    def test_progress_through_phases(self):
        """Controller progresses through phases."""
        controller = ColdStartController(min_warm_up_minutes=0.0001)
        
        controller.on_state_loaded()
        assert controller._phase == ColdStartPhase.SYNCING_DATA
        
        controller.on_data_synced(candle_count=150)
        controller.on_context_classified("trending_up")
        
        # Bias needs to stabilize
        controller.on_bias_determined("long_only")
        controller._bias_first_seen = datetime.now() - timedelta(minutes=5)
        controller.on_bias_determined("long_only")  # Same bias, now stable
        
        controller.on_exchange_healthy(score=0.9)
        
        # Force warm-up complete
        controller._started_at = datetime.now() - timedelta(minutes=10)
        
        status = controller.get_status()
        assert status.is_ready is True
    
    def test_insufficient_data_blocks(self):
        """Insufficient data keeps system not ready."""
        controller = ColdStartController()
        
        controller.on_state_loaded()
        controller.on_data_synced(candle_count=50)  # Not enough
        
        status = controller.get_status()
        assert status.is_ready is False
    
    def test_bias_instability_blocks(self):
        """Bias changes during warm-up reset stability."""
        controller = ColdStartController()
        
        controller.on_bias_determined("long_only")
        controller.on_bias_determined("short_only")  # Changed!
        
        assert controller._bias_stable is False
    
    def test_failure_on_timeout(self):
        """Startup fails after timeout."""
        controller = ColdStartController(startup_timeout_minutes=0.0001)
        controller._started_at = datetime.now() - timedelta(minutes=15)
        
        status = controller.get_status()
        
        assert status.is_failed is True
        assert "timeout" in status.failure_reason.lower()


# ============= Replay Engine Tests =============

class TestDeterministicReplayEngine:
    """Tests for deterministic replay functionality."""
    
    def test_recording_session(self):
        """Events and decisions are recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = DeterministicReplayEngine(data_dir=tmpdir)
            
            session_id = engine.start_recording("BTC/USDT")
            assert engine.is_recording
            
            engine.record_event('price', 'BTC/USDT', {'price': 40000})
            engine.record_event('price', 'BTC/USDT', {'price': 40100})
            engine.record_decision('context', 'BTC/USDT', 'RANGING', {})
            
            session = engine.stop_recording()
            
            assert session.event_count == 2
            assert session.decision_count == 1
    
    def test_session_persistence(self):
        """Sessions are saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = DeterministicReplayEngine(data_dir=tmpdir)
            
            session_id = engine.start_recording("BTC/USDT")
            engine.record_event('price', 'BTC/USDT', {'price': 40000})
            engine.stop_recording()
            
            # Load session
            loaded = engine.load_session(session_id)
            
            assert loaded is not None
            assert len(loaded.events) == 1
            assert loaded.events[0].data['price'] == 40000
    
    def test_list_sessions(self):
        """All sessions can be listed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = DeterministicReplayEngine(data_dir=tmpdir)
            
            engine.start_recording("BTC/USDT")
            engine.stop_recording()
            
            engine.start_recording("ETH/USDT")
            engine.stop_recording()
            
            sessions = engine.list_sessions()
            assert len(sessions) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
