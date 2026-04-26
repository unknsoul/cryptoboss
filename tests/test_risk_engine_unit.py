"""Unit coverage for the scalper risk engine."""

from src.risk.scalper_risk_engine import ScalperRiskConfig, ScalperRiskEngine


def test_can_open_position_false_when_max_positions_hit() -> None:
    """The engine should reject new positions once the max position cap is reached."""
    engine = ScalperRiskEngine(ScalperRiskConfig(max_positions=3))
    assert engine.can_open_position(current_open_positions=3) is False


def test_can_open_position_false_when_daily_loss_exceeded() -> None:
    """The daily loss guard should block new entries past the configured threshold."""
    engine = ScalperRiskEngine(ScalperRiskConfig(max_daily_loss_pct=0.05))
    assert engine.can_open_position(current_open_positions=1, daily_pnl=-600.0, account_balance=10000.0) is False


def test_position_size_inversely_proportional_to_stop_distance() -> None:
    """Wider stops should lead to smaller position sizes for the same risk budget."""
    engine = ScalperRiskEngine(ScalperRiskConfig(max_risk_pct=0.01))
    tight = engine.compute_position_size(10000.0, entry_price=100.0, stop_loss=99.0)
    wide = engine.compute_position_size(10000.0, entry_price=100.0, stop_loss=95.0)
    assert tight > wide


def test_break_even_trigger() -> None:
    """Long positions should move to break-even once the trigger move is reached."""
    stop = ScalperRiskEngine.check_break_even(entry=100.0, current_price=102.0, direction="long", rr_trigger=0.015)
    assert stop == 100.0


def test_break_even_not_triggered_too_soon() -> None:
    """Break-even should stay inactive until price moves far enough."""
    stop = ScalperRiskEngine.check_break_even(entry=100.0, current_price=100.5, direction="long", rr_trigger=0.015)
    assert stop is None


def test_trailing_stop_long() -> None:
    """Long trailing stops should sit below price once activated."""
    stop = ScalperRiskEngine.compute_trailing_stop(entry=100.0, current_price=105.0, atr=1.0, direction="long", trail_mult=1.5)
    assert stop == 103.5


def test_trailing_stop_short() -> None:
    """Short trailing stops should sit above price once activated."""
    stop = ScalperRiskEngine.compute_trailing_stop(entry=100.0, current_price=95.0, atr=1.0, direction="short", trail_mult=1.5)
    assert stop == 96.5


def test_daily_loss_halt_boundary() -> None:
    """The daily loss halt should trip exactly at the configured threshold."""
    engine = ScalperRiskEngine(ScalperRiskConfig(max_daily_loss_pct=0.05))
    assert engine.daily_loss_halt(daily_pnl=-500.0, account_balance=10000.0, max_daily_loss_pct=0.05) is True
    assert engine.daily_loss_halt(daily_pnl=-499.99, account_balance=10000.0, max_daily_loss_pct=0.05) is False
