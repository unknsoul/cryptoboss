from datetime import datetime

from src.core.trade_management_engine import (
    ManagedTrade,
    PositionSide,
    TPLevel,
    TradeManagementEngine,
)


def _managed_trade() -> ManagedTrade:
    return ManagedTrade(
        trade_id="T1",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=100.0,
        stop_loss=98.0,
        size=1.0,
        opened_at=datetime.utcnow(),
        tp_levels=[
            TPLevel(tp_id="TP1", price=102.0, close_pct=40.0),
            TPLevel(tp_id="TP2", price=104.0, close_pct=35.0),
            TPLevel(tp_id="TP3", price=106.0, close_pct=25.0),
        ],
    )


def test_tp1_moves_sl_to_breakeven() -> None:
    engine = TradeManagementEngine()
    trade = _managed_trade()

    decision = engine.evaluate(trade, current_price=102.1)

    assert "TP1_hit" in decision.actions
    assert "move_sl_to_breakeven" in decision.actions
    assert trade.moved_to_breakeven is True
    assert trade.stop_loss >= trade.entry_price


def test_structure_invalidation_closes_trade() -> None:
    engine = TradeManagementEngine()
    trade = _managed_trade()

    decision = engine.evaluate(trade, current_price=99.0, structure_invalidated=True)

    assert decision.close_all is True
    assert "close_immediately" in decision.actions
    assert trade.closed is True


def test_trailing_stop_updates_after_tp2() -> None:
    engine = TradeManagementEngine(atr_trail_multiplier=1.0)
    trade = _managed_trade()

    # Hit TP1 and TP2.
    engine.evaluate(trade, current_price=102.2)
    decision = engine.evaluate(trade, current_price=104.2, last_swing_low=103.1)

    assert "TP2_hit" in decision.actions
    assert trade.trailing_active is True

    # Follow-up tick should move trailing SL higher when swing low rises.
    follow = engine.evaluate(trade, current_price=105.0, last_swing_low=103.8)
    assert "update_trailing_stop" in follow.actions or trade.stop_loss >= 103.1
