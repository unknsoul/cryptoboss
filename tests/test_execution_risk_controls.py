from datetime import datetime

from src.execution.smart_routing import smart_execute
from src.risk.controls import RiskConfig, RiskController


class DummyBroker:
    def place_order(self, symbol, side, size, order_type="MARKET", price=None, client_order_id=None):
        return {
            "status": "filled",
            "symbol": symbol,
            "side": side,
            "size": size,
            "order_type": order_type,
            "client_order_id": client_order_id or "dummy-1",
        }

    def cancel_order(self, client_order_id):
        return {"status": "cancelled", "client_order_id": client_order_id}


def test_smart_execute_rejects_trade_when_risk_gate_fails():
    controller = RiskController(
        initial_equity=10000.0,
        config=RiskConfig(max_risk_per_trade_pct=0.01, max_daily_loss_pct=0.05, stop_trading_drawdown_pct=0.15),
    )
    controller.update_after_trade(
        realized_pnl=-600.0,
        equity=9400.0,
        timestamp=datetime(2026, 1, 1, 10, 0, 0),
    )
    result = smart_execute(
        broker=DummyBroker(),
        symbol="BTC/USDT",
        side="BUY",
        size=10.0,
        bid=100.0,
        ask=101.0,
        risk_controller=controller,
        equity=10000.0,
        stop_price=90.0,
        timestamp=datetime(2026, 1, 1, 10, 30, 0),
        wait_seconds=0,
    )
    assert result["status"] == "rejected"


def test_smart_execute_allows_and_adjusts_size_under_risk_cap():
    controller = RiskController(
        initial_equity=10000.0,
        config=RiskConfig(max_risk_per_trade_pct=0.02, max_daily_loss_pct=0.05, stop_trading_drawdown_pct=0.15),
    )
    result = smart_execute(
        broker=DummyBroker(),
        symbol="BTC/USDT",
        side="BUY",
        size=25.0,
        bid=100.0,
        ask=100.2,
        risk_controller=controller,
        equity=10000.0,
        stop_price=90.0,
        wait_seconds=0,
    )
    assert result["status"] == "filled"
    assert result["size"] < 25.0
