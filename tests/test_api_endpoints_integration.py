"""Integration-style checks for key dashboard API contracts."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import dashboard.api as dashboard_api
from src.strategies.base_strategy import SignalResult
from src.smc.smc_engine import SMCSetup, SetupType


def _snapshot_state():
    """Capture the mutable dashboard state for restoration between tests."""
    snapshot = {}
    for key, value in dashboard_api.state.__dict__.items():
        try:
            snapshot[key] = copy.deepcopy(value)
        except Exception:
            snapshot[key] = value
    return snapshot


def _sample_frame() -> pd.DataFrame:
    """Build a simple multi-timeframe market frame for mocked scalper responses."""
    index = pd.date_range("2026-01-01", periods=80, freq="min", tz="UTC")
    close = pd.Series(range(80), dtype=float) * 0.1 + 100.0
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.15,
            "low": close - 0.15,
            "close": close,
            "volume": 100.0,
        },
        index=index,
    )


@pytest.fixture
def client(monkeypatch):
    """Provide an isolated TestClient for endpoint verification."""
    state_snapshot = _snapshot_state()
    original_task = dashboard_api._trading_loop_task
    original_dependency_overrides = dict(dashboard_api.app.dependency_overrides)
    strategy = dashboard_api._resolve_aggressive_scalper()
    original_strategy_enabled = strategy.config.enabled if strategy is not None else None

    monkeypatch.setattr(dashboard_api, "MARKET_DATA_AVAILABLE", False)
    dashboard_api.manager.active_connections.clear()
    dashboard_api.price_clients.clear()
    dashboard_api._trading_loop_task = None
    dashboard_api.app.dependency_overrides[dashboard_api.require_auth] = lambda: SimpleNamespace(user_id="user-1")

    if strategy is not None:
        strategy.config.enabled = True
        strategy.set_account_balance(10000.0)
        strategy.set_daily_pnl(0.0)
        strategy.set_weekly_pnl(0.0)
        strategy._trade_timestamps = []
        strategy._last_signal_at = None
        strategy._last_signal_snapshot = {}
        strategy._last_watch_signal = {}

    with TestClient(dashboard_api.app) as test_client:
        yield test_client

    dashboard_api.state.__dict__.clear()
    dashboard_api.state.__dict__.update(state_snapshot)
    dashboard_api._trading_loop_task = original_task
    dashboard_api.app.dependency_overrides.clear()
    dashboard_api.app.dependency_overrides.update(original_dependency_overrides)
    dashboard_api.manager.active_connections.clear()
    dashboard_api.price_clients.clear()
    if strategy is not None and original_strategy_enabled is not None:
        strategy.config.enabled = original_strategy_enabled
        strategy._trade_timestamps = []
        strategy._last_signal_at = None
        strategy._last_signal_snapshot = {}
        strategy._last_watch_signal = {}


def test_get_system_returns_200(client) -> None:
    """The system endpoint should return the wrapped contract and engine fields."""
    response = client.get("/api/system")
    assert response.status_code == 200
    payload = response.json()
    assert payload["timestamp"]
    assert "data" in payload
    assert "engine_status" in payload


def test_get_risk_returns_200(client) -> None:
    """The risk endpoint should expose kill switch and daily risk metrics."""
    response = client.get("/api/risk")
    assert response.status_code == 200
    payload = response.json()
    assert "data" in payload
    assert "kill_switch_active" in payload["data"]
    assert "daily_pnl" in payload["data"]


def test_get_scalper_live_returns_structure(client, monkeypatch) -> None:
    """The scalper live endpoint should surface signal and serialized setup data."""
    if not dashboard_api.V12_AVAILABLE or dashboard_api.v12_scalper is None:
        pytest.skip("v12 scalper stack unavailable")

    frame = _sample_frame()
    setup = SMCSetup(
        setup_id="SETUP_1",
        setup_type=SetupType.BULLISH_OB_RETEST,
        direction="long",
        timeframe=dashboard_api.v12_scalper.entry_timeframe,
        timestamp=pd.Timestamp.utcnow(),
        entry_price=101.0,
        stop_loss=100.0,
        take_profit_1=102.5,
        take_profit_2=103.5,
        take_profit_3=104.5,
        risk_reward=1.5,
        confidence=0.82,
        components={"order_block": {"score": 0.3}},
        invalidation=100.0,
    )

    async def fake_fetch_multi_tf_data(symbol: str, timeframes: list[str], limit: int = 500):
        return {timeframe: frame.copy() for timeframe in timeframes}

    monkeypatch.setattr(dashboard_api, "fetch_multi_tf_data", fake_fetch_multi_tf_data)
    monkeypatch.setattr(
        dashboard_api.v12_scalper,
        "generate_multi_timeframe_signal",
        lambda data, account_balance=10000.0: SignalResult(
            action="BUY",
            reason="mocked_signal",
            confidence=0.8,
            size=0.25,
            price=101.0,
            stop_loss=100.0,
            take_profit=103.5,
        ),
    )
    monkeypatch.setattr(
        dashboard_api.v12_scalper,
        "analyze_current_market",
        lambda data: {
            "session": "London",
            "in_kill_zone": True,
            "kill_zone_name": "London Open",
            "setups_count": 0,
            "top_setups": [],
        },
    )
    monkeypatch.setattr(
        dashboard_api.v12_scalper.smc_engine,
        "get_active_setups",
        lambda data, timeframe=None, limit=5: [setup],
    )

    response = client.get("/api/v2/scalper/live?symbol=BTC/USDT")
    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["signal"]["action"] == "BUY"
    assert payload["analysis"]["top_setups"][0]["id"] == "SETUP_1"
    assert payload["analysis"]["setups_count"] == 1


def test_get_prices_live_returns_symbols(client, monkeypatch) -> None:
    """The live prices endpoint should expose real symbol entries in the wrapped payload."""
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return [
                {
                    "symbol": "BTCUSDT",
                    "lastPrice": "65000",
                    "priceChangePercent": "2.1",
                    "highPrice": "66000",
                    "lowPrice": "64000",
                    "quoteVolume": "1000000",
                },
                {
                    "symbol": "ETHUSDT",
                    "lastPrice": "3200",
                    "priceChangePercent": "1.2",
                    "highPrice": "3250",
                    "lowPrice": "3150",
                    "quoteVolume": "500000",
                },
                {
                    "symbol": "BNBUSDT",
                    "lastPrice": "600",
                    "priceChangePercent": "0.8",
                    "highPrice": "610",
                    "lowPrice": "590",
                    "quoteVolume": "250000",
                },
                {
                    "symbol": "SOLUSDT",
                    "lastPrice": "150",
                    "priceChangePercent": "3.4",
                    "highPrice": "152",
                    "lowPrice": "145",
                    "quoteVolume": "350000",
                },
            ]

    import requests

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse())
    response = client.get("/api/prices/live")
    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["prices"]["BTCUSDT"]["price"] > 0
    assert "ETHUSDT" in payload["prices"]


def test_kill_switch_activation_and_reset_accepts_json_body(client) -> None:
    """The kill-switch endpoint should accept JSON body payloads and update risk state."""
    activate = client.post("/api/kill-switch", json={"active": True, "reason": "test"})
    assert activate.status_code == 200
    assert activate.json()["kill_switch_active"] is True

    risk_payload = client.get("/api/risk").json()["data"]
    assert risk_payload["kill_switch_active"] is True

    deactivate = client.post("/api/kill-switch", json={"active": False, "reason": "reset"})
    assert deactivate.status_code == 200
    assert deactivate.json()["kill_switch_active"] is False


def test_strategy_enable_disable_toggles_aggressive_scalper(client) -> None:
    """Strategy toggle endpoints should accept id-based payloads and flip runtime enablement."""
    strategy = dashboard_api._resolve_aggressive_scalper()
    if strategy is None:
        pytest.skip("aggressive scalper unavailable")

    disable = client.post("/api/strategy/disable", json={"id": "aggressive_scalper_v1"})
    assert disable.status_code == 200
    assert strategy.config.enabled is False

    enable = client.post("/api/strategy/enable", json={"id": "aggressive_scalper_v1"})
    assert enable.status_code == 200
    assert strategy.config.enabled is True


def test_trades_empty_state_response_includes_message(client, monkeypatch) -> None:
    """The trades endpoint should explain an empty account state instead of looking broken."""
    import src.core.database.repository as repository_module

    class FakeRepository:
        def get_active_account_id(self, user_id: str):
            return "account-1"

        def get_trades(self, user_id: str, account_id: str, limit: int = 50):
            return []

    monkeypatch.setattr(repository_module, "get_repository", lambda: FakeRepository())
    response = client.get("/api/trades")
    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["trades"] == []
    assert payload["message"] == "No trades yet. Start the engine and wait for signals."
