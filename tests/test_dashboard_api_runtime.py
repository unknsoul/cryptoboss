import asyncio
import copy
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import dashboard.api as dashboard_api


def _snapshot_state():
    snapshot = {}
    for key, value in dashboard_api.state.__dict__.items():
        try:
            snapshot[key] = copy.deepcopy(value)
        except Exception:
            snapshot[key] = value
    return snapshot


@pytest.fixture
def client(monkeypatch):
    state_snapshot = _snapshot_state()
    original_task = dashboard_api._trading_loop_task
    original_heartbeat = dashboard_api.ROOT_WS_HEARTBEAT_SECONDS
    original_risk_settings = copy.deepcopy(dashboard_api.risk_settings)
    original_log_entries = dashboard_api.live_log_stream.snapshot(limit=800)
    original_dependency_overrides = dict(dashboard_api.app.dependency_overrides)
    original_runtime_telemetry = {
        "last_api_roundtrip_ms": dashboard_api.runtime_telemetry.last_api_roundtrip_ms,
        "last_api_request_at": dashboard_api.runtime_telemetry.last_api_request_at,
        "last_websocket_heartbeat_at": dashboard_api.runtime_telemetry.last_websocket_heartbeat_at,
        "last_binance_data_at": dashboard_api.runtime_telemetry.last_binance_data_at,
    }
    original_env = {
        "locked": dashboard_api.env_signature._locked,
        "mode": dashboard_api.env_signature._mode,
        "started_at": dashboard_api.env_signature._started_at,
        "checksum": dashboard_api.env_signature._checksum,
    }

    monkeypatch.setattr(dashboard_api, "MARKET_DATA_AVAILABLE", False)
    dashboard_api.manager.active_connections.clear()
    dashboard_api.price_clients.clear()
    dashboard_api.live_log_stream.reset()
    dashboard_api._trading_loop_task = None
    dashboard_api.runtime_telemetry.last_api_roundtrip_ms = None
    dashboard_api.runtime_telemetry.last_api_request_at = None
    dashboard_api.runtime_telemetry.last_websocket_heartbeat_at = None
    dashboard_api.runtime_telemetry.last_binance_data_at = None

    with TestClient(dashboard_api.app) as test_client:
        yield test_client

    dashboard_api.state.__dict__.clear()
    dashboard_api.state.__dict__.update(state_snapshot)
    dashboard_api.ROOT_WS_HEARTBEAT_SECONDS = original_heartbeat
    dashboard_api._trading_loop_task = original_task
    dashboard_api.risk_settings.clear()
    dashboard_api.risk_settings.update(original_risk_settings)
    dashboard_api.app.dependency_overrides.clear()
    dashboard_api.app.dependency_overrides.update(original_dependency_overrides)
    dashboard_api.env_signature._locked = original_env["locked"]
    dashboard_api.env_signature._mode = original_env["mode"]
    dashboard_api.env_signature._started_at = original_env["started_at"]
    dashboard_api.env_signature._checksum = original_env["checksum"]
    dashboard_api.manager.active_connections.clear()
    dashboard_api.price_clients.clear()
    dashboard_api.live_log_stream.reset()
    for entry in original_log_entries:
        dashboard_api.live_log_stream.publish_nowait(entry)
    dashboard_api.runtime_telemetry.last_api_roundtrip_ms = original_runtime_telemetry["last_api_roundtrip_ms"]
    dashboard_api.runtime_telemetry.last_api_request_at = original_runtime_telemetry["last_api_request_at"]
    dashboard_api.runtime_telemetry.last_websocket_heartbeat_at = original_runtime_telemetry["last_websocket_heartbeat_at"]
    dashboard_api.runtime_telemetry.last_binance_data_at = original_runtime_telemetry["last_binance_data_at"]


def test_engine_start_requires_validated_exchange(client):
    dashboard_api.state.api_validated = False
    dashboard_api.state.exchange_client = None
    dashboard_api.state.connection_status = "disconnected"
    dashboard_api._set_engine_status("stopped")

    response = client.post(
        "/api/engine/start",
        json={"mode": "testnet", "capital": 1000, "symbols": ["BTC/USDT"], "strategy": "dca"},
    )

    assert response.status_code == 409
    assert "validate" in response.json()["detail"].lower()


def test_engine_status_tracks_start_pause_resume_and_stop(client, monkeypatch):
    dashboard_api.state.api_validated = True
    dashboard_api.state.exchange_client = object()
    dashboard_api.state.connection_status = "connected"
    dashboard_api.state.mode = "testnet"
    dashboard_api._set_engine_status("stopped")

    start_loop = AsyncMock(side_effect=lambda: dashboard_api._set_engine_status("running"))
    stop_loop = AsyncMock()
    monkeypatch.setattr(dashboard_api, "start_real_trading_loop", start_loop)
    monkeypatch.setattr(dashboard_api, "stop_real_trading_loop", stop_loop)

    assert client.get("/api/status").json()["status"] == "stopped"

    start_response = client.post(
        "/api/engine/start",
        json={"mode": "testnet", "capital": 1000, "symbols": ["BTC/USDT"], "strategy": "dca"},
    )
    assert start_response.status_code == 200
    assert start_response.json()["status"] == "running"
    assert client.get("/api/status").json()["status"] == "running"

    pause_response = client.post("/api/operator/pause", json={"reason": "pause runtime smoke"})
    assert pause_response.status_code == 200
    assert client.get("/api/status").json()["status"] == "paused"

    resume_response = client.post("/api/operator/resume", json={"reason": "resume runtime smoke"})
    assert resume_response.status_code == 200
    assert client.get("/api/status").json()["status"] == "running"

    stop_response = client.post("/api/engine/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["status"] == "stopped"
    assert client.get("/api/status").json()["status"] == "stopped"

    assert start_loop.await_count == 2
    assert stop_loop.await_count == 2


def test_system_endpoint_reports_real_incident_state(client):
    dashboard_api.state.incident_state = "INCIDENT_FREEZE"
    dashboard_api.state.incident_reason = "Exchange health degraded"
    dashboard_api.state.trading_paused = True
    dashboard_api._set_engine_status("paused")
    dashboard_api.runtime_telemetry.note_websocket_heartbeat()
    dashboard_api.runtime_telemetry.note_binance_data()

    payload = client.get("/api/system").json()

    assert payload["incident_state"] == "INCIDENT_FREEZE"
    assert payload["trading_paused"] is True
    assert payload["engine_status"] == "paused"
    assert payload["exchange_health_stage"] == "CLOSE_ONLY"
    assert isinstance(payload["component_health"], list)
    assert isinstance(payload["startup_checklist"], list)
    assert payload["latency"]["last_binance_data_at"] is not None


def test_root_websocket_emits_heartbeat_without_client_ping(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "ROOT_WS_HEARTBEAT_SECONDS", 0.01)

    with client.websocket_connect("/ws") as websocket:
        init_message = websocket.receive_json()
        heartbeat = websocket.receive_json()

    assert init_message["type"] == "init"
    assert heartbeat["type"] == "heartbeat"
    assert "status" in heartbeat


def test_health_latency_endpoint_reports_runtime_metrics(client):
    client.get("/api/system")
    dashboard_api.runtime_telemetry.note_websocket_heartbeat()
    dashboard_api.runtime_telemetry.note_binance_data()

    response = client.get("/api/health/latency")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_roundtrip_ms"] is not None
    assert payload["last_websocket_heartbeat_at"] is not None
    assert payload["last_binance_data_at"] is not None


def test_logs_snapshot_and_sse_stream(client):
    dashboard_api.live_log_stream.reset()
    dashboard_api.logger.warning("runtime smoke entry for live logs")

    snapshot_response = client.get("/api/logs?levels=WARNING")

    assert snapshot_response.status_code == 200
    snapshot_payload = snapshot_response.json()
    assert snapshot_payload["count"] >= 1
    assert any("runtime smoke entry for live logs" in entry["message"] for entry in snapshot_payload["entries"])

    class FakeRequest:
        def __init__(self):
            self._checks = 0

        async def is_disconnected(self):
            self._checks += 1
            return self._checks > 1

    async def collect_stream_text():
        response = await dashboard_api.stream_logs(FakeRequest(), limit=10, levels="WARNING")
        assert response.media_type == "text/event-stream"
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return "".join(chunks)

    streamed_text = asyncio.run(collect_stream_text())

    assert "event: snapshot" in streamed_text
    assert "runtime smoke entry for live logs" in streamed_text


def test_update_risk_settings_updates_settings_and_risk_views(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_save_risk_settings", lambda: None)
    dashboard_api.state.initial_capital = 10000.0
    dashboard_api.state.capital = 10000.0

    response = client.put(
        "/api/settings/risk",
        json={
            "daily_loss_limit": 250.0,
            "weekly_loss_limit": 900.0,
            "max_drawdown": 8.5,
            "max_positions": 4,
            "max_exposure": 7500.0,
            "trades_per_day": 12,
            "trades_per_context": 4,
            "losses_per_bias": 3,
        },
    )

    assert response.status_code == 200
    assert response.json()["risk"]["daily_loss_limit"] == 250.0

    settings_payload = client.get("/api/settings").json()
    risk_payload = client.get("/api/risk").json()

    assert settings_payload["risk"]["daily_loss_limit"] == 250.0
    assert settings_payload["risk"]["trades_per_day"] == 12
    assert risk_payload["limits"]["daily_loss_limit"] == 250.0
    assert risk_payload["limits"]["weekly_loss_limit"] == 900.0
    assert dashboard_api.state.operator_action_log[-1]["action"] == "UPDATE_RISK_SETTINGS"


def test_select_account_reconnects_runtime_and_starts_engine(client, monkeypatch):
    user = type("User", (), {"user_id": "user-123"})()
    account = type(
        "Account",
        (),
        {
            "exchange_account_id": "account-123",
            "environment": "testnet",
            "to_dict": lambda self: {
                "exchange_account_id": "account-123",
                "environment": "testnet",
                "label": "Primary",
            },
        },
    )()

    class FakeAccountService:
        def get_account(self, user_id, exchange_account_id):
            return type("Result", (), {"success": True, "account": account})()

    async def fake_attach_exchange_client_for_account(user_id, selected_account):
        dashboard_api.state.exchange_client = object()
        dashboard_api.state.api_validated = True
        dashboard_api.state.connection_status = "connected"
        dashboard_api.state.initial_capital = 1200.0
        dashboard_api.state.capital = 1200.0
        return {"USDT": 1200.0}

    start_loop = AsyncMock(side_effect=lambda: dashboard_api._set_engine_status("running"))
    stop_loop = AsyncMock()

    dashboard_api.app.dependency_overrides[dashboard_api.require_auth] = lambda: user
    monkeypatch.setattr(dashboard_api, "get_account_service", lambda: FakeAccountService())
    monkeypatch.setattr(dashboard_api, "_attach_exchange_client_for_account", fake_attach_exchange_client_for_account)
    monkeypatch.setattr(dashboard_api, "start_real_trading_loop", start_loop)
    monkeypatch.setattr(dashboard_api, "stop_real_trading_loop", stop_loop)
    monkeypatch.setattr(dashboard_api, "BOT_INSTANCE_AVAILABLE", False)
    monkeypatch.setattr(dashboard_api, "SCOPED_STATE_AVAILABLE", False)

    response = client.post("/api/accounts/select", json={"exchange_account_id": "account-123"})

    assert response.status_code == 200
    payload = response.json()["data"]

    assert payload["success"] is True
    assert payload["engine_status"] == "running"
    assert payload["connection_status"] == "connected"
    assert payload["balances"]["USDT"] == 1200.0
    assert dashboard_api.state.active_exchange_account_id == "account-123"
    assert start_loop.await_count == 1
    assert stop_loop.await_count == 1


def test_active_account_endpoint_restores_runtime_from_persisted_selection(client, monkeypatch):
    user = type("User", (), {"user_id": "user-restore"})()
    account = type(
        "Account",
        (),
        {
            "exchange_account_id": "account-restore",
            "environment": "testnet",
            "to_dict": lambda self: {
                "exchange_account_id": "account-restore",
                "environment": "testnet",
                "label": "Recovered",
            },
        },
    )()

    class FakeAccountService:
        def get_account(self, user_id, exchange_account_id):
            return type("Result", (), {"success": True, "account": account})()

        def get_key_fingerprint(self, user_id, exchange_account_id):
            return "abcd1234"

    async def fake_restore_runtime(current_user, selected_account):
        dashboard_api.state.active_user_id = current_user.user_id
        dashboard_api.state.active_exchange_account_id = selected_account.exchange_account_id
        dashboard_api.state.exchange_client = object()
        dashboard_api.state.api_validated = True
        dashboard_api.state.connection_status = "connected"
        dashboard_api._set_engine_status("running")
        return {"USDT": 900.0}

    dashboard_api.state.active_exchange_account_id = None
    dashboard_api.state.exchange_client = None
    dashboard_api.state.api_validated = False
    dashboard_api.state.connection_status = "disconnected"
    dashboard_api._set_engine_status("stopped")

    dashboard_api.app.dependency_overrides[dashboard_api.require_auth] = lambda: user
    monkeypatch.setattr(dashboard_api, "get_account_service", lambda: FakeAccountService())
    monkeypatch.setattr(dashboard_api, "_lookup_persisted_active_account_id", lambda user_id: "account-restore")
    monkeypatch.setattr(dashboard_api, "_restore_active_account_runtime", fake_restore_runtime)

    response = client.get("/api/accounts/active")

    assert response.status_code == 200
    payload = response.json()["data"]

    assert payload["active"] is True
    assert payload["runtime_restored"] is True
    assert payload["connection_status"] == "connected"
    assert payload["engine_status"] == "running"
    assert payload["account"]["api_key_fingerprint"] == "abcd1234"


def test_open_managed_trade_creates_runtime_position(client):
    """Opening a managed trade should populate runtime trade and position state."""
    if not dashboard_api.TRADE_MANAGEMENT_AVAILABLE or not dashboard_api.AGGRESSIVE_SCALPER_AVAILABLE:
        pytest.skip("Managed trade stack is unavailable in this environment")

    class FakeExchangeClient:
        async def create_order(self, **kwargs):
            return {
                "id": "entry-1",
                "average": 100.0,
                "price": 100.0,
                "fee": {"cost": 0.05},
                **kwargs,
            }

    dashboard_api.state.exchange_client = FakeExchangeClient()
    dashboard_api.state.initial_capital = 1000.0
    dashboard_api.state.capital = 1000.0
    strategy = dashboard_api.AggressiveScalper()
    strategy.set_account_balance(1000.0)

    signal = type(
        "Signal",
        (),
        {
            "action": "BUY",
            "size": 0.2,
            "stop_loss": 99.0,
            "take_profit": 102.8,
            "confidence": 0.81,
            "reason": "test_entry",
            "metadata": {"tp1": 101.4, "tp2": 102.8, "tp3": 104.2},
        },
    )()

    async def run_open():
        event = await dashboard_api._open_managed_trade(
            strategy=strategy,
            symbol="BTC/USDT",
            signal=signal,
            current_price=100.0,
            balance_map={"USDT": {"free": 1000.0, "total": 1000.0}},
        )
        dashboard_api._sync_open_positions_view({"BTC/USDT": 100.0})
        return event

    event = asyncio.run(run_open())

    assert event is not None
    assert event["event_type"] == "OPEN"
    assert len(dashboard_api.state.managed_trades) == 1
    assert dashboard_api.state.position == pytest.approx(0.2)
    assert dashboard_api.state.position_entry_price == pytest.approx(100.0)
    assert dashboard_api.state.trades[-1]["trade_id"] == event["trade_id"]


def test_manage_trade_closes_only_remaining_quantity(client):
    """Managed exits should partial close correctly and only flatten the remaining size."""
    if not dashboard_api.TRADE_MANAGEMENT_AVAILABLE or not dashboard_api.AGGRESSIVE_SCALPER_AVAILABLE:
        pytest.skip("Managed trade stack is unavailable in this environment")

    class FakeExchangeClient:
        def __init__(self):
            self.order_count = 0

        async def create_order(self, **kwargs):
            self.order_count += 1
            if self.order_count == 1:
                return {"id": "entry-1", "average": 100.0, "price": 100.0, "fee": {"cost": 0.05}, **kwargs}
            if self.order_count == 2:
                return {"id": "tp1-1", "average": 101.4, "price": 101.4, "fee": {"cost": 0.03}, **kwargs}
            return {"id": "close-1", "average": 101.2, "price": 101.2, "fee": {"cost": 0.03}, **kwargs}

    dashboard_api.state.exchange_client = FakeExchangeClient()
    dashboard_api.state.initial_capital = 1000.0
    dashboard_api.state.capital = 1000.0
    strategy = dashboard_api.AggressiveScalper()
    strategy.set_account_balance(1000.0)

    signal = type(
        "Signal",
        (),
        {
            "action": "BUY",
            "size": 0.2,
            "stop_loss": 99.0,
            "take_profit": 102.8,
            "confidence": 0.81,
            "reason": "test_entry",
            "metadata": {"tp1": 101.4, "tp2": 102.8, "tp3": 104.2},
        },
    )()

    async def run_management():
        await dashboard_api._open_managed_trade(
            strategy=strategy,
            symbol="BTC/USDT",
            signal=signal,
            current_price=100.0,
            balance_map={"USDT": {"free": 1000.0, "total": 1000.0}},
        )
        trade_id, trade = next(iter(dashboard_api.state.managed_trades.items()))
        await dashboard_api._manage_trade_for_symbol(
            trade_id=trade_id,
            trade=trade,
            strategy=strategy,
            current_price=101.4,
            atr_value=1.0,
        )
        remaining_trade = dashboard_api.state.managed_trades[trade_id]
        await dashboard_api._manage_trade_for_symbol(
            trade_id=trade_id,
            trade=remaining_trade,
            strategy=strategy,
            current_price=101.2,
            atr_value=1.0,
            opposing_signal="SELL",
        )

    asyncio.run(run_management())

    partial_events = [event for event in dashboard_api.state.trades if event.get("event_type") == "PARTIAL_CLOSE"]
    close_events = [event for event in dashboard_api.state.trades if event.get("event_type") == "CLOSE"]

    assert len(partial_events) == 1
    assert partial_events[0]["amount"] == pytest.approx(0.1)
    assert len(close_events) == 1
    assert close_events[0]["amount"] == pytest.approx(0.1)
    assert dashboard_api.state.managed_trades == {}
