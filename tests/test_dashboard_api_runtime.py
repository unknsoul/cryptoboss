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
    original_dependency_overrides = dict(dashboard_api.app.dependency_overrides)
    original_env = {
        "locked": dashboard_api.env_signature._locked,
        "mode": dashboard_api.env_signature._mode,
        "started_at": dashboard_api.env_signature._started_at,
        "checksum": dashboard_api.env_signature._checksum,
    }

    monkeypatch.setattr(dashboard_api, "MARKET_DATA_AVAILABLE", False)
    dashboard_api.manager.active_connections.clear()
    dashboard_api.price_clients.clear()
    dashboard_api._trading_loop_task = None

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

    payload = client.get("/api/system").json()

    assert payload["incident_state"] == "INCIDENT_FREEZE"
    assert payload["trading_paused"] is True
    assert payload["engine_status"] == "paused"


def test_root_websocket_emits_heartbeat_without_client_ping(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "ROOT_WS_HEARTBEAT_SECONDS", 0.01)

    with client.websocket_connect("/ws") as websocket:
        init_message = websocket.receive_json()
        heartbeat = websocket.receive_json()

    assert init_message["type"] == "init"
    assert heartbeat["type"] == "heartbeat"
    assert "status" in heartbeat


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
