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
