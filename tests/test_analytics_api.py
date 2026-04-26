from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

import dashboard.api as dashboard_api
from src.analytics.performance_analytics import TradeAnalyticsService


def _sample_trades() -> list[dict]:
    base = datetime(2026, 4, 20, 8, 0, tzinfo=UTC)
    return [
        {
            "trade_id": "t-1",
            "symbol": "BTCUSDT",
            "side": "buy",
            "price": 84000,
            "quantity": 0.01,
            "pnl": 120.0,
            "timestamp": (base - timedelta(hours=3)).isoformat(),
            "closed_at": (base - timedelta(hours=3)).isoformat(),
            "strategy_id": "smc_scalper",
            "rr": 2.1,
        },
        {
            "trade_id": "t-2",
            "symbol": "ETHUSDT",
            "side": "sell",
            "price": 3100,
            "quantity": 0.5,
            "pnl": -45.0,
            "timestamp": (base - timedelta(hours=2)).isoformat(),
            "closed_at": (base - timedelta(hours=2)).isoformat(),
            "strategy_id": "liquidity_sweep",
            "rr": -0.8,
        },
        {
            "trade_id": "t-3",
            "symbol": "BTCUSDT",
            "side": "buy",
            "price": 84500,
            "quantity": 0.012,
            "pnl": 80.0,
            "timestamp": (base - timedelta(days=3)).isoformat(),
            "closed_at": (base - timedelta(days=3)).isoformat(),
            "strategy_id": "smc_scalper",
            "rr": 1.5,
        },
        {
            "trade_id": "t-4",
            "symbol": "SOLUSDT",
            "side": "buy",
            "price": 150,
            "quantity": 10,
            "pnl": -70.0,
            "timestamp": (base - timedelta(days=10)).isoformat(),
            "closed_at": (base - timedelta(days=10)).isoformat(),
            "strategy_id": "range_reversion",
            "rr": -1.0,
        },
        {
            "trade_id": "t-5",
            "symbol": "BNBUSDT",
            "side": "sell",
            "price": 620,
            "quantity": 2,
            "pnl": 150.0,
            "timestamp": (base - timedelta(days=17)).isoformat(),
            "closed_at": (base - timedelta(days=17)).isoformat(),
            "strategy_id": "bos_continuation",
            "rr": 2.8,
        },
    ]


def test_trade_analytics_service_builds_summary_structures() -> None:
    service = TradeAnalyticsService()
    trades = _sample_trades()
    now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)

    today = service.today_summary(trades, initial_capital=10000.0, now=now)
    hourly = service.hourly_performance(trades, now=now)
    symbols = service.symbol_performance(trades)
    strategies = service.strategy_breakdown(trades)
    weekly = service.weekly_equity(trades, initial_capital=10000.0)
    drawdowns = service.drawdown_periods(trades, initial_capital=10000.0)

    assert today["trades"] == 2
    assert today["best_trade"]["symbol"] == "BTCUSDT"
    assert len(hourly["hourly"]) == 24
    assert len(hourly["heatmap"]) == 7
    assert symbols[0]["symbol"] in {"BNBUSDT", "BTCUSDT"}
    assert strategies[0]["trades"] >= 1
    assert weekly["points"]
    assert drawdowns


def test_analytics_endpoints_return_wrapped_payloads(monkeypatch) -> None:
    user = type("User", (), {"user_id": "analytics-user"})()
    original_dependency_overrides = dict(dashboard_api.app.dependency_overrides)

    dashboard_api.app.dependency_overrides[dashboard_api.require_auth] = lambda: user
    monkeypatch.setattr(
        dashboard_api,
        "_load_analytics_trade_records",
        lambda current_user, limit=5000: ("account-1", _sample_trades(), 10000.0),
    )

    with TestClient(dashboard_api.app) as client:
        today = client.get("/api/analytics/today")
        hourly = client.get("/api/analytics/hourly-performance")
        symbols = client.get("/api/analytics/symbol-performance")
        strategies = client.get("/api/analytics/strategy-breakdown")
        weekly = client.get("/api/analytics/weekly-equity")
        drawdowns = client.get("/api/analytics/drawdown-periods")

    dashboard_api.app.dependency_overrides.clear()
    dashboard_api.app.dependency_overrides.update(original_dependency_overrides)

    assert today.status_code == 200
    assert hourly.status_code == 200
    assert symbols.status_code == 200
    assert strategies.status_code == 200
    assert weekly.status_code == 200
    assert drawdowns.status_code == 200

    today_payload = today.json()
    hourly_payload = hourly.json()
    symbols_payload = symbols.json()
    strategies_payload = strategies.json()
    weekly_payload = weekly.json()
    drawdowns_payload = drawdowns.json()

    assert "data" in today_payload
    assert today_payload["data"]["exchange_account_id"] == "account-1"
    assert len(hourly_payload["data"]["hourly"]) == 24
    assert len(hourly_payload["data"]["heatmap"]) == 7
    assert isinstance(symbols_payload["data"]["symbols"], list)
    assert isinstance(strategies_payload["data"]["strategies"], list)
    assert isinstance(weekly_payload["data"]["points"], list)
    assert isinstance(drawdowns_payload["data"]["periods"], list)
