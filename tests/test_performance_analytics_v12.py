from datetime import datetime, timedelta

from src.analysis.performance_analytics import PerformanceAnalyticsEngine


def _trades() -> list[dict]:
    base = datetime(2025, 1, 1)
    return [
        {
            "pnl_usdt": 120.0,
            "return_pct": 1.2,
            "direction": "long",
            "symbol": "BTCUSDT",
            "strategy": "SMC_TREND_FOLLOW",
            "session": "LONDON",
            "regime": "TRENDING_BULLISH",
            "hold_minutes": 90,
            "rr_achieved": 2.4,
            "rr_planned": 2.0,
            "closed_at": base.isoformat(),
        },
        {
            "pnl_usdt": -80.0,
            "return_pct": -0.8,
            "direction": "short",
            "symbol": "ETHUSDT",
            "strategy": "SMC_SCALPER",
            "session": "NEW_YORK",
            "regime": "RANGING",
            "hold_minutes": 35,
            "rr_achieved": -1.0,
            "rr_planned": 2.0,
            "closed_at": (base + timedelta(days=1)).isoformat(),
        },
        {
            "pnl_usdt": 60.0,
            "return_pct": 0.6,
            "direction": "long",
            "symbol": "BTCUSDT",
            "strategy": "RANGE_SCALP",
            "session": "OVERLAP",
            "regime": "RANGING",
            "hold_minutes": 25,
            "rr_achieved": 1.8,
            "rr_planned": 1.5,
            "closed_at": (base + timedelta(days=2)).isoformat(),
        },
    ]


def test_compute_snapshot_metrics() -> None:
    engine = PerformanceAnalyticsEngine()
    snapshot = engine.compute_snapshot(_trades(), initial_capital=10000)

    assert snapshot.total_trades == 3
    assert "profit_factor" in snapshot.metrics
    assert "monthly_returns" in snapshot.metrics
    assert snapshot.metrics["total_pnl_usdt"] == 100.0
    assert snapshot.metrics["win_rate"] > 0
    assert "BTCUSDT" in snapshot.metrics["pnl_by_symbol"]
