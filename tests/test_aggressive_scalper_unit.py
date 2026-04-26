"""Unit coverage for the aggressive scalper strategy runtime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.aggressive_scalper import AggressiveScalper, ScalperParams


def _market_frame(rows: int = 80, direction: str = "up") -> pd.DataFrame:
    """Build a deterministic OHLCV frame for scalper signal tests."""
    index = pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC")
    drift = np.linspace(0.0, 2.0, rows) if direction == "up" else np.linspace(2.0, 0.0, rows)
    close = 100.0 + drift
    open_ = close - 0.08 if direction == "up" else close + 0.08
    high = np.maximum(open_, close) + 0.12
    low = np.minimum(open_, close) - 0.12
    volume = np.full(rows, 100.0)
    volume[-1] = 300.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _series(rows: int, last_two: tuple[float, float], base: float = 50.0) -> pd.Series:
    """Build a numeric series with controlled final values."""
    values = [base] * rows
    values[-2], values[-1] = last_two
    return pd.Series(values)


def _patch_buy_confluence(strategy: AggressiveScalper, monkeypatch, adx_value: float = 30.0) -> None:
    """Patch internals to create a high-confidence buy setup."""
    monkeypatch.setattr(
        strategy,
        "_rsi",
        lambda close, period: _series(len(close), (30.0, 34.0), 35.0)
        if period == strategy.params.rsi_fast_period
        else _series(len(close), (24.0, 20.0), 40.0),
    )
    monkeypatch.setattr(strategy, "_macd_histogram", lambda close: _series(len(close), (-0.15, 0.25), -0.05))
    monkeypatch.setattr(strategy, "_adx", lambda df, period: pd.Series([adx_value] * len(df)))
    monkeypatch.setattr(strategy, "_ema_crossover_signal", lambda close: 1)
    monkeypatch.setattr(strategy, "_price_structure", lambda close: 1)
    monkeypatch.setattr(strategy, "_volume_delta", lambda df: 0.45)


def _patch_sell_confluence(strategy: AggressiveScalper, monkeypatch, adx_value: float = 28.0) -> None:
    """Patch internals to create a high-confidence sell setup."""
    monkeypatch.setattr(
        strategy,
        "_rsi",
        lambda close, period: _series(len(close), (70.0, 66.0), 65.0)
        if period == strategy.params.rsi_fast_period
        else _series(len(close), (68.0, 75.0), 60.0),
    )
    monkeypatch.setattr(strategy, "_macd_histogram", lambda close: _series(len(close), (0.2, -0.22), 0.05))
    monkeypatch.setattr(strategy, "_adx", lambda df, period: pd.Series([adx_value] * len(df)))
    monkeypatch.setattr(strategy, "_ema_crossover_signal", lambda close: -1)
    monkeypatch.setattr(strategy, "_price_structure", lambda close: -1)
    monkeypatch.setattr(strategy, "_volume_delta", lambda df: -0.42)


def test_hold_on_insufficient_data() -> None:
    """The scalper should hold when there are not enough candles to score a trade."""
    strategy = AggressiveScalper()
    signal = strategy.generate_signal(_market_frame(rows=12), 11, 101.0)
    assert signal.action == "HOLD"
    assert signal.reason == "insufficient_data"


def test_buy_signal_has_size_and_levels(monkeypatch) -> None:
    """A confirmed buy signal should size the trade and place risk levels below/above entry."""
    strategy = AggressiveScalper(params=ScalperParams(cooldown_seconds=0, max_trades_per_hour=10))
    strategy.set_account_balance(10000.0)
    _patch_buy_confluence(strategy, monkeypatch)

    frame = _market_frame(direction="up")
    signal = strategy.generate_signal(frame, len(frame) - 1, float(frame["close"].iloc[-1]))

    risk = abs(signal.price - float(signal.stop_loss or 0.0))
    reward = abs(float(signal.take_profit or 0.0) - signal.price)
    assert signal.action == "BUY"
    assert signal.size > 0
    assert float(signal.stop_loss or 0.0) < signal.price
    assert float(signal.take_profit or 0.0) > signal.price
    assert reward / risk >= 2.5


def test_sell_signal_places_stop_above_entry(monkeypatch) -> None:
    """A confirmed sell signal should mirror levels for a short entry."""
    strategy = AggressiveScalper(params=ScalperParams(cooldown_seconds=0, max_trades_per_hour=10))
    strategy.set_account_balance(10000.0)
    _patch_sell_confluence(strategy, monkeypatch)

    frame = _market_frame(direction="down")
    signal = strategy.generate_signal(frame, len(frame) - 1, float(frame["close"].iloc[-1]))

    assert signal.action == "SELL"
    assert signal.size > 0
    assert float(signal.stop_loss or 0.0) > signal.price
    assert float(signal.take_profit or 0.0) < signal.price


def test_hold_when_adx_too_low(monkeypatch) -> None:
    """Weak trend strength should block an otherwise valid setup."""
    strategy = AggressiveScalper(params=ScalperParams(cooldown_seconds=0, max_trades_per_hour=10))
    _patch_buy_confluence(strategy, monkeypatch, adx_value=15.0)

    frame = _market_frame(direction="up")
    signal = strategy.generate_signal(frame, len(frame) - 1, float(frame["close"].iloc[-1]))

    assert signal.action == "HOLD"
    assert signal.reason == "no_confluence"


def test_hourly_cap_enforcement(monkeypatch) -> None:
    """Confirmed entries should stop once the per-hour cap is hit."""
    strategy = AggressiveScalper(params=ScalperParams(cooldown_seconds=0, max_trades_per_hour=1))
    _patch_buy_confluence(strategy, monkeypatch)
    frame = _market_frame(direction="up")

    first = strategy.generate_signal(frame, len(frame) - 1, float(frame["close"].iloc[-1]))
    second = strategy.generate_signal(frame, len(frame) - 1, float(frame["close"].iloc[-1]))

    assert first.action == "BUY"
    assert second.action == "HOLD"
    assert second.reason == "hourly_trade_cap_reached"


def test_daily_loss_halt(monkeypatch) -> None:
    """The daily loss guard should halt new entries once the threshold is breached."""
    strategy = AggressiveScalper(params=ScalperParams(cooldown_seconds=0))
    strategy.set_account_balance(10000.0)
    strategy.set_daily_pnl(-600.0)
    _patch_buy_confluence(strategy, monkeypatch)

    frame = _market_frame(direction="up")
    signal = strategy.generate_signal(frame, len(frame) - 1, float(frame["close"].iloc[-1]))

    assert signal.action == "HOLD"
    assert signal.reason == "daily_loss_halt"


def test_cooldown_enforcement(monkeypatch) -> None:
    """The strategy should reject back-to-back entries inside the cooldown window."""
    strategy = AggressiveScalper(params=ScalperParams(cooldown_seconds=120, max_trades_per_hour=10))
    _patch_buy_confluence(strategy, monkeypatch)
    frame = _market_frame(direction="up")

    first = strategy.generate_signal(frame, len(frame) - 1, float(frame["close"].iloc[-1]))
    second = strategy.generate_signal(frame, len(frame) - 1, float(frame["close"].iloc[-1]))

    assert first.action == "BUY"
    assert second.action == "HOLD"
    assert second.reason == "cooldown_active"
