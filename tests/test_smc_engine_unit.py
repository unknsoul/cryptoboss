"""Focused unit coverage for SMC order-block and setup plumbing."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.smc.order_blocks import OrderBlockDetector
from src.smc.smc_engine import SMCEngine


def _smc_frame() -> pd.DataFrame:
    """Build a frame with a clear impulse for SMC detectors."""
    rows = []
    price = 100.0
    for idx in range(40):
        open_price = price
        close_price = price + 0.2
        high = close_price + 0.15
        low = open_price - 0.15
        volume = 100 + idx
        rows.append((open_price, high, low, close_price, volume))
        price += 0.1

    rows[10] = (101.4, 101.6, 100.6, 100.7, 120.0)
    rows[11] = (100.8, 102.4, 100.7, 102.1, 210.0)
    rows[12] = (102.1, 103.6, 102.0, 103.2, 220.0)

    index = pd.date_range("2026-01-01", periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=index)


def test_order_block_detected_after_impulse() -> None:
    """A strong impulse should leave at least one order block candidate."""
    detector = OrderBlockDetector(timeframe="15m")
    blocks = detector.detect(_smc_frame())
    assert len(blocks) >= 1


def test_use_body_only_flag_changes_zone_bounds() -> None:
    """Body-only order blocks should clamp their bounds to open and close."""
    detector = OrderBlockDetector(timeframe="15m", use_body_only=True)
    blocks = detector.detect(_smc_frame())
    assert blocks
    first = blocks[0]
    assert first.top <= max(first.open, first.close)
    assert first.bottom >= min(first.open, first.close)


def test_smc_engine_reads_body_only_flag(monkeypatch) -> None:
    """The SMC engine should propagate the config flag into its order-block detectors."""
    monkeypatch.setattr(SMCEngine, "_load_use_body_only_flag", staticmethod(lambda: True))
    engine = SMCEngine(timeframes=["15m"])
    assert engine.ob_detectors["15m"].use_body_only is True


def test_get_active_setups_returns_timeframe_filtered_results() -> None:
    """Setup retrieval should respect the requested timeframe filter."""
    engine = SMCEngine(timeframes=["15m", "1h"])
    frame = _smc_frame()
    setups = engine.get_active_setups({"15m": frame, "1h": frame.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()}, timeframe="15m", limit=5)
    assert all(setup.timeframe == "15m" for setup in setups)
