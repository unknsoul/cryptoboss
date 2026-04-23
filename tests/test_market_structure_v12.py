from datetime import datetime, timedelta

import pandas as pd

from src.analysis.market_structure import (
    MarketStructureEngine,
    StructureEvent,
    StructureEventType,
    SwingPoint,
    TrendState,
)


def _sample_frame(rows: int = 120) -> pd.DataFrame:
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(rows)]

    # Build a gently rising market with pullbacks.
    base = [100 + (i * 0.15) for i in range(rows)]
    close = [v + (0.8 if i % 10 < 5 else -0.8) for i, v in enumerate(base)]
    open_ = [c - 0.2 for c in close]
    high = [max(o, c) + 0.6 for o, c in zip(open_, close)]
    low = [min(o, c) - 0.6 for o, c in zip(open_, close)]

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        },
        index=pd.to_datetime(timestamps),
    )


def test_structure_snapshot_has_trend_and_events() -> None:
    frame = _sample_frame()
    engine = MarketStructureEngine(
        swing_lookback_candles=20,
        swing_pivot_confirmation_candles=2,
        minimum_swing_size_atr_multiplier=0.1,
    )

    snapshot = engine.get_structure_snapshot(frame, timeframe="15m")

    assert snapshot.timeframe == "15m"
    assert snapshot.trend_state in set(TrendState)
    assert isinstance(snapshot.swings, list)
    assert isinstance(snapshot.events, list)


def test_classify_choch_from_bos_direction_flip() -> None:
    now = pd.Timestamp(datetime.utcnow())
    bos_events = [
        StructureEvent(
            event_type=StructureEventType.BOS,
            direction="bullish",
            timestamp=now,
            level=100.0,
            body_close_confirmed=True,
            weight=0.7,
        ),
        StructureEvent(
            event_type=StructureEventType.BOS,
            direction="bearish",
            timestamp=now + pd.Timedelta(minutes=15),
            level=99.0,
            body_close_confirmed=True,
            weight=0.7,
        ),
    ]

    engine = MarketStructureEngine()
    events = engine.classify_choch(bos_events)

    assert any(event.event_type == StructureEventType.CHOCH for event in events)
    assert any(event.event_type == StructureEventType.MSB for event in events)


def test_trend_state_bullish_sequence() -> None:
    now = pd.Timestamp(datetime.utcnow())
    swings = [
        SwingPoint(index=1, timestamp=now, kind="swing_low", price=95.0),
        SwingPoint(index=2, timestamp=now + pd.Timedelta(minutes=15), kind="swing_high", price=105.0),
        SwingPoint(index=3, timestamp=now + pd.Timedelta(minutes=30), kind="swing_low", price=97.0),
        SwingPoint(index=4, timestamp=now + pd.Timedelta(minutes=45), kind="swing_high", price=108.0),
    ]

    engine = MarketStructureEngine()
    trend = engine.get_trend_state(swings, events=[])

    assert trend == TrendState.BULLISH_TRENDING
