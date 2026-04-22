import numpy as np
import pandas as pd

from src.analysis.indicators import IndicatorEngine


def _sample_ohlcv(rows: int = 300) -> pd.DataFrame:
    index = np.arange(rows)
    base = 20000 + (index * 2) + np.sin(index / 8.0) * 40
    close = pd.Series(base)
    open_ = close.shift(1).fillna(close.iloc[0])

    return pd.DataFrame(
        {
            "open": open_,
            "high": close + 10,
            "low": close - 10,
            "close": close,
            "volume": 1000 + (index % 20) * 10,
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h"),
        }
    )


def test_all_indicators_computed_no_nan_after_warmup():
    engine = IndicatorEngine()
    df = _sample_ohlcv(320)

    enriched = engine.compute_all(df)

    required = [
        "EMA_20",
        "EMA_50",
        "EMA_200",
        "RSI_14",
        "MACD_hist",
        "ATR_14",
        "OBV",
        "VWAP",
    ]

    for col in required:
        assert col in enriched.columns

    warm = enriched.iloc[220:]
    assert warm[required].isna().sum().sum() == 0


def test_swing_detection_accuracy_on_known_data():
    closes = [10, 12, 15, 11, 9, 11, 16, 12, 10]
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [v + 0.5 for v in closes],
            "low": [v - 0.5 for v in closes],
            "close": closes,
            "volume": [100] * len(closes),
            "timestamp": pd.date_range("2025-01-01", periods=len(closes), freq="h"),
        }
    )

    engine = IndicatorEngine()
    highs = engine.find_swing_highs(df, lookback=1)
    lows = engine.find_swing_lows(df, lookback=1)

    assert any(point["index"] == 2 for point in highs)
    assert any(point["index"] == 6 for point in highs)
    assert any(point["index"] == 4 for point in lows)


def test_order_block_detection():
    df = pd.DataFrame(
        {
            "open": [100, 99, 98, 97, 96, 101, 103, 104],
            "high": [101, 100, 99, 98, 100, 104, 106, 107],
            "low": [99, 98, 97, 95, 94, 100, 102, 103],
            "close": [99.5, 98.5, 97.5, 96.5, 99.8, 103.5, 105.5, 106.5],
            "volume": [120, 130, 140, 150, 220, 240, 210, 200],
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="h"),
        }
    )

    engine = IndicatorEngine()
    blocks = engine.find_order_blocks(df)

    assert len(blocks) >= 1
    assert {"direction", "index", "low", "high"}.issubset(blocks[0].keys())
