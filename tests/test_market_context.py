import numpy as np
import pandas as pd

from src.analysis.market_context import MarketContextEngine, RegimeEnum


def _trend_df(rows: int = 320, slope: float = 3.0, noise: float = 8.0) -> pd.DataFrame:
    idx = np.arange(rows)
    base = 25000 + (idx * slope) + np.sin(idx / 9.0) * noise
    close = pd.Series(base)

    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 12,
            "low": close - 12,
            "close": close,
            "volume": 1000 + (idx % 25) * 25,
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h"),
        }
    )


def test_regime_classification_on_labeled_historical_data():
    engine = MarketContextEngine()
    df = _trend_df(rows=420, slope=4.0, noise=6.0)

    regime, confidence = engine.classify_regime(df)

    assert regime in {RegimeEnum.STRONG_UPTREND, RegimeEnum.WEAK_UPTREND}
    assert 0.0 <= confidence <= 1.0


def test_structure_bias_correct_hh_hl_pattern():
    engine = MarketContextEngine()

    swing_highs = [
        {"index": 20, "price": 100},
        {"index": 40, "price": 110},
        {"index": 60, "price": 120},
    ]
    swing_lows = [
        {"index": 25, "price": 90},
        {"index": 45, "price": 96},
        {"index": 65, "price": 102},
    ]

    bias = engine.determine_structure_bias(swing_highs, swing_lows)

    assert bias == "BULLISH"


def test_key_level_detection():
    engine = MarketContextEngine()
    df = _trend_df(rows=320, slope=2.5, noise=10.0)

    context = engine.analyze(df, symbol="BTC/USDT", timeframe="1h")

    assert context.key_levels["nearest_support"] is not None
    assert context.key_levels["nearest_resistance"] is not None
    assert context.key_levels["vwap"] is not None
    assert context.key_levels["daily_open"] is not None

    near_support = engine.is_near_key_level(
        context.key_levels["nearest_support"],
        context,
        tolerance_atr=1.0,
    )
    assert near_support is True
