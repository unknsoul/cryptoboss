import numpy as np
import pandas as pd

from src.analysis.indicators import IndicatorEngine


def test_indicator_engine_computes_core_columns():
    periods = 300
    close = np.linspace(60000, 63000, periods) + np.sin(np.linspace(0, 12, periods)) * 400
    df = pd.DataFrame(
        {
            "open": close - 25,
            "high": close + 125,
            "low": close - 125,
            "close": close,
            "volume": np.linspace(100, 500, periods),
        }
    )

    enriched = IndicatorEngine().compute_all(df)

    expected_columns = [
        "EMA_20",
        "EMA_50",
        "EMA_200",
        "RSI_14",
        "MACD",
        "ATR_14",
        "VWAP",
        "OBV",
    ]

    for column in expected_columns:
        assert column in enriched.columns
        assert pd.notna(enriched.iloc[-1][column])
