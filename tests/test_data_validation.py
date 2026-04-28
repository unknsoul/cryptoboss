import pandas as pd
import pytest

from src.data.validation import validate_ohlcv


def _sample_ohlcv(rows: int = 50) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="1min", tz="UTC")
    data = {
        "timestamp": dates,
        "open": [100 + i * 0.1 for i in range(rows)],
        "high": [100 + i * 0.1 + 0.5 for i in range(rows)],
        "low": [100 + i * 0.1 - 0.5 for i in range(rows)],
        "close": [100 + i * 0.1 + 0.2 for i in range(rows)],
        "volume": [10 + i for i in range(rows)],
    }
    return pd.DataFrame(data)


def test_validate_ohlcv_ok():
    df = _sample_ohlcv()
    validate_ohlcv(df)


def test_validate_ohlcv_nan():
    df = _sample_ohlcv()
    df.loc[0, "close"] = None
    with pytest.raises(ValueError):
        validate_ohlcv(df)
