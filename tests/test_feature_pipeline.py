import pandas as pd

from src.features.pipeline import FeaturePipeline


def _sample_ohlcv(rows: int = 300) -> pd.DataFrame:
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


def test_feature_pipeline_builds_features():
    df = _sample_ohlcv()
    pipeline = FeaturePipeline()
    orderbook = {"bids": [[100.0, 2.0], [99.5, 1.0]], "asks": [[100.5, 1.5], [101.0, 2.0]]}

    features = pipeline.build_features(df, orderbook=orderbook)

    assert "volatility_atr_14" in features.columns
    assert "smc_trend" in features.columns
    assert "returns_zscore_50" in features.columns
    assert "spread" in features.columns
