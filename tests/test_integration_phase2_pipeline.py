import asyncio

import numpy as np
import pandas as pd

from src.analysis.bias_engine import BiasEngine
from src.analysis.market_context import MarketContextEngine
from src.analysis.sentiment_engine import SentimentEngine


def _dataset(rows: int, slope: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.arange(rows)
    close = 28000 + (idx * slope) + rng.normal(0, 9, size=rows)
    close = pd.Series(close)

    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 14,
            "low": close - 14,
            "close": close,
            "volume": 1200 + (idx % 18) * 35,
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h"),
        }
    )


def test_context_bias_sentiment_pipeline_integration():
    context_engine = MarketContextEngine()
    bias_engine = BiasEngine()

    df_4h = _dataset(rows=360, slope=2.4, seed=11)
    df_1d = _dataset(rows=360, slope=1.3, seed=17)

    ctx_4h = context_engine.analyze(df_4h, symbol="BTC/USDT", timeframe="4h")
    ctx_1d = context_engine.analyze(df_1d, symbol="BTC/USDT", timeframe="1d")

    bias = bias_engine.compute_bias({"4h": ctx_4h, "1d": ctx_1d})

    async def _sentiment():
        engine = SentimentEngine(
            news_provider=lambda: ["Broad risk appetite improves as macro pressure eases"],
            fear_greed_provider=lambda: 60,
            social_provider=lambda: {"upvote_ratio": 0.67, "mention_velocity": 0.58},
            funding_provider=lambda: 0.002,
            oi_provider=lambda: 0.011,
        )
        return await engine.refresh_all_sources()

    sentiment = asyncio.run(_sentiment())

    assert ctx_4h.symbol == "BTC/USDT"
    assert 0.0 <= ctx_4h.regime_confidence <= 1.0
    assert bias.primary_bias in {"LONG", "SHORT", "NEUTRAL"}
    assert -1.0 <= sentiment.composite_score <= 1.0

    if bias.primary_bias == "LONG":
        assert bias_engine.is_trade_direction_permitted(bias, "LONG") is True
