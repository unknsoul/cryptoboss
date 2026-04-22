import asyncio

from src.analysis.sentiment_engine import SentimentEngine, SentimentScore


def test_finbert_returns_float_between_neg1_pos1():
    engine = SentimentEngine()
    score = engine.analyze_news_with_finbert(
        [
            "Bitcoin rallies as institutions accumulate aggressively.",
            "Market faces liquidation risk after leverage spike.",
            "Macro uncertainty remains elevated for risk assets.",
        ]
    )

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_composite_score_within_range():
    engine = SentimentEngine()
    score = engine.compute_composite_score(
        {
            "fear_greed_raw": 62,
            "news_sentiment": 0.35,
            "social_buzz_score": 0.61,
            "funding_rate": 0.004,
            "open_interest_delta": 0.02,
        }
    )

    assert -1.0 <= score <= 1.0


def test_extreme_sentiment_flag_triggers_correctly():
    engine = SentimentEngine()
    score = SentimentScore(
        composite_score=0.88,
        fear_greed_raw=91,
        news_sentiment=0.7,
        social_buzz_score=0.86,
        funding_rate=0.018,
        funding_rate_signal="LONG_SQUEEZE_RISK",
        extreme_sentiment_flag=True,
    )

    assert engine.is_extreme_sentiment(score) is True


def test_refresh_all_sources_returns_schema():
    async def _run():
        engine = SentimentEngine(
            news_provider=lambda: ["Bullish breakout with strong ETF inflows"],
            fear_greed_provider=lambda: 58,
            social_provider=lambda: {"upvote_ratio": 0.72, "mention_velocity": 0.63},
            funding_provider=lambda: 0.003,
            oi_provider=lambda: 0.015,
        )
        return await engine.refresh_all_sources()

    result = asyncio.run(_run())

    assert isinstance(result.composite_score, float)
    assert -1.0 <= result.composite_score <= 1.0
    assert result.funding_rate_signal in {"LONG_SQUEEZE_RISK", "SHORT_SQUEEZE_RISK", "NEUTRAL"}
