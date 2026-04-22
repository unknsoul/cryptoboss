"""Multi-source sentiment engine with deterministic scoring."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np


@dataclass(slots=True)
class SentimentScore:
    composite_score: float
    fear_greed_raw: int
    news_sentiment: float
    social_buzz_score: float
    funding_rate: float
    funding_rate_signal: str
    extreme_sentiment_flag: bool


class SentimentEngine:
    """Aggregate sentiment from news, market, and social-derived sources."""

    def __init__(
        self,
        news_provider: Callable[[], Any] | None = None,
        fear_greed_provider: Callable[[], Any] | None = None,
        social_provider: Callable[[], Any] | None = None,
        funding_provider: Callable[[], Any] | None = None,
        oi_provider: Callable[[], Any] | None = None,
        contrarian_threshold: float = 0.01,
    ) -> None:
        self.news_provider = news_provider
        self.fear_greed_provider = fear_greed_provider
        self.social_provider = social_provider
        self.funding_provider = funding_provider
        self.oi_provider = oi_provider
        self.contrarian_threshold = contrarian_threshold

    async def refresh_all_sources(self) -> SentimentScore:
        """Refresh all source components and return composite sentiment score."""
        articles = await self._resolve_provider(self.news_provider, default=[])
        fear_greed = int(await self._resolve_provider(self.fear_greed_provider, default=50))
        social_payload = await self._resolve_provider(self.social_provider, default={})
        funding_rate = float(await self._resolve_provider(self.funding_provider, default=0.0))
        open_interest_delta = float(await self._resolve_provider(self.oi_provider, default=0.0))

        news_sentiment = self.analyze_news_with_finbert(list(articles))
        social_buzz = self._compute_social_buzz(dict(social_payload))

        if funding_rate >= self.contrarian_threshold:
            funding_signal = "LONG_SQUEEZE_RISK"
        elif funding_rate <= -self.contrarian_threshold:
            funding_signal = "SHORT_SQUEEZE_RISK"
        else:
            funding_signal = "NEUTRAL"

        composite = self.compute_composite_score(
            {
                "fear_greed_raw": fear_greed,
                "news_sentiment": news_sentiment,
                "social_buzz_score": social_buzz,
                "funding_rate": funding_rate,
                "open_interest_delta": open_interest_delta,
            }
        )

        provisional = SentimentScore(
            composite_score=composite,
            fear_greed_raw=fear_greed,
            news_sentiment=news_sentiment,
            social_buzz_score=social_buzz,
            funding_rate=funding_rate,
            funding_rate_signal=funding_signal,
            extreme_sentiment_flag=False,
        )
        provisional.extreme_sentiment_flag = self.is_extreme_sentiment(provisional)
        return provisional

    def analyze_news_with_finbert(self, articles: List[str]) -> float:
        """Return sentiment score in [-1, 1] using deterministic lexical fallback."""
        if not articles:
            return 0.0

        positives = {
            "surge",
            "rally",
            "bullish",
            "accumulate",
            "breakout",
            "growth",
            "strong",
            "inflow",
            "recovery",
            "optimism",
        }
        negatives = {
            "crash",
            "bearish",
            "liquidation",
            "panic",
            "risk",
            "decline",
            "selloff",
            "outflow",
            "loss",
            "uncertainty",
        }

        article_scores: list[float] = []
        for article in articles:
            tokens = [token.strip(".,!?;:()[]{}\"'").lower() for token in article.split()]
            pos = sum(1 for token in tokens if token in positives)
            neg = sum(1 for token in tokens if token in negatives)
            denom = max(pos + neg, 1)
            article_scores.append((pos - neg) / denom)

        return float(np.clip(np.mean(article_scores), -1.0, 1.0))

    def compute_composite_score(self, components: Dict) -> float:
        """Compute weighted composite sentiment in [-1, 1]."""
        fear = int(components.get("fear_greed_raw", 50))
        news = float(components.get("news_sentiment", 0.0))
        social = float(components.get("social_buzz_score", 0.5))
        funding = float(components.get("funding_rate", 0.0))
        oi_delta = float(components.get("open_interest_delta", 0.0))

        fear_norm = np.clip((fear - 50) / 50.0, -1.0, 1.0)
        social_norm = np.clip((social * 2.0) - 1.0, -1.0, 1.0)
        funding_contrarian = float(np.clip(-np.tanh(funding * 100), -1.0, 1.0))
        oi_component = float(np.clip(np.tanh(oi_delta * 12), -1.0, 1.0))

        score = (
            0.35 * news
            + 0.25 * fear_norm
            + 0.20 * social_norm
            + 0.10 * funding_contrarian
            + 0.10 * oi_component
        )
        return float(np.clip(score, -1.0, 1.0))

    def is_extreme_sentiment(self, score: SentimentScore) -> bool:
        """Flag extreme sentiment readings suitable for contrarian gating."""
        if abs(score.composite_score) >= 0.8:
            return True
        if score.fear_greed_raw <= 15 or score.fear_greed_raw >= 85:
            return True
        if abs(score.funding_rate) >= 0.02:
            return True
        return False

    @staticmethod
    async def _resolve_provider(provider: Callable[[], Any] | None, default: Any) -> Any:
        if provider is None:
            return default

        result = provider()
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _compute_social_buzz(payload: Dict[str, Any]) -> float:
        upvote_ratio = float(payload.get("upvote_ratio", 0.5))
        mention_velocity = float(payload.get("mention_velocity", 0.5))
        comment_sentiment = float(payload.get("comment_sentiment", 0.0))

        comment_component = np.clip((comment_sentiment + 1.0) / 2.0, 0.0, 1.0)
        score = (0.45 * upvote_ratio) + (0.35 * mention_velocity) + (0.20 * comment_component)
        return float(np.clip(score, 0.0, 1.0))
