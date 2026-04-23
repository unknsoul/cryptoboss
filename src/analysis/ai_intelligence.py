"""AI intelligence advisory layer.

This module is advisory-only by design. It never emits direct entry/exit signals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Protocol


class AIJsonClient(Protocol):
    """Protocol for pluggable AI clients used by the advisor."""

    def generate_json(self, prompt: str, timeout_seconds: int = 10) -> dict[str, Any]:
        """Return model output as JSON-compatible dict."""


@dataclass(slots=True)
class AIAdvisoryResult:
    """Standardized AI advisory response envelope."""

    success: bool
    advisory_type: str
    model: str
    latency_ms: int
    payload: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AIIntelligenceAdvisor:
    """Advisory AI wrapper with graceful fallback behavior."""

    def __init__(
        self,
        model: str = "gpt-4o",
        enabled: bool = True,
        timeout_seconds: int = 10,
        client: AIJsonClient | None = None,
    ) -> None:
        self.model = model
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.client = client

    def market_narrative_analysis(self, symbol: str, context: dict[str, Any]) -> AIAdvisoryResult:
        """Generate advisory market narrative."""
        start = perf_counter()

        if not self.enabled:
            return self._result(
                advisory_type="market_narrative",
                start=start,
                payload=self._heuristic_market_narrative(symbol, context),
                success=False,
                error="AI advisor disabled",
            )

        prompt = self._build_market_prompt(symbol, context)

        if self.client is None:
            return self._result(
                advisory_type="market_narrative",
                start=start,
                payload=self._heuristic_market_narrative(symbol, context),
                success=False,
                error="No AI client configured; using heuristic fallback",
            )

        try:
            payload = self.client.generate_json(prompt, timeout_seconds=self.timeout_seconds)
            payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
            payload.setdefault("advisory_only", True)
            return self._result(
                advisory_type="market_narrative",
                start=start,
                payload=payload,
                success=True,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback path.
            return self._result(
                advisory_type="market_narrative",
                start=start,
                payload=self._heuristic_market_narrative(symbol, context),
                success=False,
                error=f"AI request failed: {exc}",
            )

    def news_sentiment_filter(self, symbol: str, headlines: list[str]) -> AIAdvisoryResult:
        """Generate advisory sentiment score from headlines."""
        start = perf_counter()

        if not headlines:
            return self._result(
                advisory_type="news_sentiment",
                start=start,
                payload={
                    "symbol": symbol,
                    "sentiment_score": 0.0,
                    "key_themes": [],
                    "risk_event_detected": False,
                    "advisory_only": True,
                },
                success=True,
            )

        prompt = self._build_sentiment_prompt(symbol, headlines)

        if self.enabled and self.client is not None:
            try:
                payload = self.client.generate_json(prompt, timeout_seconds=self.timeout_seconds)
                payload.setdefault("symbol", symbol)
                payload.setdefault("advisory_only", True)
                return self._result(
                    advisory_type="news_sentiment",
                    start=start,
                    payload=payload,
                    success=True,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback path.
                fallback = self._heuristic_sentiment(symbol, headlines)
                return self._result(
                    advisory_type="news_sentiment",
                    start=start,
                    payload=fallback,
                    success=False,
                    error=f"AI request failed: {exc}",
                )

        return self._result(
            advisory_type="news_sentiment",
            start=start,
            payload=self._heuristic_sentiment(symbol, headlines),
            success=False,
            error="No AI client configured; using heuristic fallback",
        )

    @staticmethod
    def _build_market_prompt(symbol: str, context: dict[str, Any]) -> str:
        return (
            "You are a professional SMC trader. Analyze symbol and return JSON only.\n"
            f"Symbol: {symbol}\n"
            f"Context: {context}\n"
            "Return keys: next_target, key_levels, premium_or_discount, direction_confidence_0_to_1, "
            "narrative_summary"
        )

    @staticmethod
    def _build_sentiment_prompt(symbol: str, headlines: list[str]) -> str:
        return (
            "Rate crypto market sentiment from -1 to +1 and return JSON only.\n"
            f"Symbol: {symbol}\n"
            f"Headlines: {headlines}\n"
            "Return keys: sentiment_score, key_themes, risk_event_detected"
        )

    @staticmethod
    def _heuristic_market_narrative(symbol: str, context: dict[str, Any]) -> dict[str, Any]:
        bias = str(context.get("bias", "NEUTRAL")).upper()
        regime = str(context.get("regime", "RANGING")).upper()
        key_levels = context.get("key_levels", [])

        if bias == "BULLISH":
            next_target = context.get("nearest_bsl") or "next_buyside_liquidity"
            premium_discount = "DISCOUNT" if regime.startswith("TRENDING") else "EQUILIBRIUM"
            confidence = 0.65
            summary = "Bias favors continuation to buyside liquidity if displacement confirms."
        elif bias == "BEARISH":
            next_target = context.get("nearest_ssl") or "next_sellside_liquidity"
            premium_discount = "PREMIUM" if regime.startswith("TRENDING") else "EQUILIBRIUM"
            confidence = 0.65
            summary = "Bias favors continuation to sellside liquidity if bearish structure holds."
        else:
            next_target = "range_extremes"
            premium_discount = "EQUILIBRIUM"
            confidence = 0.35
            summary = "No clear directional edge; wait for sweep and structure confirmation."

        return {
            "symbol": symbol,
            "next_target": next_target,
            "key_levels": key_levels,
            "premium_or_discount": premium_discount,
            "direction_confidence_0_to_1": confidence,
            "narrative_summary": summary,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "advisory_only": True,
        }

    @staticmethod
    def _heuristic_sentiment(symbol: str, headlines: list[str]) -> dict[str, Any]:
        positive_tokens = {
            "surge",
            "rally",
            "gain",
            "record",
            "approval",
            "inflow",
            "breakout",
            "bull",
        }
        negative_tokens = {
            "hack",
            "lawsuit",
            "ban",
            "dump",
            "loss",
            "outflow",
            "exploit",
            "bear",
            "liquidation",
        }

        score = 0
        themes: list[str] = []
        risk_event = False

        for raw in headlines:
            headline = raw.lower()
            for token in positive_tokens:
                if token in headline:
                    score += 1
                    themes.append(token)
            for token in negative_tokens:
                if token in headline:
                    score -= 1
                    themes.append(token)
                    if token in {"hack", "lawsuit", "ban", "exploit"}:
                        risk_event = True

        normalized = 0.0
        if headlines:
            normalized = max(-1.0, min(1.0, score / max(1.0, len(headlines))))

        # Deduplicate while preserving order.
        seen: set[str] = set()
        deduped_themes: list[str] = []
        for theme in themes:
            if theme in seen:
                continue
            seen.add(theme)
            deduped_themes.append(theme)

        return {
            "symbol": symbol,
            "sentiment_score": normalized,
            "key_themes": deduped_themes[:8],
            "risk_event_detected": risk_event,
            "advisory_only": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _result(
        self,
        advisory_type: str,
        start: float,
        payload: dict[str, Any],
        success: bool,
        error: str | None = None,
    ) -> AIAdvisoryResult:
        elapsed_ms = int((perf_counter() - start) * 1000)
        return AIAdvisoryResult(
            success=success,
            advisory_type=advisory_type,
            model=self.model,
            latency_ms=elapsed_ms,
            payload=payload,
            error=error,
        )
