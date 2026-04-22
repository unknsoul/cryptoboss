"""Higher-timeframe directional bias engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from src.analysis.market_context import MarketContext, RegimeEnum


@dataclass(slots=True)
class Bias:
    symbol: str
    primary_bias: str
    bias_strength: float
    bias_timeframe: str
    conditions_for_long: list[str]
    conditions_for_short: list[str]
    invalidation_level: float
    next_significant_level: float


class BiasEngine:
    """Determine HTF directional bias from 4H and 1D contexts."""

    LONG_REGIMES = {RegimeEnum.STRONG_UPTREND, RegimeEnum.WEAK_UPTREND}
    SHORT_REGIMES = {RegimeEnum.STRONG_DOWNTREND, RegimeEnum.WEAK_DOWNTREND}

    def compute_bias(self, contexts: Dict[str, MarketContext]) -> Bias:
        """Compute directional bias from higher timeframe contexts."""
        ctx_4h = contexts.get("4h") or next(iter(contexts.values()))
        ctx_1d = contexts.get("1d") or ctx_4h

        long_reasons = self._long_reasons(ctx_4h, ctx_1d)
        short_reasons = self._short_reasons(ctx_4h, ctx_1d)

        long_score = len(long_reasons)
        short_score = len(short_reasons)

        if long_score >= 4 and long_score > short_score:
            primary = "LONG"
            strength = long_score / 5.0
            next_level = float(ctx_1d.key_levels.get("nearest_resistance", ctx_4h.key_levels.get("nearest_resistance", 0.0)))
            invalidation = min(
                float(ctx_4h.key_levels.get("nearest_support", 0.0)),
                float(ctx_1d.key_levels.get("nearest_support", 0.0)),
            )
        elif short_score >= 4 and short_score > long_score:
            primary = "SHORT"
            strength = short_score / 5.0
            next_level = float(ctx_1d.key_levels.get("nearest_support", ctx_4h.key_levels.get("nearest_support", 0.0)))
            invalidation = max(
                float(ctx_4h.key_levels.get("nearest_resistance", 0.0)),
                float(ctx_1d.key_levels.get("nearest_resistance", 0.0)),
            )
        else:
            primary = "NEUTRAL"
            strength = max(long_score, short_score) / 5.0
            invalidation = float(ctx_1d.key_levels.get("last_price", ctx_4h.key_levels.get("last_price", 0.0)))
            next_level = invalidation

        bias_timeframe = "4h" if ctx_4h.regime_confidence >= ctx_1d.regime_confidence else "1d"

        return Bias(
            symbol=ctx_4h.symbol,
            primary_bias=primary,
            bias_strength=float(min(max(strength, 0.0), 1.0)),
            bias_timeframe=bias_timeframe,
            conditions_for_long=long_reasons,
            conditions_for_short=short_reasons,
            invalidation_level=float(invalidation),
            next_significant_level=float(next_level),
        )

    def is_trade_direction_permitted(self, bias: Bias, direction: str) -> bool:
        """Return whether a proposed direction is permitted by current bias."""
        normalized = direction.upper()
        if bias.primary_bias == "NEUTRAL":
            return normalized in {"LONG", "SHORT"}
        if bias.primary_bias == "LONG":
            return normalized == "LONG"
        if bias.primary_bias == "SHORT":
            return normalized == "SHORT"
        return False

    def get_bias_invalidation_level(self, bias: Bias, df_daily: pd.DataFrame) -> float:
        """Compute invalidation from daily structure as a fallback helper."""
        if bias.primary_bias == "LONG":
            return float(df_daily["low"].tail(20).min())
        if bias.primary_bias == "SHORT":
            return float(df_daily["high"].tail(20).max())
        return float(df_daily["close"].iloc[-1])

    def _long_reasons(self, ctx_4h: MarketContext, ctx_1d: MarketContext) -> list[str]:
        reasons: list[str] = []

        last_1d = float(ctx_1d.key_levels.get("last_price", 0.0))
        ema200_1d = float(ctx_1d.key_levels.get("ema200", last_1d))
        prev_day_high = float(ctx_1d.key_levels.get("prev_day_high", last_1d))
        last_4h = float(ctx_4h.key_levels.get("last_price", 0.0))
        vwap_4h = float(ctx_4h.key_levels.get("vwap", last_4h))

        if last_1d > ema200_1d:
            reasons.append("1D price above EMA200")
        if ctx_4h.structure_bias == "BULLISH":
            reasons.append("4H structure shows HH + HL")
        if ctx_4h.regime in self.LONG_REGIMES:
            reasons.append("4H regime is STRONG_UPTREND or WEAK_UPTREND")
        if last_4h > vwap_4h:
            reasons.append("4H price above VWAP")
        if last_1d > prev_day_high:
            reasons.append("Daily close above previous day high (momentum)")

        return reasons

    def _short_reasons(self, ctx_4h: MarketContext, ctx_1d: MarketContext) -> list[str]:
        reasons: list[str] = []

        last_1d = float(ctx_1d.key_levels.get("last_price", 0.0))
        ema200_1d = float(ctx_1d.key_levels.get("ema200", last_1d))
        prev_day_low = float(ctx_1d.key_levels.get("prev_day_low", last_1d))
        last_4h = float(ctx_4h.key_levels.get("last_price", 0.0))
        vwap_4h = float(ctx_4h.key_levels.get("vwap", last_4h))

        if last_1d < ema200_1d:
            reasons.append("1D price below EMA200")
        if ctx_4h.structure_bias == "BEARISH":
            reasons.append("4H structure shows LH + LL")
        if ctx_4h.regime in self.SHORT_REGIMES:
            reasons.append("4H regime is STRONG_DOWNTREND or WEAK_DOWNTREND")
        if last_4h < vwap_4h:
            reasons.append("4H price below VWAP")
        if last_1d < prev_day_low:
            reasons.append("Daily close below previous day low")

        return reasons
