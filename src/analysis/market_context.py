"""Market context engine for regime and structure-aware gating."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.analysis.indicators import IndicatorEngine
from src.analysis.regime_detector_advanced import AdvancedRegimeDetector, MarketRegime


class RegimeEnum(str, Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    RANGE_HIGH_VOLUME = "RANGE_HIGH_VOLUME"
    RANGE_LOW_VOLUME = "RANGE_LOW_VOLUME"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    HIGH_VOLATILITY_EXPANSION = "HIGH_VOLATILITY_EXPANSION"
    CONSOLIDATION_BREAKOUT_PENDING = "CONSOLIDATION_BREAKOUT_PENDING"


@dataclass(slots=True)
class MarketContext:
    symbol: str
    timeframe: str
    timestamp: datetime
    regime: RegimeEnum
    regime_confidence: float
    trend_direction: str
    trend_strength: float
    volatility_level: str
    volume_character: str
    key_levels: Dict[str, Any]
    structure_bias: str
    atr_value: float


class MarketContextEngine:
    """Classify market context for downstream bias/permission engines."""

    def __init__(
        self,
        indicator_engine: IndicatorEngine | None = None,
        regime_detector: AdvancedRegimeDetector | None = None,
    ) -> None:
        self.indicator_engine = indicator_engine or IndicatorEngine()
        self.regime_detector = regime_detector or AdvancedRegimeDetector()

    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> MarketContext:
        """Analyze a symbol/timeframe and return the full market context payload."""
        enriched = self.indicator_engine.compute_all(df)
        regime, regime_confidence = self.classify_regime(enriched)

        swing_highs = self.indicator_engine.find_swing_highs(enriched, lookback=10)
        swing_lows = self.indicator_engine.find_swing_lows(enriched, lookback=10)
        structure_bias = self.determine_structure_bias(swing_highs, swing_lows)

        trend_direction = self._trend_direction(enriched)
        trend_strength = self._trend_strength_normalized(enriched)
        atr_value = float(enriched["ATR_14"].iloc[-1]) if not pd.isna(enriched["ATR_14"].iloc[-1]) else 0.0

        context = MarketContext(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            regime=regime,
            regime_confidence=regime_confidence,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility_level=self._volatility_level(enriched),
            volume_character=self._volume_character(enriched),
            key_levels={},
            structure_bias=structure_bias,
            atr_value=atr_value,
        )

        context.key_levels = self.find_key_levels(enriched, context)
        return context

    def classify_regime(self, df: pd.DataFrame) -> Tuple[RegimeEnum, float]:
        """Classify regime and return confidence between 0 and 1."""
        detector_info = self.regime_detector.detect_regime(df)
        mapped = self._map_regime(detector_info.regime, detector_info.confidence)

        # Directional override: low-vol environments can still trend cleanly.
        if mapped == RegimeEnum.CONSOLIDATION_BREAKOUT_PENDING:
            direction = self._trend_direction(df)
            trailing_return = float((df["close"].iloc[-1] / df["close"].iloc[-50]) - 1.0) if len(df) >= 50 else 0.0
            if direction == "UP" and trailing_return > 0.003:
                mapped = RegimeEnum.WEAK_UPTREND
            elif direction == "DOWN" and trailing_return < -0.003:
                mapped = RegimeEnum.WEAK_DOWNTREND

        if mapped in {RegimeEnum.RANGE_HIGH_VOLUME, RegimeEnum.RANGE_LOW_VOLUME}:
            vol_ratio = self._volume_ratio(df)
            mapped = RegimeEnum.RANGE_HIGH_VOLUME if vol_ratio >= 1.0 else RegimeEnum.RANGE_LOW_VOLUME

        return mapped, float(np.clip(detector_info.confidence, 0.0, 1.0))

    def determine_structure_bias(self, swing_highs, swing_lows) -> str:
        """Determine structure bias from HH/HL or LH/LL sequence."""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "NEUTRAL"

        h_prev, h_last = swing_highs[-2]["price"], swing_highs[-1]["price"]
        l_prev, l_last = swing_lows[-2]["price"], swing_lows[-1]["price"]

        if h_last > h_prev and l_last > l_prev:
            return "BULLISH"
        if h_last < h_prev and l_last < l_prev:
            return "BEARISH"
        return "NEUTRAL"

    def find_key_levels(self, df: pd.DataFrame, context: MarketContext) -> Dict[str, Any]:
        """Find nearest structure and flow levels used by downstream gating."""
        price = float(df["close"].iloc[-1])
        highs = self.indicator_engine.find_swing_highs(df, lookback=5)
        lows = self.indicator_engine.find_swing_lows(df, lookback=5)

        low_prices = [float(point["price"]) for point in lows]
        high_prices = [float(point["price"]) for point in highs]

        below = [p for p in low_prices if p <= price]
        above = [p for p in high_prices if p >= price]

        nearest_support = max(below) if below else (min(low_prices) if low_prices else price)
        nearest_resistance = min(above) if above else (max(high_prices) if high_prices else price)

        ob = self.indicator_engine.find_order_blocks(df)
        fvg = self.indicator_engine.find_fair_value_gaps(df)

        day_start_open = float(df["open"].iloc[-24]) if len(df) >= 24 else float(df["open"].iloc[0])
        ema200 = float(df["EMA_200"].iloc[-1]) if "EMA_200" in df.columns else price

        prev_day_slice = df.iloc[-48:-24] if len(df) >= 48 else df.iloc[:-1]
        prev_day_high = float(prev_day_slice["high"].max()) if len(prev_day_slice) else price
        prev_day_low = float(prev_day_slice["low"].min()) if len(prev_day_slice) else price

        return {
            "nearest_support": float(nearest_support),
            "nearest_resistance": float(nearest_resistance),
            "active_order_blocks": ob[-6:],
            "active_fvg": fvg[-6:],
            "vwap": float(df["VWAP"].iloc[-1]) if "VWAP" in df.columns else price,
            "daily_open": day_start_open,
            "ema200": ema200,
            "prev_day_high": prev_day_high,
            "prev_day_low": prev_day_low,
            "last_price": price,
        }

    def is_near_key_level(self, price: float, context: MarketContext, tolerance_atr: float = 0.5) -> bool:
        """Check if a given price is near major support/resistance."""
        tolerance = max(context.atr_value * tolerance_atr, max(price, 1.0) * 0.001)
        support = float(context.key_levels.get("nearest_support", price))
        resistance = float(context.key_levels.get("nearest_resistance", price))

        return abs(price - support) <= tolerance or abs(price - resistance) <= tolerance

    def _map_regime(self, base_regime: MarketRegime, confidence: float) -> RegimeEnum:
        if base_regime == MarketRegime.TRENDING_UP:
            return RegimeEnum.STRONG_UPTREND if confidence >= 0.75 else RegimeEnum.WEAK_UPTREND
        if base_regime == MarketRegime.TRENDING_DOWN:
            return RegimeEnum.STRONG_DOWNTREND if confidence >= 0.75 else RegimeEnum.WEAK_DOWNTREND
        if base_regime == MarketRegime.HIGH_VOLATILITY:
            return RegimeEnum.HIGH_VOLATILITY_EXPANSION
        if base_regime == MarketRegime.LOW_VOLATILITY:
            return RegimeEnum.CONSOLIDATION_BREAKOUT_PENDING
        return RegimeEnum.RANGE_LOW_VOLUME

    @staticmethod
    def _trend_direction(df: pd.DataFrame) -> str:
        if "EMA_20" in df.columns and "EMA_50" in df.columns:
            ema20 = float(df["EMA_20"].iloc[-1])
            ema50 = float(df["EMA_50"].iloc[-1])
        else:
            ema20 = float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        if ema20 > ema50 * 1.001:
            return "UP"
        if ema20 < ema50 * 0.999:
            return "DOWN"
        return "NEUTRAL"

    @staticmethod
    def _trend_strength_normalized(df: pd.DataFrame) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr = tr.rolling(14).mean().replace(0, np.nan)
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr

        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
        adx = dx.rolling(14).mean().fillna(0.0)

        return float(np.clip(adx.iloc[-1] / 100.0, 0.0, 1.0))

    @staticmethod
    def _volume_ratio(df: pd.DataFrame) -> float:
        rolling = df["volume"].rolling(20).mean()
        baseline = float(rolling.iloc[-1]) if not pd.isna(rolling.iloc[-1]) else float(df["volume"].mean())
        if baseline <= 0:
            return 1.0
        return float(df["volume"].iloc[-1] / baseline)

    def _volume_character(self, df: pd.DataFrame) -> str:
        ratio = self._volume_ratio(df)
        if ratio >= 1.8:
            return "SPIKE"
        if ratio >= 1.1:
            return "INCREASING"
        if ratio <= 0.8:
            return "DECREASING"
        return "NORMAL"

    @staticmethod
    def _volatility_level(df: pd.DataFrame) -> str:
        atr = float(df["ATR_14"].iloc[-1]) if "ATR_14" in df.columns and not pd.isna(df["ATR_14"].iloc[-1]) else 0.0
        close = max(float(df["close"].iloc[-1]), 1e-9)
        atr_pct = atr / close

        if atr_pct >= 0.03:
            return "EXTREME"
        if atr_pct >= 0.015:
            return "HIGH"
        if atr_pct <= 0.006:
            return "LOW"
        return "NORMAL"
