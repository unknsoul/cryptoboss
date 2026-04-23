"""Market structure engine for BOS/CHoCH/MSB and trend state classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import pandas as pd


class StructureEventType(str, Enum):
    """Supported structural event types."""

    BOS = "BOS"
    CHOCH = "CHoCH"
    MSB = "MSB"


class TrendState(str, Enum):
    """High-level trend classifications."""

    BULLISH_TRENDING = "BULLISH_TRENDING"
    BEARISH_TRENDING = "BEARISH_TRENDING"
    RANGING = "RANGING"
    DISTRIBUTION = "DISTRIBUTION"
    ACCUMULATION = "ACCUMULATION"


@dataclass(slots=True)
class SwingPoint:
    """Represents a pivot swing point."""

    index: int
    timestamp: pd.Timestamp
    kind: str  # swing_high | swing_low
    price: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(slots=True)
class StructureEvent:
    """Represents a detected structure event."""

    event_type: StructureEventType
    direction: str  # bullish | bearish
    timestamp: pd.Timestamp
    level: float
    body_close_confirmed: bool
    weight: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        return data


@dataclass(slots=True)
class MarketStructureSnapshot:
    """Full market structure snapshot for one timeframe."""

    timeframe: str
    trend_state: TrendState
    swings: list[SwingPoint]
    events: list[StructureEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "trend_state": self.trend_state.value,
            "swings": [s.to_dict() for s in self.swings],
            "events": [e.to_dict() for e in self.events],
        }


class MarketStructureEngine:
    """Detects swings, BOS/CHoCH/MSB, and classifies trend state."""

    def __init__(
        self,
        swing_lookback_candles: int = 20,
        swing_pivot_confirmation_candles: int = 3,
        minimum_swing_size_atr_multiplier: float = 0.5,
    ) -> None:
        self.swing_lookback_candles = swing_lookback_candles
        self.swing_pivot_confirmation_candles = swing_pivot_confirmation_candles
        self.minimum_swing_size_atr_multiplier = minimum_swing_size_atr_multiplier

    def detect_swings(self, frame: pd.DataFrame) -> list[SwingPoint]:
        """Detect local swing highs and lows with pivot confirmation."""
        self._validate_frame(frame)
        if len(frame) < (self.swing_pivot_confirmation_candles * 2 + 5):
            return []

        highs = frame["high"].astype(float)
        lows = frame["low"].astype(float)
        atr = self._atr(frame, period=14)
        piv = self.swing_pivot_confirmation_candles

        swings: list[SwingPoint] = []

        for i in range(piv, len(frame) - piv):
            window_high = highs.iloc[i - piv : i + piv + 1]
            window_low = lows.iloc[i - piv : i + piv + 1]

            is_swing_high = highs.iloc[i] >= window_high.max()
            is_swing_low = lows.iloc[i] <= window_low.min()

            if not (is_swing_high or is_swing_low):
                continue

            # Filter tiny pivots against local ATR to reduce noise.
            local_atr = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
            local_range = float(highs.iloc[i] - lows.iloc[i])
            if local_atr > 0 and local_range < local_atr * self.minimum_swing_size_atr_multiplier:
                continue

            ts = frame.index[i]
            if is_swing_high:
                swings.append(
                    SwingPoint(
                        index=i,
                        timestamp=pd.Timestamp(ts),
                        kind="swing_high",
                        price=float(highs.iloc[i]),
                    )
                )
            if is_swing_low:
                swings.append(
                    SwingPoint(
                        index=i,
                        timestamp=pd.Timestamp(ts),
                        kind="swing_low",
                        price=float(lows.iloc[i]),
                    )
                )

        return swings

    def classify_bos(self, frame: pd.DataFrame, swings: list[SwingPoint]) -> list[StructureEvent]:
        """Classify BOS events from swing breaks using close confirmation."""
        self._validate_frame(frame)
        if not swings:
            return []

        closes = frame["close"].astype(float)
        events: list[StructureEvent] = []

        last_high: SwingPoint | None = None
        last_low: SwingPoint | None = None

        for swing in swings:
            if swing.kind == "swing_high":
                last_high = swing
            elif swing.kind == "swing_low":
                last_low = swing

            idx = swing.index
            if idx >= len(frame):
                continue

            close_price = float(closes.iloc[idx])

            if last_high and close_price > last_high.price:
                events.append(
                    StructureEvent(
                        event_type=StructureEventType.BOS,
                        direction="bullish",
                        timestamp=swing.timestamp,
                        level=last_high.price,
                        body_close_confirmed=True,
                        weight=0.7,
                    )
                )
            if last_low and close_price < last_low.price:
                events.append(
                    StructureEvent(
                        event_type=StructureEventType.BOS,
                        direction="bearish",
                        timestamp=swing.timestamp,
                        level=last_low.price,
                        body_close_confirmed=True,
                        weight=0.7,
                    )
                )

        return self._dedupe_events(events)

    def classify_choch(self, bos_events: list[StructureEvent]) -> list[StructureEvent]:
        """Classify CHoCH and MSB from BOS direction changes."""
        if len(bos_events) < 2:
            return []

        events: list[StructureEvent] = []
        previous = bos_events[0]

        for current in bos_events[1:]:
            if current.direction == previous.direction:
                previous = current
                continue

            choch = StructureEvent(
                event_type=StructureEventType.CHOCH,
                direction=current.direction,
                timestamp=current.timestamp,
                level=current.level,
                body_close_confirmed=current.body_close_confirmed,
                weight=0.9,
            )
            events.append(choch)

            msb = StructureEvent(
                event_type=StructureEventType.MSB,
                direction=current.direction,
                timestamp=current.timestamp,
                level=current.level,
                body_close_confirmed=current.body_close_confirmed,
                weight=1.0,
            )
            events.append(msb)
            previous = current

        return events

    def get_trend_state(
        self,
        swings: list[SwingPoint],
        events: list[StructureEvent],
    ) -> TrendState:
        """Infer trend state from recent swing sequence and latest events."""
        highs = [s for s in swings if s.kind == "swing_high"]
        lows = [s for s in swings if s.kind == "swing_low"]

        if len(highs) < 2 or len(lows) < 2:
            return TrendState.RANGING

        recent_highs = highs[-2:]
        recent_lows = lows[-2:]

        highs_rising = recent_highs[-1].price > recent_highs[-2].price
        lows_rising = recent_lows[-1].price > recent_lows[-2].price
        highs_falling = recent_highs[-1].price < recent_highs[-2].price
        lows_falling = recent_lows[-1].price < recent_lows[-2].price

        last_event = events[-1] if events else None

        if highs_rising and lows_rising:
            return TrendState.BULLISH_TRENDING
        if highs_falling and lows_falling:
            return TrendState.BEARISH_TRENDING

        if last_event and last_event.direction == "bullish":
            return TrendState.ACCUMULATION
        if last_event and last_event.direction == "bearish":
            return TrendState.DISTRIBUTION

        return TrendState.RANGING

    def get_structure_snapshot(
        self,
        frame: pd.DataFrame,
        timeframe: str,
    ) -> MarketStructureSnapshot:
        """Build a complete structure snapshot for a timeframe."""
        swings = self.detect_swings(frame)
        bos = self.classify_bos(frame, swings)
        choch_msb = self.classify_choch(bos)
        events = sorted([*bos, *choch_msb], key=lambda e: e.timestamp)

        trend = self.get_trend_state(swings, events)

        return MarketStructureSnapshot(
            timeframe=timeframe,
            trend_state=trend,
            swings=swings[-50:],
            events=events[-50:],
        )

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Frame missing required columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Frame is empty")

    @staticmethod
    def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)

        tr = pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    @staticmethod
    def _dedupe_events(events: list[StructureEvent]) -> list[StructureEvent]:
        """Remove duplicate events emitted for the same timestamp+direction+type."""
        deduped: list[StructureEvent] = []
        seen: set[tuple[str, str, str]] = set()

        for event in events:
            key = (
                event.timestamp.isoformat(),
                event.direction,
                event.event_type.value,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(event)

        deduped.sort(key=lambda e: e.timestamp)
        return deduped
