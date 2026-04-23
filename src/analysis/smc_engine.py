"""Smart Money Concepts (SMC) engine for OB/FVG/liquidity/premium-discount analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import pandas as pd


class PremiumDiscountZone(str, Enum):
    """Premium/discount classification."""

    PREMIUM = "PREMIUM"
    DISCOUNT = "DISCOUNT"
    EQUILIBRIUM = "EQUILIBRIUM"


@dataclass(slots=True)
class OrderBlock:
    """Detected order block."""

    direction: str
    timeframe: str
    timestamp: pd.Timestamp
    low: float
    high: float
    impulse_strength: float
    score: float
    mitigated: bool = False
    invalidated: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(slots=True)
class FairValueGap:
    """Detected fair value gap."""

    direction: str
    timeframe: str
    timestamp: pd.Timestamp
    bottom: float
    top: float
    midpoint: float
    score: float
    filled_ratio: float = 0.0
    invalidated: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(slots=True)
class LiquidityZone:
    """Detected BSL/SSL area."""

    zone_type: str  # BSL | SSL
    timeframe: str
    timestamp: pd.Timestamp
    price: float
    touches: int
    swept: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(slots=True)
class SweepEvent:
    """Detected liquidity sweep event."""

    zone_type: str
    timeframe: str
    timestamp: pd.Timestamp
    wick_extension_atr: float
    confirmed: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(slots=True)
class SMCSnapshot:
    """Composite SMC state for one timeframe."""

    timeframe: str
    order_blocks: list[OrderBlock]
    fvgs: list[FairValueGap]
    bsl_zones: list[LiquidityZone]
    ssl_zones: list[LiquidityZone]
    last_sweep: SweepEvent | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "order_blocks": [o.to_dict() for o in self.order_blocks],
            "fvgs": [f.to_dict() for f in self.fvgs],
            "bsl_zones": [z.to_dict() for z in self.bsl_zones],
            "ssl_zones": [z.to_dict() for z in self.ssl_zones],
            "last_sweep": self.last_sweep.to_dict() if self.last_sweep else None,
        }


class SMCEngine:
    """Detects SMC primitives and returns structured snapshots."""

    def __init__(
        self,
        equal_hl_tolerance_pct: float = 0.15,
        equal_hl_min_touches: int = 2,
        equal_hl_lookback: int = 50,
        min_impulse_atr_mult: float = 1.5,
        min_gap_atr_mult: float = 0.3,
        min_gap_price_pct: float = 0.05,
        premium_threshold_fib: float = 0.618,
        discount_threshold_fib: float = 0.382,
        oez_low: float = 0.618,
        oez_high: float = 0.786,
    ) -> None:
        self.equal_hl_tolerance_pct = equal_hl_tolerance_pct
        self.equal_hl_min_touches = equal_hl_min_touches
        self.equal_hl_lookback = equal_hl_lookback
        self.min_impulse_atr_mult = min_impulse_atr_mult
        self.min_gap_atr_mult = min_gap_atr_mult
        self.min_gap_price_pct = min_gap_price_pct
        self.premium_threshold_fib = premium_threshold_fib
        self.discount_threshold_fib = discount_threshold_fib
        self.oez_low = oez_low
        self.oez_high = oez_high

    def detect_obs(self, frame: pd.DataFrame, timeframe: str) -> list[OrderBlock]:
        """Detect bullish and bearish order blocks."""
        self._validate_frame(frame)
        if len(frame) < 5:
            return []

        atr = self._atr(frame)
        events: list[OrderBlock] = []

        for i in range(2, len(frame)):
            prev = frame.iloc[i - 1]
            curr = frame.iloc[i]
            curr_atr = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
            body = abs(float(curr["close"] - curr["open"]))

            if curr_atr <= 0:
                continue

            impulse_mult = body / curr_atr
            if impulse_mult < self.min_impulse_atr_mult:
                continue

            # Bullish OB: bearish candle before bullish displacement.
            if float(prev["close"]) < float(prev["open"]) and float(curr["close"]) > float(curr["open"]):
                score = min(1.0, 0.4 + impulse_mult / 4.0)
                events.append(
                    OrderBlock(
                        direction="bullish",
                        timeframe=timeframe,
                        timestamp=pd.Timestamp(frame.index[i - 1]),
                        low=float(prev["low"]),
                        high=float(prev["high"]),
                        impulse_strength=impulse_mult,
                        score=score,
                    )
                )

            # Bearish OB: bullish candle before bearish displacement.
            if float(prev["close"]) > float(prev["open"]) and float(curr["close"]) < float(curr["open"]):
                score = min(1.0, 0.4 + impulse_mult / 4.0)
                events.append(
                    OrderBlock(
                        direction="bearish",
                        timeframe=timeframe,
                        timestamp=pd.Timestamp(frame.index[i - 1]),
                        low=float(prev["low"]),
                        high=float(prev["high"]),
                        impulse_strength=impulse_mult,
                        score=score,
                    )
                )

        return events[-80:]

    @staticmethod
    def is_price_inside_ob(price: float, ob: OrderBlock) -> bool:
        """Return True when price is inside the OB range."""
        return ob.low <= price <= ob.high

    @staticmethod
    def get_ob_mitigation_level(ob: OrderBlock) -> float:
        """Return OB midpoint mitigation level."""
        return ob.low + (ob.high - ob.low) * 0.5

    @staticmethod
    def invalidate_ob(ob: OrderBlock, close_price: float) -> bool:
        """Invalidate OB on close through."""
        if ob.direction == "bullish" and close_price < ob.low:
            ob.invalidated = True
        if ob.direction == "bearish" and close_price > ob.high:
            ob.invalidated = True
        return ob.invalidated

    @staticmethod
    def score_ob(ob: OrderBlock, current_price: float | None = None) -> float:
        """Return quality score for an OB."""
        if current_price is None:
            return max(0.0, min(1.0, ob.score))

        distance = abs(current_price - ((ob.low + ob.high) / 2.0))
        width = max(1e-12, ob.high - ob.low)
        proximity = max(0.0, 1.0 - (distance / (5.0 * width)))
        return max(0.0, min(1.0, 0.7 * ob.score + 0.3 * proximity))

    def detect_fvgs(self, frame: pd.DataFrame, timeframe: str) -> list[FairValueGap]:
        """Detect 3-candle FVG structures."""
        self._validate_frame(frame)
        if len(frame) < 3:
            return []

        atr = self._atr(frame)
        fvgs: list[FairValueGap] = []

        for i in range(1, len(frame) - 1):
            c0 = frame.iloc[i - 1]
            c1 = frame.iloc[i]
            c2 = frame.iloc[i + 1]

            local_atr = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0

            # Bullish FVG
            if float(c0["high"]) < float(c2["low"]):
                gap = float(c2["low"] - c0["high"])
                if self._gap_valid(gap, float(c1["close"]), local_atr):
                    score = self._score_gap(gap, local_atr)
                    fvgs.append(
                        FairValueGap(
                            direction="bullish",
                            timeframe=timeframe,
                            timestamp=pd.Timestamp(frame.index[i]),
                            bottom=float(c0["high"]),
                            top=float(c2["low"]),
                            midpoint=float((c0["high"] + c2["low"]) / 2.0),
                            score=score,
                        )
                    )

            # Bearish FVG
            if float(c0["low"]) > float(c2["high"]):
                gap = float(c0["low"] - c2["high"])
                if self._gap_valid(gap, float(c1["close"]), local_atr):
                    score = self._score_gap(gap, local_atr)
                    fvgs.append(
                        FairValueGap(
                            direction="bearish",
                            timeframe=timeframe,
                            timestamp=pd.Timestamp(frame.index[i]),
                            bottom=float(c2["high"]),
                            top=float(c0["low"]),
                            midpoint=float((c2["high"] + c0["low"]) / 2.0),
                            score=score,
                        )
                    )

        return fvgs[-120:]

    def is_fvg_filled(self, fvg: FairValueGap, current_price: float) -> bool:
        """Check if FVG has been fully filled by current price."""
        if fvg.direction == "bullish":
            fvg.filled_ratio = max(0.0, min(1.0, (fvg.top - current_price) / max(1e-12, fvg.top - fvg.bottom)))
            if current_price <= fvg.bottom:
                fvg.invalidated = True
        else:
            fvg.filled_ratio = max(0.0, min(1.0, (current_price - fvg.bottom) / max(1e-12, fvg.top - fvg.bottom)))
            if current_price >= fvg.top:
                fvg.invalidated = True
        return fvg.invalidated

    @staticmethod
    def get_fvg_midpoint(fvg: FairValueGap) -> float:
        """Return FVG midpoint."""
        return fvg.midpoint

    @staticmethod
    def score_fvg(fvg: FairValueGap) -> float:
        """Return quality score for FVG."""
        freshness = max(0.0, 1.0 - fvg.filled_ratio)
        return max(0.0, min(1.0, 0.7 * fvg.score + 0.3 * freshness))

    def find_bsl_zones(self, frame: pd.DataFrame, timeframe: str) -> list[LiquidityZone]:
        """Find buyside liquidity (equal highs)."""
        self._validate_frame(frame)
        highs = frame["high"].astype(float).tail(self.equal_hl_lookback)
        return self._find_equal_levels(
            series=highs,
            zone_type="BSL",
            timeframe=timeframe,
            index=frame.index[-len(highs) :],
        )

    def find_ssl_zones(self, frame: pd.DataFrame, timeframe: str) -> list[LiquidityZone]:
        """Find sellside liquidity (equal lows)."""
        self._validate_frame(frame)
        lows = frame["low"].astype(float).tail(self.equal_hl_lookback)
        return self._find_equal_levels(
            series=lows,
            zone_type="SSL",
            timeframe=timeframe,
            index=frame.index[-len(lows) :],
        )

    def detect_sweep(
        self,
        frame: pd.DataFrame,
        bsl_zones: list[LiquidityZone],
        ssl_zones: list[LiquidityZone],
        timeframe: str,
    ) -> SweepEvent | None:
        """Detect if the latest candle swept nearby liquidity."""
        self._validate_frame(frame)
        if frame.empty:
            return None

        last = frame.iloc[-1]
        ts = pd.Timestamp(frame.index[-1])
        atr = float(self._atr(frame).iloc[-1]) if len(frame) >= 2 else 0.0
        atr = max(1e-12, atr)

        for zone in bsl_zones:
            if float(last["high"]) > zone.price and float(last["close"]) < zone.price:
                extension = float(last["high"] - zone.price) / atr
                zone.swept = True
                return SweepEvent(
                    zone_type="BSL",
                    timeframe=timeframe,
                    timestamp=ts,
                    wick_extension_atr=extension,
                    confirmed=extension >= 0.3,
                )

        for zone in ssl_zones:
            if float(last["low"]) < zone.price and float(last["close"]) > zone.price:
                extension = float(zone.price - last["low"]) / atr
                zone.swept = True
                return SweepEvent(
                    zone_type="SSL",
                    timeframe=timeframe,
                    timestamp=ts,
                    wick_extension_atr=extension,
                    confirmed=extension >= 0.3,
                )

        return None

    @staticmethod
    def get_nearest_liquidity(
        price: float,
        bsl_zones: list[LiquidityZone],
        ssl_zones: list[LiquidityZone],
    ) -> tuple[LiquidityZone | None, LiquidityZone | None]:
        """Return nearest BSL above and SSL below current price."""
        bsl_candidates = [z for z in bsl_zones if z.price >= price]
        ssl_candidates = [z for z in ssl_zones if z.price <= price]

        nearest_bsl = min(bsl_candidates, key=lambda z: z.price - price) if bsl_candidates else None
        nearest_ssl = min(ssl_candidates, key=lambda z: price - z.price) if ssl_candidates else None
        return nearest_bsl, nearest_ssl

    def get_zone(self, price: float, swing_high: float, swing_low: float) -> PremiumDiscountZone:
        """Classify current price in premium/discount/equilibrium terms."""
        level = self.get_fib_level(price, swing_high, swing_low)
        if level >= self.premium_threshold_fib:
            return PremiumDiscountZone.PREMIUM
        if level <= self.discount_threshold_fib:
            return PremiumDiscountZone.DISCOUNT
        return PremiumDiscountZone.EQUILIBRIUM

    @staticmethod
    def get_fib_level(price: float, swing_high: float, swing_low: float) -> float:
        """Return normalized fib level from swing range."""
        if swing_high <= swing_low:
            return 0.5
        return (price - swing_low) / (swing_high - swing_low)

    def is_in_oez(self, fib_level: float) -> bool:
        """Return True when fib level is inside optimal entry zone."""
        return self.oez_low <= fib_level <= self.oez_high

    def build_snapshot(self, frame: pd.DataFrame, timeframe: str) -> SMCSnapshot:
        """Build complete SMC snapshot for one timeframe."""
        obs = self.detect_obs(frame, timeframe=timeframe)
        fvgs = self.detect_fvgs(frame, timeframe=timeframe)
        bsl = self.find_bsl_zones(frame, timeframe=timeframe)
        ssl = self.find_ssl_zones(frame, timeframe=timeframe)
        sweep = self.detect_sweep(frame, bsl, ssl, timeframe=timeframe)

        current_price = float(frame["close"].iloc[-1])
        for ob in obs:
            self.invalidate_ob(ob, current_price)
        for fvg in fvgs:
            self.is_fvg_filled(fvg, current_price)

        return SMCSnapshot(
            timeframe=timeframe,
            order_blocks=obs,
            fvgs=fvgs,
            bsl_zones=bsl,
            ssl_zones=ssl,
            last_sweep=sweep,
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
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    def _gap_valid(self, gap: float, reference_price: float, atr: float) -> bool:
        min_gap_atr = atr * self.min_gap_atr_mult if atr > 0 else 0.0
        min_gap_pct = reference_price * (self.min_gap_price_pct / 100.0)
        return gap >= max(min_gap_atr, min_gap_pct)

    @staticmethod
    def _score_gap(gap: float, atr: float) -> float:
        if atr <= 0:
            return 0.5
        return max(0.0, min(1.0, 0.35 + (gap / atr) / 4.0))

    def _find_equal_levels(
        self,
        series: pd.Series,
        zone_type: str,
        timeframe: str,
        index: pd.Index,
    ) -> list[LiquidityZone]:
        values = series.tolist()
        ts_values = [pd.Timestamp(ts) for ts in index]
        if not values:
            return []

        levels: list[LiquidityZone] = []
        used: set[int] = set()

        for i, value in enumerate(values):
            if i in used:
                continue

            touches = [i]
            tolerance = abs(value) * (self.equal_hl_tolerance_pct / 100.0)

            for j in range(i + 1, len(values)):
                if abs(values[j] - value) <= tolerance:
                    touches.append(j)

            if len(touches) >= self.equal_hl_min_touches:
                used.update(touches)
                avg_price = sum(values[k] for k in touches) / len(touches)
                levels.append(
                    LiquidityZone(
                        zone_type=zone_type,
                        timeframe=timeframe,
                        timestamp=ts_values[touches[-1]],
                        price=float(avg_price),
                        touches=len(touches),
                    )
                )

        return levels[-40:]
