"""SMC scalper strategy for 5m/1m execution windows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis.market_structure import MarketStructureEngine, TrendState
from src.analysis.smc_engine import PremiumDiscountZone, SMCEngine

from .base_strategy import BaseStrategy, SignalResult, StrategyConfig


@dataclass(slots=True)
class SMCScalperSettings:
    """Config settings for SMCScalperStrategy."""

    timeframe: str = "5m"
    lookback_bars: int = 180
    min_confluence: float = 0.65
    max_sl_pct: float = 0.8


class SMCScalperStrategy(BaseStrategy):
    """High-frequency SMC strategy constrained by context and confluence."""

    def __init__(
        self,
        strategy_id: str = "smc_scalper",
        symbol: str = "BTC/USDT",
        settings: SMCScalperSettings | None = None,
    ) -> None:
        self.settings = settings or SMCScalperSettings()
        self.structure_engine = MarketStructureEngine(
            swing_lookback_candles=12,
            swing_pivot_confirmation_candles=2,
            minimum_swing_size_atr_multiplier=0.2,
        )
        self.smc_engine = SMCEngine()

        super().__init__(
            StrategyConfig(
                strategy_id=strategy_id,
                version="12.0",
                symbol=symbol,
                min_confidence=self.settings.min_confluence,
                cooldown_seconds=15,
            )
        )

    def generate_signal(self, df: pd.DataFrame, i: int, current_price: float) -> SignalResult:
        if i < 40:
            return SignalResult(action="HOLD", reason="warmup: insufficient candles")

        window = df.iloc[max(0, i - self.settings.lookback_bars + 1) : i + 1].copy()
        snapshot = self.structure_engine.get_structure_snapshot(window, timeframe=self.settings.timeframe)
        smc = self.smc_engine.build_snapshot(window, timeframe=self.settings.timeframe)

        if len(window) < 20:
            return SignalResult(action="HOLD", reason="not enough bars for micro context")

        highs = [s.price for s in snapshot.swings if s.kind == "swing_high"]
        lows = [s.price for s in snapshot.swings if s.kind == "swing_low"]
        if not highs or not lows:
            return SignalResult(action="HOLD", reason="missing swing highs/lows")

        zone = self.smc_engine.get_zone(
            current_price,
            swing_high=highs[-1],
            swing_low=lows[-1],
        )

        bullish_obs = [o for o in smc.order_blocks if o.direction == "bullish" and not o.invalidated]
        bearish_obs = [o for o in smc.order_blocks if o.direction == "bearish" and not o.invalidated]

        in_bull_ob = any(self.smc_engine.is_price_inside_ob(current_price, ob) for ob in bullish_obs)
        in_bear_ob = any(self.smc_engine.is_price_inside_ob(current_price, ob) for ob in bearish_obs)

        momentum = self._micro_momentum(window)

        if snapshot.trend_state in {TrendState.BULLISH_TRENDING, TrendState.ACCUMULATION, TrendState.RANGING}:
            confluence = 0.0
            if zone in {PremiumDiscountZone.DISCOUNT, PremiumDiscountZone.EQUILIBRIUM}:
                confluence += 0.25
            if in_bull_ob:
                confluence += 0.30
            if momentum > 0:
                confluence += 0.20
            if smc.last_sweep and smc.last_sweep.zone_type == "SSL" and smc.last_sweep.confirmed:
                confluence += 0.25

            if confluence >= self.settings.min_confluence and in_bull_ob:
                stop_loss = current_price * (1.0 - self.settings.max_sl_pct / 100.0)
                take_profit = current_price + ((current_price - stop_loss) * 2.0)
                return SignalResult(
                    action="BUY",
                    reason="micro bullish scalp setup",
                    confidence=min(1.0, confluence),
                    size=1.0,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_strength=confluence,
                    metadata={
                        "zone": zone.value,
                        "momentum": momentum,
                        "trend": snapshot.trend_state.value,
                    },
                )

        if snapshot.trend_state in {TrendState.BEARISH_TRENDING, TrendState.DISTRIBUTION, TrendState.RANGING}:
            confluence = 0.0
            if zone in {PremiumDiscountZone.PREMIUM, PremiumDiscountZone.EQUILIBRIUM}:
                confluence += 0.25
            if in_bear_ob:
                confluence += 0.30
            if momentum < 0:
                confluence += 0.20
            if smc.last_sweep and smc.last_sweep.zone_type == "BSL" and smc.last_sweep.confirmed:
                confluence += 0.25

            if confluence >= self.settings.min_confluence and in_bear_ob:
                stop_loss = current_price * (1.0 + self.settings.max_sl_pct / 100.0)
                take_profit = current_price - ((stop_loss - current_price) * 2.0)
                return SignalResult(
                    action="SELL",
                    reason="micro bearish scalp setup",
                    confidence=min(1.0, confluence),
                    size=1.0,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_strength=confluence,
                    metadata={
                        "zone": zone.value,
                        "momentum": momentum,
                        "trend": snapshot.trend_state.value,
                    },
                )

        return SignalResult(action="HOLD", reason="no valid scalp setup")

    @staticmethod
    def _micro_momentum(frame: pd.DataFrame) -> float:
        close = frame["close"].astype(float)
        fast = close.ewm(span=5, adjust=False).mean().iloc[-1]
        slow = close.ewm(span=13, adjust=False).mean().iloc[-1]
        return float(fast - slow)
