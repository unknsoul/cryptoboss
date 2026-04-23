"""SMC trend-follow strategy (HTF bias and structure aligned entries)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis.market_structure import MarketStructureEngine, TrendState
from src.analysis.smc_engine import PremiumDiscountZone, SMCEngine

from .base_strategy import BaseStrategy, SignalResult, StrategyConfig


@dataclass(slots=True)
class SMCTrendFollowSettings:
    """Config settings for SMC trend-follow strategy."""

    timeframe: str = "15m"
    lookback_bars: int = 240
    min_confluence: float = 0.65
    max_sl_pct: float = 2.0


class SMCTrendFollowStrategy(BaseStrategy):
    """Trend-following SMC strategy for OB/FVG/liquidity confluence."""

    def __init__(
        self,
        strategy_id: str = "smc_trend_follow",
        symbol: str = "BTC/USDT",
        settings: SMCTrendFollowSettings | None = None,
    ) -> None:
        self.settings = settings or SMCTrendFollowSettings()
        self.structure_engine = MarketStructureEngine()
        self.smc_engine = SMCEngine()

        super().__init__(
            StrategyConfig(
                strategy_id=strategy_id,
                version="12.0",
                symbol=symbol,
                min_confidence=self.settings.min_confluence,
                cooldown_seconds=30,
            )
        )

    def generate_signal(self, df: pd.DataFrame, i: int, current_price: float) -> SignalResult:
        if i < 60:
            return SignalResult(action="HOLD", reason="warmup: insufficient candles")

        window = df.iloc[max(0, i - self.settings.lookback_bars + 1) : i + 1].copy()

        snapshot = self.structure_engine.get_structure_snapshot(window, timeframe=self.settings.timeframe)
        smc = self.smc_engine.build_snapshot(window, timeframe=self.settings.timeframe)

        highs = [s.price for s in snapshot.swings if s.kind == "swing_high"]
        lows = [s.price for s in snapshot.swings if s.kind == "swing_low"]
        if not highs or not lows:
            return SignalResult(action="HOLD", reason="no valid swing context")

        swing_high = highs[-1]
        swing_low = lows[-1]
        zone = self.smc_engine.get_zone(current_price, swing_high=swing_high, swing_low=swing_low)

        bullish_obs = [o for o in smc.order_blocks if o.direction == "bullish" and not o.invalidated]
        bearish_obs = [o for o in smc.order_blocks if o.direction == "bearish" and not o.invalidated]

        bullish_in_zone = [o for o in bullish_obs if self.smc_engine.is_price_inside_ob(current_price, o)]
        bearish_in_zone = [o for o in bearish_obs if self.smc_engine.is_price_inside_ob(current_price, o)]

        nearest_bsl, nearest_ssl = self.smc_engine.get_nearest_liquidity(
            current_price,
            smc.bsl_zones,
            smc.ssl_zones,
        )

        trend = snapshot.trend_state

        if trend in {TrendState.BULLISH_TRENDING, TrendState.ACCUMULATION}:
            confluence = 0.0
            if zone in {PremiumDiscountZone.DISCOUNT, PremiumDiscountZone.EQUILIBRIUM}:
                confluence += 0.35
            if bullish_in_zone:
                confluence += 0.30
            if nearest_ssl and nearest_ssl.swept:
                confluence += 0.20
            if smc.last_sweep and smc.last_sweep.zone_type == "SSL" and smc.last_sweep.confirmed:
                confluence += 0.15

            if confluence >= self.settings.min_confluence and bullish_in_zone:
                best_ob = max(bullish_in_zone, key=lambda ob: self.smc_engine.score_ob(ob, current_price))
                stop_loss = min(best_ob.low, current_price * (1.0 - (self.settings.max_sl_pct / 100.0)))
                risk = max(1e-12, current_price - stop_loss)
                take_profit = current_price + risk * 2.5

                return SignalResult(
                    action="BUY",
                    reason="bullish trend + discount OB confluence",
                    confidence=min(1.0, confluence),
                    size=1.0,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_strength=confluence,
                    metadata={
                        "trend": trend.value,
                        "zone": zone.value,
                        "nearest_liquidity": nearest_bsl.price if nearest_bsl else None,
                    },
                )

        if trend in {TrendState.BEARISH_TRENDING, TrendState.DISTRIBUTION}:
            confluence = 0.0
            if zone in {PremiumDiscountZone.PREMIUM, PremiumDiscountZone.EQUILIBRIUM}:
                confluence += 0.35
            if bearish_in_zone:
                confluence += 0.30
            if nearest_bsl and nearest_bsl.swept:
                confluence += 0.20
            if smc.last_sweep and smc.last_sweep.zone_type == "BSL" and smc.last_sweep.confirmed:
                confluence += 0.15

            if confluence >= self.settings.min_confluence and bearish_in_zone:
                best_ob = max(bearish_in_zone, key=lambda ob: self.smc_engine.score_ob(ob, current_price))
                stop_loss = max(best_ob.high, current_price * (1.0 + (self.settings.max_sl_pct / 100.0)))
                risk = max(1e-12, stop_loss - current_price)
                take_profit = current_price - risk * 2.5

                return SignalResult(
                    action="SELL",
                    reason="bearish trend + premium OB confluence",
                    confidence=min(1.0, confluence),
                    size=1.0,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_strength=confluence,
                    metadata={
                        "trend": trend.value,
                        "zone": zone.value,
                        "nearest_liquidity": nearest_ssl.price if nearest_ssl else None,
                    },
                )

        return SignalResult(action="HOLD", reason="no valid trend-follow setup")
