from datetime import datetime, timezone

from src.analysis.bias_engine import BiasEngine
from src.analysis.market_context import MarketContext, RegimeEnum


def _context(
    timeframe: str,
    regime: RegimeEnum,
    trend_direction: str,
    structure_bias: str,
    last_price: float,
    ema200: float,
    vwap: float,
    support: float,
    resistance: float,
    adx_norm: float,
) -> MarketContext:
    return MarketContext(
        symbol="BTC/USDT",
        timeframe=timeframe,
        timestamp=datetime.now(timezone.utc),
        regime=regime,
        regime_confidence=0.8,
        trend_direction=trend_direction,
        trend_strength=adx_norm,
        volatility_level="NORMAL",
        volume_character="INCREASING",
        key_levels={
            "nearest_support": support,
            "nearest_resistance": resistance,
            "active_order_blocks": [],
            "active_fvg": [],
            "vwap": vwap,
            "daily_open": ema200,
            "ema200": ema200,
            "prev_day_high": last_price - 20,
            "prev_day_low": last_price - 140,
            "last_price": last_price,
        },
        structure_bias=structure_bias,
        atr_value=45.0,
    )


def test_bias_long_in_uptrend_above_ema200():
    engine = BiasEngine()
    contexts = {
        "4h": _context(
            timeframe="4h",
            regime=RegimeEnum.STRONG_UPTREND,
            trend_direction="UP",
            structure_bias="BULLISH",
            last_price=10420,
            ema200=9900,
            vwap=10200,
            support=10100,
            resistance=10850,
            adx_norm=0.82,
        ),
        "1d": _context(
            timeframe="1d",
            regime=RegimeEnum.WEAK_UPTREND,
            trend_direction="UP",
            structure_bias="BULLISH",
            last_price=10420,
            ema200=9800,
            vwap=10050,
            support=9900,
            resistance=11200,
            adx_norm=0.74,
        ),
    }

    bias = engine.compute_bias(contexts)

    assert bias.primary_bias == "LONG"
    assert 0.0 <= bias.bias_strength <= 1.0


def test_bias_blocks_short_in_strong_uptrend():
    engine = BiasEngine()
    contexts = {
        "4h": _context("4h", RegimeEnum.STRONG_UPTREND, "UP", "BULLISH", 10420, 9900, 10200, 10100, 10850, 0.82),
        "1d": _context("1d", RegimeEnum.WEAK_UPTREND, "UP", "BULLISH", 10420, 9800, 10050, 9900, 11200, 0.74),
    }

    bias = engine.compute_bias(contexts)

    assert engine.is_trade_direction_permitted(bias, "SHORT") is False
    assert engine.is_trade_direction_permitted(bias, "LONG") is True


def test_bias_neutral_in_range_adc_below_20():
    engine = BiasEngine()
    contexts = {
        "4h": _context(
            timeframe="4h",
            regime=RegimeEnum.RANGE_LOW_VOLUME,
            trend_direction="NEUTRAL",
            structure_bias="NEUTRAL",
            last_price=10000,
            ema200=10010,
            vwap=10000,
            support=9950,
            resistance=10050,
            adx_norm=0.12,
        ),
        "1d": _context(
            timeframe="1d",
            regime=RegimeEnum.RANGE_HIGH_VOLUME,
            trend_direction="NEUTRAL",
            structure_bias="NEUTRAL",
            last_price=10000,
            ema200=10020,
            vwap=10005,
            support=9930,
            resistance=10070,
            adx_norm=0.15,
        ),
    }

    bias = engine.compute_bias(contexts)

    assert bias.primary_bias == "NEUTRAL"
