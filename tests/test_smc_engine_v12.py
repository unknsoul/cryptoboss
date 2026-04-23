from datetime import datetime, timedelta

import pandas as pd

from src.analysis.smc_engine import PremiumDiscountZone, SMCEngine


def _smc_frame() -> pd.DataFrame:
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(40)]

    # Crafted to produce impulses + potential FVGs.
    close = [100 + (i * 0.2) for i in range(40)]
    close[10] = 102.5
    close[11] = 105.2
    close[12] = 106.8
    close[20] = 108.2
    close[21] = 105.4
    close[22] = 103.5

    open_ = [c + (0.35 if i % 7 == 0 else -0.3) for i, c in enumerate(close)]

    # Force explicit bearish -> bullish impulse pair for OB detection.
    open_[10] = close[10] + 0.9
    open_[11] = close[11] - 1.4

    high = [max(o, c) + 0.5 for o, c in zip(open_, close)]
    low = [min(o, c) - 0.5 for o, c in zip(open_, close)]

    # Force an explicit bullish FVG around candle 11.
    high[10] = 103.0
    low[12] = 104.2

    # Force an equal-high cluster for BSL.
    high[30] = 110.0
    high[32] = 110.01

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        },
        index=pd.to_datetime(timestamps),
    )


def test_detect_obs_and_fvgs() -> None:
    frame = _smc_frame()
    engine = SMCEngine(min_impulse_atr_mult=0.8, min_gap_atr_mult=0.05, min_gap_price_pct=0.01)

    obs = engine.detect_obs(frame, timeframe="15m")
    fvgs = engine.detect_fvgs(frame, timeframe="15m")

    assert isinstance(obs, list)
    assert isinstance(fvgs, list)
    assert any(ob.direction in {"bullish", "bearish"} for ob in obs)
    assert any(fvg.direction in {"bullish", "bearish"} for fvg in fvgs)


def test_liquidity_and_zone_classification() -> None:
    frame = _smc_frame()
    engine = SMCEngine(equal_hl_tolerance_pct=0.2)

    bsl = engine.find_bsl_zones(frame, timeframe="15m")
    ssl = engine.find_ssl_zones(frame, timeframe="15m")

    assert isinstance(bsl, list)
    assert isinstance(ssl, list)

    zone = engine.get_zone(price=107.0, swing_high=112.0, swing_low=100.0)
    assert zone in set(PremiumDiscountZone)

    fib_level = engine.get_fib_level(price=107.0, swing_high=112.0, swing_low=100.0)
    assert 0.0 <= fib_level <= 1.0


def test_build_snapshot_contains_expected_keys() -> None:
    frame = _smc_frame()
    engine = SMCEngine()

    snapshot = engine.build_snapshot(frame, timeframe="15m")
    as_dict = snapshot.to_dict()

    assert as_dict["timeframe"] == "15m"
    assert "order_blocks" in as_dict
    assert "fvgs" in as_dict
    assert "bsl_zones" in as_dict
    assert "ssl_zones" in as_dict
