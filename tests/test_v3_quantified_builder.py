import numpy as np
import pandas as pd

from src.v3.config import SignalEngineConfig
from src.v3.quantified_smc import detect_bos, detect_fvg, detect_order_block
from src.v3.signal_engine import SignalEngine
from src.v3.strategy_builder_engine import StrategyBuilderEngine, StrategyValidationEngine


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")


def test_detect_bos_quantified_break_with_volume_spike():
    n = 50
    idx = _index(n)
    close = np.linspace(100.0, 101.0, n)
    high = close + 0.05
    low = close - 0.05
    open_ = close - 0.01
    vol = np.full(n, 100.0)

    # Force final candle to break prior swing high with strong volume.
    close[-1] = high[:-1].max() + 0.02
    high[-1] = close[-1] + 0.03
    vol[-1] = 400.0

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": vol,
        },
        index=idx,
    )

    scored = detect_bos(frame, lookback=20, break_threshold=0.0001, volume_spike_mult=1.5)
    assert int(scored["bos"].iloc[-1]) == 1


def test_detect_fvg_and_order_block_quantified():
    # Crafted candles for bullish FVG at i=2 and OB on candle before impulse.
    frame = pd.DataFrame(
        {
            "open": [1.00, 0.98, 1.04, 1.21, 1.23, 1.25, 1.24, 1.23, 1.22, 1.21, 1.20, 1.19],
            "high": [1.02, 1.00, 1.26, 1.24, 1.26, 1.27, 1.26, 1.25, 1.24, 1.23, 1.22, 1.21],
            "low": [0.99, 0.97, 1.04, 1.19, 1.21, 1.22, 1.21, 1.20, 1.19, 1.18, 1.17, 1.16],
            "close": [1.01, 0.97, 1.24, 1.23, 1.25, 1.26, 1.25, 1.24, 1.23, 1.22, 1.21, 1.20],
            "tick_volume": [80, 85, 500, 120, 110, 105, 100, 95, 90, 88, 85, 82],
        },
        index=_index(12),
    )

    with_fvg = detect_fvg(frame, min_gap=0.01)
    with_ob = detect_order_block(with_fvg, impulse_multiplier=1.5, lookback=5, volume_multiplier=1.1)

    assert int(with_fvg["fvg"].iloc[2]) == 1
    assert int((with_ob["ob"] != 0).sum()) >= 1


def test_signal_engine_force_fallback_prevents_hold_only():
    # Smooth uptrend with one large impulse for positive momentum direction.
    prices = np.linspace(100, 101.5, 80)
    frame = pd.DataFrame(
        {
            "open": prices - 0.02,
            "high": prices + 0.05,
            "low": prices - 0.05,
            "close": prices,
        },
        index=_index(80),
    )

    market_structure = {
        "bos": {"confirmed": True, "htf": [{"timestamp": frame.index[-1], "direction": "bullish"}], "ltf": []},
        "choch": {"confirmed": False, "htf": [], "ltf": []},
        "trend_detection": {"trend_alignment": True},
        "latest_structure_direction": "bullish",
    }
    smart_money = {
        "ob_signal": 1,
        "fvg_signal": 0,
        "liquidity_sweep": False,
    }

    cfg = SignalEngineConfig(
        buy_threshold=12,
        sell_threshold=12,
        min_trade_score=20,
        force_trade_if_no_signal=True,
    )
    engine = SignalEngine(cfg)
    signal = engine.evaluate(market_structure, smart_money, frame, win_probability=0.8)

    assert signal.action == "BUY"


def test_strategy_builder_engine_rule_composer_and_robustness_score():
    engine = StrategyBuilderEngine()
    strategy = engine.build(
        name="rule_test",
        rules=["bos", "ob_touch", "momentum"],
        weights={"bos": 2.0, "ob_touch": 3.0, "momentum": 1.0},
        buy_threshold=4,
        sell_threshold=4,
    )

    frame = pd.DataFrame(
        {
            "bos": [1, -1, 0],
            "ob_touch": [1, -1, 0],
            "momentum": [1, -1, 0],
        }
    )
    scored = engine.evaluate_dataframe(frame, strategy)

    assert "strategy_score" in scored.columns
    assert scored["signal"].iloc[0] == "BUY"
    assert scored["signal"].iloc[1] == "SELL"

    wf = {
        "folds": [
            {"oos_profit_factor": 1.3, "oos_sharpe": 1.1},
            {"oos_profit_factor": 1.2, "oos_sharpe": 0.9},
            {"oos_profit_factor": 1.4, "oos_sharpe": 1.0},
        ]
    }
    robustness = StrategyValidationEngine.robustness_score(wf)
    assert robustness > 0
