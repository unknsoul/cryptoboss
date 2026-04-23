from datetime import datetime, timedelta

import pandas as pd

from src.strategies.range_scalp import RangeScalpStrategy
from src.strategies.smc_scalper import SMCScalperStrategy
from src.strategies.smc_trend_follow import SMCTrendFollowStrategy


def _strategy_frame(rows: int = 320) -> pd.DataFrame:
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(rows)]

    base = [100 + i * 0.05 for i in range(rows)]
    close = [v + (0.8 if i % 12 < 6 else -0.6) for i, v in enumerate(base)]
    open_ = [c - 0.2 for c in close]
    high = [max(o, c) + 0.45 for o, c in zip(open_, close)]
    low = [min(o, c) - 0.45 for o, c in zip(open_, close)]

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        },
        index=pd.to_datetime(timestamps),
    )


def test_smc_trend_follow_signal_contract() -> None:
    frame = _strategy_frame()
    strategy = SMCTrendFollowStrategy(symbol="BTC/USDT")

    signal = strategy.generate_signal(frame, len(frame) - 1, float(frame["close"].iloc[-1]))
    assert signal.action in {"BUY", "SELL", "HOLD"}
    assert isinstance(signal.reason, str)


def test_smc_scalper_signal_contract() -> None:
    frame = _strategy_frame()
    strategy = SMCScalperStrategy(symbol="ETH/USDT")

    signal = strategy.generate_signal(frame, len(frame) - 1, float(frame["close"].iloc[-1]))
    assert signal.action in {"BUY", "SELL", "HOLD"}
    assert isinstance(signal.reason, str)


def test_range_scalp_signal_contract() -> None:
    frame = _strategy_frame()
    strategy = RangeScalpStrategy(symbol="SOL/USDT")

    signal = strategy.generate_signal(frame, len(frame) - 1, float(frame["close"].iloc[-1]))
    assert signal.action in {"BUY", "SELL", "HOLD"}
    assert isinstance(signal.reason, str)
