import numpy as np
import pandas as pd

from src.v3 import BacktestingEngine, IntradayScalperV3System, StrategyBuilderService


def _sample_ohlc(n: int = 1800) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")

    base = 100 + np.cumsum(np.random.randn(n) * 0.05)
    open_price = pd.Series(base).shift(1).fillna(base[0]).values
    close_price = base

    high = np.maximum(open_price, close_price) + np.random.rand(n) * 0.05
    low = np.minimum(open_price, close_price) - np.random.rand(n) * 0.05

    frame = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close_price,
            "tick_volume": np.random.randint(100, 1000, n),
            "spread": np.random.rand(n) * 0.03,
        },
        index=index,
    )
    return frame


def test_v3_cycle_smoke():
    system = IntradayScalperV3System()
    frame = _sample_ohlc()

    result = system.run_cycle(
        symbol="BTCUSDT",
        frames_by_timeframe={"1m": frame},
        account_state={"equity": 10000.0, "balance": 10000.0, "last_price": float(frame["close"].iloc[-1])},
    )

    assert result["version"] == "3.0_intraday_scalper"
    assert result["architecture"] == "modular_microservices"
    assert len(result["modules"]) == 10
    assert result["signal"]["action"] in {"BUY", "SELL", "HOLD"}


def test_v3_strategy_builder_custom_strategy():
    service = StrategyBuilderService()
    artifact = service.create_custom_strategy(
        name="Test V3 Strategy",
        indicators=[
            {"indicator": "ema", "operator": "cross_above", "threshold": 0, "params": {"fast": 9, "slow": 21}},
            {"indicator": "rsi", "operator": ">", "threshold": 50, "params": {"period": 14}},
        ],
        smc_conditions=["order_block", "fvg", "bos"],
        risk_rules={"max_risk_pct": 0.01, "max_positions": 2},
        multi_timeframe_rules={"htf": "15m", "ltf": "1m"},
    )

    assert artifact.name == "Test V3 Strategy"
    assert "order_block" in artifact.smc_conditions
    assert artifact.risk_rules["max_risk_pct"] == 0.01


def test_v3_backtesting_engine_smoke():
    engine = BacktestingEngine()
    frame = _sample_ohlc(3000)

    def simple_signal(df_slice: pd.DataFrame):
        if len(df_slice) < 80:
            return {"action": "HOLD"}

        ema_fast = df_slice["close"].ewm(span=9, adjust=False).mean().iloc[-1]
        ema_slow = df_slice["close"].ewm(span=21, adjust=False).mean().iloc[-1]
        price = float(df_slice["close"].iloc[-1])

        if ema_fast > ema_slow * 1.001:
            return {
                "action": "BUY",
                "entry": price,
                "sl": price * 0.997,
                "tp1": price * 1.004,
                "tp2": price * 1.008,
            }

        if ema_fast < ema_slow * 0.999:
            return {
                "action": "SELL",
                "entry": price,
                "sl": price * 1.003,
                "tp1": price * 0.996,
                "tp2": price * 0.992,
            }

        return {"action": "HOLD"}

    result = engine.run(frame, simple_signal, strategy_name="Smoke", strategy_id="SMOKE")

    assert "summary" in result
    assert "metrics" in result
    assert set(result["summary"].keys()) >= {"win_rate", "profit_factor", "drawdown", "sharpe_ratio"}
