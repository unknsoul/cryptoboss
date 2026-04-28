import pandas as pd

from src.backtest.engine import BacktestExecutionConfig, RealBacktestEngine
from src.risk.controls import RiskConfig


class BuyHoldOnceStrategy:
    def generate_signal(self, df, i, price):
        if i == 1:
            return {"action": "BUY", "stop_loss": price * 0.99}
        if i == len(df) - 1:
            return {"action": "SELL", "reason": "final_bar"}
        return {"action": "HOLD"}


def _sample_df(rows: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    close = [100 + (i * 0.2) for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": [c + 0.5 for c in close],
            "low": [c - 0.5 for c in close],
            "close": close,
            "volume": [1000 + i for i in range(rows)],
        }
    )


def test_real_backtest_accounts_for_costs_and_drawdown():
    df = _sample_df()
    engine = RealBacktestEngine(
        initial_capital=10000,
        execution_config=BacktestExecutionConfig(
            commission_pct=0.001,
            spread_bps=4.0,
            slippage_bps=8.0,
        ),
    )
    result = engine.run(df, BuyHoldOnceStrategy())

    assert result.num_trades >= 1
    assert result.total_fees > 0
    assert result.total_spread_cost > 0
    assert result.total_slippage > 0
    assert len(result.equity_curve) > 2
    assert result.max_drawdown_pct >= 0


def test_real_backtest_halts_when_daily_loss_breached():
    df = _sample_df(rows=60)
    df.loc[:, "close"] = [100 - (i * 1.5) for i in range(len(df))]
    df.loc[:, "open"] = df["close"]
    df.loc[:, "high"] = df["close"] + 0.2
    df.loc[:, "low"] = df["close"] - 0.2

    engine = RealBacktestEngine(
        initial_capital=10000,
        risk_config=RiskConfig(
            max_risk_per_trade_pct=0.02,
            max_daily_loss_pct=0.001,
            stop_trading_drawdown_pct=0.20,
        ),
    )
    result = engine.run(df, BuyHoldOnceStrategy())

    assert result.halted is True
