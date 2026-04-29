"""Monte Carlo backtest utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.backtest.metrics import max_drawdown, sharpe_ratio


def monte_carlo_backtest(trade_log: pd.DataFrame, n_simulations: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Shuffle trade outcomes to stress-test performance."""
    if trade_log.empty or "pnl" not in trade_log.columns:
        raise ValueError("trade_log must include a 'pnl' column")

    rng = np.random.default_rng(seed)
    results = []

    for _ in range(n_simulations):
        shuffled = trade_log.sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 1e9)))
        equity_curve = shuffled["pnl"].cumsum()
        results.append(
            {
                "final_return": float(equity_curve.iloc[-1]),
                "max_drawdown": float(max_drawdown(equity_curve)),
                "sharpe": float(sharpe_ratio(equity_curve)),
            }
        )

    return pd.DataFrame(results)
