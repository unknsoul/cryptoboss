"""Backtest metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(equity_curve: pd.Series | np.ndarray, periods_per_year: int = 365) -> float:
    """Compute annualized Sharpe ratio."""
    series = pd.Series(equity_curve).astype(float)
    returns = series.diff().dropna()
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def sortino_ratio(equity_curve: pd.Series | np.ndarray, periods_per_year: int = 365) -> float:
    """Compute annualized Sortino ratio."""
    series = pd.Series(equity_curve).astype(float)
    returns = series.diff().dropna()
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0
    return float((returns.mean() / downside.std()) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series | np.ndarray) -> float:
    """Compute maximum drawdown as a fraction."""
    series = pd.Series(equity_curve).astype(float)
    peak = series.cummax()
    drawdown = (peak - series) / peak.replace(0, np.nan)
    return float(drawdown.max(skipna=True))


def calmar_ratio(equity_curve: pd.Series | np.ndarray, periods_per_year: int = 365) -> float:
    """Compute Calmar ratio."""
    series = pd.Series(equity_curve).astype(float)
    total_return = series.iloc[-1] - series.iloc[0]
    avg_return = total_return / max(len(series) - 1, 1)
    annual_return = avg_return * periods_per_year
    drawdown = max_drawdown(series)
    if drawdown == 0:
        return 0.0
    return float(annual_return / drawdown)


def profit_factor(trade_pnls: pd.Series | np.ndarray) -> float:
    """Compute profit factor from trade PnL series."""
    series = pd.Series(trade_pnls).astype(float)
    gross_profit = series[series > 0].sum()
    gross_loss = series[series < 0].abs().sum()
    if gross_loss == 0:
        return float("inf")
    return float(gross_profit / gross_loss)
