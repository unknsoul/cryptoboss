"""Value at Risk estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def value_at_risk(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Compute VaR from return series."""
    series = pd.Series(returns).dropna()
    if series.empty:
        return 0.0

    if method == "historical":
        return float(np.percentile(series, (1.0 - confidence) * 100))

    if method == "parametric":
        mean = series.mean()
        std = series.std(ddof=1)
        z = np.percentile(np.random.normal(size=100000), (1.0 - confidence) * 100)
        return float(mean + z * std)

    raise ValueError("Unknown VaR method: %s" % method)


def expected_shortfall(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Compute Expected Shortfall (CVaR)."""
    series = pd.Series(returns).dropna()
    if series.empty:
        return 0.0

    var = value_at_risk(series, confidence=confidence, method="historical")
    tail = series[series <= var]
    return float(tail.mean()) if not tail.empty else float(var)
