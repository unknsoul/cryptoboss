"""Dynamic drawdown-based risk control."""

from __future__ import annotations


def dynamic_risk_multiplier(
    current_equity: float,
    peak_equity: float,
    thresholds: tuple[float, float, float] = (0.05, 0.10, 0.15),
) -> float:
    """Return risk multiplier based on drawdown levels."""
    if peak_equity <= 0:
        return 0.0

    drawdown_pct = (peak_equity - current_equity) / peak_equity

    if drawdown_pct > thresholds[2]:
        return 0.0
    if drawdown_pct > thresholds[1]:
        return 0.25
    if drawdown_pct > thresholds[0]:
        return 0.5
    return 1.0
