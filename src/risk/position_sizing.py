"""Position sizing utilities."""

from __future__ import annotations

from dataclasses import dataclass


def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """Compute fractional Kelly position size."""
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    kelly = (win_rate / avg_loss) - ((1.0 - win_rate) / avg_win)
    return max(0.0, kelly * fraction)


@dataclass
class FractionalKellySizer:
    """Kelly-based sizing wrapper."""

    fraction: float = 0.25

    def size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        return kelly_position_size(win_rate, avg_win, avg_loss, self.fraction)


def fixed_fractional_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float,
) -> float:
    """Position size from fixed-fractional risk and stop distance."""
    if equity <= 0 or entry_price <= 0 or risk_pct <= 0:
        return 0.0
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0.0
    risk_amount = equity * risk_pct
    return max(0.0, risk_amount / stop_distance)
