"""Emergency stop and kill-switch helpers."""

from __future__ import annotations

from dataclasses import dataclass

from src.risk.advanced_sizing import KillSwitch
from src.risk.drawdown_control import dynamic_risk_multiplier


@dataclass
class EmergencyStopState:
    """Current emergency stop state."""

    should_halt: bool
    reason: str | None
    risk_multiplier: float


class EmergencyStop:
    """Unified emergency stop control."""

    def __init__(self, initial_equity: float) -> None:
        self.kill_switch = KillSwitch(initial_equity)
        self.peak_equity = initial_equity

    def update(self, current_equity: float, last_trade_pnl: float | None = None) -> EmergencyStopState:
        result = self.kill_switch.check_halt_conditions(current_equity, last_trade_pnl)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        multiplier = dynamic_risk_multiplier(current_equity, self.peak_equity)
        return EmergencyStopState(
            should_halt=bool(result.get("should_halt")),
            reason=result.get("reason"),
            risk_multiplier=multiplier,
        )
