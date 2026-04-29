"""Portfolio risk controls shared by backtest and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(slots=True)
class RiskConfig:
    """Hard risk constraints."""

    max_risk_per_trade_pct: float = 0.02
    max_daily_loss_pct: float = 0.05
    stop_trading_drawdown_pct: float = 0.15
    min_risk_per_trade_pct: float = 0.005


@dataclass(slots=True)
class RiskCheckResult:
    """Risk gate decision for one proposed trade."""

    allowed: bool
    reason: Optional[str] = None
    adjusted_size: Optional[float] = None


@dataclass(slots=True)
class RiskController:
    """Tracks session risk and validates order proposals."""

    initial_equity: float
    config: RiskConfig = field(default_factory=RiskConfig)

    current_day: Optional[str] = None
    daily_start_equity: float = 0.0
    daily_realized_pnl: float = 0.0
    peak_equity: float = 0.0
    trading_enabled: bool = True
    halt_reason: Optional[str] = None

    def __post_init__(self) -> None:
        self.daily_start_equity = float(self.initial_equity)
        self.peak_equity = float(self.initial_equity)

    def update_after_trade(self, realized_pnl: float, equity: float, timestamp: datetime) -> None:
        """Update state after a trade close/fill."""
        self._roll_day(timestamp, equity)
        self.daily_realized_pnl += float(realized_pnl)
        self.peak_equity = max(self.peak_equity, float(equity))
        self._refresh_halts(float(equity))

    def validate_trade(
        self,
        *,
        equity: float,
        proposed_risk_amount: float,
        timestamp: datetime,
    ) -> RiskCheckResult:
        """Check if a trade can be opened under hard limits."""
        self._roll_day(timestamp, equity)
        self._refresh_halts(float(equity))

        if not self.trading_enabled:
            return RiskCheckResult(allowed=False, reason=self.halt_reason)

        max_risk_amount = float(equity) * self.config.max_risk_per_trade_pct
        if proposed_risk_amount <= max_risk_amount:
            return RiskCheckResult(allowed=True)

        if max_risk_amount <= 0:
            return RiskCheckResult(allowed=False, reason="equity too low for risk allocation")

        adjusted_factor = max_risk_amount / max(proposed_risk_amount, 1e-9)
        return RiskCheckResult(
            allowed=True,
            reason="position size reduced by risk cap",
            adjusted_size=adjusted_factor,
        )

    def dynamic_risk_pct(self, equity: float) -> float:
        """Risk budget decays as drawdown grows."""
        if self.peak_equity <= 0:
            return self.config.min_risk_per_trade_pct
        drawdown = max(0.0, (self.peak_equity - equity) / self.peak_equity)
        if drawdown >= self.config.stop_trading_drawdown_pct:
            return 0.0
        ratio = drawdown / max(self.config.stop_trading_drawdown_pct, 1e-9)
        risk_span = self.config.max_risk_per_trade_pct - self.config.min_risk_per_trade_pct
        return max(self.config.min_risk_per_trade_pct, self.config.max_risk_per_trade_pct - (risk_span * ratio))

    def _roll_day(self, timestamp: datetime, equity: float) -> None:
        day_key = timestamp.date().isoformat()
        if self.current_day is None:
            self.current_day = day_key
            self.daily_start_equity = float(equity)
            self.daily_realized_pnl = 0.0
            return
        if day_key != self.current_day:
            self.current_day = day_key
            self.daily_start_equity = float(equity)
            self.daily_realized_pnl = 0.0
            if self.halt_reason == "max daily loss breached":
                self.trading_enabled = True
                self.halt_reason = None

    def _refresh_halts(self, equity: float) -> None:
        if self.daily_start_equity > 0:
            daily_loss_pct = max(0.0, -self.daily_realized_pnl / self.daily_start_equity)
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                self.trading_enabled = False
                self.halt_reason = "max daily loss breached"
                return

        if self.peak_equity > 0:
            drawdown = max(0.0, (self.peak_equity - equity) / self.peak_equity)
            if drawdown >= self.config.stop_trading_drawdown_pct:
                self.trading_enabled = False
                self.halt_reason = "stop trading drawdown breached"
