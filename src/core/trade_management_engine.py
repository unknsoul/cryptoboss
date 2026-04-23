"""Trade management engine for breakeven, partial exits, and trailing behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PositionSide(str, Enum):
    """Position direction."""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(slots=True)
class TPLevel:
    """Take-profit level configuration."""

    tp_id: str
    price: float
    close_pct: float
    hit: bool = False


@dataclass(slots=True)
class ManagedTrade:
    """Managed trade state model."""

    trade_id: str
    symbol: str
    side: PositionSide
    entry_price: float
    stop_loss: float
    size: float
    opened_at: datetime
    tp_levels: list[TPLevel] = field(default_factory=list)
    remaining_size_pct: float = 100.0
    trailing_active: bool = False
    moved_to_breakeven: bool = False
    closed: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["side"] = self.side.value
        data["opened_at"] = self.opened_at.isoformat()
        return data


@dataclass(slots=True)
class TradeManagementDecision:
    """Result of one management evaluation step."""

    actions: list[str]
    close_pct: float = 0.0
    new_stop_loss: float | None = None
    close_all: bool = False
    reason: str = ""


class TradeManagementEngine:
    """Executes trade lifecycle rules after entry."""

    def __init__(
        self,
        move_sl_to_entry_plus_fees_pct: float = 0.01,
        atr_trail_multiplier: float = 1.5,
        min_lock_in_profit_pct: float = 0.5,
    ) -> None:
        self.move_sl_to_entry_plus_fees_pct = move_sl_to_entry_plus_fees_pct
        self.atr_trail_multiplier = atr_trail_multiplier
        self.min_lock_in_profit_pct = min_lock_in_profit_pct

    def evaluate(
        self,
        trade: ManagedTrade,
        current_price: float,
        *,
        atr_value: float | None = None,
        last_swing_low: float | None = None,
        last_swing_high: float | None = None,
        structure_invalidated: bool = False,
        opposite_signal: bool = False,
    ) -> TradeManagementDecision:
        """Evaluate one market tick against management rules."""
        if trade.closed:
            return TradeManagementDecision(actions=["already_closed"], reason="trade already closed")

        actions: list[str] = []
        close_pct = 0.0
        new_sl: float | None = None

        # Hard protection: close immediately on invalid structure.
        if structure_invalidated:
            trade.closed = True
            trade.remaining_size_pct = 0.0
            return TradeManagementDecision(
                actions=["close_immediately"],
                close_pct=100.0,
                close_all=True,
                reason="structure invalidated",
            )

        if opposite_signal:
            actions.append("evaluate_close_on_opposite_signal")

        # Stop-loss check.
        if self._stop_hit(trade, current_price):
            trade.closed = True
            trade.remaining_size_pct = 0.0
            return TradeManagementDecision(
                actions=["stop_loss_hit", "close_remainder"],
                close_pct=100.0,
                close_all=True,
                reason="stop loss reached",
            )

        # Process TP hits in order.
        for idx, tp in enumerate(trade.tp_levels):
            if tp.hit:
                continue
            if not self._tp_hit(trade, current_price, tp.price):
                continue

            tp.hit = True
            close_pct += tp.close_pct
            trade.remaining_size_pct = max(0.0, trade.remaining_size_pct - tp.close_pct)
            actions.append(f"{tp.tp_id}_hit")
            actions.append(f"partial_close_{tp.close_pct:.0f}pct")

            if idx == 0 and not trade.moved_to_breakeven:
                new_sl = self._breakeven_sl(trade)
                trade.stop_loss = new_sl
                trade.moved_to_breakeven = True
                actions.append("move_sl_to_breakeven")

            if idx == 1 and not trade.trailing_active:
                trade.trailing_active = True
                actions.append("activate_trailing_stop")

        # Trailing stop behavior once active.
        if trade.trailing_active and not trade.closed:
            trail_sl = self._compute_trailing_sl(
                trade=trade,
                current_price=current_price,
                atr_value=atr_value,
                last_swing_low=last_swing_low,
                last_swing_high=last_swing_high,
            )
            if trail_sl is not None and self._trail_improves(trade, trail_sl):
                trade.stop_loss = trail_sl
                new_sl = trail_sl
                actions.append("update_trailing_stop")

        # Close any tiny remainder when all TPs were hit.
        if trade.remaining_size_pct <= 0.0:
            trade.closed = True
            actions.append("close_remainder")
            return TradeManagementDecision(
                actions=actions,
                close_pct=max(close_pct, 100.0),
                new_stop_loss=new_sl,
                close_all=True,
                reason="all targets completed",
            )

        if not actions:
            actions.append("hold")

        return TradeManagementDecision(
            actions=actions,
            close_pct=min(close_pct, 100.0),
            new_stop_loss=new_sl,
            close_all=False,
            reason="management update",
        )

    def _tp_hit(self, trade: ManagedTrade, current_price: float, tp_price: float) -> bool:
        if trade.side == PositionSide.LONG:
            return current_price >= tp_price
        return current_price <= tp_price

    def _stop_hit(self, trade: ManagedTrade, current_price: float) -> bool:
        if trade.side == PositionSide.LONG:
            return current_price <= trade.stop_loss
        return current_price >= trade.stop_loss

    def _breakeven_sl(self, trade: ManagedTrade) -> float:
        fee_buffer = trade.entry_price * (self.move_sl_to_entry_plus_fees_pct / 100.0)
        if trade.side == PositionSide.LONG:
            return trade.entry_price + fee_buffer
        return trade.entry_price - fee_buffer

    def _compute_trailing_sl(
        self,
        *,
        trade: ManagedTrade,
        current_price: float,
        atr_value: float | None,
        last_swing_low: float | None,
        last_swing_high: float | None,
    ) -> float | None:
        # Prefer structure-based trail when a confirmed swing is available.
        if trade.side == PositionSide.LONG and last_swing_low is not None:
            floor_lock = trade.entry_price * (1.0 + self.min_lock_in_profit_pct / 100.0)
            return max(float(last_swing_low), floor_lock)
        if trade.side == PositionSide.SHORT and last_swing_high is not None:
            ceiling_lock = trade.entry_price * (1.0 - self.min_lock_in_profit_pct / 100.0)
            return min(float(last_swing_high), ceiling_lock)

        if atr_value is None or atr_value <= 0:
            return None

        # Fallback ATR trail.
        if trade.side == PositionSide.LONG:
            return current_price - (atr_value * self.atr_trail_multiplier)
        return current_price + (atr_value * self.atr_trail_multiplier)

    @staticmethod
    def _trail_improves(trade: ManagedTrade, candidate_sl: float) -> bool:
        if trade.side == PositionSide.LONG:
            return candidate_sl > trade.stop_loss
        return candidate_sl < trade.stop_loss
