"""Risk engine for v3 intraday scalper architecture."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Dict, List, Optional

from .config import RiskEngineConfig
from .models import RiskDecision, SignalOutput


class RiskEngine:
    """Applies per-trade and portfolio risk constraints."""

    def __init__(self, config: Optional[RiskEngineConfig] = None):
        self.config = config or RiskEngineConfig()
        self._trades_per_day = defaultdict(int)
        self._peak_equity = 0.0

    def evaluate(
        self,
        signal: SignalOutput,
        account_state: Dict[str, float],
        smart_money: Dict[str, object],
        timestamp: Optional[datetime] = None,
    ) -> RiskDecision:
        now = timestamp or datetime.utcnow()
        day_key = now.date().isoformat()

        if signal.action not in ("BUY", "SELL"):
            return RiskDecision(approved=False, reason="Signal is HOLD")

        if self._trades_per_day[day_key] >= self.config.max_trades_per_day:
            return RiskDecision(
                approved=False,
                reason=f"Max trades per day reached ({self.config.max_trades_per_day})",
            )

        equity = float(account_state.get("equity", account_state.get("balance", 0.0)))
        if equity <= 0:
            return RiskDecision(approved=False, reason="Invalid account equity")

        if self._peak_equity <= 0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

        drawdown_pct = ((self._peak_equity - equity) / self._peak_equity) * 100.0
        if drawdown_pct > self.config.max_drawdown:
            return RiskDecision(
                approved=False,
                reason=f"Max drawdown exceeded ({drawdown_pct:.2f}% > {self.config.max_drawdown:.2f}%)",
                metadata={"drawdown_pct": drawdown_pct},
            )

        entry = float(signal.entry_price or account_state.get("last_price", 0.0))
        if entry <= 0:
            return RiskDecision(approved=False, reason="No valid entry price")

        stop_loss = self._derive_stop_loss(signal, entry, smart_money)
        take_profit = self._derive_take_profit(signal, entry, stop_loss, smart_money)

        if stop_loss is None or take_profit is None:
            return RiskDecision(approved=False, reason="Unable to derive stop loss / take profit")

        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return RiskDecision(approved=False, reason="Invalid stop-loss distance")

        risk_amount = equity * (self.config.risk_per_trade / 100.0)
        position_size = risk_amount / stop_distance

        risk_decision = RiskDecision(
            approved=True,
            reason="Risk checks passed",
            position_size=position_size,
            risk_pct=self.config.risk_per_trade,
            rr_ratio=self.config.rr_ratio,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "drawdown_pct": drawdown_pct,
                "risk_amount": risk_amount,
                "max_trades_per_day": self.config.max_trades_per_day,
            },
        )
        return risk_decision

    def register_trade(self, timestamp: Optional[datetime] = None) -> None:
        now = timestamp or datetime.utcnow()
        day_key = now.date().isoformat()
        self._trades_per_day[day_key] += 1

    def reset_day(self, day: Optional[date] = None) -> None:
        key = (day or datetime.utcnow().date()).isoformat()
        self._trades_per_day[key] = 0

    def _derive_stop_loss(self, signal: SignalOutput, entry: float, smart_money: Dict[str, object]) -> Optional[float]:
        order_blocks = smart_money.get("order_blocks", [])
        if not isinstance(order_blocks, list):
            order_blocks = []

        if signal.action == "BUY":
            bullish_obs = [ob for ob in order_blocks if str(ob.get("type", "")) == "bullish"]
            structure_floor = min((float(ob.get("bottom", entry)) for ob in bullish_obs), default=entry * 0.998)
            return structure_floor * 0.999

        bearish_obs = [ob for ob in order_blocks if str(ob.get("type", "")) == "bearish"]
        structure_cap = max((float(ob.get("top", entry)) for ob in bearish_obs), default=entry * 1.002)
        return structure_cap * 1.001

    def _derive_take_profit(
        self,
        signal: SignalOutput,
        entry: float,
        stop_loss: float,
        smart_money: Dict[str, object],
    ) -> Optional[float]:
        liquidity = smart_money.get("liquidity", {})
        levels = liquidity.get("levels", {}) if isinstance(liquidity, dict) else {}
        equal_levels = levels.get("equal_highs_lows", []) if isinstance(levels, dict) else []

        if signal.action == "BUY":
            target_candidates: List[float] = [
                float(level.get("price"))
                for level in equal_levels
                if float(level.get("price", 0.0)) > entry
            ]
            if target_candidates:
                return min(target_candidates)

            risk = abs(entry - stop_loss)
            return entry + risk * self.config.rr_ratio

        target_candidates = [
            float(level.get("price"))
            for level in equal_levels
            if float(level.get("price", 0.0)) < entry
        ]
        if target_candidates:
            return max(target_candidates)

        risk = abs(entry - stop_loss)
        return entry - risk * self.config.rr_ratio
