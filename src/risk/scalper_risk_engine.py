"""Risk engine for intraday scalping strategies."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ScalperRiskConfig:
    max_risk_pct: float = 0.0075
    max_daily_loss_pct: float = 0.03
    max_positions: int = 3
    partial_exit_pct: float = 0.4
    break_even_rr_trigger: float = 1.0
    trailing_atr_mult: float = 1.0
    max_portfolio_heat_pct: float = 0.03


class ScalperRiskEngine:
    """Handles position sizing, exposure limits, and portfolio heat for scalpers."""

    def __init__(self, config: ScalperRiskConfig):
        self.config = config
        self._open_risk_amounts: List[float] = []

    def can_open_position(
        self,
        current_open_positions: int,
        daily_pnl: float = 0.0,
        account_balance: float = 0.0,
    ) -> bool:
        if current_open_positions >= self.config.max_positions:
            return False
        if account_balance > 0 and self.daily_loss_halt(daily_pnl, account_balance, self.config.max_daily_loss_pct):
            return False
        return True

    def register_open_risk(self, risk_amount: float) -> None:
        self._open_risk_amounts.append(risk_amount)

    def close_risk(self, risk_amount: float) -> None:
        try:
            self._open_risk_amounts.remove(risk_amount)
        except ValueError:
            if self._open_risk_amounts:
                self._open_risk_amounts.pop(0)

    def portfolio_heat(self, account_balance: float) -> float:
        if account_balance <= 0:
            return 0.0
        return sum(self._open_risk_amounts) / account_balance

    def can_add_heat(self, account_balance: float, new_risk: float) -> bool:
        current = self.portfolio_heat(account_balance)
        projected = current + (new_risk / max(account_balance, 0.0001))
        return projected <= self.config.max_portfolio_heat_pct

    def heat_adjusted_size(self, base_size: float, account_balance: float) -> float:
        heat = self.portfolio_heat(account_balance)
        max_heat = self.config.max_portfolio_heat_pct
        if max_heat <= 0:
            return base_size
        ratio = max(1.0 - (heat / max_heat), 0.2)
        return base_size * ratio

    @staticmethod
    def check_break_even(
        entry: float,
        current_price: float,
        direction: str,
        rr_trigger: float,
    ) -> Optional[float]:
        if entry <= 0 or rr_trigger <= 0:
            return None
        move_pct = abs(current_price - entry) / entry
        if direction.lower() == "long" and current_price > entry and move_pct >= rr_trigger:
            return entry
        if direction.lower() == "short" and current_price < entry and move_pct >= rr_trigger:
            return entry
        return None

    @staticmethod
    def compute_trailing_stop(
        entry: float,
        current_price: float,
        atr: float,
        direction: str,
        trail_mult: float,
    ) -> Optional[float]:
        if atr <= 0 or trail_mult <= 0:
            return None
        distance = atr * trail_mult
        if direction.lower() == "long":
            return max(entry, current_price - distance)
        if direction.lower() == "short":
            return min(entry, current_price + distance)
        return None

    @staticmethod
    def daily_loss_halt(daily_pnl: float, account_balance: float, max_daily_loss_pct: float) -> bool:
        if account_balance <= 0:
            return False
        threshold_pct = max_daily_loss_pct * 100.0 if max_daily_loss_pct <= 1 else max_daily_loss_pct
        loss_pct = 0.0
        if daily_pnl < 0:
            loss_pct = (-daily_pnl / account_balance) * 100.0
        return loss_pct >= threshold_pct

    @staticmethod
    def session_adjusted_risk(base_risk: float, session_weight: float) -> float:
        return max(base_risk * max(session_weight, 0.0), 0.0)

    def compute_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        session_weight: float = 1.0,
    ) -> float:
        adjusted_risk_pct = self.session_adjusted_risk(self.config.max_risk_pct, session_weight)
        risk_amount = account_balance * adjusted_risk_pct
        price_risk = abs(entry_price - stop_loss)
        if price_risk <= 0:
            return 0.0
        size = risk_amount / price_risk
        size = self.heat_adjusted_size(size, account_balance)
        return round(max(size, 0.0), 6)

    def max_loss_for_trade(self, account_balance: float) -> float:
        return account_balance * self.config.max_risk_pct
