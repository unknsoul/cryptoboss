"""Risk engine for intraday scalping strategies."""

from dataclasses import dataclass


@dataclass
class ScalperRiskConfig:
    max_risk_pct: float = 0.005
    max_positions: int = 3
    partial_exit_pct: float = 0.5


class ScalperRiskEngine:
    """Handles position sizing and basic exposure limits for scalpers."""

    def __init__(self, config: ScalperRiskConfig):
        self.config = config

    def can_open_position(self, current_open_positions: int) -> bool:
        return current_open_positions < self.config.max_positions

    def compute_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        session_weight: float = 1.0,
    ) -> float:
        risk_amount = account_balance * self.config.max_risk_pct * max(session_weight, 0.0)
        price_risk = abs(entry_price - stop_loss)
        if price_risk <= 0:
            return 0.0
        size = risk_amount / price_risk
        return round(max(size, 0.0), 6)

    def max_loss_for_trade(self, account_balance: float) -> float:
        return account_balance * self.config.max_risk_pct
