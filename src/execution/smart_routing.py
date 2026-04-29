"""Smart order routing utilities."""

from __future__ import annotations

from typing import Dict, Optional
from datetime import datetime

import time

from src.execution.broker import Broker
from src.risk.controls import RiskController


def smart_execute(
    broker: Broker,
    symbol: str,
    side: str,
    size: float,
    bid: float,
    ask: float,
    urgency: str = "normal",
    wait_seconds: int = 30,
    risk_controller: Optional[RiskController] = None,
    equity: Optional[float] = None,
    stop_price: Optional[float] = None,
    timestamp: Optional[datetime] = None,
) -> Dict:
    """Limit-first execution with fallback to market."""
    if risk_controller is not None and equity is not None and stop_price is not None:
        mid = (bid + ask) / 2.0
        proposed_risk = size * abs(mid - stop_price)
        decision = risk_controller.validate_trade(
            equity=equity,
            proposed_risk_amount=proposed_risk,
            timestamp=timestamp or datetime.utcnow(),
        )
        if not decision.allowed:
            return {
                "status": "rejected",
                "reason": decision.reason or "risk gate rejection",
                "symbol": symbol,
                "side": side,
            }
        if decision.adjusted_size is not None:
            size = max(0.0, size * float(decision.adjusted_size))
            if size == 0.0:
                return {"status": "rejected", "reason": "risk-adjusted size is zero", "symbol": symbol, "side": side}

    if urgency == "high":
        return broker.place_order(symbol, side, size, order_type="MARKET")

    spread = max(ask - bid, 0.0)
    limit_price = bid + (0.5 * spread) if side.lower() == "buy" else ask - (0.5 * spread)

    order = broker.place_order(symbol, side, size, order_type="LIMIT", price=limit_price)
    time.sleep(wait_seconds)

    if order.get("status") == "filled":
        return order

    broker.cancel_order(order["client_order_id"])
    return broker.place_order(symbol, side, size, order_type="MARKET")
