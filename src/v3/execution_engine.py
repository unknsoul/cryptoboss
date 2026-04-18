"""Execution engine for v3 intraday scalper architecture."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from src.execution.live_broker import LiveBroker, OrderStatus

from .config import ExecutionEngineConfig
from .models import ExecutionReport, RiskDecision, SignalOutput


class ExecutionEngine:
    """Executes approved trades with spread/slippage protections."""

    def __init__(self, config: Optional[ExecutionEngineConfig] = None, broker: Optional[LiveBroker] = None):
        self.config = config or ExecutionEngineConfig()
        self.broker = broker or LiveBroker()

    def execute(
        self,
        symbol: str,
        signal: SignalOutput,
        risk_decision: RiskDecision,
        market_snapshot: Dict[str, float],
        order_type: str = "market",
    ) -> ExecutionReport:
        if signal.action not in ("BUY", "SELL"):
            return ExecutionReport(
                accepted=False,
                status="rejected",
                action=signal.action,
                symbol=symbol,
                reason="Signal is not actionable",
            )

        if not risk_decision.approved:
            return ExecutionReport(
                accepted=False,
                status="rejected",
                action=signal.action,
                symbol=symbol,
                reason=f"Risk rejected: {risk_decision.reason}",
            )

        normalized_order_type = order_type.lower()
        if normalized_order_type not in self.config.order_types:
            return ExecutionReport(
                accepted=False,
                status="rejected",
                action=signal.action,
                symbol=symbol,
                order_type=normalized_order_type,
                reason=f"Unsupported order type: {order_type}",
            )

        spread_pct = float(market_snapshot.get("spread_pct", 0.0))
        if self.config.spread_filter and spread_pct > self.config.max_spread_pct:
            return ExecutionReport(
                accepted=False,
                status="rejected",
                action=signal.action,
                symbol=symbol,
                order_type=normalized_order_type,
                reason=f"Spread filter blocked trade ({spread_pct:.4f}% > {self.config.max_spread_pct:.4f}%)",
                metadata={"spread_pct": spread_pct},
            )

        requested_price = float(signal.entry_price or market_snapshot.get("last_price", 0.0))
        if requested_price <= 0:
            return ExecutionReport(
                accepted=False,
                status="rejected",
                action=signal.action,
                symbol=symbol,
                reason="No valid price available for execution",
            )

        expected_slippage_pct = float(market_snapshot.get("expected_slippage_pct", 0.0))
        if self.config.slippage_control and expected_slippage_pct > self.config.max_slippage_pct:
            return ExecutionReport(
                accepted=False,
                status="rejected",
                action=signal.action,
                symbol=symbol,
                order_type=normalized_order_type,
                reason=(
                    f"Slippage control blocked trade "
                    f"({expected_slippage_pct:.4f}% > {self.config.max_slippage_pct:.4f}%)"
                ),
                metadata={"expected_slippage_pct": expected_slippage_pct},
            )

        raw_order = self.broker.place_order(
            symbol=symbol,
            side=signal.action,
            size=float(risk_decision.position_size),
            order_type=normalized_order_type.upper(),
            price=requested_price,
        )

        status = str(raw_order.get("status", "failed"))
        accepted = status != OrderStatus.FAILED.value
        filled_price = float(raw_order.get("filled_price", requested_price))
        realized_slippage = abs(filled_price - requested_price) / requested_price * 100.0

        reason = "Order accepted" if accepted else str(raw_order.get("error", "Order rejected by broker"))
        if accepted and self.config.slippage_control and realized_slippage > self.config.max_slippage_pct:
            reason = (
                f"Order accepted but realized slippage was high "
                f"({realized_slippage:.4f}% > {self.config.max_slippage_pct:.4f}%)"
            )

        return ExecutionReport(
            accepted=accepted,
            status=status,
            order_id=raw_order.get("client_order_id"),
            action=signal.action,
            symbol=symbol,
            order_type=normalized_order_type,
            requested_price=requested_price,
            filled_price=filled_price,
            slippage=realized_slippage,
            timestamp=datetime.utcnow(),
            reason=reason,
            metadata={
                "platform": self.config.platform,
                "execution_speed": self.config.execution_speed,
                "raw_order": raw_order,
            },
        )
