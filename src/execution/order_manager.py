"""Order lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class OrderRecord:
    """Order metadata record."""

    client_order_id: str
    symbol: str
    side: str
    size: float
    order_type: str
    status: str
    price: Optional[float] = None
    filled_price: Optional[float] = None
    created_at: str = ""
    updated_at: str = ""
    lifecycle: list[dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class OrderManager:
    """Keep track of order lifecycle updates."""

    def __init__(self) -> None:
        self.orders: Dict[str, OrderRecord] = {}

    def register_order(self, order_data: Dict) -> OrderRecord:
        now = datetime.utcnow().isoformat()
        initial_status = order_data.get("status", "pending")
        record = OrderRecord(
            client_order_id=order_data["client_order_id"],
            symbol=order_data["symbol"],
            side=order_data["side"],
            size=order_data["size"],
            order_type=order_data.get("order_type", "MARKET"),
            status=initial_status,
            price=order_data.get("price"),
            filled_price=order_data.get("filled_price"),
            created_at=now,
            updated_at=now,
            lifecycle=[{"status": initial_status, "timestamp": now}],
            metadata={k: v for k, v in order_data.items() if k not in {"client_order_id", "symbol", "side", "size", "order_type", "status", "price", "filled_price"}},
        )
        self.orders[record.client_order_id] = record
        return record

    def update_status(self, client_order_id: str, status: str, filled_price: Optional[float] = None) -> None:
        record = self.orders.get(client_order_id)
        if record is None:
            return
        record.status = status
        now = datetime.utcnow().isoformat()
        record.updated_at = now
        record.lifecycle.append({"status": status, "timestamp": now})
        if filled_price is not None:
            record.filled_price = filled_price

    def get_order(self, client_order_id: str) -> Optional[OrderRecord]:
        return self.orders.get(client_order_id)
