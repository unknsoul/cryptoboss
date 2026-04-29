"""Broker abstraction layer."""

from __future__ import annotations

from typing import Dict, Optional

from src.execution.live_broker import LiveBroker


class Broker:
    """Abstract broker interface."""

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict:
        raise NotImplementedError

    def cancel_order(self, client_order_id: str) -> Dict:
        raise NotImplementedError


class LiveBrokerAdapter(Broker):
    """Adapter for the existing LiveBroker."""

    def __init__(self, broker: LiveBroker) -> None:
        self.broker = broker

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict:
        return self.broker.place_order(symbol, side, size, order_type, price, client_order_id)

    def cancel_order(self, client_order_id: str) -> Dict:
        if client_order_id in self.broker.order_cache:
            order = self.broker.order_cache[client_order_id]
            order["status"] = "cancelled"
            return order
        return {"status": "cancelled", "client_order_id": client_order_id}
