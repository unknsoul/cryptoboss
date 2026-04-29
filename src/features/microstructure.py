"""Microstructure feature helpers."""

from __future__ import annotations

from typing import Optional

from src.data.orderbook.microstructure import orderbook_features


def microstructure_features(orderbook: dict, atr_value: Optional[float] = None) -> dict:
    """Return spread and order book imbalance features."""
    metrics = orderbook_features(orderbook)
    if atr_value and atr_value > 0:
        metrics["spread_ratio"] = metrics.get("spread", 0.0) / atr_value
    else:
        metrics["spread_ratio"] = 0.0
    return metrics
