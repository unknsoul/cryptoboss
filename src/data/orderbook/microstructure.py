"""Order book microstructure features."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def best_bid_ask(bids: Iterable[Iterable[float]], asks: Iterable[Iterable[float]]) -> Tuple[float, float]:
    """Return best bid and ask from order book arrays."""
    bid_price = float(bids[0][0]) if bids else 0.0
    ask_price = float(asks[0][0]) if asks else 0.0
    return bid_price, ask_price


def spread(bids: Iterable[Iterable[float]], asks: Iterable[Iterable[float]]) -> float:
    """Compute spread from top-of-book quotes."""
    bid_price, ask_price = best_bid_ask(bids, asks)
    if bid_price <= 0 or ask_price <= 0:
        return 0.0
    return max(ask_price - bid_price, 0.0)


def book_imbalance(bids: Iterable[Iterable[float]], asks: Iterable[Iterable[float]], depth: int = 5) -> float:
    """Compute normalized order book imbalance."""
    bid_depth = sum(float(level[1]) for level in list(bids)[:depth])
    ask_depth = sum(float(level[1]) for level in list(asks)[:depth])
    total = bid_depth + ask_depth
    if total == 0:
        return 0.0
    return float((bid_depth - ask_depth) / total)


def depth_metrics(bids: Iterable[Iterable[float]], asks: Iterable[Iterable[float]], depth: int = 10) -> dict:
    """Return depth and imbalance metrics."""
    bid_depth = sum(float(level[1]) for level in list(bids)[:depth])
    ask_depth = sum(float(level[1]) for level in list(asks)[:depth])
    return {
        "bid_depth": float(bid_depth),
        "ask_depth": float(ask_depth),
        "depth_imbalance": book_imbalance(bids, asks, depth=depth),
    }


def orderbook_features(orderbook: dict, depth: int = 5) -> dict:
    """Return derived microstructure features from an order book snapshot."""
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    bid_price, ask_price = best_bid_ask(bids, asks)

    metrics = {
        "best_bid": bid_price,
        "best_ask": ask_price,
        "spread": spread(bids, asks),
        "book_imbalance": book_imbalance(bids, asks, depth=depth),
    }
    metrics.update(depth_metrics(bids, asks, depth=depth))
    return metrics


def cumulative_volume_delta(trades: pd.DataFrame, side_col: str = "side", volume_col: str = "size") -> pd.Series:
    """Compute cumulative volume delta from trade data."""
    if trades.empty:
        return pd.Series(dtype=float)

    side = trades[side_col].str.lower().fillna("")
    volume = pd.to_numeric(trades[volume_col], errors="coerce").fillna(0.0)
    signed = np.where(side.eq("buy"), volume, -volume)
    return pd.Series(signed, index=trades.index).cumsum()
