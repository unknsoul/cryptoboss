"""Triple barrier labeling (Lopez de Prado)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Configuration for triple barrier labeling."""

    tp_pct: float = 0.02
    sl_pct: float = 0.01
    max_holding_bars: int = 50
    side: str = "long"  # long, short, or both
    include_touch: bool = True


def _first_hit(path: np.ndarray, tp: float, sl: float, include_touch: bool) -> Tuple[int, int]:
    """Return (tp_index, sl_index) where -1 means not hit."""
    if include_touch:
        tp_hits = np.where(path >= tp)[0]
        sl_hits = np.where(path <= sl)[0]
    else:
        tp_hits = np.where(path > tp)[0]
        sl_hits = np.where(path < sl)[0]

    tp_index = int(tp_hits[0]) if tp_hits.size else -1
    sl_index = int(sl_hits[0]) if sl_hits.size else -1
    return tp_index, sl_index


def _label_one(
    entry: float,
    path: np.ndarray,
    tp_pct: float,
    sl_pct: float,
    side: str,
    include_touch: bool,
) -> Tuple[int, Optional[int]]:
    """Return (label, hit_index) for one entry."""
    if side == "long":
        tp = entry * (1.0 + tp_pct)
        sl = entry * (1.0 - sl_pct)
        tp_idx, sl_idx = _first_hit(path, tp, sl, include_touch)
        if tp_idx == -1 and sl_idx == -1:
            return 0, None
        if tp_idx != -1 and (sl_idx == -1 or tp_idx < sl_idx):
            return 1, tp_idx
        return -1, sl_idx

    if side == "short":
        tp = entry * (1.0 - tp_pct)
        sl = entry * (1.0 + sl_pct)
        # Use manual checks for short to keep logic explicit.
        if include_touch:
            tp_hits = np.where(path <= tp)[0]
            sl_hits = np.where(path >= sl)[0]
        else:
            tp_hits = np.where(path < tp)[0]
            sl_hits = np.where(path > sl)[0]
        tp_idx = int(tp_hits[0]) if tp_hits.size else -1
        sl_idx = int(sl_hits[0]) if sl_hits.size else -1

        if tp_idx == -1 and sl_idx == -1:
            return 0, None
        if tp_idx != -1 and (sl_idx == -1 or tp_idx < sl_idx):
            return -1, tp_idx
        return 1, sl_idx

    raise ValueError("side must be 'long' or 'short'")


def triple_barrier_labels(
    prices: Iterable[float] | pd.Series,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    max_holding_bars: int = 50,
    side: str = "long",
    include_touch: bool = True,
) -> pd.Series:
    """Compute triple barrier labels for a price series."""
    series = pd.Series(prices).reset_index(drop=True)
    values = series.to_numpy(dtype=float)

    labels = np.zeros(len(values), dtype=int)

    for i in range(len(values)):
        end = min(i + max_holding_bars + 1, len(values))
        path = values[i + 1 : end]
        if path.size == 0:
            labels[i] = 0
            continue

        if side == "both":
            long_label, long_hit = _label_one(values[i], path, tp_pct, sl_pct, "long", include_touch)
            short_label, short_hit = _label_one(values[i], path, tp_pct, sl_pct, "short", include_touch)
            if long_label == 0 and short_label == 0:
                labels[i] = 0
            elif long_label != 0 and short_label == 0:
                labels[i] = long_label
            elif short_label != 0 and long_label == 0:
                labels[i] = short_label
            else:
                # Both hit: pick the earliest barrier.
                if long_hit is None and short_hit is None:
                    labels[i] = 0
                elif short_hit is None:
                    labels[i] = long_label
                elif long_hit is None:
                    labels[i] = short_label
                else:
                    labels[i] = long_label if long_hit <= short_hit else short_label
        else:
            labels[i], _ = _label_one(values[i], path, tp_pct, sl_pct, side, include_touch)

    return pd.Series(labels, index=series.index)


def apply_triple_barrier(
    df: pd.DataFrame,
    config: Optional[TripleBarrierConfig] = None,
    price_column: str = "close",
) -> pd.Series:
    """Apply triple barrier labels to a DataFrame of prices."""
    config = config or TripleBarrierConfig()
    if price_column not in df.columns:
        raise ValueError(f"Missing price column: {price_column}")

    return triple_barrier_labels(
        df[price_column],
        tp_pct=config.tp_pct,
        sl_pct=config.sl_pct,
        max_holding_bars=config.max_holding_bars,
        side=config.side,
        include_touch=config.include_touch,
    )
