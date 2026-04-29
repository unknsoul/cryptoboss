"""Outcome-based labeling facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.labels.triple_barrier import TripleBarrierConfig, apply_triple_barrier
from src.ml.dynamic_labeling import DynamicLabeler


@dataclass
class OutcomeLabelerConfig:
    """Configuration for label creation."""

    method: str = "triple_barrier"  # triple_barrier or dynamic_atr
    triple_barrier: Optional[TripleBarrierConfig] = None
    dynamic_lookforward_bars: int = 50


class OutcomeLabeler:
    """Create outcome-based labels for ML training."""

    def __init__(self, config: Optional[OutcomeLabelerConfig] = None) -> None:
        self.config = config or OutcomeLabelerConfig()
        self._dynamic_labeler = DynamicLabeler()

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create labels based on the configured method."""
        if self.config.method == "triple_barrier":
            return apply_triple_barrier(df, self.config.triple_barrier)

        if self.config.method == "dynamic_atr":
            return self._dynamic_labeler.create_labels(
                df,
                lookforward_bars=self.config.dynamic_lookforward_bars,
            )

        raise ValueError("Unknown label method: %s" % self.config.method)
