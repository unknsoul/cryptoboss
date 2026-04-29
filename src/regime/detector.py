"""Unified regime detection interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.analysis.regime_detector_advanced import AdvancedRegimeDetector, MarketRegime
from src.regime.hmm_regime import HMMRegimeModel


@dataclass
class RegimeSnapshot:
    """Normalized regime output for downstream consumers."""

    regime: str
    confidence: float
    details: Dict[str, float]
    hmm_state: Optional[int] = None


class RegimeDetector:
    """Combines indicator regime detection with optional HMM states."""

    def __init__(
        self,
        use_hmm: bool = True,
        crisis_drawdown_pct: float = 0.2,
        high_vol_percentile: float = 0.9,
    ) -> None:
        self._detector = AdvancedRegimeDetector()
        self._hmm = HMMRegimeModel() if use_hmm else None
        self.crisis_drawdown_pct = crisis_drawdown_pct
        self.high_vol_percentile = high_vol_percentile

    def fit_hmm(self, feature_df: pd.DataFrame) -> None:
        """Fit the HMM model on supplied features."""
        if self._hmm is None:
            return
        self._hmm.fit(feature_df)

    def detect(self, df: pd.DataFrame, hmm_features: Optional[pd.DataFrame] = None) -> RegimeSnapshot:
        """Detect current market regime."""
        if df.empty:
            return RegimeSnapshot(regime="RANGING", confidence=0.0, details={})

        info = self._detector.detect_regime(df)
        mapped_regime = self._map_regime(info.regime)

        drawdown_pct = self._drawdown_pct(df["close"].values)
        if drawdown_pct >= self.crisis_drawdown_pct:
            mapped_regime = "CRISIS"

        details = {
            "adx": float(info.adx_value),
            "atr_percentile": float(info.atr_percentile),
            "drawdown_pct": float(drawdown_pct),
        }

        hmm_state = None
        if self._hmm is not None and hmm_features is not None:
            hmm_result = self._hmm.predict(hmm_features.tail(1))
            hmm_state = int(hmm_result.states[-1])

        return RegimeSnapshot(
            regime=mapped_regime,
            confidence=float(info.confidence),
            details=details,
            hmm_state=hmm_state,
        )

    def _map_regime(self, regime: MarketRegime) -> str:
        if regime == MarketRegime.TRENDING_UP:
            return "TRENDING_BULL"
        if regime == MarketRegime.TRENDING_DOWN:
            return "TRENDING_BEAR"
        if regime == MarketRegime.HIGH_VOLATILITY:
            return "HIGH_VOL"
        if regime == MarketRegime.LOW_VOLATILITY:
            return "RANGING"
        return "RANGING"

    @staticmethod
    def _drawdown_pct(prices: np.ndarray) -> float:
        if prices.size == 0:
            return 0.0
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / np.maximum(peak, 1e-9)
        return float(drawdown.max())
