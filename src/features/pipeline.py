"""Unified feature pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.data.feature_engineering import FeatureEngine
from src.data.schema import standardize_ohlcv
from src.features.indicators import build_indicator_features
from src.features.smc_features import build_smc_features
from src.features.statistical import rolling_zscore, volatility_cluster, rolling_hurst
from src.features.microstructure import microstructure_features
from src.features.alternative import (
    funding_rate_features,
    open_interest_features,
    liquidation_features,
    options_skew_features,
    onchain_features,
)


@dataclass
class FeaturePipelineConfig:
    include_indicators: bool = True
    include_smc: bool = True
    include_statistical: bool = True
    include_microstructure: bool = True
    include_alternative: bool = True


class FeaturePipeline:
    """Build a unified feature set for training and live scoring."""

    def __init__(self, config: Optional[FeaturePipelineConfig] = None) -> None:
        self.config = config or FeaturePipelineConfig()
        self._engine = FeatureEngine()

    def build_features(
        self,
        ohlcv: pd.DataFrame,
        orderbook: Optional[dict] = None,
        funding_rates: Optional[pd.DataFrame] = None,
        open_interest: Optional[pd.DataFrame] = None,
        liquidations: Optional[pd.DataFrame] = None,
        options_data: Optional[pd.DataFrame] = None,
        onchain_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build the full feature DataFrame."""
        ohlcv = standardize_ohlcv(ohlcv, keep_extra_columns=True)
        base = self._engine.generate_features(ohlcv)
        base = self._ensure_timestamp(base)

        if self.config.include_indicators:
            indicators = build_indicator_features(ohlcv, include_ohlcv=False)
            base = self._merge_feature_block(base, indicators)

        if self.config.include_statistical:
            returns = base["close"].pct_change()
            base["returns_zscore_50"] = rolling_zscore(returns, window=50)
            base["vol_cluster"] = volatility_cluster(returns)
            base["hurst_200"] = rolling_hurst(base["close"], window=200)

        if self.config.include_smc:
            smc_features = build_smc_features(ohlcv)
            base = self._merge_feature_block(base, smc_features)

        if self.config.include_microstructure and orderbook:
            atr_value = float(base["volatility_atr_14"].iloc[-1]) if "volatility_atr_14" in base.columns else None
            metrics = microstructure_features(orderbook, atr_value=atr_value)
            for key, value in metrics.items():
                base.loc[base.index[-1], key] = value

        if self.config.include_alternative:
            base = self._merge_alternative(base, funding_rates, open_interest, liquidations, options_data, onchain_data)

        return base

    @staticmethod
    def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df.index, utc=True, errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df

    @staticmethod
    def _merge_feature_block(base: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
        if block.empty:
            return base
        aligned = block.copy()
        if len(aligned) != len(base):
            aligned = aligned.tail(len(base)).reset_index(drop=True)
        aligned.index = base.index
        for column in aligned.columns:
            if column in base.columns:
                continue
            base[column] = aligned[column]
        return base

    def _merge_alternative(
        self,
        base: pd.DataFrame,
        funding_rates: Optional[pd.DataFrame],
        open_interest: Optional[pd.DataFrame],
        liquidations: Optional[pd.DataFrame],
        options_data: Optional[pd.DataFrame],
        onchain_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        merged = base.sort_values("timestamp")

        if funding_rates is not None and not funding_rates.empty:
            features = funding_rate_features(funding_rates)
            merged = self._merge_asof(merged, features)

        if open_interest is not None and not open_interest.empty:
            features = open_interest_features(open_interest)
            merged = self._merge_asof(merged, features)

        if liquidations is not None and not liquidations.empty:
            features = liquidation_features(liquidations)
            merged = self._merge_asof(merged, features)

        if options_data is not None and not options_data.empty:
            features = options_skew_features(options_data)
            merged = self._merge_asof(merged, features)

        if onchain_data is not None and not onchain_data.empty:
            features = onchain_features(onchain_data)
            merged = self._merge_asof(merged, features)

        return merged

    @staticmethod
    def _merge_asof(base: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        right = features.copy()
        right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")
        base_sorted = base.sort_values("timestamp")
        right_sorted = right.sort_values("timestamp")
        return pd.merge_asof(base_sorted, right_sorted, on="timestamp", direction="backward")
