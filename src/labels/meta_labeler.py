"""Meta-model labeler to filter primary signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class MetaLabelerReport:
    """Training summary for meta labeler."""

    auc: float
    positive_rate: float


class MetaLabeler:
    """Train and apply a meta-model that filters primary signals."""

    def __init__(self, model: Optional[LogisticRegression] = None) -> None:
        self.model = model or LogisticRegression(max_iter=1000, class_weight="balanced")

    @staticmethod
    def build_features(base_features: pd.DataFrame, primary_proba: np.ndarray) -> pd.DataFrame:
        """Combine base features with primary model confidence."""
        features = base_features.copy()
        features["primary_confidence"] = primary_proba
        return features

    def fit(
        self,
        base_features: pd.DataFrame,
        primary_proba: np.ndarray,
        labels: pd.Series,
    ) -> MetaLabelerReport:
        """Fit the meta-model to predict when to trust primary signals."""
        features = self.build_features(base_features, primary_proba)
        valid_mask = ~(features.isna().any(axis=1) | labels.isna())
        X = features[valid_mask]
        y = labels[valid_mask].astype(int)

        self.model.fit(X, y)
        proba = self.model.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else 0.5
        positive_rate = float(y.mean()) if len(y) else 0.0

        return MetaLabelerReport(auc=auc, positive_rate=positive_rate)

    def predict_proba(self, base_features: pd.DataFrame, primary_proba: np.ndarray) -> np.ndarray:
        """Predict meta confidence for each sample."""
        features = self.build_features(base_features, primary_proba)
        return self.model.predict_proba(features)[:, 1]
