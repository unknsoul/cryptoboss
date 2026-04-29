"""XGBoost model wrapper."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


class XGBoostModel:
    """Thin wrapper around XGBoost classifier."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        if xgb is None:
            raise ImportError("xgboost is not installed")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X, y) -> "XGBoostModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def feature_importances(self, feature_names: Optional[list[str]] = None) -> dict:
        if not hasattr(self.model, "feature_importances_"):
            return {}
        importances = self.model.feature_importances_
        if feature_names:
            return {name: float(val) for name, val in zip(feature_names, importances)}
        return {str(idx): float(val) for idx, val in enumerate(importances)}
