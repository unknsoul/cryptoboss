"""CatBoost model wrapper."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None


class CatBoostModel:
    """Thin wrapper around CatBoost classifier."""

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 6,
        eval_metric: str = "AUC",
        random_state: int = 42,
        **kwargs,
    ) -> None:
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed")
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            eval_metric=eval_metric,
            random_state=random_state,
            verbose=False,
            **kwargs,
        )

    def fit(self, X, y) -> "CatBoostModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def feature_importances(self, feature_names: Optional[list[str]] = None) -> dict:
        if not hasattr(self.model, "get_feature_importance"):
            return {}
        importances = self.model.get_feature_importance()
        if feature_names:
            return {name: float(val) for name, val in zip(feature_names, importances)}
        return {str(idx): float(val) for idx, val in enumerate(importances)}
