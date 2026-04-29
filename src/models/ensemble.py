"""Ensemble model helpers."""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class EnsembleModel:
    """Weighted probability ensemble."""

    def __init__(self, models: List, weights: Optional[List[float]] = None) -> None:
        if not models:
            raise ValueError("Ensemble requires at least one model")
        self.models = models
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights

    def fit(self, X, y) -> "EnsembleModel":
        for model in self.models:
            if hasattr(model, "fit"):
                model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        weighted = None
        for model, weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            proba = np.asarray(proba, dtype=float)
            if proba.ndim > 1:
                proba = proba[:, 1]
            weighted = proba * weight if weighted is None else weighted + (proba * weight)
        return weighted if weighted is not None else np.zeros(len(X))

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
