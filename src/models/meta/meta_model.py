"""Meta-model wrapper for signal filtering."""

from __future__ import annotations

import numpy as np

from sklearn.linear_model import LogisticRegression


class MetaModel:
    """Simple logistic regression meta-model."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, X, y) -> "MetaModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
