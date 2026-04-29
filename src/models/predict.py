"""Prediction helpers for registered models."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.models.registry import ModelRegistry


def predict_from_registry(
    features: pd.DataFrame,
    version_id: Optional[str] = None,
    registry: Optional[ModelRegistry] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Load a model from the registry and generate predictions."""
    registry = registry or ModelRegistry()
    if version_id is None:
        raise ValueError("version_id is required")

    bundle = registry.load_model(version_id)
    model = bundle["model"]
    feature_list = bundle.get("features", [])
    scaler = bundle.get("scaler")

    X = features[feature_list] if feature_list else features
    if scaler is not None:
        X = scaler.transform(X)

    proba = model.predict_proba(X)
    if proba.ndim > 1:
        proba = proba[:, 1]
    return (proba >= threshold).astype(int)
