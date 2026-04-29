"""Hidden Markov Model based regime detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover
    GaussianHMM = None

from sklearn.mixture import GaussianMixture


@dataclass
class HMMRegimeResult:
    """Result from HMM regime inference."""

    states: np.ndarray
    model_type: str


class HMMRegimeModel:
    """HMM wrapper with a GaussianMixture fallback."""

    def __init__(
        self,
        n_states: int = 4,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._hmm = None
        self._gmm = None

    def fit(self, features: pd.DataFrame | np.ndarray) -> None:
        """Fit the HMM or fallback model."""
        X = self._prepare_features(features)
        if GaussianHMM is not None:
            self._hmm = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_iter=200,
            )
            self._hmm.fit(X)
        else:
            self._gmm = GaussianMixture(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
            )
            self._gmm.fit(X)

    def predict(self, features: pd.DataFrame | np.ndarray) -> HMMRegimeResult:
        """Predict regime states for each observation."""
        X = self._prepare_features(features)
        if self._hmm is not None:
            states = self._hmm.predict(X)
            return HMMRegimeResult(states=states, model_type="hmmlearn")
        if self._gmm is not None:
            states = self._gmm.predict(X)
            return HMMRegimeResult(states=states, model_type="gmm")
        raise RuntimeError("Regime model has not been fit")

    @staticmethod
    def _prepare_features(features: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            X = features.to_numpy(dtype=float)
        else:
            X = np.asarray(features, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X
