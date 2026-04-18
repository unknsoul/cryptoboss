"""AI optimizer service for v3 intraday scalper architecture."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .config import AIOptimizerConfig

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None


class AIOptimizer:
    """Optimizes parameters and estimates trade success probabilities."""

    def __init__(self, config: Optional[AIOptimizerConfig] = None):
        self.config = config or AIOptimizerConfig()
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.trained = False
        self._initialize_models()

    def _initialize_models(self) -> None:
        requested = set(self.config.models)

        if "xgboost" in requested and xgb is not None:
            self.models["xgboost"] = xgb.XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
                eval_metric="logloss",
            )

        if "lightgbm" in requested and lgb is not None:
            self.models["lightgbm"] = lgb.LGBMClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )

        if "neural_network" in requested:
            self.models["neural_network"] = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                max_iter=400,
                random_state=42,
            )

        if not self.models:
            self.models["fallback_logistic"] = LogisticRegression(max_iter=1000)

        weight = 1.0 / len(self.models)
        self.weights = {name: weight for name in self.models}

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, object]:
        X = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)
        y = labels.values if isinstance(labels, pd.Series) else np.asarray(labels)

        trained_models: List[str] = []
        failed_models: Dict[str, str] = {}

        for name, model in self.models.items():
            try:
                model.fit(X, y)
                trained_models.append(name)
            except Exception as error:
                failed_models[name] = str(error)

        self.trained = bool(trained_models)

        if self.trained:
            weight = 1.0 / len(trained_models)
            self.weights = {name: weight for name in trained_models}

        return {
            "trained": self.trained,
            "models": trained_models,
            "failed_models": failed_models,
            "purposes": self.config.purposes,
        }

    def predict_trade_success_probability(self, feature_row: Iterable[float]) -> float:
        if not self.trained:
            return 0.5

        X = np.asarray(list(feature_row), dtype=float).reshape(1, -1)

        probs: List[float] = []
        used_weights: List[float] = []

        for name, model in self.models.items():
            if name not in self.weights:
                continue

            try:
                proba = model.predict_proba(X)[0]
                classes = list(getattr(model, "classes_", range(len(proba))))
                if 1 in classes:
                    success_prob = float(proba[classes.index(1)])
                elif len(proba) > 1:
                    success_prob = float(proba[-1])
                else:
                    success_prob = float(proba[0])

                probs.append(success_prob)
                used_weights.append(self.weights[name])
            except Exception:
                continue

        if not probs:
            return 0.5

        weights = np.asarray(used_weights, dtype=float)
        weights = weights / weights.sum()
        return float(np.dot(np.asarray(probs), weights))

    def filter_false_signals(
        self,
        signals: List[Dict[str, Any]],
        feature_matrix: pd.DataFrame,
        threshold: float = 0.55,
    ) -> List[Dict[str, Any]]:
        if not signals:
            return []

        if len(signals) != len(feature_matrix):
            raise ValueError("signals length must match feature rows")

        filtered: List[Dict[str, Any]] = []
        for idx, signal in enumerate(signals):
            row = feature_matrix.iloc[idx].values
            probability = self.predict_trade_success_probability(row)
            if probability >= threshold:
                enriched = dict(signal)
                enriched["success_probability"] = probability
                filtered.append(enriched)

        return filtered

    def optimize_parameters(
        self,
        evaluator: Callable[[Dict[str, Any]], float],
        parameter_space: Dict[str, List[Any]],
        n_trials: int = 30,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        if not parameter_space:
            return {"best_params": {}, "best_score": 0.0, "trials": []}

        rng = np.random.default_rng(random_state)
        best_params: Dict[str, Any] = {}
        best_score = float("-inf")
        trials: List[Dict[str, Any]] = []

        keys = list(parameter_space.keys())
        for _ in range(n_trials):
            params = {key: rng.choice(parameter_space[key]) for key in keys}
            score = float(evaluator(params))
            trial = {"params": params, "score": score}
            trials.append(trial)

            if score > best_score:
                best_score = score
                best_params = dict(params)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "trials": trials,
            "purpose": "optimize_parameters",
        }
