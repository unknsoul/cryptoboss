"""Load per-regime adaptive parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml


DEFAULT_PARAMS: Dict[str, Dict] = {
    "TRENDING_BULL": {
        "strategy_weights": {"momentum": 0.6, "smc": 0.3, "mean_reversion": 0.1},
        "risk_per_trade": 0.02,
        "max_open_trades": 5,
    },
    "TRENDING_BEAR": {
        "strategy_weights": {"momentum": 0.5, "smc": 0.4, "mean_reversion": 0.1},
        "risk_per_trade": 0.015,
        "max_open_trades": 3,
    },
    "RANGING": {
        "strategy_weights": {"momentum": 0.1, "smc": 0.2, "mean_reversion": 0.7},
        "risk_per_trade": 0.01,
        "max_open_trades": 4,
    },
    "HIGH_VOL": {
        "strategy_weights": {"momentum": 0.2, "smc": 0.5, "mean_reversion": 0.3},
        "risk_per_trade": 0.005,
        "max_open_trades": 2,
    },
    "CRISIS": {
        "strategy_weights": {},
        "risk_per_trade": 0.0,
        "max_open_trades": 0,
    },
}


def load_regime_params(path: str = "configs/regime_params.yaml") -> Dict[str, Dict]:
    """Load regime parameters from YAML, falling back to defaults."""
    file_path = Path(path)
    if not file_path.exists():
        return DEFAULT_PARAMS.copy()

    with open(file_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    merged = DEFAULT_PARAMS.copy()
    for key, value in data.items():
        merged[key] = value
    return merged


def get_regime_params(regime: str, config: Dict[str, Dict]) -> Dict:
    """Return parameters for a specific regime."""
    return config.get(regime, DEFAULT_PARAMS.get(regime, {}))
