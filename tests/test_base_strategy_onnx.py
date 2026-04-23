from datetime import datetime, timedelta

import pandas as pd

from src.strategies.base_strategy import BaseStrategy, SignalResult, StrategyConfig


def _frame(rows: int = 64) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    timestamps = [start + timedelta(seconds=i) for i in range(rows)]
    base = [100.0 + (i * 0.05) for i in range(rows)]

    close = [b + (0.2 if i % 2 == 0 else -0.1) for i, b in enumerate(base)]
    open_ = [c - 0.05 for c in close]
    high = [max(o, c) + 0.08 for o, c in zip(open_, close)]
    low = [min(o, c) - 0.08 for o, c in zip(open_, close)]
    volume = [1000 + ((i % 10) * 25) for i in range(rows)]

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.to_datetime(timestamps),
    )


class _RuleOnlyStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig) -> None:
        self.rule_calls = 0
        super().__init__(config)

    def generate_signal(self, df: pd.DataFrame, i: int, current_price: float) -> SignalResult:
        self.rule_calls += 1
        return SignalResult(
            action="BUY",
            reason="rule_signal",
            confidence=0.85,
            size=0.15,
            price=current_price,
        )


class _FakeModel:
    model_id = "fake_predator_v1"

    def predict(self, feature_vector):
        return {
            "direction": 0.9,
            "urgency": 0.8,
            "size": 0.3,
            "confidence": 0.92,
            "order_type": "market",
        }


class _FaultyModel:
    def predict(self, feature_vector):
        raise RuntimeError("inference_failed")


def test_external_model_drives_intent_when_available() -> None:
    frame = _frame()
    strategy = _RuleOnlyStrategy(
        StrategyConfig(
            strategy_id="onnx_first_strategy",
            symbol="BTC/USDT",
            min_confidence=0.5,
            external_model=_FakeModel(),
        )
    )

    intent = strategy.generate_intent(frame, len(frame) - 1, float(frame["close"].iloc[-1]))

    assert intent is not None
    assert intent.direction.value == "long"
    assert intent.ml_model_id == "fake_predator_v1"
    assert strategy.rule_calls == 0
    assert strategy.get_metrics()["model_signals_generated"] == 1


def test_external_model_failure_falls_back_to_strategy_logic() -> None:
    frame = _frame()
    strategy = _RuleOnlyStrategy(
        StrategyConfig(
            strategy_id="onnx_fallback_strategy",
            symbol="ETH/USDT",
            min_confidence=0.5,
            external_model=_FaultyModel(),
        )
    )

    intent = strategy.generate_intent(frame, len(frame) - 1, float(frame["close"].iloc[-1]))

    assert intent is not None
    assert intent.direction.value == "long"
    assert strategy.rule_calls == 1


def test_legacy_strategy_path_unchanged_without_external_model() -> None:
    frame = _frame()
    strategy = _RuleOnlyStrategy(
        StrategyConfig(
            strategy_id="legacy_only_strategy",
            symbol="SOL/USDT",
            min_confidence=0.5,
        )
    )

    intent = strategy.generate_intent(frame, len(frame) - 1, float(frame["close"].iloc[-1]))

    assert intent is not None
    assert strategy.rule_calls == 1
    assert strategy.get_metrics()["external_model_enabled"] is False
