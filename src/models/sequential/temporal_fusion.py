"""Temporal Fusion Transformer wrapper (optional)."""

from __future__ import annotations


class TemporalFusionModel:
    """Placeholder wrapper for TFT model training."""

    def __init__(self) -> None:
        try:
            import pytorch_forecasting  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pytorch-forecasting is required for TFT models") from exc

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError("Provide a TimeSeriesDataSet and Trainer to fit TFT")

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Provide a trained TFT model for prediction")
