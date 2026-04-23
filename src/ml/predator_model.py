"""Predator model adapter for ONNX/TensorRT-compatible inference artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None


logger = logging.getLogger(__name__)


class PredatorModel:
    """
    Lightweight runtime adapter for externally-trained ONNX models.

    The model is expected to output either:
    - Named heads: direction, urgency, size, confidence (recommended), or
    - A single vector where the first three values map to direction/urgency/size.
    """

    def __init__(
        self,
        model_path: str,
        input_name: Optional[str] = None,
        output_names: Optional[Sequence[str]] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_name = input_name
        self.output_names = list(output_names) if output_names else []
        self.providers = list(providers) if providers else []

        self.model_id = self.model_path.name
        self._session: Optional[Any] = None
        self._resolved_outputs: List[str] = []

        self._load()

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    def _load(self) -> None:
        if ort is None:
            logger.warning(
                "onnxruntime is not installed; PredatorModel disabled for %s",
                self.model_path,
            )
            return

        if not self.model_path.exists():
            logger.warning("ONNX model file not found: %s", self.model_path)
            return

        try:
            providers = self.providers or ort.get_available_providers()
            self._session = ort.InferenceSession(str(self.model_path), providers=providers)

            if not self.input_name:
                self.input_name = self._session.get_inputs()[0].name

            if self.output_names:
                self._resolved_outputs = list(self.output_names)
            else:
                self._resolved_outputs = [out.name for out in self._session.get_outputs()]

            logger.info(
                "PredatorModel loaded: %s (providers=%s)",
                self.model_path,
                providers,
            )
        except Exception as exc:
            logger.error("Failed to initialize PredatorModel %s: %s", self.model_path, exc)
            self._session = None

    def predict(self, feature_vector: Sequence[float]) -> Dict[str, Any]:
        """Run one inference pass and normalize output heads."""
        if not self._session or not self.input_name:
            return {
                "ready": False,
                "error": "session_unavailable",
            }

        tensor = np.asarray(feature_vector, dtype=np.float32)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)

        outputs = self._session.run(self._resolved_outputs or None, {self.input_name: tensor})
        names = self._resolved_outputs or [f"output_{idx}" for idx in range(len(outputs))]
        output_map = {name: value for name, value in zip(names, outputs)}

        normalized = self._normalize_heads(output_map, outputs)
        normalized["ready"] = True
        normalized["model_id"] = self.model_id
        return normalized

    def _normalize_heads(self, output_map: Dict[str, Any], outputs: List[Any]) -> Dict[str, Any]:
        """Map model outputs to direction/urgency/size/confidence semantics."""
        result: Dict[str, Any] = {
            "direction": 0.0,
            "urgency": 0.0,
            "size": 0.0,
            "confidence": 0.0,
            "order_type": "limit",
        }

        lowered = {name.lower(): value for name, value in output_map.items()}

        direction = self._first_scalar(
            lowered.get("direction")
            or lowered.get("direction_head")
            or lowered.get("direction_score")
        )
        urgency = self._first_scalar(
            lowered.get("urgency")
            or lowered.get("urgency_head")
            or lowered.get("aggressiveness")
        )
        size = self._first_scalar(
            lowered.get("size")
            or lowered.get("size_head")
            or lowered.get("position_size")
        )
        confidence = self._first_scalar(
            lowered.get("confidence")
            or lowered.get("confidence_head")
            or lowered.get("probability")
        )

        if direction is None or urgency is None or size is None:
            if outputs:
                flat = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
                if flat.size >= 1 and direction is None:
                    direction = float(flat[0])
                if flat.size >= 2 and urgency is None:
                    urgency = float(flat[1])
                if flat.size >= 3 and size is None:
                    size = float(flat[2])
                if flat.size >= 4 and confidence is None:
                    confidence = float(flat[3])

        direction = float(direction or 0.0)
        urgency = float(np.clip(urgency or 0.0, 0.0, 1.0))
        size = max(0.0, float(size or 0.0))

        if confidence is None:
            confidence = min(1.0, max(0.0, abs(direction) * max(urgency, 0.1)))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        result["direction"] = direction
        result["urgency"] = urgency
        result["size"] = size
        result["confidence"] = confidence

        if urgency > 0.7:
            result["order_type"] = "market"
        elif urgency > 0.3:
            result["order_type"] = "aggressive_limit"

        return result

    @staticmethod
    def _first_scalar(value: Any) -> Optional[float]:
        if value is None:
            return None

        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None

        return float(arr[0])
