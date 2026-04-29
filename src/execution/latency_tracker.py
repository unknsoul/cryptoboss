"""Execution latency and slippage tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SlippageSample:
    """Single slippage measurement."""

    expected_fill: float
    actual_fill: float

    @property
    def slippage_bps(self) -> float:
        if self.expected_fill == 0:
            return 0.0
        return (self.actual_fill - self.expected_fill) / self.expected_fill * 10000


class LatencyTracker:
    """Track slippage across orders."""

    def __init__(self) -> None:
        self.samples: List[SlippageSample] = []

    def record(self, expected_fill: float, actual_fill: float) -> float:
        sample = SlippageSample(expected_fill=expected_fill, actual_fill=actual_fill)
        self.samples.append(sample)
        return sample.slippage_bps

    def average_slippage_bps(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.slippage_bps for s in self.samples) / len(self.samples)
