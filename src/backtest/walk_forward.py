"""Walk-forward validation splits with optional purging and embargo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class WalkForwardSplit:
    """Indices for one walk-forward split."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


def walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step_size: int,
    gap: int = 0,
    embargo: int = 0,
) -> Iterable[WalkForwardSplit]:
    """Yield walk-forward splits with optional gap and embargo."""
    start = 0
    while start + train_size + gap + test_size <= n_samples:
        train_start = start
        train_end = start + train_size
        test_start = train_end + gap
        test_end = test_start + test_size

        # Apply embargo by shrinking the training end.
        if embargo > 0:
            train_end = max(train_start, train_end - embargo)

        if train_end > train_start:
            yield WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

        start += step_size


def purge_overlap(train_indices: np.ndarray, test_start: int, test_end: int) -> np.ndarray:
    """Remove training indices that overlap test window."""
    return train_indices[(train_indices < test_start) | (train_indices >= test_end)]
