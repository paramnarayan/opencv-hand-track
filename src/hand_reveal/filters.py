from __future__ import annotations

import math
import time

import numpy as np


class OneEuroFilter:
    """Adaptive low-pass filter based on Casiez et al., CHI 2012."""

    def __init__(self, min_cutoff: float = 2.0, beta: float = 0.5, d_cutoff: float = 1.0):
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("Filter cutoffs must be positive")
        if beta < 0:
            raise ValueError("Filter beta cannot be negative")

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._raw_previous: np.ndarray | None = None
        self._filtered: np.ndarray | None = None
        self._filtered_derivative: np.ndarray | None = None
        self._timestamp: float | None = None

    @staticmethod
    def _alpha(cutoff: float | np.ndarray, dt: float) -> float | np.ndarray:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def smooth(self, points: np.ndarray, timestamp: float | None = None) -> np.ndarray:
        now = time.monotonic() if timestamp is None else timestamp
        raw = np.asarray(points, dtype=np.float64)

        if self._filtered is None:
            self._raw_previous = raw.copy()
            self._filtered = raw.copy()
            self._filtered_derivative = np.zeros_like(raw)
            self._timestamp = now
            return raw.astype(np.float32)

        assert self._raw_previous is not None
        assert self._filtered_derivative is not None
        assert self._timestamp is not None

        dt = max(now - self._timestamp, 1e-6)
        derivative = (raw - self._raw_previous) / dt
        derivative_alpha = self._alpha(self.d_cutoff, dt)
        self._filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self._filtered_derivative
        )

        cutoff = self.min_cutoff + self.beta * np.abs(self._filtered_derivative)
        position_alpha = self._alpha(cutoff, dt)
        self._filtered = position_alpha * raw + (1.0 - position_alpha) * self._filtered
        self._raw_previous = raw.copy()
        self._timestamp = now
        return self._filtered.astype(np.float32)

    def reset(self) -> None:
        self._raw_previous = None
        self._filtered = None
        self._filtered_derivative = None
        self._timestamp = None
