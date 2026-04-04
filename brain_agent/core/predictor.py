"""Predictive Coding — prediction error and surprise computation.

Maintains a prediction of the next input embedding. When actual input
arrives, computes surprise as cosine distance between prediction and
reality. Surprise feeds into ACh (novelty) and DA (prediction error).

References:
  - Friston (2005): A theory of cortical responses (Free Energy)
  - Rao & Ballard (1999): Predictive coding in visual cortex
  - Clark (2013): Whatever Next? Predictive brains, situated agents
"""
from __future__ import annotations

import math


class Predictor:
    """Lightweight prediction error module.

    Stores the predicted embedding for next input and computes
    surprise when actual input arrives. No LLM needed — pure cosine distance.
    """

    def __init__(self) -> None:
        self._prediction: list[float] | None = None

    @property
    def last_prediction(self) -> list[float] | None:
        return self._prediction

    def store_prediction(self, embedding: list[float]) -> None:
        """Store predicted embedding for next input."""
        self._prediction = embedding

    def compute_surprise(self, actual_embedding: list[float]) -> float:
        """Compute surprise as 1 - cosine_similarity(prediction, actual).

        Returns 0.0 (no surprise) to 1.0 (maximum surprise).
        Returns 0.5 (neutral) if no prediction exists.
        """
        if self._prediction is None:
            return 0.5

        sim = self._cosine_sim(self._prediction, actual_embedding)
        surprise = (1.0 - sim) / 2.0
        return max(0.0, min(1.0, surprise))

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        min_len = min(len(a), len(b))
        dot = sum(a[i] * b[i] for i in range(min_len))
        norm_a = math.sqrt(sum(x * x for x in a[:min_len]))
        norm_b = math.sqrt(sum(x * x for x in b[:min_len]))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
