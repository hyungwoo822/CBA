from __future__ import annotations

import math
from dataclasses import dataclass

RECENCY_DECAY_CONSTANT = 10.0


@dataclass
class RetrievalConfig:
    """Weights for the multi-factor retrieval scoring function.

    alpha  -- recency
    beta   -- relevance
    gamma  -- importance
    delta  -- frequency (access count)
    epsilon -- context similarity
    """

    alpha: float = 0.20
    beta: float = 0.25
    gamma: float = 0.20
    delta: float = 0.10
    epsilon: float = 0.15
    zeta: float = 0.10


class RetrievalEngine:
    """Computes composite retrieval scores for memory candidates.

    Combines recency, relevance, importance, frequency, and context
    similarity into a single score using configurable weights.
    """

    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()

    def compute_score(
        self,
        recency_distance: float,
        relevance: float,
        importance: float,
        access_count: int,
        context_similarity: float,
        activation_boost: float = 0.0,
    ) -> float:
        """Return a weighted composite retrieval score."""
        c = self.config
        recency = math.exp(-recency_distance / RECENCY_DECAY_CONSTANT)
        frequency = min(1.0, math.log(access_count + 1) / 4.6)
        score = (
            c.alpha * recency
            + c.beta * max(0.0, relevance)
            + c.gamma * max(0.0, min(1.0, importance))
            + c.delta * frequency
            + c.epsilon * max(0.0, context_similarity)
            + c.zeta * max(0.0, min(1.0, activation_boost))
        )

        # Pattern completion (CA3, Rolls 2013): nonlinear boost
        # when context similarity exceeds threshold — partial cues
        # trigger full memory reconstruction
        completion_threshold = 0.7
        if context_similarity > completion_threshold:
            excess = context_similarity - completion_threshold
            completion_bonus = excess * excess * 2.0  # Quadratic boost
            score += completion_bonus

        return score
