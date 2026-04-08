"""Multi-factor memory retrieval scoring.

Neuroscience: extended from Park et al. (2023) Generative Agents.
Added confidence_bonus based on Signal Detection Theory (Green & Swets 1966).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RECENCY_DECAY_CONSTANT = 10.0

CONFIDENCE_SCORES = {
    "EXTRACTED": 1.0,
    "INFERRED": 0.6,
    "AMBIGUOUS": 0.2,
}


@dataclass
class RetrievalConfig:
    alpha: float = 0.18     # recency
    beta: float = 0.23      # relevance
    gamma: float = 0.18     # importance
    delta: float = 0.09     # frequency
    epsilon: float = 0.14   # context_match
    zeta: float = 0.09      # activation_boost
    eta: float = 0.09       # confidence_bonus


class RetrievalEngine:
    """Score memory candidates for retrieval.

    Seven factors weighted to sum to 1.0.
    CA3 pattern completion bonus for high context similarity (Rolls 2013).
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
        confidence_bonus: float = 0.0,
    ) -> float:
        cfg = self.config
        recency = math.exp(-recency_distance / RECENCY_DECAY_CONSTANT)
        frequency = min(1.0, math.log(access_count + 1) / 4.6)

        score = (
            cfg.alpha * recency
            + cfg.beta * max(0.0, relevance)
            + cfg.gamma * max(0.0, min(1.0, importance))
            + cfg.delta * frequency
            + cfg.epsilon * max(0.0, context_similarity)
            + cfg.zeta * max(0.0, min(1.0, activation_boost))
            + cfg.eta * max(0.0, min(1.0, confidence_bonus))
        )

        # Pattern completion bonus (CA3, Rolls 2013)
        if context_similarity > 0.7:
            excess = context_similarity - 0.7
            score += excess * excess * 2.0

        return score
