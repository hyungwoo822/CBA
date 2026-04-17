from __future__ import annotations

import math

INTERFERENCE_THRESHOLD = 0.85
INTERFERENCE_FACTOR = 0.3
RIF_FACTOR = 0.15
HOMEOSTATIC_FACTOR = 0.97
PRUNING_THRESHOLD = 0.05


class ForgettingEngine:
    """Implements biologically-inspired forgetting mechanisms.

    Supports exponential decay, retroactive interference,
    retrieval-induced forgetting, and homeostatic scaling with pruning.
    """

    def retention(self, distance: float, strength: float) -> float:
        """Compute retention probability using exponential decay."""
        if strength <= 0:
            return 0.0
        return math.exp(-distance / strength)

    def apply_interference(
        self,
        old_strength: float,
        similarity: float,
        decay_policy: str = "normal",
        never_decay: bool = False,
        importance_score: float = 0.5,
    ) -> float:
        """Reduce strength when a similar competing memory exceeds threshold.

        Phase 6 adds workspace protection and importance scaling. A fully
        important memory loses half as much strength as a baseline memory.
        """
        if never_decay:
            return old_strength
        if decay_policy == "none":
            return old_strength
        if similarity < INTERFERENCE_THRESHOLD:
            return old_strength
        importance_factor = 1.0 - importance_score * 0.5
        base_loss = similarity * INTERFERENCE_FACTOR
        adjusted_loss = base_loss * importance_factor
        return max(0.0, old_strength * (1.0 - adjusted_loss))

    def retrieval_induced_forgetting(self, competitor_strength: float) -> float:
        """Suppress competing memories when a related memory is retrieved."""
        return competitor_strength * (1.0 - RIF_FACTOR)

    def homeostatic_scale(
        self,
        strengths: list[float],
        factor: float = HOMEOSTATIC_FACTOR,
        threshold: float = PRUNING_THRESHOLD,
    ) -> list[float]:
        """Scale all strengths down and prune those below threshold."""
        return [s * factor for s in strengths if s * factor >= threshold]
