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

    def apply_interference(self, old_strength: float, similarity: float) -> float:
        """Reduce strength when a similar competing memory exceeds threshold."""
        if similarity < INTERFERENCE_THRESHOLD:
            return old_strength
        return max(0.0, old_strength * (1.0 - similarity * INTERFERENCE_FACTOR))

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
