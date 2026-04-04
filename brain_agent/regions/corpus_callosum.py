"""Corpus Callosum — Inter-hemispheric communication.

Brain mapping: Largest white matter tract connecting left and right hemispheres.
~200 million axons transferring information between homologous cortical areas.

AI agent function: Integrates results from left-hemisphere (analytical/sequential)
and right-hemisphere (holistic/emotional) processing into unified representations.

References:
  - Gazzaniga (2005): Split-brain studies
  - Schulte & Mueller (2005): Callosal transfer
"""
from __future__ import annotations

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal


class CorpusCallosum(BrainRegion):
    """Inter-hemispheric integration bridge.

    Merges left (analytical/sequential) and right (holistic/emotional)
    processing results into a unified representation.
    """

    def __init__(self) -> None:
        super().__init__(
            name="corpus_callosum",
            position=Vec3(0, 15, 5),
            lobe=Lobe.SUBCORTICAL,
            hemisphere=Hemisphere.BILATERAL,
        )
        self.transfer_count: int = 0

    # ── Core integration logic ───────────────────────────────────────

    def integrate(self, left_result: dict, right_result: dict) -> dict:
        """Merge left-hemisphere (analytical) and right-hemisphere (emotional/holistic) results.

        Conflict resolution: when both hemispheres provide a value for the
        same key, the hemisphere with higher ``confidence`` wins.  If
        confidence is equal the left (analytical) result is preferred as a
        tie-breaker.
        """
        self.transfer_count += 1

        left_conf = left_result.get("confidence", 0.5)
        right_conf = right_result.get("confidence", 0.5)

        unified: dict = {}

        # Start with all keys from both sides
        all_keys = set(left_result.keys()) | set(right_result.keys())

        for key in all_keys:
            in_left = key in left_result
            in_right = key in right_result

            if in_left and not in_right:
                unified[key] = left_result[key]
            elif in_right and not in_left:
                unified[key] = right_result[key]
            else:
                # Both hemispheres provide a value — resolve conflict
                if left_result[key] == right_result[key]:
                    unified[key] = left_result[key]
                elif left_conf >= right_conf:
                    unified[key] = left_result[key]
                else:
                    unified[key] = right_result[key]

        # Always preserve both perspectives for downstream consumers
        unified["left_perspective"] = left_result
        unified["right_perspective"] = right_result
        unified["integration_source"] = "corpus_callosum"

        return unified

    # ── Signal processing ────────────────────────────────────────────

    async def process(self, signal: Signal) -> Signal | None:
        left_data = signal.metadata.get("left_result")
        right_data = signal.metadata.get("right_result")

        if left_data is not None and right_data is not None:
            unified = self.integrate(left_data, right_data)
            signal.metadata["integrated_result"] = unified
            self.emit_activation(0.8)
        else:
            # Nothing to integrate — pass through with low activation
            self.emit_activation(0.1)

        return signal
