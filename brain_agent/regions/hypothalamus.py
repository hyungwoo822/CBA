from __future__ import annotations
from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType
from brain_agent.core.neuromodulators import Neuromodulators


class Hypothalamus(BrainRegion):
    """Resource monitoring and neuromodulator control. Spec ref: Section 4.2 Hypothalamus."""

    def __init__(self, neuromodulators: Neuromodulators):
        super().__init__(name="hypothalamus", position=Vec3(0, -10, -15), lobe=Lobe.DIENCEPHALON, hemisphere=Hemisphere.BILATERAL)
        self._neuromod = neuromodulators
        self.pending_requests = 0
        self.staging_count = 0
        self.error_rate = 0.0

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type == SignalType.RESOURCE_STATUS:
            self.pending_requests = signal.payload.get("pending_requests", 0)
            self.staging_count = signal.payload.get("staging_count", 0)
            self.error_rate = signal.payload.get("error_rate", 0.0)
            # Neuromodulator updates are now handled by NeuromodulatorController
            # in the pipeline via on_system_state(). Hypothalamus just reports state.
            self.emit_activation(0.3)
            # Trigger consolidation if memory pressure high
            if self.staging_count > 20:
                return Signal(
                    type=SignalType.CONSOLIDATION_TRIGGER,
                    source=self.name,
                    payload={"staging_count": self.staging_count},
                )
        return None
