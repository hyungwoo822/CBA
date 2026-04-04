"""Insula — interoceptive monitoring and emotional awareness.

Brain mapping: Insular cortex (anterior and posterior), bilateral.

AI function: Monitors internal body state (neuromodulator levels) and
computes interoceptive signals — stress, energy, emotional awareness,
and risk sensitivity. Feeds into ACC and PFC for decision-making.

References:
  - Craig (2009): How Do You Feel — Now? The Anterior Insula and Human Awareness
  - Critchley et al. (2004): Neural systems supporting interoceptive awareness
  - Singer et al. (2009): Anterior insula integrates interoception with emotion
  - Paulus & Stein (2006): Insula and risk/uncertainty processing
"""
from __future__ import annotations

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal
from brain_agent.core.neuromodulators import Neuromodulators


class Insula(BrainRegion):
    """Insular cortex — interoceptive monitoring and body-state awareness."""

    def __init__(self, neuromodulators: Neuromodulators) -> None:
        super().__init__(
            name="insula",
            position=Vec3(35, 10, 5),
            lobe=Lobe.INSULAR,
            hemisphere=Hemisphere.BILATERAL,
        )
        self._nm = neuromodulators

    async def process(self, signal: Signal) -> Signal:
        nm = self._nm
        arousal = signal.emotional_tag.arousal if signal.emotional_tag else 0.0
        valence = signal.emotional_tag.valence if signal.emotional_tag else 0.0

        # Stress level: cortisol + epinephrine weighted (Critchley 2004)
        stress_level = nm.cortisol * 0.6 + nm.epinephrine * 0.4

        # Energy level: DA energizes, CORT depletes, NE alerts
        energy_level = max(0.0, min(1.0,
            nm.dopamine * 0.4 + (1.0 - nm.cortisol) * 0.3 + nm.norepinephrine * 0.3
        ))

        # Emotional awareness: arousal amplifies, stress sharpens (Craig 2009)
        emotional_awareness = min(1.0, arousal * 0.5 + abs(valence) * 0.3 + stress_level * 0.2)

        # Risk sensitivity: high stress + low GABA + negative valence (Paulus & Stein 2006)
        gaba_deficit = max(0.0, 0.5 - nm.gaba)
        risk_sensitivity = min(1.0,
            stress_level * 0.4 + gaba_deficit * 0.3 + max(0.0, -valence) * 0.3
        )

        activation = min(1.0, stress_level * 0.4 + arousal * 0.3 + abs(valence) * 0.3)
        self.emit_activation(activation)

        signal.metadata["interoceptive_state"] = {
            "stress_level": round(stress_level, 3),
            "energy_level": round(energy_level, 3),
            "emotional_awareness": round(emotional_awareness, 3),
            "risk_sensitivity": round(risk_sensitivity, 3),
        }
        return signal
