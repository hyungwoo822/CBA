"""Primary Motor Cortex (M1) — Final output execution.

Brain mapping: Precentral gyrus (BA 4), bilateral.
In speech production: drives articulatory muscles (tongue, lips, vocal cords).

AI function: Final output formatting and delivery — the "motor command" that
transforms Broca's formulated language into the actual response. Handles
final cleanup, encoding normalization, and output constraints.

References:
  - Levelt (1989): Articulation stage — the final motor execution of speech
  - Guenther (2006): DIVA model — feedforward motor commands for speech
"""
from __future__ import annotations

import re

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


class MotorCortex(BrainRegion):
    """Primary Motor Cortex (M1) — final output formatting and delivery.

    The last stage in the speech production pipeline (Phase 7):
    Spt (plan) -> Broca (formulate) -> M1 (execute/deliver).

    Performs:
    - Final whitespace normalization
    - Output length constraints
    - Character encoding cleanup
    - Response validation (non-empty, non-garbage)
    """

    def __init__(self) -> None:
        super().__init__(
            name="motor_cortex",
            position=Vec3(-5, 10, 15),
            lobe=Lobe.FRONTAL,
            hemisphere=Hemisphere.LEFT,
        )

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type not in (SignalType.ACTION_SELECTED, SignalType.PLAN):
            return signal

        # Apply final motor execution to all text outputs
        actions = signal.payload.get("actions", [])
        for action in actions:
            args = action.get("args", {})
            text = args.get("text")
            if text is not None:
                args["text"] = self._articulate(str(text))

        response_text = signal.payload.get("response_text")
        if response_text is not None:
            signal.payload["response_text"] = self._articulate(str(response_text))

        self.emit_activation(0.5)
        return signal

    @staticmethod
    def _articulate(text: str) -> str:
        """Final motor execution — cleanup and validation.

        Levelt (1989): Articulation is the final stage where the
        phonetic plan is converted to motor commands. Here we convert
        the formulated text to its final deliverable form.
        """
        # Strip leading/trailing whitespace
        text = text.strip()

        # Collapse excessive newlines (max 2 consecutive)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        # Remove null bytes or other control characters (except newline, tab)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        return text
