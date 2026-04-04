"""Brainstem — Arousal regulation and consciousness state management.
Brain mapping: Reticular activating system (RAS) in the brainstem.
AI function: Tracks system arousal state (awake/drowsy/sleep) based on
input activity; gates signal processing based on consciousness level.
"""
from __future__ import annotations

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


class Brainstem(BrainRegion):
    """Brainstem — arousal regulation via reticular activating system."""

    def __init__(self):
        super().__init__(
            name="brainstem",
            position=Vec3(0, -30, -25),
            lobe=Lobe.BRAINSTEM,
            hemisphere=Hemisphere.BILATERAL,
        )
        self.arousal_state: str = "awake"
        self._idle_count: int = 0

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type in (SignalType.EXTERNAL_INPUT, SignalType.TEXT_INPUT, SignalType.IMAGE_INPUT, SignalType.AUDIO_INPUT):
            # Reticular activating system — external input wakes the system
            self.arousal_state = "awake"
            self._idle_count = 0
            self.emit_activation(0.8)
            signal.metadata["arousal_state"] = self.arousal_state
            return signal

        if signal.type == SignalType.RESOURCE_STATUS:
            # Track idle periods and transition arousal states
            self._idle_count += 1
            if self._idle_count > 10:
                self.arousal_state = "sleep"
                self.emit_activation(0.1)
            elif self._idle_count > 5:
                self.arousal_state = "drowsy"
                self.emit_activation(0.3)
            else:
                self.emit_activation(0.5)

            signal.metadata["arousal_state"] = self.arousal_state
            return signal

        # Pass through other signals with current arousal state
        signal.metadata["arousal_state"] = self.arousal_state
        return signal
