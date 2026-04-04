"""Ventral Tegmental Area (VTA) — Dopamine source and reward signaling.
Brain mapping: Midbrain VTA, projects to NAcc and PFC (mesolimbic/mesocortical).
AI function: Generates dopaminergic activation proportional to prediction errors;
provides reward/salience signals to guide learning and goal pursuit.
"""
from __future__ import annotations

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


class VentralTegmentalArea(BrainRegion):
    """Ventral Tegmental Area — dopamine source and reward prediction error."""

    def __init__(self):
        super().__init__(
            name="vta",
            position=Vec3(0, -20, -20),
            lobe=Lobe.MIDBRAIN,
            hemisphere=Hemisphere.BILATERAL,
        )

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type == SignalType.PREDICTION_ERROR:
            # High activation proportional to prediction error magnitude
            error_magnitude = abs(float(signal.payload.get("error", 0.0)))
            activation = min(1.0, error_magnitude)
            self.emit_activation(activation)
            signal.metadata["vta_activation"] = self.activation_level
            return signal

        if signal.type == SignalType.ACTION_RESULT:
            # Low tonic activation for completed actions
            self.emit_activation(0.2)
            signal.metadata["vta_activation"] = self.activation_level
            return signal

        return signal
