"""Angular Gyrus — Semantic integration and cross-modal binding.

Brain mapping: Left inferior parietal lobule (BA 39).
Junction of temporal, parietal, and occipital lobes.

Functions:
  - Semantic integration: binding meaning across modalities
  - Reading comprehension: orthographic->semantic mapping
  - Numerical cognition (excluded per user spec)
  - Cross-modal transfer: connecting visual, auditory, linguistic representations

AI agent function: When multiple modalities are present (text + image, text + audio),
integrates them into a unified semantic representation before PFC processing.

References:
  - Ramachandran (2011): Angular gyrus as cross-modal abstraction hub
  - Price (2010): Reading and the angular gyrus
"""
from __future__ import annotations

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal


# Recognised modality keys in signal payloads / metadata
_MODALITY_KEYS = {
    "text": ("text",),
    "visual": ("visual_features", "image_features", "image_data"),
    "auditory": ("prosody", "audio_features", "audio_data"),
}


class AngularGyrus(BrainRegion):
    """Cross-modal semantic integration hub.

    Detects which modalities are present in an incoming signal, extracts
    semantic features from each, and produces a unified
    ``semantic_integration`` metadata block.
    """

    def __init__(self) -> None:
        super().__init__(
            name="angular_gyrus",
            position=Vec3(-35, -25, 30),
            lobe=Lobe.PARIETAL,
            hemisphere=Hemisphere.LEFT,
        )

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _detect_modalities(signal: Signal) -> list[str]:
        """Return list of modality names present in the signal."""
        combined = {**signal.payload, **signal.metadata}
        found: list[str] = []
        for modality, keys in _MODALITY_KEYS.items():
            if any(k in combined and combined[k] for k in keys):
                found.append(modality)
        return sorted(found)  # deterministic order

    @staticmethod
    def _extract_text_semantics(signal: Signal) -> dict:
        """Basic text comprehension features."""
        text = signal.payload.get("text", "")
        return {
            "text_length": len(text),
            "word_count": len(text.split()) if text else 0,
            "text_snippet": text[:120],
        }

    @staticmethod
    def _extract_visual_semantics(signal: Signal) -> dict:
        """Extract visual features from payload/metadata."""
        combined = {**signal.payload, **signal.metadata}
        for key in _MODALITY_KEYS["visual"]:
            if key in combined and combined[key]:
                data = combined[key]
                if isinstance(data, dict):
                    return dict(data)
                return {"raw_visual": data}
        return {}

    @staticmethod
    def _extract_auditory_semantics(signal: Signal) -> dict:
        """Extract auditory/prosodic features from payload/metadata."""
        combined = {**signal.payload, **signal.metadata}
        for key in _MODALITY_KEYS["auditory"]:
            if key in combined and combined[key]:
                data = combined[key]
                if isinstance(data, dict):
                    return dict(data)
                return {"raw_auditory": data}
        return {}

    def _compute_confidence(self, modalities: list[str], binding: dict) -> float:
        """Higher confidence when more modalities agree / are present."""
        n = len(modalities)
        if n == 0:
            return 0.0
        # Base confidence from modality count
        base = min(1.0, 0.4 + 0.2 * n)
        # Boost when binding has richer data
        richness = min(1.0, len(binding) / 10.0)
        return round(min(1.0, base + 0.1 * richness), 4)

    # ── Signal processing ────────────────────────────────────────────

    async def process(self, signal: Signal) -> Signal | None:
        modalities = self._detect_modalities(signal)

        if not modalities:
            # Nothing to integrate
            self.emit_activation(0.05)
            return signal

        # Build cross-modal binding
        binding: dict = {}
        if "text" in modalities:
            binding["text_comprehension"] = self._extract_text_semantics(signal)
        if "visual" in modalities:
            binding["visual_semantics"] = self._extract_visual_semantics(signal)
        if "auditory" in modalities:
            binding["auditory_semantics"] = self._extract_auditory_semantics(signal)

        confidence = self._compute_confidence(modalities, binding)

        signal.metadata["semantic_integration"] = {
            "modalities_present": modalities,
            "cross_modal_binding": binding,
            "integration_confidence": confidence,
        }

        # Higher activation for multi-modal signals
        activation = 0.3 + 0.2 * len(modalities)
        self.emit_activation(min(1.0, activation))

        return signal
