"""Posterior Superior Temporal Sulcus (pSTS) — Multisensory integration hub.

Brain mapping: Posterior bank of the superior temporal sulcus, bilateral
(typically stronger left hemisphere for language-dominant binding).

AI function: Merges ventral and dorsal stream outputs into a unified
multisensory representation before attention gating and memory encoding.

References:
  - Beauchamp et al. (2004): Audiovisual integration in pSTS
  - Wheeler et al. (2000): Multisensory memory retrieval reactivates
    original perceptual patterns — pSTS as binding hub
  - Calvert et al. (2000): pSTS response to congruent audiovisual stimuli
"""
from __future__ import annotations

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal


class PosteriorSuperiorTemporalSulcus(BrainRegion):
    """pSTS — multisensory integration and binding.

    Merges the outputs from dual-stream processing:
      - Ventral visual ("what"): object/content recognition from IT
      - Ventral auditory ("what"): semantic analysis from Wernicke
      - Dorsal auditory ("how"): production plan from Spt
      - Dorsal visual ("where/how"): spatial info from parietal

    Produces a unified multisensory representation with:
      - cross_modal_binding: merged features from all streams
      - congruence_score: how well streams agree
      - dominant_modality: which stream carries most information
    """

    def __init__(self) -> None:
        super().__init__(
            name="psts",
            position=Vec3(-12, -3, 3),
            lobe=Lobe.TEMPORAL,
            hemisphere=Hemisphere.LEFT,
        )

    async def process(self, signal: Signal) -> Signal | None:
        return self.integrate(signal)

    def integrate(self, signal: Signal) -> Signal:
        """Merge ventral and dorsal stream outputs.

        Replaces AngularGyrus's minimal aggregation with genuine
        multisensory binding that detects cross-modal congruence.
        """
        # Collect stream outputs
        ventral = signal.metadata.get("ventral_result", {})
        dorsal = signal.metadata.get("dorsal_result", {})
        comprehension = signal.payload.get("comprehension", {})
        visual_features = signal.payload.get("visual_features", {})
        prosody = signal.payload.get("prosody", {})

        # Detect active modalities
        modalities: list[str] = []
        binding: dict = {}

        # Auditory ventral (Wernicke comprehension)
        if comprehension:
            modalities.append("auditory_ventral")
            binding["semantic"] = {
                "intent": comprehension.get("intent", "statement"),
                "keywords": comprehension.get("keywords", []),
                "complexity": comprehension.get("complexity", "simple"),
                "discourse_type": comprehension.get("discourse_type"),
                "semantic_roles": comprehension.get("semantic_roles"),
            }

        # Visual ventral (object/content recognition)
        if visual_features:
            modalities.append("visual_ventral")
            binding["visual"] = (
                dict(visual_features) if isinstance(visual_features, dict)
                else {"raw": visual_features}
            )

        # Auditory dorsal (prosody / production mapping from Spt)
        if prosody:
            modalities.append("auditory_dorsal")
            binding["prosodic"] = (
                dict(prosody) if isinstance(prosody, dict)
                else {"raw": prosody}
            )

        # Dorsal stream production plan
        production_plan = dorsal.get("production_plan") or signal.metadata.get("production_plan")
        if production_plan:
            modalities.append("dorsal_motor")
            binding["motor_plan"] = production_plan

        # Text content (always present as base)
        text = signal.payload.get("text", "")
        if text and "auditory_ventral" not in modalities:
            modalities.append("text")
            binding["text"] = {"content": text[:200], "length": len(text)}

        # Compute cross-modal congruence (Calvert et al. 2000)
        congruence = self._compute_congruence(binding)

        # Determine dominant modality
        dominant = self._determine_dominant(modalities, binding)

        signal.metadata["multisensory_binding"] = {
            "modalities": modalities,
            "binding": binding,
            "congruence_score": congruence,
            "dominant_modality": dominant,
            "num_streams": len(modalities),
        }

        # Activation proportional to integration richness
        activation = min(1.0, 0.3 + 0.15 * len(modalities))
        self.emit_activation(activation)
        return signal

    @staticmethod
    def _compute_congruence(binding: dict) -> float:
        """Estimate cross-modal congruence.

        Higher when multiple streams carry consistent information.
        Calvert et al. (2000): pSTS responds superadditively to
        congruent audiovisual stimuli.
        """
        n = len(binding)
        if n <= 1:
            return 0.5  # Single modality — no conflict, no boost

        # Base congruence from modality count
        score = min(1.0, 0.4 + 0.2 * n)

        # Bonus: if semantic keywords overlap with visual description
        semantic = binding.get("semantic", {})
        visual = binding.get("visual", {})
        if semantic and visual:
            kw = set(semantic.get("keywords", []))
            vis_desc = str(visual.get("description", "")).lower()
            overlap = sum(1 for k in kw if k in vis_desc)
            if overlap > 0:
                score = min(1.0, score + 0.1 * overlap)

        return round(score, 3)

    @staticmethod
    def _determine_dominant(modalities: list[str], binding: dict) -> str:
        """Determine which modality carries the most information."""
        if not modalities:
            return "none"

        # Prefer semantic (Wernicke) when available — language-dominant agent
        if "auditory_ventral" in modalities:
            return "auditory_ventral"
        if "visual_ventral" in modalities:
            return "visual_ventral"
        if "text" in modalities:
            return "text"
        return modalities[0]
