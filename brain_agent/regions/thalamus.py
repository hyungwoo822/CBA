"""Thalamus — Sensory relay nucleus.

Brain mapping: Bilateral diencephalon, central relay hub.

The thalamus is NOT a classifier — it is a relay station. All sensory
information (except olfaction) passes through thalamic nuclei before
reaching cortex. The thalamus gates and relays, it does not interpret.

  - LGN (lateral geniculate nucleus): relays visual input to V1
  - MGN (medial geniculate nucleus): relays auditory input to A1
  - VPL/VPM: relays somatosensory input

Classification and interpretation are cortical functions (Wernicke for
language, Amygdala for emotion, PFC for reasoning).

References:
  - Sherman & Guillery (2006): Exploring the Thalamus and Its Role
    in Cortical Function — thalamus as relay, not processor.
  - Sherman (2007): The thalamus is more than just a relay.
"""
from __future__ import annotations

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType


class Thalamus(BrainRegion):
    """Thalamic relay — preprocesses and gates sensory input.

    Strips whitespace, tags modality, and passes through.
    Does NOT classify intent or content — that is cortical work
    (Wernicke, Amygdala, PFC).
    """

    def __init__(self):
        super().__init__(
            name="thalamus",
            position=Vec3(0, 0, 0),
            lobe=Lobe.DIENCEPHALON,
            hemisphere=Hemisphere.BILATERAL,
        )

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type in (
            SignalType.EXTERNAL_INPUT,
            SignalType.TEXT_INPUT,
            SignalType.IMAGE_INPUT,
            SignalType.AUDIO_INPUT,
        ):
            # Relay preprocessing: normalize input, tag modality
            text = signal.payload.get("text", "")
            signal.payload["text"] = text.strip()
            signal.payload["modality"] = signal.payload.get("modality", "text")
            signal.payload["word_count"] = len(text.split()) if text.strip() else 0
            signal.payload["has_question"] = "?" in text

            self.emit_activation(0.6)
            return signal

        return signal

    async def process_with_attention(
        self,
        signal: Signal,
        goal_embedding: list[float] | None = None,
        current_arousal: float = 0.0,
    ) -> Signal | None:
        """Thalamic reticular nucleus attention gating (McAlonan et al. 2008).

        Dual attention streams:
          - Top-down (dorsal): goal relevance via keyword overlap (Corbetta & Shulman 2002)
          - Bottom-up (ventral): salience via arousal

        Does NOT block signals — assigns attention_weight for downstream use.
        """
        signal = await self.process(signal)
        if signal is None:
            return signal

        text = signal.payload.get("text", "")
        words = set(text.lower().split())

        # Top-down: keyword overlap with current goals
        goal_keywords = set(signal.metadata.get("goal_keywords", []))
        if goal_keywords and words:
            overlap = len(words & goal_keywords) / max(len(goal_keywords), 1)
            top_down = min(1.0, overlap * 2.0)
        else:
            top_down = 0.3  # Default moderate relevance

        # Bottom-up: arousal-driven salience
        arousal = current_arousal
        if signal.emotional_tag:
            arousal = max(arousal, signal.emotional_tag.arousal)
        bottom_up = arousal

        attention_weight = top_down * 0.5 + bottom_up * 0.5
        signal.metadata["attention_weight"] = round(min(1.0, attention_weight), 3)
        return signal
