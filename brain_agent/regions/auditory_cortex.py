"""Auditory Cortex — Audio input processing.
Brain mapping: Temporal lobe, superior temporal gyrus.
AI function: Left hemisphere processes speech/language (A1->Wernicke pathway)
             via Whisper STT. Right hemisphere processes prosody/emotional tone.

References:
  - Hickok & Poeppel (2007): Dual-stream model, A1 is entry point
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal

if TYPE_CHECKING:
    from brain_agent.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class AuditoryCortexLeft(BrainRegion):
    """Left auditory cortex — speech processing via Whisper STT.

    When raw audio bytes are present and no transcript exists,
    calls Whisper STT API to transcribe speech to text.
    """

    def __init__(self, llm_provider: LLMProvider | None = None):
        super().__init__(
            name="auditory_cortex_left",
            position=Vec3(-35, -10, 10),
            lobe=Lobe.TEMPORAL,
            hemisphere=Hemisphere.LEFT,
            llm_provider=llm_provider,
        )

    async def process(self, signal: Signal) -> Signal | None:
        audio_data = signal.payload.get("audio_data")
        if audio_data is None:
            return signal

        transcript = signal.payload.get("transcript", "")

        # If no transcript and we have raw audio + LLM provider, transcribe
        if not transcript and isinstance(audio_data, (bytes, bytearray)) and self.llm_provider:
            transcript = await self._transcribe(audio_data)

        if transcript:
            signal.payload["text"] = transcript
            signal.payload["transcript"] = transcript

        signal.payload["modality"] = "audio"
        self.emit_activation(0.7 if transcript else 0.4)
        return signal

    async def _transcribe(self, audio_data: bytes) -> str:
        """Call Whisper STT API for speech-to-text."""
        try:
            import litellm

            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            try:
                with open(temp_path, "rb") as audio_file:
                    response = await litellm.atranscription(
                        model="whisper-1",
                        file=audio_file,
                    )
                return response.text or ""
            finally:
                os.unlink(temp_path)
        except Exception as e:
            logger.warning("A1 STT transcription failed: %s", e)
            return ""


class AuditoryCortexRight(BrainRegion):
    """Right auditory cortex — prosody and emotional tone processing."""

    def __init__(self):
        super().__init__(
            name="auditory_cortex_right",
            position=Vec3(35, -10, 10),
            lobe=Lobe.TEMPORAL,
            hemisphere=Hemisphere.RIGHT,
        )

    async def process(self, signal: Signal) -> Signal | None:
        audio_data = signal.payload.get("audio_data")
        if audio_data is None:
            return signal

        emotional_tone = signal.payload.get("emotional_tone", "neutral")
        stress_level = signal.payload.get("stress_level", 0.0)

        signal.payload["emotional_tone"] = emotional_tone
        signal.payload["stress_level"] = max(0.0, min(1.0, float(stress_level)))
        signal.payload["modality"] = "audio"

        # Compute basic audio features if raw bytes
        if isinstance(audio_data, (bytes, bytearray)):
            signal.payload["audio_features"] = {
                "size_bytes": len(audio_data),
                "duration_estimate": round(len(audio_data) / 16000, 1),  # rough estimate
            }

        self.emit_activation(0.5 + float(stress_level) * 0.4)
        return signal
