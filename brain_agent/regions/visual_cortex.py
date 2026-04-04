"""Visual Cortex — Image input processing via Vision LLM.
Brain mapping: Occipital lobe (V1->V2->ventral stream).
AI function: Calls vision-capable LLM to extract scene description,
objects, spatial layout, and text/OCR from images.

References:
  - Hubel & Wiesel (1959): V1 feature extraction
  - Ungerleider & Mishkin (1982): Ventral "what" stream
"""
from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType

if TYPE_CHECKING:
    from brain_agent.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_VISION_SYSTEM_PROMPT = """\
You are V1 (primary visual cortex) in a brain-inspired AI system.
Analyze the image and return a JSON object:
{
  "description": "detailed scene description",
  "objects": ["object1", "object2"],
  "text_content": "any text/writing visible in the image",
  "spatial_layout": "brief spatial description",
  "emotional_tone": "neutral|positive|negative|threatening|calming"
}
Return ONLY valid JSON."""


class VisualCortex(BrainRegion):
    """Primary visual cortex — processes image inputs via vision LLM."""

    def __init__(self, llm_provider: LLMProvider | None = None, vision_model: str = ""):
        super().__init__(
            name="visual_cortex",
            position=Vec3(0, -40, -10),
            lobe=Lobe.OCCIPITAL,
            hemisphere=Hemisphere.BILATERAL,
            llm_provider=llm_provider,
        )
        self._vision_model = vision_model

    async def process(self, signal: Signal) -> Signal | None:
        image_data = signal.payload.get("image_data")
        if image_data is None:
            return signal

        signal.payload["modality"] = "visual"

        if self.llm_provider is not None:
            features = await self._analyze_with_vision_llm(image_data)
        else:
            features = self._basic_features(image_data)

        signal.payload["visual_features"] = features
        if features.get("description") and not signal.payload.get("text"):
            signal.payload["text"] = features["description"]

        self.emit_activation(0.8 if self.llm_provider else 0.5)
        return signal

    async def _analyze_with_vision_llm(self, image_data: bytes | str) -> dict:
        """Call vision LLM to analyze image."""
        try:
            if isinstance(image_data, (bytes, bytearray)):
                b64 = base64.b64encode(image_data).decode()
                image_url = f"data:image/jpeg;base64,{b64}"
            else:
                image_url = image_data

            messages = [
                {"role": "system", "content": _VISION_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Analyze this image."},
                ]},
            ]

            response = await self.llm_provider.chat(
                messages,
                model=self._vision_model or None,
                max_tokens=500,
                temperature=0.1,
            )

            if response.content:
                text = response.content.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    text = "\n".join(lines).strip()
                return json.loads(text)
        except Exception as e:
            logger.warning("V1 vision LLM failed: %s", e)

        return self._basic_features(image_data)

    @staticmethod
    def _basic_features(image_data) -> dict:
        features = {}
        if isinstance(image_data, (bytes, bytearray)):
            features["size_bytes"] = len(image_data)
            features["description"] = f"Image ({len(image_data)} bytes)"
        return features
