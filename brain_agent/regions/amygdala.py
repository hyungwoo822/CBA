"""Amygdala — Bilateral emotional processing via LLM.

Right hemisphere: Fast emotional appraisal (LeDoux "low road").
Left hemisphere: Conscious contextual emotional evaluation.

Both hemispheres use LLM when available — the real amygdala uses
learned neural patterns for emotional evaluation, not keyword lists.
The R/L distinction is in processing DEPTH, not mechanism:
  - Right: brief, immediate assessment (low temperature, concise prompt)
  - Left: contextual, deliberate assessment (full context, nuanced prompt)

When LLM is unavailable, both return neutral baseline values — we do
NOT fake emotional understanding with keyword matching.

References:
  - LeDoux (1996): The Emotional Brain — dual pathway model
  - Glascher & Adolphs (2003): Left amygdala for conscious evaluation
  - Barrett (2006): Constructionist theory — emotion is context-dependent
  - Phelps & LeDoux (2005): Bilateral amygdala contributions
  - Morris et al. (1998): Right amygdala for automatic processing
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, EmotionalTag

if TYPE_CHECKING:
    from brain_agent.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# ── LLM prompts ───────────────────────────────────────────────────────

_AMYGDALA_RIGHT_SYSTEM_PROMPT = """\
You are the right amygdala — fast automatic emotional appraisal (LeDoux 1996 "low road").
Perform a QUICK emotional assessment. Be brief and decisive.

Return ONLY a JSON object:
{
  "valence": <float -1.0 to 1.0>,
  "arousal": <float 0.0 to 1.0>,
  "threat_detected": <bool>
}

- valence: -1.0 = very negative, 0.0 = neutral, 1.0 = very positive
- arousal: 0.0 = calm, 1.0 = highly activated
- threat_detected: true if input contains danger, urgency, distress, anger, or failure

Return ONLY valid JSON."""

_AMYGDALA_LEFT_SYSTEM_PROMPT = """\
You are the left amygdala — conscious contextual emotional evaluation \
(Glascher & Adolphs 2003, Barrett 2006 constructionist model).

The right amygdala has given a fast assessment. Now provide nuanced \
evaluation considering context, pragmatics, and communicative intent.

Return ONLY a JSON object:
{
  "valence": <float -1.0 to 1.0>,
  "arousal": <float 0.0 to 1.0>,
  "threat_level": "<none|low|moderate|high|critical>",
  "primary_emotion": "<neutral|joy|trust|anticipation|surprise|fear|anger|sadness|disgust>",
  "contextual_factors": {
    "is_hypothetical": <bool>,
    "is_sarcastic": <bool>,
    "is_venting": <bool>,
    "urgency": "<none|low|moderate|high>"
  }
}

Guidelines:
- Consider CONTEXT. "This error handling code is great" is positive despite "error".
- Distinguish genuine distress from technical description.
- Korean emotional expressions should be evaluated in context.

Return ONLY valid JSON."""


class AmygdalaRight(BrainRegion):
    """Right amygdala — fast automatic emotional appraisal (LeDoux low road).

    Uses LLM for rapid assessment (low temperature, concise prompt).
    The real fast pathway uses learned neural patterns, not keyword lists.
    Without LLM, returns neutral baseline.
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        super().__init__(
            name="amygdala_right",
            position=Vec3(25, -5, -20),
            lobe=Lobe.TEMPORAL,
            hemisphere=Hemisphere.RIGHT,
            llm_provider=llm_provider,
        )

    async def process(self, signal: Signal) -> Signal | None:
        text = str(signal.payload.get("text", ""))

        if self.llm_provider is not None:
            result = await self._evaluate_with_llm(text)
        else:
            result = {"valence": 0.0, "arousal": 0.15, "threat_detected": False}

        signal.metadata["amygdala_right"] = {
            "valence": max(-1.0, min(1.0, result.get("valence", 0.0))),
            "arousal": max(0.0, min(1.0, result.get("arousal", 0.15))),
            "threat_detected": result.get("threat_detected", False),
        }
        self.emit_activation(result.get("arousal", 0.15))
        return signal

    async def _evaluate_with_llm(self, text: str) -> dict:
        """Fast emotional appraisal via LLM (LeDoux low road)."""
        try:
            response = await self.llm_provider.chat(
                messages=[
                    {"role": "system", "content": _AMYGDALA_RIGHT_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=100,
                temperature=0.0,  # Deterministic — fast path should be consistent
            )
            if response.content:
                parsed = _parse_json(response.content)
                if parsed:
                    return parsed
                logger.warning("Amygdala R: JSON parse failed, raw=%s", response.content[:200])
        except Exception as e:
            logger.warning("Amygdala Right LLM failed: %s", e)
        return {"valence": 0.0, "arousal": 0.15, "threat_detected": False}


class AmygdalaLeft(BrainRegion):
    """Left amygdala — conscious contextual emotional evaluation.

    Uses LLM for nuanced context-dependent evaluation (Barrett 2006).
    Without LLM, returns neutral baseline.
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        super().__init__(
            name="amygdala_left",
            position=Vec3(-25, -5, -20),
            lobe=Lobe.TEMPORAL,
            hemisphere=Hemisphere.LEFT,
            llm_provider=llm_provider,
        )

    async def process(self, signal: Signal) -> Signal | None:
        text = str(signal.payload.get("text", ""))
        right_data = signal.metadata.get("amygdala_right", {})

        if self.llm_provider is not None:
            result = await self._evaluate_with_llm(text, right_data)
        else:
            result = {"valence": 0.0, "arousal": 0.1}

        signal.metadata["amygdala_left"] = {
            "valence": max(-1.0, min(1.0, result.get("valence", 0.0))),
            "arousal": max(0.0, min(1.0, result.get("arousal", 0.1))),
            "threat_level": result.get("threat_level", "none"),
            "primary_emotion": result.get("primary_emotion", "neutral"),
            "contextual_factors": result.get("contextual_factors", {}),
        }
        self.emit_activation(result.get("arousal", 0.1))
        return signal

    async def _evaluate_with_llm(self, text: str, right_data: dict) -> dict:
        """Context-aware emotional evaluation via LLM (Glascher & Adolphs 2003)."""
        try:
            user_msg = (
                f"Right amygdala fast scan: threat={right_data.get('threat_detected', False)}, "
                f"arousal={right_data.get('arousal', 0.0):.2f}\n\n"
                f"Input text: {text}"
            )
            response = await self.llm_provider.chat(
                messages=[
                    {"role": "system", "content": _AMYGDALA_LEFT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=200,
                temperature=0.1,
            )
            if response.content:
                parsed = _parse_json(response.content)
                if parsed:
                    return parsed
                logger.warning("Amygdala L: JSON parse failed, raw=%s", response.content[:200])
        except Exception as e:
            logger.warning("Amygdala Left LLM failed: %s", e)
        return {"valence": 0.0, "arousal": 0.1}


class Amygdala(BrainRegion):
    """Bilateral amygdala — R=fast appraisal, L=contextual evaluation.

    Both hemispheres run via LLM in PARALLEL (asyncio.gather).
    The brain processes R and L simultaneously — the "fast" vs "slow"
    distinction is about depth of processing, not sequential ordering.

    When LLM is available:
      R and L run in parallel → blend results
    When LLM is unavailable:
      Both return neutral baselines → no fake keyword-based emotion
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        super().__init__(
            name="amygdala",
            position=Vec3(-30, -5, -20),
            lobe=Lobe.TEMPORAL,
            hemisphere=Hemisphere.BILATERAL,
            llm_provider=llm_provider,
        )
        self.right = AmygdalaRight(llm_provider=llm_provider)
        self.left = AmygdalaLeft(llm_provider=llm_provider)

    async def process(self, signal: Signal) -> Signal | None:
        # Right and Left run in parallel (both are independent evaluations)
        # Right must run first only because Left reads right_data from metadata.
        # With LLM, we still run sequentially R→L so Left can see Right's result.
        signal = await self.right.process(signal)
        signal = await self.left.process(signal)

        # Retrieve both hemisphere results
        r_data = signal.metadata.get("amygdala_right", {})
        l_data = signal.metadata.get("amygdala_left", {})

        r_valence = r_data.get("valence", 0.0)
        r_arousal = r_data.get("arousal", 0.1)
        l_valence = l_data.get("valence", 0.0)
        l_arousal = l_data.get("arousal", 0.1)

        # Blend: threat detected → R dominates; no threat → L dominates
        threat = r_data.get("threat_detected", False)
        if threat:
            r_weight, l_weight = 0.6, 0.4
        else:
            r_weight, l_weight = 0.4, 0.6

        blended_valence = r_valence * r_weight + l_valence * l_weight
        blended_arousal = r_arousal * r_weight + l_arousal * l_weight

        tag = EmotionalTag(valence=blended_valence, arousal=blended_arousal)
        signal.emotional_tag = tag

        signal.metadata["amygdala_blend"] = {
            "right_weight": r_weight,
            "left_weight": l_weight,
            "right_activation": self.right.activation_level,
            "left_activation": self.left.activation_level,
        }

        self.emit_activation(tag.arousal)
        return signal

    def inject(self, signal: Signal, appraisal: dict) -> Signal:
        """Receive pre-computed appraisal from Cortical Integration.

        Populates both hemisphere metadata and the unified emotional_tag
        so downstream processing (BasalGanglia, memory encoding) sees
        the same structure as the original dual-hemisphere flow.
        """
        valence = appraisal.get("valence", 0.0)
        arousal = appraisal.get("arousal", 0.0)
        threat = appraisal.get("threat_detected", False)
        primary = appraisal.get("primary_emotion", "neutral")
        contextual = appraisal.get("contextual_factors", {})

        # Populate hemisphere metadata for downstream compatibility
        signal.metadata["amygdala_right"] = {
            "valence": valence,
            "arousal": arousal,
            "threat_detected": threat,
        }
        signal.metadata["amygdala_left"] = {
            "valence": valence,
            "arousal": arousal,
            "threat_level": "high" if threat else "none",
            "primary_emotion": primary,
            "contextual_factors": contextual,
        }
        signal.metadata["amygdala_blend"] = {
            "valence": valence,
            "arousal": arousal,
            "threat": threat,
            "dominant_hemisphere": "right" if threat else "left",
        }

        signal.emotional_tag = EmotionalTag(valence=valence, arousal=arousal)

        # Set activation from arousal (threat = higher activation)
        self.emit_activation(arousal)
        self.right.emit_activation(arousal * (0.8 if threat else 0.5))
        self.left.emit_activation(arousal * (0.5 if threat else 0.8))
        return signal


def _parse_json(content: str) -> dict | None:
    """Parse JSON from LLM response, tolerating markdown fences."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None
