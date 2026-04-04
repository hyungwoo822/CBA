"""Sylvian Parietal-Temporal area (Spt) — Auditory-motor interface.

Brain mapping: Left Sylvian fissure at the parietal-temporal boundary.
Dorsal auditory stream node connecting comprehension to production.

AI function: Bridges Wernicke's comprehension output to Broca's production
input. Generates a "production plan" that specifies HOW the response should
be formulated (register, emphasis, structure) based on input analysis.

References:
  - Hickok & Poeppel (2007): Spt as the auditory-motor interface in
    the dorsal stream — maps acoustic/phonological representations to
    articulatory motor plans.
  - Hickok et al. (2003): Spt activation during speech production planning.
  - Buchsbaum et al. (2011): Spt role in verbal working memory and
    sensorimotor integration.
"""
from __future__ import annotations

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal


class SylvianParietalTemporal(BrainRegion):
    """Spt — auditory-motor interface (dorsal stream).

    Transforms comprehension data into a production plan:
    - register: formal/informal/technical based on input style
    - structure: paragraph/list/code based on content type
    - emphasis: key points to highlight
    - language: match input language

    This is the bridge between understanding (Wernicke) and speaking (Broca).
    """

    def __init__(self) -> None:
        super().__init__(
            name="spt",
            position=Vec3(-10, -5, 8),
            lobe=Lobe.PARIETAL,
            hemisphere=Hemisphere.LEFT,
        )

    async def process(self, signal: Signal) -> Signal | None:
        comprehension = signal.payload.get("comprehension", {})
        emotional_tag = signal.emotional_tag

        # Build production plan (articulatory mapping)
        plan = self._build_production_plan(comprehension, emotional_tag)
        signal.metadata["production_plan"] = plan

        self.emit_activation(0.5)
        return signal

    @staticmethod
    def _build_production_plan(
        comprehension: dict,
        emotional_tag=None,
    ) -> dict:
        """Map comprehension to articulatory plan.

        Hickok & Poeppel (2007): Spt maps acoustic targets to the motor
        actions needed to achieve them — here we map the user's communicative
        intent to the response strategy.
        """
        intent = comprehension.get("intent", "statement")
        complexity = comprehension.get("complexity", "simple")
        keywords = comprehension.get("keywords", [])
        language = comprehension.get("language", "en")
        discourse_type = comprehension.get("discourse_type")

        # Register selection (how formal/informal to respond)
        register = "neutral"
        if intent in ("greeting", "emotional_expression"):
            register = "informal"
        elif intent == "command" or discourse_type == "technical":
            register = "technical"
        elif complexity == "complex" or discourse_type == "argument":
            register = "formal"

        # Structure selection (how to organize the response)
        structure = "paragraph"
        if discourse_type == "technical" or intent == "command":
            structure = "structured"  # May use lists, code blocks
        elif intent == "question" and complexity == "simple":
            structure = "concise"  # Short direct answer

        # Emphasis points (what to highlight)
        emphasis = keywords[:3] if keywords else []

        # Emotional modulation for tone
        tone = "neutral"
        if emotional_tag:
            if emotional_tag.arousal > 0.6:
                tone = "urgent" if emotional_tag.valence < -0.2 else "energetic"
            elif emotional_tag.valence > 0.3:
                tone = "warm"
            elif emotional_tag.valence < -0.3:
                tone = "empathetic"

        return {
            "register": register,
            "structure": structure,
            "emphasis": emphasis,
            "tone": tone,
            "language": language,
            "response_intent": _map_response_intent(intent),
        }


def _map_response_intent(user_intent: str) -> str:
    """Map user's intent to the expected response type.

    Questions need answers, commands need confirmations/results,
    greetings need reciprocation, etc.
    """
    mapping = {
        "question": "answer",
        "command": "confirmation_and_result",
        "request": "confirmation_and_result",
        "inform": "acknowledgment",
        "greeting": "reciprocate",
        "emotional_expression": "empathize",
        "statement": "engage",
    }
    return mapping.get(user_intent, "engage")
