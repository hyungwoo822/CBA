"""Broca's Area — Language production (phonological assembly + syntactic planning).

Brain mapping: Inferior frontal gyrus, left hemisphere (BA 44/45).

AI function: Takes PFC's cognitive output and produces well-formed natural
language via LLM-based formulation, or applies basic formatting as fallback.

References:
  - Levelt (1989): Speaking: From Intention to Articulation
    (conceptualization -> formulation -> articulation)
  - Hickok & Poeppel (2007): Dorsal stream terminates at IFG (Broca's area)
    for articulatory-motor planning
  - Friederici (2011): Broca's area in syntactic processing
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType

if TYPE_CHECKING:
    from brain_agent.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# ── LLM prompt for Broca language production ───────────────────────────

_BROCA_SYSTEM_PROMPT = """\
You are Broca's area — the language production module in a neural agent's brain.
Your role follows Levelt's (1989) speech production model: formulation and articulation.

You receive the cognitive output from the prefrontal cortex (PFC) and must formulate \
it into natural, authentic language that reflects this brain's identity and emotional state.

This is NOT a generic chatbot. This brain has a growing identity, memories, and a \
real relationship with its user. Your output should feel like a person speaking — \
not an AI generating text.

Instructions:
1. **Voice authenticity**: This brain has personality. It's warm, curious, self-aware. \
Avoid generic assistant phrases ("I'd be happy to help", "Certainly!", "Great question!"). \
Speak naturally, like someone who knows the user.
2. **Register adaptation**: Match the user's style naturally \
(casual → casual, formal → formal, Korean → Korean, mixed → mixed).
3. **Emotional tone modulation**: The emotional context from the amygdala tells you \
how to color the response — warmth, concern, excitement, empathy. Use it.
4. **Syntactic planning**: Structure clearly — but conversationally. This isn't a report.
5. **Conciseness**: Don't pad. Preserve all substance from PFC. Cut filler.
6. **Language**: Respond in the same language as input. Korean → Korean. English → English.
7. **Memory awareness**: If the PFC references shared history or user context, \
weave it in naturally — like remembering, not reciting.

Output ONLY the final response text. No meta-commentary, no "Here's my response:"."""


class BrocaArea(BrainRegion):
    """Broca's area — language production and response formulation.

    Performs Levelt's (1989) formulation stage: transforms PFC's cognitive
    output into well-formed natural language. Uses LLM when available
    for register adaptation, emotional tone modulation, and syntactic
    planning. Falls back to basic formatting otherwise.

    In the dorsal stream model (Hickok & Poeppel 2007), Broca's area is
    the endpoint where articulatory plans are assembled.
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        super().__init__(
            name="broca_area",
            position=Vec3(-30, 40, 15),
            lobe=Lobe.FRONTAL,
            hemisphere=Hemisphere.LEFT,
            llm_provider=llm_provider,
        )

    async def process(self, signal: Signal) -> Signal | None:
        if signal.type not in (SignalType.ACTION_SELECTED, SignalType.PLAN):
            return signal

        if self.llm_provider is not None:
            await self._produce_with_llm(signal)
        else:
            self._format_heuristic(signal)

        self.emit_activation(0.7 if self.llm_provider else 0.6)
        return signal

    def inject_refined(self, signal: Signal, refined_text: str | None) -> None:
        """Receive refined response from Post-Synaptic Consolidation."""
        if refined_text is None:
            self.emit_activation(0.3)
            return

        actions = signal.payload.get("actions", [])
        for action in actions:
            args = action.get("args", {})
            if "text" in args:
                args["text"] = self._clean_text(refined_text)
                break

        response_text = signal.payload.get("response_text")
        if response_text is not None:
            signal.payload["response_text"] = self._clean_text(refined_text)

        self.emit_activation(0.7)

    # ── LLM path (Levelt 1989: formulation stage) ─────────────────────

    async def _produce_with_llm(self, signal: Signal) -> None:
        """LLM-based language production.

        Takes PFC's raw cognitive output and applies:
        - Register adaptation (formal/informal matching)
        - Emotional tone modulation (from Amygdala emotional_tag)
        - Syntactic planning (structure, flow, coherence)
        """
        # Extract PFC's raw output
        pfc_output = self._extract_pfc_output(signal)
        if not pfc_output:
            return

        # Build production context from upstream regions
        comprehension = signal.metadata.get("comprehension", {})
        emotional_tag = signal.emotional_tag
        production_plan = signal.metadata.get("production_plan", {})

        context_parts = [f"PFC cognitive output:\n{pfc_output}"]

        if comprehension:
            intent = comprehension.get("intent", "unknown")
            lang = comprehension.get("language", "en")
            context_parts.append(f"User intent: {intent}, Language: {lang}")

        if emotional_tag:
            context_parts.append(
                f"Emotional context: valence={emotional_tag.valence:.2f}, "
                f"arousal={emotional_tag.arousal:.2f}"
            )

        if production_plan:
            register = production_plan.get("register", "neutral")
            emphasis = production_plan.get("emphasis", [])
            context_parts.append(f"Register: {register}")
            if emphasis:
                context_parts.append(f"Emphasis points: {', '.join(emphasis)}")

        try:
            response = await self.llm_provider.chat(
                messages=[
                    {"role": "system", "content": _BROCA_SYSTEM_PROMPT},
                    {"role": "user", "content": "\n\n".join(context_parts)},
                ],
                max_tokens=2048,
                temperature=0.3,
            )

            if response.content and response.content.strip():
                formatted = response.content.strip()
                self._apply_to_signal(signal, formatted)
                return

        except Exception as e:
            logger.warning("Broca LLM production failed, falling back: %s", e)

        # Fallback to basic formatting
        self._format_heuristic(signal)

    # ── Heuristic path (original formatting logic as fallback) ─────────

    def _format_heuristic(self, signal: Signal) -> None:
        """Basic text cleanup — original Broca behavior as fallback."""
        actions = signal.payload.get("actions", [])
        for action in actions:
            args = action.get("args", {})
            text = args.get("text")
            if text is not None:
                args["text"] = self._clean_text(str(text))

        response_text = signal.payload.get("response_text")
        if response_text is not None:
            signal.payload["response_text"] = self._clean_text(str(response_text))

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_pfc_output(signal: Signal) -> str | None:
        """Extract the raw text output from PFC's plan signal."""
        # Try actions first
        actions = signal.payload.get("actions", [])
        for action in actions:
            text = action.get("args", {}).get("text")
            if text:
                return str(text)
        # Fallback to response_text
        return signal.payload.get("response_text")

    @staticmethod
    def _apply_to_signal(signal: Signal, formatted: str) -> None:
        """Write the formatted output back into the signal."""
        actions = signal.payload.get("actions", [])
        for action in actions:
            args = action.get("args", {})
            if "text" in args:
                args["text"] = formatted
        if "response_text" in signal.payload:
            signal.payload["response_text"] = formatted

    async def format_response(
        self,
        pfc_output: str | None,
        response_mode: str = "normal",
        clarification_questions: list[str] | None = None,
        language: str = "en",
    ) -> str:
        """Unified response formatter for normal and block-mode output."""
        questions = clarification_questions or []
        if response_mode == "block" and questions:
            return self._format_questions(questions, language)
        if pfc_output is None:
            return ""
        return self._clean_text(str(pfc_output))

    def _format_questions(self, questions: list[str], language: str) -> str:
        """Render clarification questions without changing their language."""
        cleaned = [q.strip() for q in questions if q and q.strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        return "\n".join(f"- {q}" for q in cleaned)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic text cleanup (whitespace normalization)."""
        text = text.strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text
