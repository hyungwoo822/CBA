"""Wernicke's Area — Language comprehension and semantic parsing.

Brain mapping: Posterior superior temporal gyrus, left hemisphere.
Auditory ventral stream endpoint (STG -> STS -> MTG).

AI function: Deep semantic analysis of input text via LLM. The real
Wernicke's area performs lexical-semantic processing using billions of
neurons — keyword matching is not a valid model of this process.

When llm_provider is unavailable, returns a minimal structural parse
(word count, basic punctuation cues) without pretending to understand
intent or semantics.

References:
  - Hickok & Poeppel (2007): Dual-stream model — Wernicke's area is the
    endpoint of the auditory ventral stream ("what" pathway) responsible
    for lexical-semantic processing.
  - Friederici (2011): Temporal cortex in language comprehension.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal

if TYPE_CHECKING:
    from brain_agent.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# ── LLM prompt for Wernicke comprehension ──────────────────────────────

_WERNICKE_SYSTEM_PROMPT = """\
You are the language comprehension module (Wernicke's area) in a brain-inspired AI system.
Your role is to perform deep semantic analysis of the input text, functioning as the \
auditory ventral stream endpoint (Hickok & Poeppel 2007).

Analyze the input and return ONLY a JSON object with these fields:
{
  "intent": "<question|command|request|inform|greeting|emotional_expression|statement>",
  "complexity": "<simple|moderate|complex>",
  "keywords": ["keyword1", "keyword2", ...],
  "semantic_roles": {"agent": "...", "action": "...", "patient": "...", "topic": "..."},
  "discourse_type": "<request|narrative|argument|description|social|technical>",
  "language": "<en|ko|mixed>"
}

Rules:
- intent: The communicative purpose. "question" = seeking info, "command" = requesting action, \
"request" = polite action request, "inform" = conveying info, "greeting" = social opening/closing, \
"emotional_expression" = venting/expressing feelings, "statement" = neutral declaration.
- complexity: "simple" = single clause/short, "moderate" = 2-3 clauses, "complex" = nested/multi-topic.
- keywords: 3-8 content-bearing terms (not stopwords, not particles).
- semantic_roles: Core participants. Use null for absent roles.
- discourse_type: Overall communicative frame.
- language: Primary language detected.

Return ONLY valid JSON. No markdown, no explanation."""


class WernickeArea(BrainRegion):
    """Wernicke's area — language comprehension and semantic parsing.

    Auditory ventral stream endpoint (Hickok & Poeppel 2007).
    Uses LLM for genuine semantic analysis. Without LLM, returns only
    structural metadata (word count, punctuation) — no fake intent
    classification via keyword matching.
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        super().__init__(
            name="wernicke_area",
            position=Vec3(-40, -20, 15),
            lobe=Lobe.TEMPORAL,
            hemisphere=Hemisphere.LEFT,
            llm_provider=llm_provider,
        )

    async def process(self, signal: Signal) -> Signal | None:
        text = signal.payload.get("text")
        if text is None:
            return signal

        text_str = str(text).strip()
        if not text_str:
            return signal

        if self.llm_provider is not None:
            comprehension = await self._comprehend_with_llm(text_str)
        else:
            comprehension = self._structural_parse(text_str)

        signal.payload["comprehension"] = comprehension
        complexity = comprehension.get("complexity", "simple")
        self.emit_activation(0.5 + (0.3 if complexity != "simple" else 0.0))
        return signal

    # ── LLM path (Hickok & Poeppel 2007: ventral stream semantic analysis) ──

    async def _comprehend_with_llm(self, text: str) -> dict:
        """Deep semantic comprehension via LLM.

        The auditory ventral stream performs lexical-semantic processing
        far beyond what keyword matching can achieve — understanding
        pragmatic intent, discourse structure, and semantic roles.
        """
        try:
            response = await self.llm_provider.chat(
                messages=[
                    {"role": "system", "content": _WERNICKE_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=300,
                temperature=0.1,
            )

            if response.content:
                parsed = self._parse_llm_response(response.content)
                if parsed:
                    words = text.split()
                    parsed.setdefault("word_count", len(words))
                    parsed.setdefault("avg_word_length",
                                      round(sum(len(w) for w in words) / max(len(words), 1), 2))
                    return parsed

        except Exception as e:
            logger.warning("Wernicke LLM comprehension failed: %s", e)

        return self._structural_parse(text)

    @staticmethod
    def _parse_llm_response(content: str) -> dict | None:
        """Parse JSON from LLM response, tolerating markdown fences."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "intent" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    # ── Structural parse (no LLM — minimal, honest) ───────────────────

    @staticmethod
    def _structural_parse(text: str) -> dict:
        """Minimal structural parse when LLM is unavailable.

        Returns only what can be determined WITHOUT semantic understanding:
        word count, punctuation cues, basic morphological stats. Does NOT
        attempt intent classification — that requires genuine comprehension.
        """
        words = text.split()
        word_count = len(words)
        avg_word_len = sum(len(w) for w in words) / max(word_count, 1)

        return {
            "intent": "statement",  # Honest default — we can't classify without LLM
            "word_count": word_count,
            "complexity": "complex" if word_count >= 15 else "simple",
            "avg_word_length": round(avg_word_len, 2),
            "keywords": [],  # Cannot extract meaningful keywords without understanding
        }
