"""Stage 5: Broca-style response refinement for the personal workspace."""
from __future__ import annotations

import logging

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_REFINE_SYSTEM_PROMPT = (
    "You are the Broca-area response polisher. Polish the draft response for "
    "natural conversational quality while preserving meaning exactly. If no "
    "polishing is needed, return the single word 'null'. Return only the final text."
)


class Refiner:
    def __init__(self, llm_provider: LLMProvider, config: ExtractionConfig):
        self._llm = llm_provider
        self._cfg = config

    async def refine(self, agent_response: str, language: str, workspace: dict | None) -> str:
        if not agent_response:
            return ""
        if not workspace or not (
            workspace.get("id") == "personal" or workspace.get("name") == "Personal Knowledge"
        ):
            return ""

        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _REFINE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Draft response: {agent_response}\nLanguage: {language}"},
                ],
                model=self._resolve_model(),
                max_tokens=500,
                temperature=0.3,
            )
        except Exception as exc:
            logger.warning("Refiner LLM call failed: %s", exc)
            return agent_response

        polished = (response.content or "").strip()
        if not polished or polished.lower() == "null":
            return agent_response
        return polished

    def _resolve_model(self) -> str:
        if self._cfg.refine_model == "auto":
            return self._llm.get_default_model()
        return self._cfg.refine_model
