"""Stage 2.5: temporal update resolution (C3)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.types import TemporalResolveResult
from brain_agent.providers.base import LLMProvider

logger = logging.getLogger(__name__)

TEMPORAL_MARKERS_CURRENT = ["now", "currently", "today", "지금", "현재", "이제", "오늘"]
TEMPORAL_MARKERS_PAST = ["before", "used to", "previously", "earlier", "과거", "예전", "이전", "옛날"]

_CLASSIFY_PROMPT_TEMPLATE = """Given old_fact and new_fact with the same subject and relation but different targets, classify whether the user is expressing a state change, a contradiction, or uncertainty.

old_fact: {old_fact}
new_fact: {new_fact}
context: {context}

Return one word only: update, contradiction, or ambiguous."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TemporalResolver:
    def __init__(self, semantic_store, llm_provider: LLMProvider, config: ExtractionConfig):
        self._semantic = semantic_store
        self._llm = llm_provider
        self._cfg = config

    async def resolve(self, edges: list[dict], narrative_chunk: str, workspace_id: str) -> TemporalResolveResult:
        result = TemporalResolveResult()
        lowered = narrative_chunk.lower()
        has_temporal_marker = any(marker in lowered for marker in TEMPORAL_MARKERS_CURRENT + TEMPORAL_MARKERS_PAST)

        for edge in edges:
            existing = await self._semantic.get_relationships(edge["source"], workspace_id=workspace_id)
            same_subject_relation = [
                item
                for item in existing
                if item.get("source") == edge.get("source")
                and item.get("relation") == edge.get("relation")
                and item.get("valid_to") is None
            ]

            if not same_subject_relation:
                result.new_edges.append(edge)
                continue

            live_edge = same_subject_relation[0]
            if live_edge.get("target") == edge.get("target"):
                result.reinforced_edges.append(live_edge)
                continue

            if has_temporal_marker:
                result.update_ops.append({"type": "supersede", "edge_id": live_edge["id"], "valid_to": _now_iso()})
                new_edge = dict(edge)
                new_edge["valid_from"] = _now_iso()
                result.new_edges.append(new_edge)
                continue

            judgment = await self._classify(live_edge, edge, narrative_chunk)
            if judgment == "update":
                result.update_ops.append({"type": "supersede", "edge_id": live_edge["id"], "valid_to": _now_iso()})
                new_edge = dict(edge)
                new_edge["valid_from"] = _now_iso()
                result.new_edges.append(new_edge)
            elif judgment == "contradiction":
                result.new_edges.append(edge)
            else:
                tagged = dict(edge)
                tagged["temporal_ambiguous"] = True
                result.new_edges.append(tagged)

        return result

    async def _classify(self, old_fact: dict, new_fact: dict, context: str) -> str:
        prompt = _CLASSIFY_PROMPT_TEMPLATE.format(
            old_fact={"source": old_fact.get("source"), "relation": old_fact.get("relation"), "target": old_fact.get("target")},
            new_fact={"source": new_fact.get("source"), "relation": new_fact.get("relation"), "target": new_fact.get("target")},
            context=context,
        )
        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self._resolve_model(),
            max_tokens=10,
            temperature=0.0,
        )
        word = (response.content or "").strip().lower().strip(".!,?\"' ")
        if word not in {"update", "contradiction", "ambiguous"}:
            logger.debug("Unexpected temporal classification response: %r", response.content)
            return "ambiguous"
        return word

    def _resolve_model(self) -> str:
        if self._cfg.temporal_classify_model == "auto":
            return self._llm.get_default_model()
        return self._cfg.temporal_classify_model
