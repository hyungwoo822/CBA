"""Stage 3: validate extracted facts before persistence."""
from __future__ import annotations

import logging
from difflib import SequenceMatcher

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.extraction.types import ValidationResult

logger = logging.getLogger(__name__)

_ALIAS_SIMILARITY_THRESHOLD = 0.8


def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


class Validator:
    def __init__(self, semantic_store, ontology_store, config: ExtractionConfig):
        self._semantic = semantic_store
        self._ontology = ontology_store
        self._cfg = config

    async def validate(
        self,
        nodes: list[dict],
        edges: list[dict],
        workspace_id: str,
        narrative_chunk: str,
        input_kinds: list[str],
    ) -> ValidationResult:
        contradictions: list[dict] = []
        severe_questions: list[dict] = []
        moderate_questions: list[dict] = []

        for edge in edges:
            if edge.get("temporal_ambiguous"):
                moderate_questions.append(
                    {
                        "question": (
                            f"Is {edge.get('source')} {edge.get('relation')} "
                            f"{edge.get('target')} a current update or a separate fact?"
                        ),
                        "raised_by": "ambiguity_detector",
                        "severity": "moderate",
                        "context_input": narrative_chunk,
                        "context_node": edge.get("source", ""),
                    }
                )

            existing_edges = await self._semantic.get_relationships(
                edge["source"],
                workspace_id=workspace_id,
            )
            for existing in existing_edges:
                if existing.get("valid_to") is not None:
                    continue
                if existing.get("source") != edge.get("source"):
                    continue
                if existing.get("relation") != edge.get("relation"):
                    continue
                if existing.get("target") == edge.get("target"):
                    continue
                if (
                    _fuzzy_ratio(existing.get("target", ""), edge.get("target", ""))
                    >= _ALIAS_SIMILARITY_THRESHOLD
                ):
                    continue

                subject_is_hub = await self._is_hub_node(edge["source"], workspace_id)
                severity = self._compute_severity(
                    new_confidence=edge.get("confidence", "EXTRACTED"),
                    existing_confidence=existing.get("confidence", "EXTRACTED"),
                    subject_is_hub=subject_is_hub,
                )
                contradictions.append(
                    {
                        "subject": edge["source"],
                        "key": edge["relation"],
                        "value_a": existing.get("target"),
                        "value_b": edge["target"],
                        "severity": severity,
                        "existing_edge_id": existing.get("id"),
                        "value_a_confidence": existing.get("confidence", "EXTRACTED"),
                        "value_b_confidence": edge.get("confidence", "EXTRACTED"),
                    }
                )

        for node in nodes:
            schema = await self._get_node_schema(workspace_id, node)
            required = schema.get("required") or []
            for prop in required:
                if prop in (node.get("properties") or {}):
                    continue
                moderate_questions.append(
                    {
                        "question": f"What is the {prop} for {node.get('label', node.get('type'))}?",
                        "raised_by": "ambiguity_detector",
                        "severity": "moderate",
                        "context_input": narrative_chunk,
                        "context_node": node.get("label", ""),
                    }
                )

        if hasattr(self._semantic, "find_events_near"):
            for node in nodes:
                if (node.get("type") or "").lower() != "event":
                    continue
                happened_at = (node.get("properties") or {}).get("happened_at")
                if not happened_at:
                    continue
                candidates = await self._semantic.find_events_near(
                    workspace_id,
                    happened_at,
                    window_hours=self._cfg.pattern_separation_window_hours,
                )
                for candidate in candidates:
                    if candidate.get("id") == node.get("id"):
                        continue
                    similarity = _fuzzy_ratio(node.get("label", ""), candidate.get("label", ""))
                    if similarity > self._cfg.pattern_separation_label_similarity_threshold:
                        moderate_questions.append(
                            {
                                "question": (
                                    f"Are {node.get('label')} and "
                                    f"{candidate.get('label')} the same event?"
                                ),
                                "raised_by": "pattern_separation",
                                "severity": "moderate",
                                "context_input": narrative_chunk,
                                "context_node": node.get("label", ""),
                            }
                        )
                        break

        if "question" in input_kinds:
            try:
                hits = await self._semantic.search(
                    narrative_chunk,
                    workspace_id=workspace_id,
                    top_k=5,
                )
            except Exception:
                hits = []
            max_similarity = max((_hit_similarity(hit) for hit in hits), default=0.0)
            if max_similarity < self._cfg.fok_similarity_threshold:
                moderate_questions.append(
                    {
                        "question": (
                            f"I do not have enough stored knowledge to answer: "
                            f"{narrative_chunk[:80]}"
                        ),
                        "raised_by": "fok_pre_retrieval",
                        "severity": "moderate",
                        "context_input": narrative_chunk,
                    }
                )

        capped: list[dict] = []
        cap = self._cfg.max_open_questions_per_extraction
        for question in severe_questions:
            if len(capped) < cap:
                capped.append(question)
        for question in moderate_questions:
            if len(capped) < cap:
                capped.append(question)

        dropped = (len(severe_questions) + len(moderate_questions)) - len(capped)
        if dropped > 0:
            logger.debug("Validator dropped %d open questions past cap", dropped)

        return ValidationResult(contradictions=contradictions, open_questions=capped)

    async def _get_node_schema(self, workspace_id: str, node: dict) -> dict:
        if hasattr(self._ontology, "get_node_schema"):
            try:
                schema = await self._ontology.get_node_schema(workspace_id, node.get("type"))
                return schema or {"required": []}
            except Exception:
                return {"required": []}

        if hasattr(self._ontology, "resolve_node_type"):
            try:
                resolved = await self._ontology.resolve_node_type(workspace_id, node.get("type"))
                if resolved:
                    return resolved.get("schema") or {"required": []}
            except Exception:
                return {"required": []}

        return {"required": []}

    async def _is_hub_node(self, node: str, workspace_id: str) -> bool:
        if not hasattr(self._semantic, "is_hub_node"):
            return False
        try:
            return bool(await self._semantic.is_hub_node(node, workspace_id))
        except Exception:
            return False

    @staticmethod
    def _compute_severity(
        new_confidence: str,
        existing_confidence: str,
        subject_is_hub: bool,
    ) -> str:
        high = {"EXTRACTED"}
        if new_confidence in high and existing_confidence in high and subject_is_hub:
            return "severe"
        if new_confidence not in high or existing_confidence not in high:
            return "minor"
        return "moderate"


def _hit_similarity(hit: dict) -> float:
    if "similarity" in hit:
        return float(hit.get("similarity") or 0.0)
    if "distance" in hit:
        return max(0.0, 1.0 - float(hit.get("distance") or 1.0))
    return 0.0
