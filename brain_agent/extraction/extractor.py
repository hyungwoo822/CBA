"""Stage 2: workspace-aware structured extraction."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from brain_agent.extraction.config import ExtractionConfig
from brain_agent.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM_PROMPT_TEMPLATE = """You are a knowledge extractor. Given user input, extract structured facts conforming to the workspace ontology.

## Available Node Types
{node_types_block}

## Available Relation Types
{relation_types_block}

Return ONLY a single JSON object with this schema, no prose and no markdown fences:
{{
  "nodes": [{{"type": "<NodeTypeName>", "label": "<english_lowercase_label>", "properties": {{}}, "confidence": "EXTRACTED | INFERRED | AMBIGUOUS"}}],
  "edges": [{{"source": "<node_label>", "relation": "<relation_name>", "target": "<node_label>", "target_workspace_id": null, "confidence": "EXTRACTED | INFERRED | AMBIGUOUS", "epistemic_source": "asserted | cited | inferred | observed", "importance_score": <float 0..1>, "never_decay": 0}}],
  "new_type_proposals": [{{"kind": "node | relation", "name": "<NewName>", "definition": "<short definition>", "confidence": "EXTRACTED | INFERRED | AMBIGUOUS", "source_snippet": "<snippet from input>"}}],
  "narrative_chunk": "<original text preserved verbatim>"
}}

Rules:
- Use existing types when possible.
- If a concept does not fit, propose it in new_type_proposals.
- Labels must be English lowercase with spaces converted to underscores.
- Every edge must include epistemic_source, importance_score, and never_decay.
- Set never_decay to 1 for business logic, specs, requirements, or decisions that must not be forgotten.
"""

_RETRY_SUFFIX_TEMPLATE = """

The previous output had an error: {error}

Please correct the output and return ONLY valid JSON matching the schema above."""


@dataclass
class _ExtractOutput:
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    new_type_proposals: list[dict] = field(default_factory=list)
    narrative_chunk: str = ""


class Extractor:
    def __init__(self, ontology_store, llm_provider: LLMProvider, config: ExtractionConfig):
        self._ontology = ontology_store
        self._llm = llm_provider
        self._cfg = config

    async def extract(self, text: str, workspace_id: str) -> _ExtractOutput:
        node_types = await self._ontology.get_node_types(workspace_id)
        relation_types = await self._ontology.get_relation_types(workspace_id)
        system_prompt = _EXTRACT_SYSTEM_PROMPT_TEMPLATE.format(
            node_types_block=self._format_node_types(node_types),
            relation_types_block=self._format_relation_types(relation_types),
        )

        known_node_names = {nt["name"] for nt in node_types if nt.get("name")}
        known_rel_names = {rt["name"] for rt in relation_types if rt.get("name")}

        error: str | None = None
        for attempt in range(self._cfg.max_retry + 1):
            user_prompt = f"User input:\n{text}"
            if error is not None:
                user_prompt += _RETRY_SUFFIX_TEMPLATE.format(error=error)

            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self._resolve_model(),
                max_tokens=2000,
                temperature=0.1,
            )
            raw = _strip_markdown_fences((response.content or "").strip())

            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("top-level JSON must be an object")
            except (json.JSONDecodeError, ValueError) as exc:
                error = f"JSON parse failed: {exc}"
                logger.debug("Extractor attempt %s JSON error: %s", attempt, error)
                continue

            violation = self._validate_ontology(data, known_node_names, known_rel_names)
            if violation is not None:
                error = violation
                logger.debug("Extractor attempt %s ontology violation: %s", attempt, error)
                continue

            return _ExtractOutput(
                nodes=list(data.get("nodes", [])),
                edges=[self._apply_edge_defaults(edge) for edge in data.get("edges", [])],
                new_type_proposals=list(data.get("new_type_proposals", [])),
                narrative_chunk=data.get("narrative_chunk") or text,
            )

        logger.warning("Extractor fell back to narrative-only after %s attempts: %s", self._cfg.max_retry + 1, error)
        return _ExtractOutput(narrative_chunk=text)

    def _resolve_model(self) -> str:
        if self._cfg.extract_model == "auto":
            return self._llm.get_default_model()
        return self._cfg.extract_model

    @staticmethod
    def _format_node_types(types: list[dict]) -> str:
        lines = []
        for item in types:
            name = item.get("name")
            if not name:
                continue
            parent = item.get("parent") or item.get("parent_name")
            suffix = f" (parent: {parent})" if parent else ""
            lines.append(f"- {name}{suffix}")
        return "\n".join(lines) if lines else "(none)"

    @staticmethod
    def _format_relation_types(types: list[dict]) -> str:
        lines = []
        for item in types:
            name = item.get("name")
            if not name:
                continue
            desc = item.get("description") or item.get("definition") or ""
            suffix = f" - {desc}" if desc else ""
            lines.append(f"- {name}{suffix}")
        return "\n".join(lines) if lines else "(none)"

    @staticmethod
    def _validate_ontology(data: dict, known_nodes: set[str], known_rels: set[str]) -> str | None:
        proposals = data.get("new_type_proposals", []) or []
        proposed_node_names = {p.get("name") for p in proposals if p.get("kind") in {"node", "node_type"}}
        proposed_rel_names = {p.get("name") for p in proposals if p.get("kind") in {"relation", "relation_type"}}
        proposed_any_names = {p.get("name") for p in proposals}

        for node in data.get("nodes", []) or []:
            node_type = node.get("type")
            if node_type not in known_nodes and node_type not in proposed_node_names:
                return f"Unknown node type '{node_type}'. Add it to new_type_proposals or use one of {sorted(known_nodes)}"

        for edge in data.get("edges", []) or []:
            relation = edge.get("relation")
            if relation not in known_rels and relation not in (proposed_rel_names | proposed_any_names):
                return f"Unknown relation '{relation}'. Add it to new_type_proposals or use one of {sorted(known_rels)}"

        return None

    @staticmethod
    def _apply_edge_defaults(edge: dict) -> dict:
        out = dict(edge)
        out.setdefault("epistemic_source", "asserted")
        out.setdefault("importance_score", 0.5)
        try:
            out["importance_score"] = max(0.0, min(1.0, float(out["importance_score"])))
        except (TypeError, ValueError):
            out["importance_score"] = 0.5
        out.setdefault("never_decay", 0)
        out["never_decay"] = 1 if out.get("never_decay") in {1, True, "1", "true"} else 0
        out.setdefault("target_workspace_id", None)
        return out


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines).strip()
