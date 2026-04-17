"""MCP-compatible export preview endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


CONFIDENCE_RANK = {
    "INFERRED": 0,
    "AMBIGUOUS": 0,
    "PROVISIONAL": 0,
    "EXTRACTED": 1,
    "STABLE": 1,
    "CANONICAL": 2,
    "USER_GROUND_TRUTH": 3,
}


class ExportFilters(BaseModel):
    never_decay_only: bool = False
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    min_confidence: str = "PROVISIONAL"
    include_raw_vault: bool = False


class PreviewBody(BaseModel):
    workspace_id: str
    filters: ExportFilters = Field(default_factory=ExportFilters)


def build_router(state: dict) -> APIRouter:
    router = APIRouter(prefix="/api/export", tags=["export"])

    def _agent():
        agent = state.get("agent")
        if agent is None or not getattr(agent, "_initialized", False):
            raise HTTPException(503, "agent not initialized")
        return agent

    @router.post("/preview")
    async def preview_export(body: PreviewBody):
        agent = _agent()
        workspace = await agent.memory.workspace.get_workspace(body.workspace_id)
        if workspace is None:
            raise HTTPException(404, "workspace not found")

        filters = body.filters
        min_rank = CONFIDENCE_RANK.get(filters.min_confidence.upper(), 0)
        facts = []
        raw_refs: dict[str, dict] = {}
        for edge in await agent.memory.semantic.list_edges(body.workspace_id):
            if filters.never_decay_only and not edge.get("never_decay"):
                continue
            if float(edge.get("importance_score") or 0.0) < filters.min_importance:
                continue
            if CONFIDENCE_RANK.get(str(edge.get("confidence", "")).upper(), 0) < min_rank:
                continue

            source_id = edge.get("source_id")
            source = None
            if source_id:
                source = await agent.memory.raw_vault.get_source(source_id)
                if source and filters.include_raw_vault:
                    raw_refs[source_id] = source
            source_summary = None
            if source_id:
                source_summary = {
                    "id": source_id,
                    "kind": source.get("kind") if source else None,
                    "snippet": ((source or {}).get("extracted_text") or "")[:160],
                }
                if source and filters.include_raw_vault:
                    source_summary["content_inline"] = source.get("extracted_text") or ""

            facts.append(
                {
                    "id": edge.get("id"),
                    "subject": edge.get("subject"),
                    "relation": edge.get("relation"),
                    "target": edge.get("target"),
                    "confidence": edge.get("confidence"),
                    "importance_score": edge.get("importance_score"),
                    "never_decay": bool(edge.get("never_decay")),
                    "epistemic_source": edge.get("epistemic_source"),
                    "source": source_summary,
                    "valid_from": edge.get("valid_from"),
                    "valid_to": edge.get("valid_to"),
                }
            )

        node_types = await agent.memory.ontology.get_node_types(body.workspace_id)
        relation_types = await agent.memory.ontology.get_relation_types(body.workspace_id)
        open_questions = await agent.memory.open_questions.list_unanswered(body.workspace_id)
        contradictions = await agent.memory.contradictions.list_open(body.workspace_id)

        return {
            "workspace": {
                "id": workspace["id"],
                "name": workspace["name"],
                "template": workspace.get("template_id"),
                "decay_policy": workspace.get("decay_policy"),
            },
            "ontology": {
                "node_types": [
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "confidence": item.get("confidence"),
                        "workspace_id": item.get("workspace_id"),
                    }
                    for item in node_types
                ],
                "relation_types": [
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "confidence": item.get("confidence"),
                        "workspace_id": item.get("workspace_id"),
                    }
                    for item in relation_types
                ],
            },
            "facts": facts,
            "open_questions": open_questions,
            "unresolved_contradictions": contradictions,
            "raw_vault_refs": list(raw_refs.values()),
        }

    return router
