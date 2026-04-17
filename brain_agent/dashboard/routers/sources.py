"""Raw-vault source endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response


def build_router(state: dict) -> APIRouter:
    router = APIRouter(prefix="/api/sources", tags=["sources"])

    def _agent():
        agent = state.get("agent")
        if agent is None or not getattr(agent, "_initialized", False):
            raise HTTPException(503, "agent not initialized")
        return agent

    @router.get("/{source_id}")
    async def get_source_meta(source_id: str):
        agent = _agent()
        source = await agent.memory.raw_vault.get_source(source_id)
        if source is None:
            raise HTTPException(404, "source not found")
        return source

    @router.get("/{source_id}/raw")
    async def get_source_raw(source_id: str):
        agent = _agent()
        source = await agent.memory.raw_vault.get_source(source_id)
        if source is None:
            raise HTTPException(404, "source not found")
        if not source.get("integrity_valid", 1):
            raise HTTPException(503, "source integrity check failed")
        content = await agent.memory.raw_vault.get_raw_bytes(source_id)
        if content is None:
            raise HTTPException(404, "source bytes not found")
        return Response(
            content=content,
            media_type=source.get("mime_type") or "application/octet-stream",
        )

    @router.get("/{source_id}/text")
    async def get_source_text(source_id: str):
        agent = _agent()
        source = await agent.memory.raw_vault.get_source(source_id)
        if source is None:
            raise HTTPException(404, "source not found")
        return {"text": source.get("extracted_text") or ""}

    return router
