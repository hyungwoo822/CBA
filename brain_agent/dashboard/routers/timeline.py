"""Temporal edge timeline endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query


def build_router(state: dict) -> APIRouter:
    router = APIRouter(prefix="/api/memory", tags=["timeline"])

    def _agent():
        agent = state.get("agent")
        if agent is None or not getattr(agent, "_initialized", False):
            raise HTTPException(503, "agent not initialized")
        return agent

    @router.get("/timeline")
    async def get_timeline(
        workspace_id: str = Query(...),
        subject: str = Query(...),
    ):
        agent = _agent()
        return {
            "subject": subject,
            "workspace_id": workspace_id,
            "chain": await agent.memory.semantic.get_timeline(
                workspace_id=workspace_id,
                subject=subject,
            ),
        }

    return router
