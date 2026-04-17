"""LLM provider inventory endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from brain_agent.dashboard.providers_inventory import build_inventory


def build_router(state: dict) -> APIRouter:
    router = APIRouter(prefix="/api/llm", tags=["llm"])

    @router.get("/providers")
    async def list_providers():
        agent = state.get("agent")
        if agent is None:
            raise HTTPException(503, "agent not initialized")
        provider = getattr(agent, "_llm_provider", None)
        default_model = (
            provider.get_default_model()
            if provider is not None
            else getattr(getattr(agent, "config", None), "agent", None).model
        )
        return build_inventory(default_model or "auto")

    return router
