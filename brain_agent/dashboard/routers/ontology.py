"""Ontology curation endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class ApproveBody(BaseModel):
    approved_by: str = "user"


def build_router(state: dict) -> APIRouter:
    router = APIRouter(prefix="/api/ontology", tags=["ontology"])

    def _agent():
        agent = state.get("agent")
        if agent is None or not getattr(agent, "_initialized", False):
            raise HTTPException(503, "agent not initialized")
        return agent

    @router.get("/{workspace_id}/types")
    async def list_types(workspace_id: str):
        agent = _agent()
        return {
            "node_types": await agent.memory.ontology.get_node_types(workspace_id),
            "relation_types": await agent.memory.ontology.get_relation_types(workspace_id),
        }

    @router.get("/{workspace_id}/proposals")
    async def list_proposals(workspace_id: str):
        agent = _agent()
        return {
            "proposals": await agent.memory.ontology.list_pending(
                workspace_id=workspace_id,
            )
        }

    @router.post("/proposals/{proposal_id}/approve")
    async def approve_proposal(proposal_id: str, body: ApproveBody | None = None):
        agent = _agent()
        try:
            proposal = await agent.memory.ontology.approve_proposal(
                proposal_id,
                approved_by=(body.approved_by if body else "user"),
            )
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        from brain_agent.dashboard.emitter import DashboardEmitter

        await DashboardEmitter().proposal_decided(
            proposal_id=proposal_id,
            status="approved",
            workspace_id=proposal["workspace_id"],
        )
        return {"status": "approved", "proposal": proposal}

    @router.post("/proposals/{proposal_id}/reject")
    async def reject_proposal(proposal_id: str):
        agent = _agent()
        try:
            proposal = await agent.memory.ontology.reject_proposal(proposal_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        from brain_agent.dashboard.emitter import DashboardEmitter

        await DashboardEmitter().proposal_decided(
            proposal_id=proposal_id,
            status="rejected",
            workspace_id=proposal["workspace_id"],
        )
        return {"status": "rejected", "proposal": proposal}

    return router
