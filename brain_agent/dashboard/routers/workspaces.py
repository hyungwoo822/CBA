"""Workspace CRUD endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel


class CreateWorkspaceBody(BaseModel):
    name: str
    description: str = ""
    decay_policy: str = "normal"
    template: str | None = None


class UpdateWorkspaceBody(BaseModel):
    name: str | None = None
    description: str | None = None
    decay_policy: str | None = None


class SetCurrentBody(BaseModel):
    session_id: str = "default"
    workspace_id: str


def build_router(state: dict) -> APIRouter:
    router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])

    def _agent():
        agent = state.get("agent")
        if agent is None or not getattr(agent, "_initialized", False):
            raise HTTPException(503, "agent not initialized")
        return agent

    @router.get("")
    async def list_workspaces():
        agent = _agent()
        return {"workspaces": await agent.memory.workspace.list_workspaces()}

    @router.post("")
    async def create_workspace(body: CreateWorkspaceBody):
        agent = _agent()
        try:
            workspace = await agent.memory.workspace.create_workspace(
                name=body.name,
                description=body.description,
                decay_policy=body.decay_policy,
                template_id=body.template,
            )
            if body.template:
                await agent.memory.ontology.apply_template(
                    workspace["id"],
                    body.template,
                    workspace_store=agent.memory.workspace,
                )
                workspace = await agent.memory.workspace.get_workspace(workspace["id"])
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return workspace

    @router.get("/current")
    async def get_current(session_id: str = Query("default")):
        agent = _agent()
        workspace_id = await agent.memory.workspace.get_session_workspace(session_id)
        workspace = await agent.memory.workspace.get_workspace(workspace_id)
        return {"session_id": session_id, "workspace": workspace}

    @router.put("/current")
    async def set_current(body: SetCurrentBody):
        agent = _agent()
        try:
            from brain_agent.dashboard.emitter import DashboardEmitter

            await agent.memory.workspace.set_session_workspace(
                body.session_id,
                body.workspace_id,
                emitter=DashboardEmitter(),
            )
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        workspace = await agent.memory.workspace.get_workspace(body.workspace_id)
        if workspace is None:
            raise HTTPException(404, "workspace not found")
        return {"status": "ok", "workspace": workspace}

    @router.get("/{workspace_id}")
    async def get_workspace(workspace_id: str):
        agent = _agent()
        workspace = await agent.memory.workspace.get_workspace(workspace_id)
        if workspace is None:
            raise HTTPException(404, "workspace not found")
        stats = await _compute_stats(agent, workspace["id"])
        return {**workspace, **stats}

    @router.patch("/{workspace_id}")
    async def update_workspace(workspace_id: str, body: UpdateWorkspaceBody):
        agent = _agent()
        workspace = await agent.memory.workspace.get_workspace(workspace_id)
        if workspace is None:
            raise HTTPException(404, "workspace not found")
        fields = {key: value for key, value in body.model_dump().items() if value is not None}
        try:
            await agent.memory.workspace.update_workspace(workspace["id"], **fields)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        updated = await agent.memory.workspace.get_workspace(workspace["id"])
        return updated

    @router.delete("/{workspace_id}")
    async def delete_workspace(workspace_id: str):
        agent = _agent()
        workspace = await agent.memory.workspace.get_workspace(workspace_id)
        if workspace is None:
            raise HTTPException(404, "workspace not found")
        try:
            await agent.memory.workspace.delete_workspace(workspace["id"])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "ok"}

    return router


async def _compute_stats(agent, workspace_id: str) -> dict:
    node_count = await agent.memory.semantic.count_nodes(workspace_id=workspace_id)
    edge_count = await agent.memory.semantic.count_edges(workspace_id=workspace_id)
    pending_count = len(await agent.memory.ontology.list_pending(workspace_id=workspace_id))
    blocking_count = await agent.memory.open_questions.count_blocking(workspace_id)
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "pending_count": pending_count,
        "blocking_count": blocking_count,
    }
