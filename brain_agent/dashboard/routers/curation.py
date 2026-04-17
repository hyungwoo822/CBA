"""Question and contradiction curation endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel


class AnswerBody(BaseModel):
    answer: str
    answer_source: str = "user"


class ResolveBody(BaseModel):
    resolution: str
    resolved_by: str = "user"
    resolution_confidence: str = "EXTRACTED"


def build_router(state: dict) -> APIRouter:
    router = APIRouter(tags=["curation"])

    def _agent():
        agent = state.get("agent")
        if agent is None or not getattr(agent, "_initialized", False):
            raise HTTPException(503, "agent not initialized")
        return agent

    @router.get("/api/questions/{workspace_id}")
    async def list_questions(
        workspace_id: str,
        severity: str | None = Query(None),
    ):
        agent = _agent()
        if severity:
            items = [
                item
                for item in await agent.memory.open_questions.list_by_severity(
                    workspace_id,
                    severity,
                )
                if item.get("answered_at") is None
            ]
        else:
            items = await agent.memory.open_questions.list_unanswered(workspace_id)
        return {"questions": items}

    @router.post("/api/questions/{question_id}/answer")
    async def answer_question(question_id: str, body: AnswerBody):
        agent = _agent()
        try:
            question = await agent.memory.open_questions.answer_question(
                question_id,
                answer=body.answer,
                answer_source=body.answer_source,
            )
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        from brain_agent.dashboard.emitter import DashboardEmitter

        await DashboardEmitter().question_answered(
            question_id=question_id,
            workspace_id=question["workspace_id"],
        )
        return {"status": "answered", "question": question}

    @router.get("/api/contradictions/{workspace_id}")
    async def list_contradictions(
        workspace_id: str,
        severity: str | None = Query(None),
    ):
        agent = _agent()
        if severity:
            items = [
                item
                for item in await agent.memory.contradictions.list_by_severity(
                    workspace_id,
                    severity,
                )
                if item.get("status") == "open"
            ]
        else:
            items = await agent.memory.contradictions.list_open(workspace_id)
        return {"contradictions": items}

    @router.post("/api/contradictions/{contradiction_id}/resolve")
    async def resolve_contradiction(contradiction_id: str, body: ResolveBody):
        agent = _agent()
        try:
            contradiction = await agent.memory.contradictions.resolve(
                contradiction_id,
                resolution=body.resolution,
                resolved_by=body.resolved_by,
                resolution_confidence=body.resolution_confidence,
            )
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        from brain_agent.dashboard.emitter import DashboardEmitter

        await DashboardEmitter().contradiction_resolved(
            contradiction_id=contradiction_id,
            resolution=body.resolution,
            workspace_id=contradiction["workspace_id"],
        )
        return {"status": "resolved", "contradiction": contradiction}

    @router.post("/api/contradictions/{contradiction_id}/dismiss")
    async def dismiss_contradiction(contradiction_id: str):
        agent = _agent()
        try:
            contradiction = await agent.memory.contradictions.dismiss(contradiction_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"status": "dismissed", "contradiction": contradiction}

    return router
