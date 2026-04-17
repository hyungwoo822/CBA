"""Question and contradiction curation API tests."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from brain_agent.agent import BrainAgent
from brain_agent.dashboard.server import create_app, event_bus


@pytest.fixture
async def client(tmp_path):
    agent = BrainAgent(use_mock_embeddings=True, data_dir=str(tmp_path))
    await agent.initialize()
    app = create_app(agent=agent)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as api_client:
        yield api_client, agent
    await agent.close()


async def test_questions_list_empty(client):
    api_client, _ = client
    response = await api_client.get("/api/questions/personal")
    assert response.status_code == 200
    assert response.json()["questions"] == []


async def test_answer_question_emits(client):
    api_client, agent = client
    question = await agent.memory.open_questions.add_question(
        workspace_id="personal",
        question="why?",
        severity="moderate",
        context_input="...",
        raised_by="user",
    )
    event_bus._buffer.clear()
    response = await api_client.post(
        f"/api/questions/{question['id']}/answer",
        json={"answer": "because", "answer_source": "chat"},
    )
    assert response.status_code == 200
    event_types = [event["type"] for event in event_bus.get_recent(10)]
    assert "question_answered" in event_types


async def test_contradictions_list_filter_by_severity(client):
    api_client, _ = client
    response = await api_client.get("/api/contradictions/personal")
    assert response.status_code == 200
    assert response.json()["contradictions"] == []


async def test_resolve_contradiction_emits(client):
    api_client, agent = client
    contradiction = await agent.memory.contradictions.detect(
        workspace_id="personal",
        subject="x",
        key_or_relation="is",
        value_a="a",
        value_b="b",
        value_a_confidence="EXTRACTED",
        value_b_confidence="INFERRED",
    )
    event_bus._buffer.clear()
    response = await api_client.post(
        f"/api/contradictions/{contradiction['id']}/resolve",
        json={
            "resolution": "A",
            "resolved_by": "user",
            "resolution_confidence": "USER_GROUND_TRUTH",
        },
    )
    assert response.status_code == 200
    event_types = [event["type"] for event in event_bus.get_recent(10)]
    assert "contradiction_resolved" in event_types


async def test_dismiss_contradiction(client):
    api_client, agent = client
    contradiction = await agent.memory.contradictions.detect(
        workspace_id="personal",
        subject="y",
        key_or_relation="is",
        value_a="1",
        value_b="2",
    )
    response = await api_client.post(
        f"/api/contradictions/{contradiction['id']}/dismiss"
    )
    assert response.status_code == 200
    assert response.json()["status"] == "dismissed"
