"""Ontology dashboard API tests."""
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


async def test_list_types_returns_universal_seed(client):
    api_client, _ = client
    response = await api_client.get("/api/ontology/personal/types")
    assert response.status_code == 200
    body = response.json()
    assert len(body["node_types"]) >= 7
    assert len(body["relation_types"]) >= 10


async def test_list_proposals_empty_initially(client):
    api_client, _ = client
    response = await api_client.get("/api/ontology/personal/proposals")
    assert response.status_code == 200
    assert response.json()["proposals"] == []


async def test_approve_proposal_emits_proposal_decided(client):
    api_client, agent = client
    proposal = await agent.memory.ontology.propose_node_type(
        workspace_id="personal",
        name="CustomType",
        definition={},
        confidence="PROVISIONAL",
        source_input="...",
    )
    event_bus._buffer.clear()
    response = await api_client.post(
        f"/api/ontology/proposals/{proposal['id']}/approve"
    )
    assert response.status_code == 200
    event_types = [event["type"] for event in event_bus.get_recent(10)]
    assert "proposal_decided" in event_types


async def test_reject_proposal_emits(client):
    api_client, agent = client
    proposal = await agent.memory.ontology.propose_node_type(
        workspace_id="personal",
        name="OtherType",
        definition={},
        confidence="PROVISIONAL",
        source_input="...",
    )
    event_bus._buffer.clear()
    response = await api_client.post(
        f"/api/ontology/proposals/{proposal['id']}/reject"
    )
    assert response.status_code == 200
    event = [
        item
        for item in event_bus.get_recent(10)
        if item["type"] == "proposal_decided"
    ][0]
    assert event["payload"]["status"] == "rejected"
