"""Phase 8 end-to-end smoke over the ASGI app."""
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
        yield api_client
    await agent.close()


async def test_full_flow(client):
    response = await client.get("/api/workspaces")
    assert response.status_code == 200

    response = await client.post(
        "/api/workspaces",
        json={"name": "Smoke", "decay_policy": "none"},
    )
    assert response.status_code == 200
    workspace_id = response.json()["id"]

    event_bus._buffer.clear()
    response = await client.put(
        "/api/workspaces/current",
        json={"session_id": "smoke", "workspace_id": workspace_id},
    )
    assert response.status_code == 200
    assert any(event["type"] == "workspace_changed" for event in event_bus.get_recent(10))

    response = await client.get(f"/api/ontology/{workspace_id}/types")
    assert response.status_code == 200

    response = await client.get(f"/api/memory/knowledge-graph?workspace_id={workspace_id}")
    assert response.status_code == 200

    response = await client.post(
        "/api/export/preview",
        json={"workspace_id": workspace_id, "filters": {}},
    )
    assert response.status_code == 200
    assert response.json()["workspace"]["id"] == workspace_id

    response = await client.get("/api/llm/providers")
    assert response.status_code == 200
    assert "default_model" in response.json()
